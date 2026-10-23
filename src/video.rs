use anyhow::Result;
use ffmpeg_next::{
    codec, ffi, filter,
    format::Pixel,
    software::scaling::{context::Context as ScaleContext, flag::Flags as ScaleFlags},
    util::color::{Primaries, Range, Space, TransferCharacteristic},
};
use fltk::app;
use log::{error, warn};
use std::{
    sync::{
        Arc,
        atomic::Ordering,
        mpsc::{Receiver, SyncSender},
    },
    time::{Duration, Instant},
};

use crate::{
    clock::MasterClock,
    demux::{SeekCtl, SeekOutcome, TimedPacket, is_backpressure},
    dmabuf::DmaBufFrame,
    frame::{ChromaLayout, Message, YuvFrame, linesize, sar_corrected},
    source::SourceShared,
};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
// UI 侧用有界通道接收：UI 停顿时 send 阻塞产生背压，防止帧无限堆积；
// 每次发送后需 app::awake() 唤醒 fltk 事件循环（替代 fltk channel 的内置唤醒）。
// seek 时：flush 解码器、排空通道、把主时钟基准设到目标点并回报就绪；复位后的
// 首帧再做一次时钟对齐（解码器选中的关键帧可能与请求点有偏差），随后正常同步。
pub fn spawn_video_thread(ctx: VideoThreadCtx) -> std::thread::JoinHandle<()> {
    // 保留一份 tx 副本：出错时用它给 UI 发 End，让本源正常收尾。
    // VideoThread::new 失败时 ctx 被消费丢弃，若无此副本 UI 侧只会看到
    // 通道断开（消息泵不靠断开退出），本源会永远挂着
    let tx = ctx.tx.clone();
    // 线程永久退出时把它从 seek 就绪屏障里移除，否则 demux 会卡在
    // wait_ready 上等一个再也不会 report_ready 的线程
    let ctl = ctx.shared.ctl.clone();
    std::thread::spawn(move || {
        // 非正常退出（new/run 返回 Err 或 panic）才 abandon；正常走到 linger
        // 结束时 demux 也已退出，不需要再处理 seek
        struct AbandonGuard {
            ctl: Arc<SeekCtl>,
            normal: bool,
        }
        impl Drop for AbandonGuard {
            fn drop(&mut self) {
                if !self.normal {
                    self.ctl.abandon_decoder();
                }
            }
        }
        let mut guard = AbandonGuard { ctl, normal: false };

        // 工作线程的错误不会自动上抛，join 句柄又被丢弃：必须在这里就地记录，
        // 否则解码失败（网络中断、文件损坏、seek 复位出错）对用户完全无感——
        // 表现为画面停在最后一帧
        if let Err(e) = VideoThread::new(ctx).and_then(|t| t.run()) {
            error!("video thread exited with error: {e:#}");
            if tx.send(Message::End).is_ok() {
                app::awake();
            }
        } else {
            guard.normal = true;
        }
    })
}

// 视频线程的完整输入：由主循环一次性打包传入，
// 避免散参数顺序/个数错误（原为 10 个参数的 spawn）
pub struct VideoThreadCtx {
    pub tx: SyncSender<Message>,
    pub recycle_rx: Receiver<YuvFrame>,
    pub clock: MasterClock,
    pub shared: Arc<SourceShared>,
    pub video_rx: Receiver<TimedPacket>,
    pub video_parameters: Option<codec::parameters::Parameters>,
    pub video_time_base: Option<ffmpeg_next::Rational>,
    // 本源是否有可用的音频时钟（与 session 的 has_audio 一致）：音频播完后
    // 采样时钟不再前进，同步等待必须立刻放弃（见 process_frame）
    pub has_audio: bool,
    // 容器时间轴原点（秒）：视频 PTS 归一化用，见 demux::timeline_origin
    pub timeline_origin: f64,
}

// 严重迟到时的处理策略：先丢帧，持续落后就发起一次"追帧 seek"（跳到当前
// 音频时钟位置，避免越丢越远）；多次追帧仍无效则改为显示迟到帧（慢放），
// 保证任何情况下都不会永久停在某一帧。
enum LateAction {
    Drop,
    CatchUp,
    Display,
}

struct LateFramePolicy {
    // 连续被丢弃的迟到帧数
    streak: u32,
    // 已经发起的追帧 seek 次数
    catchups: u32,
    // 追帧多次仍不行：放弃丢帧，显示迟到帧（宁可慢放也不冻屏）
    display_late: bool,
    last_catchup: Option<Instant>,
}

impl LateFramePolicy {
    fn new() -> Self {
        Self {
            streak: 0,
            catchups: 0,
            display_late: false,
            last_catchup: None,
        }
    }

    // seek 后重新给一次机会（新的解码位置可能恢复正常）
    fn reset(&mut self) {
        *self = Self::new();
    }

    // 按时到达：清空连续迟到计数
    fn on_time(&mut self) {
        self.streak = 0;
    }

    // late_sec：已经迟到多少秒（正数）
    fn on_late(&mut self, late_sec: f64, now: Instant) -> LateAction {
        const LATE_RECOVER_SEC: f64 = 0.5; // 落后超过 0.5s 才值得追帧
        const LATE_STREAK: u32 = 30; // 连续 30 帧都迟到才动作（约 0.5~1s）
        const CATCHUP_COOLDOWN: Duration = Duration::from_secs(3);
        const MAX_CATCHUPS: u32 = 3;

        self.streak += 1;
        if self.display_late {
            return LateAction::Display;
        }
        let cooled = self
            .last_catchup
            .is_none_or(|t| now.duration_since(t) >= CATCHUP_COOLDOWN);
        if late_sec >= LATE_RECOVER_SEC && self.streak >= LATE_STREAK && cooled {
            if self.catchups < MAX_CATCHUPS {
                self.catchups += 1;
                self.streak = 0;
                self.last_catchup = Some(now);
                return LateAction::CatchUp;
            }
            // 追帧也救不回来（解码持续慢于实时）：不再无限丢帧
            self.display_late = true;
            warn!("video persistently behind; displaying late frames (slow motion)");
            return LateAction::Display;
        }
        LateAction::Drop
    }
}

// 视频解码线程的局部状态：原为 thread::spawn 内的大闭包（process_frame 等），
// 收成结构体后提炼为可独立调用的普通方法，便于阅读与复查。
// 字段与闭包捕获一一对应，职责不变（Cell 换成普通字段，&mut self 即可读写）。
struct VideoThread {
    // 无视频轨（参数缺失）时为 None，解码循环直接跳过
    decoder: Option<codec::decoder::video::Video>,
    time_base: Option<ffmpeg_next::Rational>,
    // 容器时间轴原点（秒），PTS 归一化后才是 UI/时钟使用的 0 基时间
    timeline_origin: f64,
    // 非 YUV420P 输出（10-bit、4:2:2 / 4:4:4 等）经 swscale 统一转成 420P，
    // 渲染端只需处理一种平面布局。转换器惰性创建，格式/尺寸变化时重建。
    scaler: Option<(ScaleContext, (Pixel, u32, u32))>,
    // HDR/10bit：统一转成 YUV420P10LE（保留位深），渲染端 GPU tone map
    scaler10: Option<(ScaleContext, (Pixel, u32, u32))>,
    // HDR(PQ/HLG) → SDR 的滤镜图，惰性创建、按输入特征缓存
    tonemapper: Tonemapper,
    // 转换输出帧池：swscale 输出帧的所有权要交给下游（显示/零拷贝），
    // UI 回收帧后归还到这里，下一轮直接复用 → 避免每帧一次整帧缓冲分配
    conv_pool: Vec<ffmpeg_next::util::frame::video::Video>,
    // 调试/兜底：AUV_CPU_TONEMAP=1 时强制走原 CPU tone map，便于对比 GPU 输出
    force_cpu_tonemap: bool,
    // PTS 缺失（rawvideo 等）时的单调回退：按上帧 +1 个 time_base 步进；
    // seek 复位时清空
    last_pts: Option<i64>,
    // seek 代数与复位后首帧标记
    local_epoch: u64,
    // 可接受的最小 packet 代数：成功定位后抬高到新代数（丢弃 seek 前读出的
    // 旧包）；定位失败时保持不变——读位置没变，在途的旧包仍然有效，丢了会
    // 造成画面跳变
    packet_epoch_floor: u64,
    just_seeked: bool,
    // 严重迟到时的状态机（见 LateFramePolicy）
    late_policy: LateFramePolicy,
    // 是否已让解码器丢弃非参考帧（持续迟到时启用，追上后恢复）
    decoder_skip: bool,
    // 迟到压力：按迟到/按时帧增减，避免连续迟到 streak 被偶发按时帧清零
    late_pressure: u32,
    // 是否已记录过硬解下载（确认解码器真的输出 VAAPI/CUDA 帧，而非静默回退）
    hw_download_logged: bool,
    // 是否已记录过 DMA-BUF 零拷贝路径生效
    dmabuf_logged: bool,
    // DMA-BUF 映射失败（格式不支持）只告警一次
    dmabuf_fallback_logged: bool,
    // 本流的 DRM 格式不支持导入：后续帧直接下载，避免每帧白做一次映射
    dmabuf_unsupported: bool,
    // 打开解码器时请求的硬解后端；None=软件。用于首帧检测 FFmpeg 静默回退
    hw_kind: Option<HwKind>,
    hw_fallback_logged: bool,
    tx: SyncSender<Message>,
    recycle_rx: Receiver<YuvFrame>,
    video_rx: Receiver<TimedPacket>,
    clock: MasterClock,
    shared: Arc<SourceShared>,
    has_audio: bool,
}

impl VideoThread {
    fn new(ctx: VideoThreadCtx) -> Result<Self> {
        let VideoThreadCtx {
            tx,
            recycle_rx,
            clock,
            shared,
            video_rx,
            video_parameters,
            video_time_base,
            timeline_origin,
            has_audio,
        } = ctx;
        let (decoder, hw_kind) = match video_parameters {
            Some(params) => {
                let (d, k) = Self::open_decoder(params)?;
                (Some(d), k)
            }
            None => (None, None),
        };
        Ok(Self {
            decoder,
            time_base: video_time_base,
            timeline_origin,
            scaler: None,
            scaler10: None,
            tonemapper: Tonemapper::new(),
            conv_pool: Vec::new(),
            force_cpu_tonemap: std::env::var_os("AUV_CPU_TONEMAP").is_some_and(|v| v != "0"),
            last_pts: None,
            local_epoch: 0,
            packet_epoch_floor: 0,
            just_seeked: false,
            late_policy: LateFramePolicy::new(),
            decoder_skip: false,
            late_pressure: 0,
            hw_download_logged: false,
            dmabuf_logged: false,
            dmabuf_fallback_logged: false,
            dmabuf_unsupported: false,
            hw_kind,
            hw_fallback_logged: false,
            tx,
            recycle_rx,
            video_rx,
            clock,
            shared,
            has_audio,
        })
    }

    // 打开视频解码器：按 AUV_HWACCEL 依次尝试硬解后端；设备不可用、
    // NVDEC 不支持该流或解码器打开失败时继续下一个候选，最后回退软件解码。
    // VAAPI 帧在渲染端支持时走 EGL/DMABUF 零拷贝；否则由 download_hw_frame
    // 拷回内存再进入现有管线。
    fn open_decoder(
        params: codec::parameters::Parameters,
    ) -> Result<(codec::decoder::video::Video, Option<HwKind>)> {
        let candidates = hwaccel_candidates();
        for (i, kind) in candidates.iter().copied().enumerate() {
            match Self::open_decoder_hw(params.clone(), kind) {
                Ok(decoder) => {
                    log::debug!("{} decoder opened", kind.name());
                    return Ok((decoder, Some(kind)));
                }
                Err(e) => {
                    // 还有下一个候选时说明即将换后端，用 warn 让用户知道为什么
                    // 显式指定的 cuda/vaapi 没有生效；最后一个只记 debug
                    if let Some(next) = candidates.get(i + 1) {
                        warn!("{} unavailable ({e}); trying {}", kind.name(), next.name());
                    } else {
                        log::debug!("{} unavailable: {e}", kind.name());
                    }
                }
            }
        }
        if !candidates.is_empty() {
            warn!("hardware decoding unavailable; using software decoder");
        }
        Ok((Self::open_decoder_software(params)?, None))
    }

    fn open_decoder_software(
        params: codec::parameters::Parameters,
    ) -> Result<codec::decoder::video::Video> {
        let mut ctx = codec::context::Context::from_parameters(params)?;
        // 帧级多线程解码：count=0 表示由 ffmpeg 按 CPU 自动决定。
        // 默认的 Type::None 是单线程，4K 解码会明显吃满一个核且掉帧；
        // 帧级（而非 slice 级）保证输出帧可直接整帧移交下游。
        let mut threading = codec::threading::Config::kind(codec::threading::Type::Frame);
        threading.count = 0;
        ctx.set_threading(threading);
        Ok(ctx.decoder().video()?)
    }

    fn open_decoder_hw(
        params: codec::parameters::Parameters,
        kind: HwKind,
    ) -> Result<codec::decoder::video::Video> {
        use std::ptr;
        // NVDEC 能力预检：通用 CUDA hwaccel 在 NVDEC 不可用的机器上仍能
        // “成功打开”，到首帧才报 "Hardware is lacking required capabilities"，
        // 然后静默退化成软件帧。用同一路参数打开一次 *_cuvid 解码器，
        // 让 FFmpeg 在起播前就执行 cuvidGetDecoderCaps，失败时直接换后端。
        if kind == HwKind::Cuda && matches!(cuvid_capability(&params), CuvidCapability::Unsupported)
        {
            anyhow::bail!("cuvid capability check failed");
        }
        unsafe {
            let mut device: *mut ffi::AVBufferRef = ptr::null_mut();
            let ret = ffi::av_hwdevice_ctx_create(
                &mut device,
                kind.device_type(),
                ptr::null(),
                ptr::null_mut(),
                0,
            );
            if ret < 0 || device.is_null() {
                anyhow::bail!("av_hwdevice_ctx_create({}) failed: {ret}", kind.name());
            }
            let mut ctx = match codec::context::Context::from_parameters(params) {
                Ok(c) => c,
                Err(e) => {
                    ffi::av_buffer_unref(&mut device);
                    return Err(e.into());
                }
            };
            {
                let p = ctx.as_mut_ptr();
                (*p).hw_device_ctx = device;
                (*p).get_format = Some(kind.get_format());
            }
            let mut threading = codec::threading::Config::kind(codec::threading::Type::Frame);
            threading.count = 0;
            ctx.set_threading(threading);
            // hw_device_ctx 的所有权已交给 ctx；打开失败时 ctx 的 Drop 会释放它
            Ok(ctx.decoder().video()?)
        }
    }

    // 主入口：解码循环（含 seek 复位）→ 等音频播完 → 发 End → 驻留
    fn run(mut self) -> Result<()> {
        self.run_decoding()?;
        self.wait_audio_done()?;
        self.linger()?;
        Ok(())
    }

    // 解码循环：收完 demux 的全部 packet 后通道断开，send_eof 把解码器内
    // 残留的帧（如 B 帧延迟）全部取完；无视频轨时直接跳过
    fn run_decoding(&mut self) -> Result<()> {
        if self.decoder.is_none() {
            return Ok(());
        }
        let mut eof = false;
        // 复用的接收目标帧：avcodec_receive_frame 内部先 unref 再重填，
        // 因此跨 packet 复用是安全的（原先每个 packet 新建一个 AVFrame）
        let mut target = self.take_recycled();
        // 待重试的包：send_packet 返回 EAGAIN 表示解码器**没有接收**这个包
        // （输入缓冲满），必须原样重试；直接丢掉等于丢帧（花屏/画面缺块）。
        // 取帧（receive_frame）腾出解码器内部空间后就能收下。
        let mut pending: Option<TimedPacket> = None;
        // EOF 标记同样可能因 EAGAIN 没被接受：必须重试，否则解码器里
        // 残留的帧（B 帧延迟等）永远取不出来，片尾会缺一截
        let mut pending_eof = false;
        while !eof {
            // 本源收尾（播完/关窗/异常）：尽快退出解码循环
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            // 先响应 seek（暂停中被唤醒后也会回到这里）：
            // demux 已给出处理结果——定位成功则本地复位，失败则保持现状继续播
            if let Some(decision) = self.shared.ctl.snapshot_decision(self.local_epoch) {
                self.local_epoch = decision.epoch;
                match decision.outcome {
                    SeekOutcome::Reset => {
                        self.reset_for_seek(decision.target);
                        // 复位后解码器已 flush，seek 前的包不能再发出去。
                        // Failed 分支没有 flush、读位置也没变，pending 包仍有效，
                        // 丢掉等于丢一帧（视频还会连带后续参考帧花屏）
                        pending = None;
                        pending_eof = false;
                    }
                    // 定位失败：读位置没有变化，解码器必须保持现状。
                    // 此时 flush 会清空参考帧，之后每帧都解不出来（画面直接坏掉）
                    SeekOutcome::Failed => {
                        self.shared.ctl.report_ready(decision.epoch);
                    }
                }
                continue;
            }
            // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
            // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
            //
            // 先重试上一轮没发出去的包/EOF，再收新包。EAGAIN 说明解码器
            // 输入缓冲满了，这轮不能收新包（否则新包也会被拒，白丢一个）。
            let mut blocked = false;
            if pending_eof {
                match self.decoder.as_mut().unwrap().send_eof() {
                    Ok(()) => {
                        pending_eof = false;
                        eof = true;
                    }
                    Err(e) if is_backpressure(&e) => blocked = true,
                    Err(e) => {
                        self.note_decode_error("send_eof", &e);
                        pending_eof = false;
                        eof = true;
                    }
                }
            } else if let Some(tp) = pending.take() {
                match self.decoder.as_mut().unwrap().send_packet(&tp.packet) {
                    Ok(()) => {}
                    Err(e) if is_backpressure(&e) => {
                        pending = Some(tp);
                        blocked = true;
                    }
                    Err(e) => {
                        self.note_decode_error("send_packet", &e);
                    }
                }
            }
            if !blocked && !eof {
                match self.video_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(tp) => {
                        // seek 前读出的旧包在复位后才送达：按代数丢弃。
                        // 定位失败时门槛不抬高，这些包仍然对应当前读位置，必须保留
                        if tp.epoch < self.packet_epoch_floor {
                            continue;
                        }
                        match self.decoder.as_mut().unwrap().send_packet(&tp.packet) {
                            Ok(()) => {}
                            Err(e) if is_backpressure(&e) => {
                                pending = Some(tp);
                            }
                            Err(e) => {
                                self.note_decode_error("send_packet", &e);
                            }
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        // 通道断开：给解码器送 EOF 冲掉残留帧。EOF 也可能
                        // 因 EAGAIN 被拒，记下待重试。
                        match self.decoder.as_mut().unwrap().send_eof() {
                            Ok(()) => eof = true,
                            Err(e) if is_backpressure(&e) => pending_eof = true,
                            Err(e) => {
                                self.note_decode_error("send_eof", &e);
                                eof = true;
                            }
                        }
                    }
                }
            }
            // EAGAIN（本轮没有更多帧）与 EOF 都是正常收尾，不能算解码错误；
            // 其余错误（InvalidData / External 等）计数并留一条日志。
            // process_frame 发布成功时会把 target 的像素所有权移进 YuvFrame
            // 并补上一个新的目标帧（回收来的或新建的）
            let mut drained = false;
            loop {
                let received = self.decoder.as_mut().unwrap().receive_frame(&mut target);
                match received {
                    Ok(()) => {
                        drained = true;
                        self.process_frame(&mut target);
                    }
                    Err(e) if is_backpressure(&e) => break,
                    Err(ffmpeg_next::Error::Eof) => break,
                    Err(e) => {
                        self.note_decode_error("receive_frame", &e);
                        break;
                    }
                }
            }
            // 解码器还是塞不下、这轮一帧也没取出来：歇一下再重试。
            // 否则会变成"重试→取不到帧→重试"的 100% CPU 空转
            if blocked && !drained {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }

    // 解码器报错（EAGAIN 背压除外）：计数，首个错误打一条日志。流损坏/驱动
    // 故障时原来这些错误被 `let _` 吞掉，表现为画面停住但日志和 stats 都干净
    fn note_decode_error(&self, stage: &str, e: &ffmpeg_next::Error) {
        if self
            .shared
            .stats
            .decode_errors
            .fetch_add(1, Ordering::Relaxed)
            == 0
        {
            warn!("decoder {stage} failed: {e} (further decode errors are counted in stats)");
        }
    }

    // 取一个可复用的接收目标帧：优先用 UI 回收回来的（连同其像素缓冲），
    // 没有则新建空帧
    fn take_recycled(&mut self) -> ffmpeg_next::util::frame::video::Video {
        match self.recycle_rx.try_recv() {
            Ok(f) => f
                .into_buf()
                .map_or_else(ffmpeg_next::util::frame::video::Video::empty, |b| *b),
            Err(_) => ffmpeg_next::util::frame::video::Video::empty(),
        }
    }

    // 取回 UI 回收的帧（保留 Box 所有权，用于放回转换输出池）
    fn take_recycled_box(&mut self) -> Option<Box<ffmpeg_next::util::frame::video::Video>> {
        match self.recycle_rx.try_recv() {
            Ok(f) => f.into_buf(),
            Err(_) => None,
        }
    }

    // 安排下一轮的目标帧：
    // - 直接上屏：解码帧的像素所有权已经交给 UI，target 必须换新的
    // - 转换路径：解码帧只是中间结果，仍留在 target 里，下一轮解码器直接
    //   重填（avcodec_receive_frame 先 unref 再填 → 缓冲被复用，没有整帧
    //   分配）；UI 回收回来的正是上一帧的转换输出，放回输出池复用
    fn next_target(
        &mut self,
        target: &mut ffmpeg_next::util::frame::video::Video,
        decoded_kept: bool,
    ) {
        if decoded_kept {
            if let Some(recycled) = self.take_recycled_box() {
                self.conv_pool.push(*recycled);
            }
        } else {
            *target = self.take_recycled();
        }
    }

    // 取一个可复用的转换输出帧。格式/尺寸必须与目标一致：swscale 按输出帧
    // 自己的 linesize 写入，尺寸不符就是越界写。池里的帧尺寸变了就丢弃。
    fn take_conv_dst(
        pool: &mut Vec<ffmpeg_next::util::frame::video::Video>,
        format: Pixel,
        w: u32,
        h: u32,
    ) -> ffmpeg_next::util::frame::video::Video {
        while let Some(cand) = pool.pop() {
            if cand.format() == format && cand.width() == w && cand.height() == h {
                return cand;
            }
            // 尺寸/格式变了：不能当输出缓冲用，丢弃
        }
        let mut dst = ffmpeg_next::util::frame::video::Video::empty();
        unsafe { dst.alloc(format, w, h) };
        dst
    }

    // seek 的本地复位：flush 解码器、排空通道、清 PTS 回退、置首帧标记，
    // 把主时钟基准设到目标点后回报就绪（demux 移动读位置后才发布该结果）
    fn reset_for_seek(&mut self, target: Option<f64>) {
        if let Some(d) = self.decoder.as_mut() {
            d.flush();
        }
        while self.video_rx.try_recv().is_ok() {}
        // 读位置已变：旧代数的包全部作废
        self.packet_epoch_floor = self.local_epoch;
        self.last_pts = None;
        self.just_seeked = true;
        self.late_policy.reset();
        self.late_pressure = 0;
        self.set_decoder_skip(false);
        if let Some(target_sec) = target {
            self.clock.seek_to(target_sec);
        }
        self.shared.ctl.report_ready(self.local_epoch);
    }

    // 持续迟到时让解码器直接丢弃非参考帧，省掉"解出来再丢"的成本；
    // 追上后恢复。skip_frame 是 AVCodecContext 的运行期字段，FFmpeg 会在
    // 每个 packet 解码前读取，不需要重开解码器。
    fn set_decoder_skip(&mut self, skip: bool) {
        if self.decoder_skip == skip {
            return;
        }
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        unsafe {
            let p = decoder.as_mut_ptr();
            let discard = if skip {
                ffi::AVDiscard::AVDISCARD_NONREF
            } else {
                ffi::AVDiscard::AVDISCARD_DEFAULT
            };
            (*p).skip_frame = discard;
            // 迟到压力大时连 deblocking 一起跳过；只在落后时启用，追上即恢复
            (*p).skip_loop_filter = if skip {
                ffi::AVDiscard::AVDISCARD_ALL
            } else {
                ffi::AVDiscard::AVDISCARD_DEFAULT
            };
        }
        self.decoder_skip = skip;
        log::debug!(
            "decoder skip_frame={}",
            if skip { "nonref" } else { "default" }
        );
    }

    // 处理一个解码帧：暂停门控 → PTS 计算 → 时钟对齐 → 同步睡眠 →
    // 必要时转 420P → 零拷贝移交给 UI。target 是解码器的输出目标帧，
    // 发布成功时其所有权被移入 YuvFrame（零拷贝），随后换上新的目标帧。
    fn process_frame(&mut self, target: &mut ffmpeg_next::util::frame::video::Video) {
        // 本源已收尾：本帧不再处理
        if self.shared.stop.load(Ordering::Relaxed) {
            return;
        }
        self.shared.stats.decoded.fetch_add(1, Ordering::Relaxed);
        // 暂停门控：复位后的首帧跳过（让用户立即看到 seek 结果）。
        // seek 请求会 notify 唤醒，醒来发现代数变化即丢弃本帧，
        // 交由主循环执行复位（此时仍保持暂停状态）
        if !self.just_seeked {
            let pause_start = Instant::now();
            // 先查 seek 再等待：被 seek 打断时立即返回，不必等超时；
            // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
            if self.shared.paused.wait_while_paused(|| {
                self.shared.stop.load(Ordering::Relaxed)
                    || self.shared.ctl.epoch() != self.local_epoch
            }) {
                return; // seek/收尾打断暂停：本帧作废
            }
            self.clock.add_pause_duration(pause_start.elapsed());
        }

        // PTS 缺失时回退 best_effort_timestamp；仍缺则按上帧 +1 单调递增
        let pts = target
            .pts()
            .or_else(|| target.timestamp())
            .unwrap_or_else(|| self.last_pts.map(|p| p + 1).unwrap_or(0));
        self.last_pts = Some(pts);
        // 异常 time_base（分母为 0）时按 1 秒单位处理，避免除零。
        // 容器 PTS 是绝对时间戳（TS/HLS 常见 start_time ≈ 1.4s），必须减去
        // 时间轴原点，否则首帧要等到 start_time 才显示、结尾永远等不到
        let raw_pts_sec = if let Some(tb) = self.time_base {
            if tb.denominator() > 0 {
                pts as f64 * tb.numerator() as f64 / tb.denominator() as f64
            } else {
                pts as f64
            }
        } else {
            pts as f64
        };
        let video_pts_sec = raw_pts_sec - self.timeline_origin;

        // 复位后的首帧：把主时钟对齐到该帧 pts，修正解码关键帧与请求点的偏差
        if self.just_seeked {
            self.clock.rebase(video_pts_sec);
            self.just_seeked = false;
        }

        let audio_time_sec = self.clock.get_time_sec();
        let diff = video_pts_sec - audio_time_sec;
        // 混合睡眠：远离 PTS 时先长睡（每片 ≤50ms，便于期间响应 seek），
        // 最后 ~3ms 以 250µs 短睡逼近，消除 thread::sleep 的定时器超睡
        // 抖动带来的周期性微卡顿（实测改成 1ms 粒度会增加迟到帧）
        if diff > 0.005 {
            self.late_policy.on_time();
            self.late_pressure = self.late_pressure.saturating_sub(1);
            if self.late_pressure == 0 {
                self.set_decoder_skip(false);
            }
            let mut last_now = f64::NEG_INFINITY;
            loop {
                // 等待期间若出现新 seek，立即放弃本帧：音频线程的复位会把
                // 主时钟拨回（samples 清零），若不检查会在此永久空转，
                // 导致 seek 屏障凑不齐而整体死锁
                if self.shared.ctl.epoch() != self.local_epoch {
                    return;
                }
                let now = self.clock.get_time_sec();
                if now >= video_pts_sec {
                    break;
                }
                // 音频已经播完（含解码线程异常退出）：采样计数不会再前进。
                // 主时钟此时已接管墙钟（见 MasterClock::get_time_sec），时间照常
                // 前进，节流照常进行——不能在这里无条件退出，否则剩余视频帧
                // 会以解码速度一次性放出，片尾变成快进。
                // 只有时钟确实不再前进（异常路径没能接管）时才放弃等待，
                // 否则会等一个永远到不了的时间点，把整个源挂死。
                if self.has_audio
                    && self.shared.audio_done.load(Ordering::Relaxed)
                    && now <= last_now
                {
                    break;
                }
                last_now = now;
                let remain = video_pts_sec - now;
                let nap = if remain > 0.008 {
                    (remain - 0.003).min(0.050)
                } else {
                    0.00025
                };
                std::thread::sleep(Duration::from_secs_f64(nap));
            }
        } else if diff < -0.050 {
            // 迟到：少量/短暂落后就丢帧；持续严重落后则追帧 seek；
            // 追帧也无效时改为显示迟到帧，避免永久冻屏（见 LateFramePolicy）
            let action = self.late_policy.on_late(-diff, Instant::now());
            self.late_pressure = (self.late_pressure + 1).min(1000);
            if self.late_pressure >= 6 {
                self.set_decoder_skip(true);
            }
            match action {
                LateAction::Drop => {
                    self.shared
                        .stats
                        .late_dropped
                        .fetch_add(1, Ordering::Relaxed);
                    return;
                }
                LateAction::CatchUp if self.shared.ctl.seekable() => {
                    let seek_target = self.clock.get_time_sec();
                    warn!(
                        "video {:.2}s behind; catch-up seek to {:.2}s",
                        -diff, seek_target
                    );
                    self.shared.ctl.request(seek_target);
                    // 若恰好在暂停状态，唤醒解码线程处理新 seek
                    self.shared.paused.wake();
                    self.shared.stats.catchups.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // 不可 seek 的源（如 yt-dlp 管道）追不了帧：按 Display 处理，
                // 显示迟到帧，避免永久冻屏
                LateAction::CatchUp | LateAction::Display => {}
            }
        } else {
            self.late_policy.on_time();
        }

        // 0 尺寸的帧（部分流在初始化阶段会给出）没有平面数据可读：
        // 此时 target.data(n) 可能是空指针，切片会触发 UB，直接跳过本帧。
        // 必须在 DMA-BUF 映射/下载之前判断，否则会被误判为格式不支持。
        if target.width() == 0 || target.height() == 0 {
            warn!("skipping zero-sized frame");
            return;
        }

        // 解码器声明支持硬解但输出仍是软件格式：说明驱动/后端实际不可用
        if let Some(kind) = self.hw_kind
            && !matches!(target.format(), Pixel::VAAPI | Pixel::CUDA)
            && !self.hw_fallback_logged
        {
            warn!(
                "{} decoder opened but output is {:?}; hardware decode unavailable for this stream, fell back to software",
                kind.name(),
                target.format()
            );
            self.hw_fallback_logged = true;
        }

        // VAAPI 硬件帧优先走 EGL/DMABUF 零拷贝：av_hwframe_map 只增加引用，
        // 渲染端把 DMA-BUF 平面直接导入成 GL 纹理，不再下载到内存、也不经 PBO。
        // 映射失败（格式不支持/驱动不允许）才回退下面的下载路径。
        let dmabuf = if target.format() == Pixel::VAAPI
            && crate::egl::available()
            && !self.dmabuf_unsupported
            && !(self.force_cpu_tonemap && Tonemapper::is_hdr(target))
        {
            let mapped = unsafe { DmaBufFrame::from_vaapi(target) };
            if mapped.is_none() {
                // 同一路流的 DRM 布局不会变，禁用后后续帧直接下载
                self.dmabuf_unsupported = true;
                if !self.dmabuf_fallback_logged {
                    warn!("VAAPI frame cannot be exported as DMA-BUF; falling back to download");
                    self.dmabuf_fallback_logged = true;
                }
            }
            mapped
        } else {
            None
        };
        if dmabuf.is_some() && !self.dmabuf_logged {
            log::debug!("hardware frame received as DMA-BUF (zero-copy)");
            self.dmabuf_logged = true;
        }

        // 没有可导入的 DMA-BUF 时，把 VAAPI/CUDA 硬件帧下载到系统内存，
        // 再走现有 8/10bit 转换管线（一次全帧拷贝）。
        if dmabuf.is_none() && matches!(target.format(), Pixel::VAAPI | Pixel::CUDA) {
            if !self.hw_download_logged {
                log::debug!("hardware frame received: {:?}", target.format());
                self.hw_download_logged = true;
            }
            // 优先复用 UI 回收的软件帧缓冲，避免每帧重新分配下载目标
            let mut sw = self.take_recycled();
            if download_hw_frame(target, &mut sw) {
                *target = sw;
            } else {
                warn!("failed to download hardware frame; dropping");
                *target = self.take_recycled();
                return;
            }
        }

        // 色彩信息取自解码帧（可能随流中途变化：滤镜/下转换），每帧读取
        let color_space = target.color_space();
        let color_range = target.color_range();

        // SAR 修正显示尺寸：非方形像素（变形宽银幕等 sar != 1）内容不被拉伸，
        // 渲染视口与窗口宽高比都按修正后的尺寸计算
        let (dw, dh) = sar_corrected(target.width(), target.height(), target.aspect_ratio());
        let (disp_w, disp_h) = (dw as i32, dh as i32);

        // HDR(PQ/HLG) 经滤镜图 tone map 成 SDR BT.709；其余走 swscale/零拷贝。
        // 非 420P、或负 stride（垂直翻转的行序）时经 swscale 统一成正 stride
        // 的 420P——swscale 能吃负 linesize，输出恒为正，于是 YuvFrame 对下游
        // 永远是"420P + 正 stride + 正向行序"。正 stride 的 420P 走零拷贝快路径。
        let is_hdr = Tonemapper::is_hdr(target);
        let transfer = target.color_transfer_characteristic();
        let mut primaries = target.color_primaries();
        let mut color_space = color_space;
        if is_hdr {
            // HDR10/HLG 的标准色域是 BT.2020；容器未标注时按此处理，
            // 否则 shader 会拿 709 系数去解 BT.2020 的 YUV，颜色会错
            if color_space == Space::Unspecified {
                color_space = Space::BT2020NCL;
            }
            if primaries == Primaries::Unspecified {
                primaries = Primaries::BT2020;
            }
        }

        // 零拷贝路径：色彩/显示元数据已确定，直接把 DMA-BUF 帧交给 UI。
        // YuvFrame 持有 DRM_PRIME 引用，UI 回收前解码器不会复写该 surface。
        if let Some(dma) = dmabuf {
            let (out_transfer, out_primaries) = if is_hdr {
                (transfer, primaries)
            } else {
                (TransferCharacteristic::Unspecified, Primaries::Unspecified)
            };
            let frame_data = YuvFrame::new_dmabuf(
                dma,
                video_pts_sec,
                color_space,
                color_range,
                out_transfer,
                out_primaries,
                disp_w,
                disp_h,
                self.local_epoch,
            );
            let sent = self.tx.send(Message::Frame(frame_data)).is_ok();
            // 为下一轮 receive_frame 补目标帧；硬件帧本身由 dma 持有引用
            *target = self.take_recycled();
            if sent {
                self.shared.stats.displayed.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .dmabuf_frames
                    .fetch_add(1, Ordering::Relaxed);
                app::awake();
            }
            return;
        }

        let ready_420p =
            target.format() == Pixel::YUV420P && linesize(target, 0) > 0 && linesize(target, 1) > 0;
        let ready_10 = target.format() == Pixel::YUV420P10LE
            && linesize(target, 0) > 0
            && linesize(target, 1) > 0;
        // 硬解下载后的常见格式：NV12（8bit）/ P010（10bit），两平面
        let ready_nv12 =
            target.format() == Pixel::NV12 && linesize(target, 0) > 0 && linesize(target, 1) > 0;
        let ready_p010 =
            target.format() == Pixel::P010LE && linesize(target, 0) > 0 && linesize(target, 1) > 0;
        // 直接上屏的格式要把解码帧的像素所有权交给下游（move 出去）；
        // 需要转换的格式只借用它——解码帧留在 target 里，下一轮由解码器
        // 重填（avcodec_receive_frame 先 unref 再填，缓冲被复用），
        // 稳态下没有整帧缓冲分配
        let take_owned = |target: &mut ffmpeg_next::util::frame::video::Video| {
            std::mem::replace(target, ffmpeg_next::util::frame::video::Video::empty())
        };
        // (buf, layout, bit_depth, space, range, transfer, primaries)
        type Converted = (
            Box<ffmpeg_next::util::frame::video::Video>,
            ChromaLayout,
            u8,
            Space,
            Range,
            TransferCharacteristic,
            Primaries,
        );
        let converted: Option<Converted> = if is_hdr && self.force_cpu_tonemap {
            self.tonemapper.convert(target, self.time_base).map(|b| {
                (
                    Box::new(b),
                    ChromaLayout::Planar,
                    8,
                    Space::BT709,
                    Range::MPEG,
                    TransferCharacteristic::BT709,
                    Primaries::BT709,
                )
            })
        } else if is_hdr {
            if ready_p010 {
                Some((
                    Box::new(take_owned(target)),
                    ChromaLayout::SemiPlanar,
                    10,
                    color_space,
                    color_range,
                    transfer,
                    primaries,
                ))
            } else if ready_10 {
                // HDR10/HLG 10bit 平面：零拷贝交给 GPU tone map
                Some((
                    Box::new(take_owned(target)),
                    ChromaLayout::Planar,
                    10,
                    color_space,
                    color_range,
                    transfer,
                    primaries,
                ))
            } else if ready_nv12 {
                // 8bit HDR 极少见，但 shader 同样能处理
                Some((
                    Box::new(take_owned(target)),
                    ChromaLayout::SemiPlanar,
                    8,
                    color_space,
                    color_range,
                    transfer,
                    primaries,
                ))
            } else if ready_420p {
                Some((
                    Box::new(take_owned(target)),
                    ChromaLayout::Planar,
                    8,
                    color_space,
                    color_range,
                    transfer,
                    primaries,
                ))
            } else if let Some(b) = self.scale_to_420p10(target) {
                Some((
                    b,
                    ChromaLayout::Planar,
                    10,
                    color_space,
                    color_range,
                    transfer,
                    primaries,
                ))
            } else {
                // 10bit 转换失败：退回原有 CPU tone map（颜色正确但更慢）
                warn!("HDR 10bit conversion failed, falling back to CPU tone mapping");
                self.tonemapper.convert(target, self.time_base).map(|b| {
                    (
                        Box::new(b),
                        ChromaLayout::Planar,
                        8,
                        Space::BT709,
                        Range::MPEG,
                        TransferCharacteristic::BT709,
                        Primaries::BT709,
                    )
                })
            }
        } else if ready_p010 {
            // 10bit SDR（NV12/P010 常见于硬解）：保留位深，不 tone map
            Some((
                Box::new(take_owned(target)),
                ChromaLayout::SemiPlanar,
                10,
                color_space,
                color_range,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
            ))
        } else if ready_nv12 {
            Some((
                Box::new(take_owned(target)),
                ChromaLayout::SemiPlanar,
                8,
                color_space,
                color_range,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
            ))
        } else if ready_10 {
            // 10bit SDR 平面：保留位深
            Some((
                Box::new(take_owned(target)),
                ChromaLayout::Planar,
                10,
                color_space,
                color_range,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
            ))
        } else if ready_420p {
            Some((
                Box::new(take_owned(target)),
                ChromaLayout::Planar,
                8,
                color_space,
                color_range,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
            ))
        } else {
            self.scale_to_420p(target).map(|b| {
                (
                    b,
                    ChromaLayout::Planar,
                    8,
                    color_space,
                    color_range,
                    TransferCharacteristic::Unspecified,
                    Primaries::Unspecified,
                )
            })
        };
        // 转换路径下解码帧没有被 move 出去，仍在 target 里 → 下一轮可复用
        let decoded_kept = !unsafe { target.is_empty() };
        let Some((buf, layout, bit_depth, out_space, out_range, out_transfer, out_primaries)) =
            converted
        else {
            // 转换失败只跳过本帧，不让解码线程挂掉导致播放卡死
            self.next_target(target, decoded_kept);
            return;
        };

        let frame_data = if layout == ChromaLayout::SemiPlanar {
            YuvFrame::new_semi(
                buf,
                bit_depth,
                video_pts_sec,
                out_space,
                out_range,
                out_transfer,
                out_primaries,
                disp_w,
                disp_h,
                self.local_epoch,
            )
        } else if bit_depth == 8
            && out_transfer == TransferCharacteristic::Unspecified
            && out_primaries == Primaries::Unspecified
        {
            YuvFrame::new_shared(
                buf,
                video_pts_sec,
                out_space,
                out_range,
                disp_w,
                disp_h,
                self.local_epoch,
            )
        } else {
            YuvFrame::new_hdr(
                buf,
                bit_depth,
                video_pts_sec,
                out_space,
                out_range,
                out_transfer,
                out_primaries,
                disp_w,
                disp_h,
                self.local_epoch,
            )
        };

        // 有界通道：UI 停顿时在此阻塞形成背压，防止帧无限堆积；
        // 发送后手动唤醒 fltk 事件循环
        let sent = self.tx.send(Message::Frame(frame_data)).is_ok();
        // 安排下一轮的目标帧（直接上屏换新帧 / 转换路径复用解码帧，
        // 回收帧进转换输出池）
        self.next_target(target, decoded_kept);
        if sent {
            self.shared.stats.displayed.fetch_add(1, Ordering::Relaxed);
            app::awake();
        }
    }

    // HDR/10bit：把任意格式转成 YUV420P10LE，保留 10bit 供 GPU tone map。
    // 与 scale_to_420p 分开缓存，避免 8bit/10bit 输出互相踢掉转换器。
    fn scale_to_420p10(
        &mut self,
        src: &ffmpeg_next::util::frame::video::Video,
    ) -> Option<Box<ffmpeg_next::util::frame::video::Video>> {
        let key = (src.format(), src.width(), src.height());
        if self.scaler10.as_ref().is_none_or(|(_, k)| *k != key) {
            match ScaleContext::get(
                key.0,
                key.1,
                key.2,
                Pixel::YUV420P10LE,
                key.1,
                key.2,
                ScaleFlags::BILINEAR,
            ) {
                Ok(ctx) => self.scaler10 = Some((ctx, key)),
                Err(e) => {
                    error!("swscale HDR init failed ({:?}): {}", key.0, e);
                    return None;
                }
            }
        }
        let mut dst = Self::take_conv_dst(&mut self.conv_pool, Pixel::YUV420P10LE, key.1, key.2);
        let (ctx, _) = self.scaler10.as_mut().unwrap();
        if ctx.run(src, &mut dst).is_err() {
            return None;
        }
        Some(Box::new(dst))
    }

    // 用 swscale 把任意输入（任意像素格式 / 负 stride）转成正 stride 的 420P。
    // scaler 按 (格式, 宽, 高) 缓存，格式尺寸变化时重建。返回 None 表示转换失败
    fn scale_to_420p(
        &mut self,
        src: &ffmpeg_next::util::frame::video::Video,
    ) -> Option<Box<ffmpeg_next::util::frame::video::Video>> {
        let key = (src.format(), src.width(), src.height());
        if self.scaler.as_ref().is_none_or(|(_, k)| *k != key) {
            match ScaleContext::get(
                key.0,
                key.1,
                key.2,
                Pixel::YUV420P,
                key.1,
                key.2,
                ScaleFlags::BILINEAR,
            ) {
                Ok(ctx) => self.scaler = Some((ctx, key)),
                Err(e) => {
                    error!("swscale init failed ({:?}): {}", key.0, e);
                    return None;
                }
            }
        }
        // 输出帧从池里取：它的所有权要交给下游（零拷贝），所以不能就地复用，
        // 只能靠 UI 回收后归还。仅非 420P / 负 stride 的少数流走这条路径
        let mut dst = Self::take_conv_dst(&mut self.conv_pool, Pixel::YUV420P, key.1, key.2);
        let (ctx, _) = self.scaler.as_mut().unwrap();
        if ctx.run(src, &mut dst).is_err() {
            return None;
        }
        Some(Box::new(dst))
    }

    // 视频播完后等待音频播完再结束，避免音频尾部被截断；
    // 阻塞在条件变量上，由音频线程播完时唤醒，避免忙轮询。
    // 等待期间仍响应 seek：demux 若重新发包，排空即可（解码循环已结束）
    fn wait_audio_done(&mut self) -> Result<()> {
        let (done_lock, done_cond) = &*self.shared.done_cond;
        let mut done_guard = done_lock.lock().unwrap();
        // 不能用固定 5s 硬超时：视频比音频短时会把正常的长音频尾截断，
        // 暂停在结尾时环缓冲不排空也会误判。改为"无进展"看门狗：
        // - 采样计数前进 → 正常收尾，刷新计时；
        // - 暂停中 → 采样本就不动，不算失联；
        // - 音频输出失效 → 直接结束（环不会再排空）；
        // - 连续 5s 没有任何进展 → 才判定音频线程静默失联。
        // 线程 panic/初始化失败已由 ExitGuard 无条件置 audio_done，无需再兜底
        const STALL_TIMEOUT: Duration = Duration::from_secs(5);
        let mut last_samples = self.clock.audio_samples();
        let mut last_progress = Instant::now();
        while !self.shared.audio_done.load(Ordering::Relaxed) {
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(decision) = self.shared.ctl.snapshot_decision(self.local_epoch) {
                self.local_epoch = decision.epoch;
                while self.video_rx.try_recv().is_ok() {}
                self.shared.ctl.report_ready(decision.epoch);
                // seek 后音频可能从头再播，重新计时
                last_samples = self.clock.audio_samples();
                last_progress = Instant::now();
                continue;
            }
            if self.shared.audio_dead.load(Ordering::Relaxed) {
                warn!("audio output failed while draining; ending source");
                break;
            }
            let paused = self.shared.paused.is_paused();
            let samples = self.clock.audio_samples();
            if samples != last_samples {
                last_samples = samples;
                last_progress = Instant::now();
            } else if paused {
                // 暂停期间采样不推进是预期行为，不能算卡死；恢复后重新计时
                last_progress = Instant::now();
            } else if last_progress.elapsed() >= STALL_TIMEOUT {
                warn!(
                    "audio made no progress for {}s (may have stalled); ending source",
                    STALL_TIMEOUT.as_secs()
                );
                break;
            }
            done_guard = done_cond
                .wait_timeout(done_guard, Duration::from_millis(50))
                .unwrap()
                .0;
        }
        drop(done_guard);
        if self.tx.send(Message::End).is_ok() {
            app::awake();
        }
        Ok(())
    }

    // 驻留期：本源临近结束但 demux 可能还活着（用户 seek 会重新发包）。
    // 维持就绪屏障不悬挂、丢弃再发的包；demux 退出（通道断开）即收尾。
    fn linger(&mut self) -> Result<()> {
        loop {
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(decision) = self.shared.ctl.snapshot_decision(self.local_epoch) {
                self.local_epoch = decision.epoch;
                while self.video_rx.try_recv().is_ok() {}
                self.shared.ctl.report_ready(decision.epoch);
                continue;
            }
            match self.video_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(_p) => {} // seek 后再发的包：解码循环已结束，直接丢弃
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HwKind {
    Cuda,
    Vaapi,
}

impl HwKind {
    fn name(self) -> &'static str {
        match self {
            HwKind::Cuda => "CUDA/NVDEC",
            HwKind::Vaapi => "VAAPI",
        }
    }

    fn device_type(self) -> ffi::AVHWDeviceType {
        match self {
            HwKind::Cuda => ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            HwKind::Vaapi => ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
        }
    }

    fn get_format(
        self,
    ) -> unsafe extern "C" fn(
        *mut ffi::AVCodecContext,
        *const ffi::AVPixelFormat,
    ) -> ffi::AVPixelFormat {
        match self {
            HwKind::Cuda => get_format_cuda,
            HwKind::Vaapi => get_format_vaapi,
        }
    }
}

// AUV_HWACCEL: auto(默认)=VAAPI→CUDA/NVDEC；cuda/nvdec、vaapi、0/none/off。
// 笔记本混合显卡默认优先核显 VAAPI：解码帧本来就要下载到内存再交给显示用的
// 核显 GL，走 Intel 不必唤醒 MX350，功耗/发热更低；需要 dGPU 时可显式 cuda。
// 显式 cuda 在 NVDEC 不可用/不支持该流时继续尝试 VAAPI，避免直接掉到软解。
fn hwaccel_candidates() -> Vec<HwKind> {
    match std::env::var("AUV_HWACCEL").as_deref() {
        Ok("0") | Ok("none") | Ok("off") => Vec::new(),
        Ok("cuda") | Ok("nvdec") => vec![HwKind::Cuda, HwKind::Vaapi],
        Ok("vaapi") => vec![HwKind::Vaapi],
        Ok("auto") | Err(_) => vec![HwKind::Vaapi, HwKind::Cuda],
        Ok(other) => {
            warn!("unknown AUV_HWACCEL={other:?}; using auto (VAAPI->CUDA)");
            vec![HwKind::Vaapi, HwKind::Cuda]
        }
    }
}

// CUDA/NVDEC 能力预检结果
enum CuvidCapability {
    // cuvid 解码器能打开：NVDEC 支持该编码/色度/位深
    Supported,
    // FFmpeg 有该 cuvid 解码器，但打开失败（NVDEC 不可用或格式不支持）
    Unsupported,
    // 没有对应 cuvid 解码器（FFmpeg 未编译 NVDEC）或参数转换失败：
    // 不做结论，交给通用 CUDA hwaccel 尝试
    Unknown,
}

// 用 FFmpeg 的 *_cuvid 解码器对同一路流做一次能力探测。
// cuvid 初始化内部会调用 cuvidGetDecoderCaps；NVDEC 不可用时
// avcodec_open2 立即失败，而不是等到首帧才退化。
fn cuvid_capability(params: &codec::parameters::Parameters) -> CuvidCapability {
    let Some(name) = cuvid_decoder_name(params.id()) else {
        return CuvidCapability::Unknown;
    };
    let Some(codec) = codec::decoder::find_by_name(name) else {
        log::debug!("{name} not available in this FFmpeg build");
        return CuvidCapability::Unknown;
    };
    let mut ctx = codec::context::Context::new_with_codec(codec);
    unsafe {
        let ret = ffi::avcodec_parameters_to_context(ctx.as_mut_ptr(), params.as_ptr());
        if ret < 0 {
            return CuvidCapability::Unknown;
        }
    }
    match ctx.decoder().video() {
        Ok(_) => {
            log::debug!("{name} capability probe succeeded");
            CuvidCapability::Supported
        }
        Err(e) => {
            log::debug!("{name} capability probe failed: {e}");
            CuvidCapability::Unsupported
        }
    }
}

// AVCodecID → 专用 cuvid 解码器名（mpeg1/2 的命名与 codec name 不同，需显式映射）
fn cuvid_decoder_name(id: codec::Id) -> Option<&'static str> {
    Some(match id {
        codec::Id::H264 => "h264_cuvid",
        codec::Id::HEVC => "hevc_cuvid",
        codec::Id::AV1 => "av1_cuvid",
        codec::Id::VP8 => "vp8_cuvid",
        codec::Id::VP9 => "vp9_cuvid",
        codec::Id::MPEG1VIDEO => "mpeg1_cuvid",
        codec::Id::MPEG2VIDEO => "mpeg2_cuvid",
        codec::Id::MPEG4 => "mpeg4_cuvid",
        codec::Id::VC1 => "vc1_cuvid",
        codec::Id::MJPEG => "mjpeg_cuvid",
        _ => return None,
    })
}

unsafe fn get_format_prefer(
    fmts: *const ffi::AVPixelFormat,
    want: ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        let mut p = fmts;
        while !p.is_null() && *p != ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *p == want {
                return *p;
            }
            p = p.add(1);
        }
        if fmts.is_null() {
            ffi::AVPixelFormat::AV_PIX_FMT_NONE
        } else {
            *fmts
        }
    }
}

// 编码器提供对应硬件像素格式时选择它，否则退回编码器给出的第一个格式
unsafe extern "C" fn get_format_vaapi(
    _ctx: *mut ffi::AVCodecContext,
    fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe { get_format_prefer(fmts, ffi::AVPixelFormat::AV_PIX_FMT_VAAPI) }
}

unsafe extern "C" fn get_format_cuda(
    _ctx: *mut ffi::AVCodecContext,
    fmts: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe { get_format_prefer(fmts, ffi::AVPixelFormat::AV_PIX_FMT_CUDA) }
}

// 把 VAAPI/CUDA 硬件帧下载到系统内存（格式通常 NV12/P010），并复制 PTS/色域
// 元数据。dst 优先使用 UI 回收的软件帧：格式/尺寸匹配时 transfer 直接复用其
// 缓冲，避免每帧分配；不匹配则 unref 后让 FFmpeg 重新分配。
fn download_hw_frame(
    hw: &ffmpeg_next::util::frame::video::Video,
    dst: &mut ffmpeg_next::util::frame::video::Video,
) -> bool {
    unsafe {
        let mut ret = ffi::av_hwframe_transfer_data(dst.as_mut_ptr(), hw.as_ptr(), 0);
        if ret < 0 {
            ffi::av_frame_unref(dst.as_mut_ptr());
            ret = ffi::av_hwframe_transfer_data(dst.as_mut_ptr(), hw.as_ptr(), 0);
        }
        if ret < 0 {
            return false;
        }
        ffi::av_frame_copy_props(dst.as_mut_ptr(), hw.as_ptr());
        dst.set_width(hw.width());
        dst.set_height(hw.height());
    }
    true
}

// HDR(高动态范围) → SDR 转换器：
//   zscale(PQ/HLG + BT.2020 → linear) → tonemap(hable) → zscale(→ BT.709) → yuv420p
// 输出恒为 BT.709 limited 的 YUV420P，交给既有渲染/截图路径（它们只认 SDR）。
// 滤镜图按输入 (格式, 尺寸, 色彩信息) 缓存，变化时重建；任何失败都返回 None，
// 调用方退回普通转换，保证不会因 HDR 处理失败而无法播放。
struct Tonemapper {
    cached: Option<(filter::Graph, TonemapKey)>,
}

// 构建滤镜图的输入特征（变化时才重建）
#[derive(PartialEq)]
struct TonemapKey {
    format: Pixel,
    width: u32,
    height: u32,
    space: Space,
    primaries: Primaries,
    transfer: TransferCharacteristic,
    range: Range,
    time_base: (i32, i32),
}

impl Tonemapper {
    fn new() -> Self {
        Self { cached: None }
    }

    // 只有 PQ(SMPTE2084) / HLG(ARIB_STD_B67) 需要 tone mapping；
    // SDR（BT.709 等）仍走原有零拷贝/swscale 路径
    fn is_hdr(frame: &ffmpeg_next::util::frame::video::Video) -> bool {
        matches!(
            frame.color_transfer_characteristic(),
            TransferCharacteristic::SMPTE2084 | TransferCharacteristic::ARIB_STD_B67
        )
    }

    fn convert(
        &mut self,
        frame: &ffmpeg_next::util::frame::video::Video,
        time_base: Option<ffmpeg_next::Rational>,
    ) -> Option<ffmpeg_next::util::frame::video::Video> {
        let tb = match time_base {
            Some(tb) if tb.numerator() > 0 && tb.denominator() > 0 => {
                (tb.numerator(), tb.denominator())
            }
            _ => (1, 1_000_000),
        };
        let key = TonemapKey {
            format: frame.format(),
            width: frame.width(),
            height: frame.height(),
            space: frame.color_space(),
            primaries: frame.color_primaries(),
            transfer: frame.color_transfer_characteristic(),
            range: frame.color_range(),
            time_base: tb,
        };
        if self.cached.as_ref().map(|(_, k)| k) != Some(&key) {
            match Self::build(&key) {
                Ok(g) => self.cached = Some((g, key)),
                Err(e) => {
                    error!("HDR tone map graph init failed: {e}");
                    return None;
                }
            }
        }
        let (graph, _) = self.cached.as_mut()?;
        // 1:1 滤镜链：推入一帧后 sink 应立即可取出一帧
        graph.get("in")?.source().add(frame).ok()?;
        let mut dst = ffmpeg_next::util::frame::video::Video::empty();
        graph.get("out")?.sink().frame(&mut dst).ok()?;
        Some(dst)
    }

    fn build(key: &TonemapKey) -> Result<filter::Graph, ffmpeg_next::Error> {
        let args = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect=1/1",
            key.width,
            key.height,
            ffi::AVPixelFormat::from(key.format) as i32,
            key.time_base.0,
            key.time_base.1,
        );
        // 用 ffmpeg 的规范名喂给 zscale（如 smpte2084 / bt2020nc / tv）；
        // 缺失时按 HDR 常见默认补齐（PQ + BT.2020 + limited）
        let named = |o: Option<&str>, default: &str| o.unwrap_or(default).to_string();
        let chain = format!(
            "zscale=tin={tin}:min={m}:pin={p}:rin={r}:t=linear:npl=100,\
             format=gbrpf32le,\
             zscale=pin={p}:p=bt709,\
             tonemap=tonemap=hable:desat=0,\
             zscale=t=bt709:m=bt709:r=tv,\
             format=yuv420p",
            tin = named(key.transfer.name(), "smpte2084"),
            m = named(key.space.name(), "bt2020nc"),
            p = named(key.primaries.name(), "bt2020"),
            r = named(key.range.name(), "tv"),
        );
        let buffer = filter::find("buffer").ok_or(ffmpeg_next::Error::FilterNotFound)?;
        let buffersink = filter::find("buffersink").ok_or(ffmpeg_next::Error::FilterNotFound)?;
        let mut graph = filter::Graph::new();
        graph.add(&buffer, "in", &args)?;
        graph.add(&buffersink, "out", "")?;
        graph.output("in", 0)?.input("out", 0)?.parse(&chain)?;
        graph.validate()?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 转换输出帧池：格式/尺寸必须匹配才能复用。
    // 尺寸不符的帧绝不能当输出缓冲——swscale 按输出帧自己的 linesize 写入，
    // 尺寸不符就是越界写（堆损坏）
    #[test]
    fn conv_dst_pool_rejects_mismatched_frames() {
        let mut pool: Vec<ffmpeg_next::util::frame::video::Video> = Vec::new();
        // 池里放一个尺寸不同的帧
        let mut wrong = ffmpeg_next::util::frame::video::Video::empty();
        unsafe { wrong.alloc(Pixel::YUV420P, 32, 32) };
        pool.push(wrong);

        // 取 64x48 的输出：尺寸不符的帧被丢弃，另分配一个
        let dst = VideoThread::take_conv_dst(&mut pool, Pixel::YUV420P, 64, 48);
        assert!(pool.is_empty(), "尺寸不符的帧必须从池里丢掉");
        assert_eq!(dst.format(), Pixel::YUV420P);
        assert_eq!((dst.width(), dst.height()), (64, 48));
        assert!(linesize(&dst, 0) > 0, "输出帧必须已分配缓冲");

        // 再取一个：池空 → 新分配
        let dst2 = VideoThread::take_conv_dst(&mut pool, Pixel::YUV420P, 64, 48);
        assert_eq!((dst2.width(), dst2.height()), (64, 48));

        // 尺寸/格式都对得上 → 直接复用池里的帧
        let mut pool2: Vec<ffmpeg_next::util::frame::video::Video> = Vec::new();
        let mut ok = ffmpeg_next::util::frame::video::Video::empty();
        unsafe { ok.alloc(Pixel::YUV420P, 64, 48) };
        let ok_ptr = unsafe { ok.as_ptr() };
        pool2.push(ok);
        let reused = VideoThread::take_conv_dst(&mut pool2, Pixel::YUV420P, 64, 48);
        assert!(pool2.is_empty());
        assert_eq!(
            unsafe { reused.as_ptr() },
            ok_ptr,
            "尺寸匹配时必须复用池里的帧"
        );
    }

    // 真实跑一遍 HDR tone mapping：构造一张 PQ/BT.2020 的 10bit 帧，
    // 走滤镜图后应得到同尺寸的 SDR YUV420P。依赖 ffmpeg 的 zscale/tonemap
    // 滤镜，缺失时跳过（不视为失败）
    #[test]
    fn tonemap_hdr_frame_to_420p() {
        if filter::find("zscale").is_none() || filter::find("tonemap").is_none() {
            eprintln!("zscale/tonemap unavailable, skipping");
            return;
        }
        let _ = ffmpeg_next::init();

        let mut frame = ffmpeg_next::util::frame::video::Video::new(Pixel::YUV420P10LE, 64, 48);
        frame.set_color_space(Space::BT2020NCL);
        frame.set_color_primaries(Primaries::BT2020);
        frame.set_color_transfer_characteristic(TransferCharacteristic::SMPTE2084);
        frame.set_color_range(Range::MPEG);
        frame.set_pts(Some(0));
        for plane in 0..3 {
            let buf = frame.data_mut(plane);
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i % 200) as u8;
            }
        }

        let mut tm = Tonemapper::new();
        let out = tm
            .convert(&frame, Some(ffmpeg_next::Rational::new(1, 30)))
            .expect("HDR frame should be tone mapped");
        assert_eq!(out.format(), Pixel::YUV420P);
        assert_eq!((out.width(), out.height()), (64, 48));
    }

    // NVDEC 预检用的 cuvid 解码器名映射：常见编码都要覆盖
    #[test]
    fn cuvid_names_cover_common_codecs() {
        assert_eq!(cuvid_decoder_name(codec::Id::H264), Some("h264_cuvid"));
        assert_eq!(cuvid_decoder_name(codec::Id::HEVC), Some("hevc_cuvid"));
        // mpeg1/2 的 cuvid 命名与 codec name 不同，不能靠字符串拼接
        assert_eq!(
            cuvid_decoder_name(codec::Id::MPEG2VIDEO),
            Some("mpeg2_cuvid")
        );
        assert_eq!(cuvid_decoder_name(codec::Id::VP9), Some("vp9_cuvid"));
        // 没有对应 cuvid 解码器的编码返回 None（预检不做结论，交给通用 hwaccel）
        assert_eq!(cuvid_decoder_name(codec::Id::FLV1), None);
    }

    // 迟到策略：先丢帧 → 连续 30 帧后追帧（3s 冷却）→ 3 次仍不行则改为
    // 显示迟到帧。关键是最后一定要退出"无限丢帧"，否则画面会永久冻结。
    #[test]
    fn late_policy_never_drops_forever() {
        let mut p = LateFramePolicy::new();
        let t0 = Instant::now();
        for _ in 0..29 {
            assert!(matches!(p.on_late(0.6, t0), LateAction::Drop));
        }
        assert!(matches!(p.on_late(0.6, t0), LateAction::CatchUp));

        // 冷却期内继续丢，不重复追帧
        for _ in 0..30 {
            assert!(matches!(p.on_late(0.6, t0), LateAction::Drop));
        }
        let t1 = t0 + Duration::from_secs(4);
        assert!(matches!(p.on_late(0.6, t1), LateAction::CatchUp));

        for _ in 0..30 {
            assert!(matches!(p.on_late(0.6, t1), LateAction::Drop));
        }
        let t2 = t1 + Duration::from_secs(4);
        assert!(matches!(p.on_late(0.6, t2), LateAction::CatchUp));

        for _ in 0..30 {
            assert!(matches!(p.on_late(0.6, t2), LateAction::Drop));
        }
        let t3 = t2 + Duration::from_secs(4);
        assert!(
            matches!(p.on_late(0.6, t3), LateAction::Display),
            "3 次追帧后必须放弃丢帧"
        );
        assert!(matches!(p.on_late(0.6, t3), LateAction::Display));

        // seek 后策略复位（新的解码位置可能恢复正常）
        p.reset();
        assert!(matches!(p.on_late(0.6, t3), LateAction::Drop));
    }
}

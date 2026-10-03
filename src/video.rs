use anyhow::Result;
use ffmpeg_next::{
    codec,
    codec::packet::Packet,
    ffi, filter,
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
    frame::{Message, YuvFrame, linesize, sar_corrected},
    source::SourceShared,
};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
// UI 侧用有界通道接收：UI 停顿时 send 阻塞产生背压，防止帧无限堆积；
// 每次发送后需 app::awake() 唤醒 fltk 事件循环（替代 fltk channel 的内置唤醒）。
// seek 时：flush 解码器、排空通道、把主时钟基准设到目标点并回报就绪；复位后的
// 首帧再做一次时钟对齐（解码器选中的关键帧可能与请求点有偏差），随后正常同步。
pub fn spawn_video_thread(ctx: VideoThreadCtx) {
    // 保留一份 tx 副本：出错时用它给 UI 发 End，让本源正常收尾。
    // VideoThread::new 失败时 ctx 被消费丢弃，若无此副本 UI 侧只会看到
    // 通道断开（消息泵不靠断开退出），本源会永远挂着
    let tx = ctx.tx.clone();
    std::thread::spawn(move || {
        // 工作线程的错误不会自动上抛，join 句柄又被丢弃：必须在这里就地记录，
        // 否则解码失败（网络中断、文件损坏、seek 复位出错）对用户完全无感——
        // 表现为画面停在最后一帧
        if let Err(e) = VideoThread::new(ctx).and_then(|t| t.run()) {
            error!("video thread exited with error: {e:#}");
            if tx.send(Message::End).is_ok() {
                app::awake();
            }
        }
    });
}

// 视频线程的完整输入：由主循环一次性打包传入，
// 避免散参数顺序/个数错误（原为 10 个参数的 spawn）
pub struct VideoThreadCtx {
    pub tx: SyncSender<Message>,
    pub recycle_rx: Receiver<YuvFrame>,
    pub clock: MasterClock,
    pub shared: Arc<SourceShared>,
    pub video_rx: Receiver<Packet>,
    pub video_parameters: Option<codec::parameters::Parameters>,
    pub video_time_base: Option<ffmpeg_next::Rational>,
}

// 视频解码线程的局部状态：原为 thread::spawn 内的大闭包（process_frame 等），
// 收成结构体后提炼为可独立调用的普通方法，便于阅读与复查。
// 字段与闭包捕获一一对应，职责不变（Cell 换成普通字段，&mut self 即可读写）。
struct VideoThread {
    // 无视频轨（参数缺失）时为 None，解码循环直接跳过
    decoder: Option<codec::decoder::video::Video>,
    time_base: Option<ffmpeg_next::Rational>,
    // 非 YUV420P 输出（10-bit、4:2:2 / 4:4:4 等）经 swscale 统一转成 420P，
    // 渲染端只需处理一种平面布局。转换器惰性创建，格式/尺寸变化时重建。
    scaler: Option<(ScaleContext, (Pixel, u32, u32))>,
    // HDR(PQ/HLG) → SDR 的滤镜图，惰性创建、按输入特征缓存
    tonemapper: Tonemapper,
    // PTS 缺失（rawvideo 等）时的单调回退：按上帧 +1 个 time_base 步进；
    // seek 复位时清空
    last_pts: Option<i64>,
    // seek 代数与复位后首帧标记
    local_epoch: u64,
    just_seeked: bool,
    tx: SyncSender<Message>,
    recycle_rx: Receiver<YuvFrame>,
    video_rx: Receiver<Packet>,
    clock: MasterClock,
    shared: Arc<SourceShared>,
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
        } = ctx;
        let decoder = match video_parameters {
            Some(params) => {
                let mut ctx = codec::context::Context::from_parameters(params)?;
                // 帧级多线程解码：count=0 表示由 ffmpeg 按 CPU 自动决定。
                // 默认的 Type::None 是单线程，4K 解码会明显吃满一个核且掉帧；
                // 帧级（而非 slice 级）保证输出帧可直接整帧移交下游。
                // 注意必须在 open 之前设置，否则不生效。
                // Config::kind() 生成的默认 count 就是 0（ffmpeg 自动定线程数），
                // 这里显式再赋一次是为了不依赖该默认值
                let mut threading = codec::threading::Config::kind(codec::threading::Type::Frame);
                threading.count = 0;
                ctx.set_threading(threading);
                Some(ctx.decoder().video()?)
            }
            None => None,
        };
        Ok(Self {
            decoder,
            time_base: video_time_base,
            scaler: None,
            tonemapper: Tonemapper::new(),
            last_pts: None,
            local_epoch: 0,
            just_seeked: false,
            tx,
            recycle_rx,
            video_rx,
            clock,
            shared,
        })
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
        while !eof {
            // 先响应 seek（暂停中被唤醒后也会回到这里）
            if let Some((e, t)) = self.shared.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                self.reset_for_seek(t);
                continue;
            }
            // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
            // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
            match self.video_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(packet) => {
                    let _ = self.decoder.as_mut().unwrap().send_packet(&packet);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = self.decoder.as_mut().unwrap().send_eof();
                    eof = true;
                }
            }
            while self
                .decoder
                .as_mut()
                .unwrap()
                .receive_frame(&mut target)
                .is_ok()
            {
                // process_frame 发布成功时会把 target 的像素所有权移进 YuvFrame
                // 并补上一个新的目标帧（回收来的或新建的）
                self.process_frame(&mut target);
            }
        }
        Ok(())
    }

    // 取一个可复用的接收目标帧：优先用 UI 回收回来的（连同其像素缓冲），
    // 没有则新建空帧
    fn take_recycled(&mut self) -> ffmpeg_next::util::frame::video::Video {
        match self.recycle_rx.try_recv() {
            Ok(f) => *f.into_buf(),
            Err(_) => ffmpeg_next::util::frame::video::Video::empty(),
        }
    }

    // seek 的本地复位：flush 解码器、排空通道、清 PTS 回退、置首帧标记，
    // 把主时钟基准设到目标点后回报就绪（demux 才移动读位置）
    fn reset_for_seek(&mut self, target: Option<f64>) {
        if let Some(d) = self.decoder.as_mut() {
            d.flush();
        }
        while self.video_rx.try_recv().is_ok() {}
        self.last_pts = None;
        self.just_seeked = true;
        if let Some(target_sec) = target {
            self.clock.seek_to(target_sec);
        }
        self.shared.ctl.report_ready(self.local_epoch);
    }

    // 处理一个解码帧：暂停门控 → PTS 计算 → 时钟对齐 → 同步睡眠 →
    // 必要时转 420P → 零拷贝移交给 UI。target 是解码器的输出目标帧，
    // 发布成功时其所有权被移入 YuvFrame（零拷贝），随后换上新的目标帧。
    fn process_frame(&mut self, target: &mut ffmpeg_next::util::frame::video::Video) {
        // 暂停门控：复位后的首帧跳过（让用户立即看到 seek 结果）。
        // seek 请求会 notify 唤醒，醒来发现代数变化即丢弃本帧，
        // 交由主循环执行复位（此时仍保持暂停状态）
        if !self.just_seeked {
            let pause_start = Instant::now();
            // 先查 seek 再等待：被 seek 打断时立即返回，不必等超时；
            // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
            if self
                .shared
                .paused
                .wait_while_paused(|| self.shared.ctl.epoch() != self.local_epoch)
            {
                return; // seek 打断暂停：本帧作废
            }
            self.clock.add_pause_duration(pause_start.elapsed());
        }

        // PTS 缺失时回退 best_effort_timestamp；仍缺则按上帧 +1 单调递增
        let pts = target
            .pts()
            .or_else(|| target.timestamp())
            .unwrap_or_else(|| self.last_pts.map(|p| p + 1).unwrap_or(0));
        self.last_pts = Some(pts);
        // 异常 time_base（分母为 0）时按 1 秒单位处理，避免除零
        let video_pts_sec = if let Some(tb) = self.time_base {
            if tb.denominator() > 0 {
                pts as f64 * tb.numerator() as f64 / tb.denominator() as f64
            } else {
                pts as f64
            }
        } else {
            pts as f64
        };

        // 复位后的首帧：把主时钟对齐到该帧 pts，修正解码关键帧与请求点的偏差
        if self.just_seeked {
            self.clock.rebase(video_pts_sec);
            self.just_seeked = false;
        }

        let audio_time_sec = self.clock.get_time_sec();
        let diff = video_pts_sec - audio_time_sec;
        // 混合睡眠：远离 PTS 时先长睡（每片 ≤50ms，便于期间响应 seek），
        // 最后 ~3ms 以 250µs 短睡逼近，消除 thread::sleep 的定时器超睡
        // 抖动带来的周期性微卡顿
        if diff > 0.005 {
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
                let remain = video_pts_sec - now;
                let nap = if remain > 0.008 {
                    (remain - 0.003).min(0.050)
                } else {
                    0.00025
                };
                std::thread::sleep(Duration::from_secs_f64(nap));
            }
        } else if diff < -0.050 {
            return;
        }

        // 色彩信息取自解码帧（可能随流中途变化：滤镜/下转换），每帧读取
        let color_space = target.color_space();
        let color_range = target.color_range();
        // 0 尺寸的帧（部分流在初始化阶段会给出）没有平面数据可读：
        // 此时 target.data(n) 可能是空指针，切片会触发 UB，直接跳过本帧
        if target.width() == 0 || target.height() == 0 {
            warn!("skipping zero-sized frame");
            return;
        }

        // SAR 修正显示尺寸：非方形像素（变形宽银幕等 sar != 1）内容不被拉伸，
        // 渲染视口与窗口宽高比都按修正后的尺寸计算
        let (dw, dh) = sar_corrected(target.width(), target.height(), target.aspect_ratio());
        let (disp_w, disp_h) = (dw as i32, dh as i32);

        // HDR(PQ/HLG) 经滤镜图 tone map 成 SDR BT.709；其余走 swscale/零拷贝。
        // 非 420P、或负 stride（垂直翻转的行序）时经 swscale 统一成正 stride
        // 的 420P——swscale 能吃负 linesize，输出恒为正，于是 YuvFrame 对下游
        // 永远是"420P + 正 stride + 正向行序"。正 stride 的 420P 走零拷贝快路径。
        let is_hdr = Tonemapper::is_hdr(target);
        let ready_420p =
            target.format() == Pixel::YUV420P && linesize(target, 0) > 0 && linesize(target, 1) > 0;
        let owned = std::mem::replace(target, ffmpeg_next::util::frame::video::Video::empty());
        let converted: Option<(Box<ffmpeg_next::util::frame::video::Video>, Space, Range)> =
            if is_hdr {
                match self.tonemapper.convert(&owned, self.time_base) {
                    Some(b) => Some((Box::new(b), Space::BT709, Range::MPEG)),
                    None => {
                        // tone mapping 失败：退回普通转换（颜色仍不对，但保证能播）
                        warn!("HDR tone mapping failed, falling back to plain conversion");
                        self.scale_to_420p(&owned)
                            .map(|b| (b, color_space, color_range))
                    }
                }
            } else if ready_420p {
                Some((Box::new(owned), color_space, color_range))
            } else {
                self.scale_to_420p(&owned)
                    .map(|b| (b, color_space, color_range))
            };
        let Some((buf, out_space, out_range)) = converted else {
            // 转换失败只跳过本帧，不让解码线程挂掉导致播放卡死
            *target = self.take_recycled();
            return;
        };

        let frame_data = YuvFrame::new_shared(
            buf,
            video_pts_sec,
            out_space,
            out_range,
            disp_w,
            disp_h,
            self.local_epoch,
        );

        // 有界通道：UI 停顿时在此阻塞形成背压，防止帧无限堆积；
        // 发送后手动唤醒 fltk 事件循环
        let sent = self.tx.send(Message::Frame(frame_data)).is_ok();
        // 为下一轮 receive_frame 补一个目标帧：UI 渲染完会把旧帧回收回来，
        // 稳态下走回收路径，既不分配 AVFrame 也保留其像素缓冲
        *target = self.take_recycled();
        if sent {
            app::awake();
        }
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
        // 输出帧每次新建：它的所有权要交给下游（零拷贝），不能像以前那样
        // 复用 self.conv。仅非 420P / 负 stride 的少数流走这条路径
        let mut dst = ffmpeg_next::util::frame::video::Video::empty();
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
        // 兜底超时：若音频线程 panic 退出，audio_done 永远不置位，
        // 超过时限就放弃等待、按普通收尾继续，避免本源永久挂起
        let mut deadline = Instant::now() + Duration::from_secs(5);
        while !self.shared.audio_done.load(Ordering::Relaxed) {
            if let Some((e, _t)) = self.shared.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                while self.video_rx.try_recv().is_ok() {}
                self.shared.ctl.report_ready(e);
                // seek 后音频可能从头再播，重置超时避免误判其死亡
                deadline = Instant::now() + Duration::from_secs(5);
                continue;
            }
            if Instant::now() >= deadline {
                warn!("audio thread did not finish within 5s (may have panicked); ending source");
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
            if let Some((e, _t)) = self.shared.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                while self.video_rx.try_recv().is_ok() {}
                self.shared.ctl.report_ready(e);
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
}

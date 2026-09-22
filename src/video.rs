use anyhow::Result;
use ffmpeg_next::{
    codec,
    codec::packet::Packet,
    format::Pixel,
    software::scaling::{context::Context as ScaleContext, flag::Flags as ScaleFlags},
};
use fltk::app;
use log::{error, warn};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{Receiver, SyncSender},
    },
    time::{Duration, Instant},
};

use crate::{
    clock::MasterClock,
    demux::SeekCtl,
    frame::{Message, YuvFrame},
};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
// UI 侧用有界通道接收：UI 停顿时 send 阻塞产生背压，防止帧无限堆积；
// 每次发送后需 app::awake() 唤醒 fltk 事件循环（替代 fltk channel 的内置唤醒）。
// seek 时：flush 解码器、排空通道、把主时钟基准设到目标点并回报就绪；复位后的
// 首帧再做一次时钟对齐（解码器选中的关键帧可能与请求点有偏差），随后正常同步。
pub fn spawn_video_thread(ctx: VideoThreadCtx) {
    std::thread::spawn(move || -> Result<()> { VideoThread::new(ctx)?.run() });
}

// 视频线程的完整输入：由主循环一次性打包传入，
// 避免散参数顺序/个数错误（原为 10 个参数的 spawn）
pub struct VideoThreadCtx {
    pub tx: SyncSender<Message>,
    pub recycle_rx: Receiver<YuvFrame>,
    pub clock: MasterClock,
    pub paused: Arc<(Mutex<bool>, Condvar)>,
    pub audio_done: Arc<AtomicU64>,
    pub sync: Arc<(Mutex<()>, Condvar)>,
    pub video_rx: Receiver<Packet>,
    pub video_parameters: Option<codec::parameters::Parameters>,
    pub video_time_base: Option<ffmpeg_next::Rational>,
    pub ctl: Arc<SeekCtl>,
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
    conv: ffmpeg_next::util::frame::video::Video,
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
    paused: Arc<(Mutex<bool>, Condvar)>,
    audio_done: Arc<AtomicU64>,
    sync: Arc<(Mutex<()>, Condvar)>,
    ctl: Arc<SeekCtl>,
}

impl VideoThread {
    fn new(ctx: VideoThreadCtx) -> Result<Self> {
        let VideoThreadCtx {
            tx,
            recycle_rx,
            clock,
            paused,
            audio_done,
            sync,
            video_rx,
            video_parameters,
            video_time_base,
            ctl,
        } = ctx;
        let decoder = match video_parameters {
            Some(params) => Some(
                codec::context::Context::from_parameters(params)?
                    .decoder()
                    .video()?,
            ),
            None => None,
        };
        Ok(Self {
            decoder,
            time_base: video_time_base,
            scaler: None,
            conv: ffmpeg_next::util::frame::video::Video::empty(),
            last_pts: None,
            local_epoch: 0,
            just_seeked: false,
            tx,
            recycle_rx,
            video_rx,
            clock,
            paused,
            audio_done,
            sync,
            ctl,
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
        while !eof {
            // 先响应 seek（暂停中被唤醒后也会回到这里）
            if let Some((e, t)) = self.ctl.snapshot_new(self.local_epoch) {
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
            let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
            while self
                .decoder
                .as_mut()
                .unwrap()
                .receive_frame(&mut decoded)
                .is_ok()
            {
                self.process_frame(&decoded);
            }
        }
        Ok(())
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
        self.ctl.report_ready(self.local_epoch);
    }

    // 处理一个解码帧：暂停门控 → PTS 计算 → 时钟对齐 → 同步睡眠 →
    // 转 420P → 拷贝进回收帧 → 发送给 UI。所有副作用都落在自身字段上。
    fn process_frame(&mut self, decoded: &ffmpeg_next::util::frame::video::Video) {
        // 暂停门控：复位后的首帧跳过（让用户立即看到 seek 结果）。
        // seek 请求会 notify 唤醒，醒来发现代数变化即丢弃本帧，
        // 交由主循环执行复位（此时仍保持暂停状态）
        if !self.just_seeked {
            let pause_start = Instant::now();
            let (pause_lock, pause_cond) = &*self.paused;
            let mut guard = pause_lock.lock().unwrap();
            while *guard {
                // 先查 seek 再等待：被 seek 打断时立即返回，不必等超时；
                // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
                if self.ctl.epoch() != self.local_epoch {
                    drop(guard);
                    return; // seek 打断暂停：本帧作废
                }
                let (g, _) = pause_cond
                    .wait_timeout(guard, Duration::from_millis(50))
                    .unwrap();
                guard = g;
            }
            drop(guard);
            self.clock.add_pause_duration(pause_start.elapsed());
        }

        // PTS 缺失时回退 best_effort_timestamp；仍缺则按上帧 +1 单调递增
        let pts = decoded
            .pts()
            .or_else(|| decoded.timestamp())
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
                if self.ctl.epoch() != self.local_epoch {
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

        // 需要转换时先转，之后统一按 420P 布局取平面；
        // 转换失败只跳过本帧，不让解码线程挂掉导致播放卡死
        let need_convert = decoded.format() != Pixel::YUV420P;
        if need_convert {
            let key = (decoded.format(), decoded.width(), decoded.height());
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
                        return;
                    }
                }
            }
            let (ctx, _) = self.scaler.as_mut().unwrap();
            if ctx.run(decoded, &mut self.conv).is_err() {
                return;
            }
        }

        let src = if need_convert { &self.conv } else { decoded };
        let w = src.width();
        let h = src.height() as usize;
        let y_stride = src.stride(0);
        let uv_stride = src.stride(1);
        let y_len = y_stride * h;
        let uv_len = uv_stride * (h / 2);

        // SAR 修正显示尺寸：非方形像素（变形宽银幕等 sar != 1）内容不被拉伸，
        // 渲染视口与窗口宽高比都按修正后的尺寸计算
        let (disp_w, disp_h) = {
            let sar = decoded.aspect_ratio();
            if sar.numerator() > 0 && sar.denominator() > 0 && sar.numerator() != sar.denominator()
            {
                let dw =
                    (w as i64 * sar.numerator() as i64 / sar.denominator() as i64).max(1) as i32;
                (dw, src.height() as i32)
            } else {
                (w as i32, src.height() as i32)
            }
        };

        // 回收帧尺寸不匹配（如中途换分辨率）时重新分配，避免越界 panic
        let mut frame_data = match self.recycle_rx.try_recv() {
            Ok(f) if f.y.len() == y_len && f.u.len() == uv_len && f.v.len() == uv_len => f,
            _ => YuvFrame {
                y: vec![0; y_len],
                u: vec![0; uv_len],
                v: vec![0; uv_len],
                width: w as i32,
                height: h as i32,
                y_stride: y_stride as i32,
                uv_stride: uv_stride as i32,
                pts_sec: video_pts_sec,
                frame_gen: 0, // 发送前由 next_frame_gen 覆盖
                color_space: decoded.color_space(),
                color_range: decoded.color_range(),
                disp_w,
                disp_h,
            },
        };

        frame_data.y.copy_from_slice(&src.data(0)[..y_len]);
        frame_data.u.copy_from_slice(&src.data(1)[..uv_len]);
        frame_data.v.copy_from_slice(&src.data(2)[..uv_len]);
        frame_data.width = w as i32;
        frame_data.height = h as i32;
        frame_data.disp_w = disp_w;
        frame_data.disp_h = disp_h;
        // 色彩信息可能随流中途变化（滤镜/下转换），每次刷新
        frame_data.color_space = decoded.color_space();
        frame_data.color_range = decoded.color_range();
        frame_data.y_stride = y_stride as i32;
        frame_data.uv_stride = uv_stride as i32;
        frame_data.pts_sec = video_pts_sec;
        frame_data.frame_gen = crate::frame::next_frame_gen();

        // 有界通道：UI 停顿时在此阻塞形成背压，防止帧无限堆积；
        // 发送后手动唤醒 fltk 事件循环
        if self.tx.send(Message::Frame(frame_data)).is_ok() {
            app::awake();
        }
    }

    // 视频播完后等待音频播完再结束，避免音频尾部被截断；
    // 阻塞在条件变量上，由音频线程播完时唤醒，避免忙轮询。
    // 等待期间仍响应 seek：demux 若重新发包，排空即可（解码循环已结束）
    fn wait_audio_done(&mut self) -> Result<()> {
        let (done_lock, done_cond) = &*self.sync;
        let mut done_guard = done_lock.lock().unwrap();
        // 兜底超时：若音频线程 panic 退出，audio_done 永远不置位，
        // 超过时限就放弃等待、按普通收尾继续，避免本源永久挂起
        let mut deadline = Instant::now() + Duration::from_secs(5);
        while self.audio_done.load(Ordering::Relaxed) == 0 {
            if let Some((e, _t)) = self.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                while self.video_rx.try_recv().is_ok() {}
                self.ctl.report_ready(e);
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
            if let Some((e, _t)) = self.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                while self.video_rx.try_recv().is_ok() {}
                self.ctl.report_ready(e);
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

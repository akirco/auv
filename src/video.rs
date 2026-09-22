use anyhow::Result;
use ffmpeg_next::codec;
use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling::{context::Context as ScaleContext, flag::Flags as ScaleFlags};
use fltk::app;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::clock::MasterClock;
use crate::demux::SeekCtl;
use crate::frame::{Message, YuvFrame};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
// UI 侧用有界通道接收：UI 停顿时 send 阻塞产生背压，防止帧无限堆积；
// 每次发送后需 app::awake() 唤醒 fltk 事件循环（替代 fltk channel 的内置唤醒）。
// seek 时：flush 解码器、排空通道、把主时钟基准设到目标点并回报就绪；复位后的
// 首帧再做一次时钟对齐（解码器选中的关键帧可能与请求点有偏差），随后正常同步。
#[allow(clippy::too_many_arguments)]
pub fn spawn_video_thread(
    tx: SyncSender<Message>,
    recycle_rx: Receiver<YuvFrame>,
    clock: MasterClock,
    paused: Arc<(Mutex<bool>, Condvar)>,
    audio_done: Arc<AtomicU64>,
    sync: Arc<(Mutex<()>, Condvar)>,
    video_rx: Receiver<Packet>,
    video_parameters: Option<codec::parameters::Parameters>,
    video_time_base: Option<ffmpeg_next::Rational>,
    ctl: Arc<SeekCtl>,
) {
    std::thread::spawn(move || -> Result<()> {
        // seek 代数与复位后首帧标记（Cell：process_frame 闭包与主循环都要读写）
        let local_epoch = Cell::new(0u64);
        let just_seeked = Cell::new(false);

        if let (Some(params), Some(time_base)) = (video_parameters, video_time_base) {
            let mut decoder = codec::context::Context::from_parameters(params)?
                .decoder()
                .video()?;

            // 非 YUV420P 输出（10-bit、4:2:2 / 4:4:4 等）经 swscale 统一转成 420P，
            // 渲染端只需处理一种平面布局。转换器惰性创建，格式/尺寸变化时重建。
            let mut scaler: Option<(ScaleContext, (Pixel, u32, u32))> = None;
            let mut conv = ffmpeg_next::util::frame::video::Video::empty();
            // PTS 缺失（rawvideo 等）时的单调回退：按上帧 +1 个 time_base 步进；
            // Cell 便于 seek 复位时清空（闭包与主循环共享）
            let last_pts = Cell::new(None::<i64>);

            let mut process_frame = |decoded: &ffmpeg_next::util::frame::video::Video| {
                // 暂停门控：复位后的首帧跳过（让用户立即看到 seek 结果）。
                // seek 请求会 notify 唤醒，醒来发现代数变化即丢弃本帧，
                // 交由主循环执行复位（此时仍保持暂停状态）
                if !just_seeked.get() {
                    let pause_start = Instant::now();
                    let (pause_lock, pause_cond) = &*paused;
                    let mut guard = pause_lock.lock().unwrap();
                    while *guard {
                        // 先查 seek 再等待：被 seek 打断时立即返回，不必等超时；
                        // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
                        if ctl.epoch() != local_epoch.get() {
                            drop(guard);
                            return; // seek 打断暂停：本帧作废
                        }
                        let (g, _) = pause_cond
                            .wait_timeout(guard, Duration::from_millis(50))
                            .unwrap();
                        guard = g;
                    }
                    drop(guard);
                    clock.add_pause_duration(pause_start.elapsed());
                }

                // PTS 缺失时回退 best_effort_timestamp；仍缺则按上帧 +1 单调递增
                let pts = decoded
                    .pts()
                    .or_else(|| decoded.timestamp())
                    .unwrap_or_else(|| last_pts.get().map(|p| p + 1).unwrap_or(0));
                last_pts.set(Some(pts));
                // 异常 time_base（分母为 0）时按 1 秒单位处理，避免除零
                let video_pts_sec = if time_base.denominator() > 0 {
                    pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64
                } else {
                    pts as f64
                };

                // 复位后的首帧：把主时钟对齐到该帧 pts，修正解码关键帧与请求点的偏差
                if just_seeked.get() {
                    clock.rebase(video_pts_sec);
                    just_seeked.set(false);
                }

                let audio_time_sec = clock.get_time_sec();
                let diff = video_pts_sec - audio_time_sec;
                // 混合睡眠：远离 PTS 时先长睡（每片 ≤50ms，便于期间响应 seek），
                // 最后 ~3ms 以 250µs 短睡逼近，消除 thread::sleep 的定时器超睡
                // 抖动带来的周期性微卡顿
                if diff > 0.005 {
                    loop {
                        // 等待期间若出现新 seek，立即放弃本帧：音频线程的复位会把
                        // 主时钟拨回（samples 清零），若不检查会在此永久空转，
                        // 导致 seek 屏障凑不齐而整体死锁
                        if ctl.epoch() != local_epoch.get() {
                            return;
                        }
                        let now = clock.get_time_sec();
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
                    if scaler.as_ref().is_none_or(|(_, k)| *k != key) {
                        match ScaleContext::get(
                            key.0,
                            key.1,
                            key.2,
                            Pixel::YUV420P,
                            key.1,
                            key.2,
                            ScaleFlags::BILINEAR,
                        ) {
                            Ok(ctx) => scaler = Some((ctx, key)),
                            Err(e) => {
                                eprintln!("swscale init failed ({:?}): {}", key.0, e);
                                return;
                            }
                        }
                    }
                    let (ctx, _) = scaler.as_mut().unwrap();
                    if ctx.run(decoded, &mut conv).is_err() {
                        return;
                    }
                }

                let src = if need_convert { &conv } else { decoded };
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
                    if sar.numerator() > 0
                        && sar.denominator() > 0
                        && sar.numerator() != sar.denominator()
                    {
                        let dw = (w as i64 * sar.numerator() as i64 / sar.denominator() as i64)
                            .max(1) as i32;
                        (dw, src.height() as i32)
                    } else {
                        (w as i32, src.height() as i32)
                    }
                };

                // 回收帧尺寸不匹配（如中途换分辨率）时重新分配，避免越界 panic
                let mut frame_data = match recycle_rx.try_recv() {
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
                if tx.send(Message::Frame(frame_data)).is_ok() {
                    app::awake();
                }
            };

            // 收完 demux 的全部 packet 后，通道断开，send_eof 把解码器内残留的
            // 帧（如 B 帧延迟）全部取完
            let mut eof = false;
            while !eof {
                // 先响应 seek（暂停中被唤醒后也会回到这里）
                if let Some((e, t)) = ctl.snapshot_new(local_epoch.get()) {
                    local_epoch.set(e);
                    // 视频侧复位：丢弃解码器内部残帧与通道内旧包
                    decoder.flush();
                    while video_rx.try_recv().is_ok() {}
                    last_pts.set(None);
                    just_seeked.set(true);
                    if let Some(target_sec) = t {
                        clock.seek_to(target_sec);
                    }
                    ctl.report_ready(e);
                    continue;
                }
                // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
                // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
                match video_rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(packet) => {
                        let _ = decoder.send_packet(&packet);
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        let _ = decoder.send_eof();
                        eof = true;
                    }
                }
                let mut decoded = ffmpeg_next::util::frame::video::Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    process_frame(&decoded);
                }
            }
        }

        // 视频播完后等待音频播完再结束，避免音频尾部被截断；
        // 阻塞在条件变量上，由音频线程播完时唤醒，避免忙轮询。
        // 等待期间仍响应 seek：demux 若重新发包，排空即可（解码循环已结束）
        let (done_lock, done_cond) = &*sync;
        let mut done_guard = done_lock.lock().unwrap();
        while audio_done.load(Ordering::Relaxed) == 0 {
            if let Some((e, _t)) = ctl.snapshot_new(local_epoch.get()) {
                local_epoch.set(e);
                while video_rx.try_recv().is_ok() {}
                ctl.report_ready(e);
                continue;
            }
            done_guard = done_cond
                .wait_timeout(done_guard, Duration::from_millis(50))
                .unwrap()
                .0;
        }
        drop(done_guard);
        if tx.send(Message::End).is_ok() {
            app::awake();
        }

        // 驻留期：本源临近结束但 demux 可能还活着（用户 seek 会重新发包）。
        // 维持就绪屏障不悬挂、丢弃再发的包；demux 退出（通道断开）即收尾。
        loop {
            if let Some((e, _t)) = ctl.snapshot_new(local_epoch.get()) {
                local_epoch.set(e);
                while video_rx.try_recv().is_ok() {}
                ctl.report_ready(e);
                continue;
            }
            match video_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(_p) => {} // seek 后再发的包：解码循环已结束，直接丢弃
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    });
}
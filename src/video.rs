use anyhow::Result;
use ffmpeg_next::codec;
use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling::{context::Context as ScaleContext, flag::Flags as ScaleFlags};
use fltk::app;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::clock::MasterClock;
use crate::frame::{Message, YuvFrame};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
// UI 侧用有界通道接收：UI 停顿时 send 阻塞产生背压，防止帧无限堆积；
// 每次发送后需 app::awake() 唤醒 fltk 事件循环（替代 fltk channel 的内置唤醒）。
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
) {
    std::thread::spawn(move || -> Result<()> {
        if let (Some(params), Some(time_base)) = (video_parameters, video_time_base) {
            let mut decoder = codec::context::Context::from_parameters(params)?
                .decoder()
                .video()?;

            // 非 YUV420P 输出（10-bit、4:2:2 / 4:4:4 等）经 swscale 统一转成 420P，
            // 渲染端只需处理一种平面布局。转换器惰性创建，格式/尺寸变化时重建。
            let mut scaler: Option<(ScaleContext, (Pixel, u32, u32))> = None;
            let mut conv = ffmpeg_next::util::frame::video::Video::empty();

            let mut process_frame = |decoded: &ffmpeg_next::util::frame::video::Video| {
                // 暂停：在锁内等待，恢复时由键盘处理器唤醒，避免忙轮询
                let (pause_lock, pause_cond) = &*paused;
                let mut guard = pause_lock.lock().unwrap();
                while *guard {
                    guard = pause_cond.wait(guard).unwrap();
                }
                drop(guard);

                let pts = decoded.pts().unwrap_or(0);
                let video_pts_sec =
                    pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;

                let audio_time_sec = clock.get_time_sec();
                let diff = video_pts_sec - audio_time_sec;
                // 混合睡眠：远离 PTS 时先长睡，最后 ~3ms 以 250µs 短睡逼近，
                // 消除 thread::sleep 的定时器超睡抖动带来的周期性微卡顿
                if diff > 0.005 {
                    if diff > 0.008 {
                        std::thread::sleep(Duration::from_secs_f64(diff - 0.003));
                    }
                    while clock.get_time_sec() < video_pts_sec {
                        std::thread::sleep(Duration::from_micros(250));
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
                    },
                };

                frame_data.y.copy_from_slice(&src.data(0)[..y_len]);
                frame_data.u.copy_from_slice(&src.data(1)[..uv_len]);
                frame_data.v.copy_from_slice(&src.data(2)[..uv_len]);
                frame_data.width = w as i32;
                frame_data.height = h as i32;
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
                match video_rx.recv() {
                    Ok(packet) => {
                        let _ = decoder.send_packet(&packet);
                    }
                    Err(_) => {
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
        // 阻塞在条件变量上，由音频线程播完时唤醒，避免忙轮询
        let (done_lock, done_cond) = &*sync;
        let mut done_guard = done_lock.lock().unwrap();
        while audio_done.load(Ordering::Relaxed) == 0 {
            done_guard = done_cond.wait(done_guard).unwrap();
        }
        drop(done_guard);
        if tx.send(Message::End).is_ok() {
            app::awake();
        }
        Ok(())
    });
}

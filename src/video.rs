use anyhow::Result;
use ffmpeg_next::codec;
use ffmpeg_next::codec::packet::Packet;
use fltk::app::Sender;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::clock::MasterClock;
use crate::frame::{Message, YuvFrame};

// 视频解码线程：从 demux 通道收 packet 解码，按主时钟同步节奏发送给 UI。
#[allow(clippy::too_many_arguments)]
pub fn spawn_video_thread(
    tx: Sender<Message>,
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

            let process_frame = |decoded: &ffmpeg_next::util::frame::video::Video| {
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
                if diff > 0.005 {
                    std::thread::sleep(Duration::from_secs_f64(diff));
                } else if diff < -0.050 {
                    return;
                }

                let h = decoded.height() as usize;
                let y_stride = decoded.stride(0);
                let uv_stride = decoded.stride(1);

                let mut frame_data = recycle_rx.try_recv().unwrap_or_else(|_| YuvFrame {
                    y: vec![0; y_stride * h],
                    u: vec![0; uv_stride * (h / 2)],
                    v: vec![0; uv_stride * (h / 2)],
                    width: decoded.width() as i32,
                    height: decoded.height() as i32,
                    y_stride: y_stride as i32,
                    uv_stride: uv_stride as i32,
                    pts_sec: video_pts_sec,
                });

                frame_data
                    .y
                    .copy_from_slice(&decoded.data(0)[..y_stride * h]);
                frame_data
                    .u
                    .copy_from_slice(&decoded.data(1)[..uv_stride * (h / 2)]);
                frame_data
                    .v
                    .copy_from_slice(&decoded.data(2)[..uv_stride * (h / 2)]);
                frame_data.width = decoded.width() as i32;
                frame_data.height = decoded.height() as i32;
                frame_data.y_stride = y_stride as i32;
                frame_data.uv_stride = uv_stride as i32;
                frame_data.pts_sec = video_pts_sec;

                tx.send(Message::Frame(frame_data));
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
        tx.send(Message::End);
        Ok(())
    });
}

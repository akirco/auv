mod audio;
mod clock;
mod demux;
mod frame;
mod render;
mod util;
mod video;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ffmpeg_next::{codec, media};
use fltk::{app, prelude::*, window::GlWindow};
use ringbuf::HeapRb;
use ringbuf::traits::*;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use clock::MasterClock;
use frame::{Message, YuvFrame};
use render::draw_frame;
use util::{calc_display_size, hyprctl_toggle_fullscreen, is_hyprland};

// 单个媒体源的流信息
type StreamInfo = (
    Option<usize>,                         // video_index
    Option<usize>,                         // audio_index
    Option<codec::parameters::Parameters>, // video_params
    Option<codec::parameters::Parameters>, // audio_params
    Option<ffmpeg_next::Rational>,         // video_time_base
    u32,                                   // video_w
    u32,                                   // video_h
);

// 从已打开的 demux 上下文提取视频/音频流信息，供各解码线程使用
fn extract_streams(ictx: &ffmpeg_next::format::context::Input) -> Result<StreamInfo> {
    let video_stream = ictx.streams().best(media::Type::Video);
    let audio_stream = ictx.streams().best(media::Type::Audio);
    let video_params = video_stream.as_ref().map(|s| s.parameters().clone());
    let audio_params = audio_stream.as_ref().map(|s| s.parameters().clone());
    let (video_w, video_h) = match video_params.as_ref() {
        Some(p) => {
            let dec = codec::context::Context::from_parameters(p.clone())?
                .decoder()
                .video()?;
            (dec.width(), dec.height())
        }
        None => return Err(anyhow::anyhow!("No video stream found")),
    };
    Ok((
        video_stream.as_ref().map(|s| s.index()),
        audio_stream.as_ref().map(|s| s.index()),
        video_params,
        audio_params,
        video_stream.as_ref().map(|s| s.time_base()),
        video_w,
        video_h,
    ))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let sources = util::parse_sources(&args[1..])?;
    if sources.is_empty() {
        return Err(anyhow::anyhow!(
            "Usage: auv <file_or_url>... [-p|--playlist <list.txt>]\nPage links (bilibili, youtube, ...) require yt-dlp installed;\nytdlp options like --cookies/--cookies-from-browser go into ~/.config/yt-dlp/config"
        ));
    }

    ffmpeg_next::init()?;
    // 压掉 ffmpeg 的 INFO 日志（如 HLS 每个分片的 "Opening ... for reading"），
    // 只保留真正的错误输出
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Error);

    let app = app::App::default();
    let (screen_w, screen_h) = app::screen_size();

    // 预创建窗口（尺寸在首个成功打开的源处确定后再调整并显示）
    let mut win = GlWindow::new(0, 0, 640, 480, "");
    win.set_mode(fltk::enums::Mode::Opengl3);
    win.end();

    // 跨源共享的句柄：当前音频流（按键处理器用）、回收通道发送端（UI 用）、帧缓存
    let stream_handle = Rc::new(RefCell::new(None::<cpal::Stream>));
    let recycle_handle = Rc::new(RefCell::new(None::<mpsc::Sender<YuvFrame>>));
    let frame_store = Rc::new(RefCell::new(None::<YuvFrame>));
    let paused = Arc::new((Mutex::new(false), Condvar::new()));

    // 音频输出设备配置只探测一次，跨源不变
    let host = cpal::default_host();
    let audio_device = host.default_output_device();
    let (sample_rate, channels, stream_config) = if let Some(device) = audio_device.as_ref() {
        match device.default_output_config() {
            Ok(cfg) => (cfg.sample_rate(), cfg.channels() as u32, Some(cfg.config())),
            Err(_) => (0, 0, None),
        }
    } else {
        (0, 0, None)
    };
    let has_audio = stream_config.is_some();

    let mut played_any = false;
    let mut gl_ready = false;

    for source in &sources {
        // 每个源独立打开一次 ictx（demux 线程会整体消费它）；
        // 页面链接（B 站等）经 yt-dlp 桥接时需持有子进程直到本源播完
        let (ictx, mut ytdlp_child) = match util::open_input(source) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Skipping {}: {}", source, e);
                continue;
            }
        };
        let (
            video_index,
            audio_index,
            video_params,
            audio_params,
            video_time_base,
            video_w,
            video_h,
        ) = match extract_streams(&ictx) {
            Ok(info) => info,
            Err(e) => {
                eprintln!("Skipping {}: {}", source, e);
                continue;
            }
        };
        let (disp_w, disp_h) = calc_display_size(video_w, video_h);
        let title = format!("AUV (A/V Sync Engine) - {}", source);

        if !gl_ready {
            // 首个源：确定窗口尺寸并显示，初始化 GL 上下文，注册绘制/按键回调
            let win_x = ((screen_w as i32 - disp_w as i32) / 2).max(0);
            let win_y = ((screen_h as i32 - disp_h as i32) / 2 - 20).max(0);
            win.resize(win_x, win_y, disp_w as i32, disp_h as i32);
            win.set_label(title.as_str());
            win.show();

            win.make_current();
            // 呈现节奏锁定显示器刷新率，避免无谓的满速 present 抢占 GPU
            win.set_swap_interval(1);
            gl::load_with(|s| win.get_proc_address(s) as *const _);
            let rs = unsafe { render::setup_opengl() };

            let draw_store = frame_store.clone();
            win.draw(move |w| {
                let frame_guard = draw_store.borrow();
                if let Some(frame) = frame_guard.as_ref() {
                    draw_frame(w, &rs, frame);
                }
            });

            let app_close = app;
            let paused_keys = paused.clone();
            let stream_handle_keys = stream_handle.clone();
            win.handle(move |w, ev| {
                if ev == fltk::enums::Event::Close {
                    app_close.quit();
                    return true;
                }
                if ev == fltk::enums::Event::KeyDown {
                    let key = app::event_key();
                    if key == fltk::enums::Key::from_char(' ') {
                        // 在锁内翻转暂停状态并通知视频线程，避免通知丢失导致其永久睡眠
                        let mut guard = paused_keys.0.lock().unwrap();
                        *guard = !*guard;
                        let paused_now = *guard;
                        if let Some(s) = stream_handle_keys.borrow().as_ref() {
                            if paused_now {
                                let _ = s.pause();
                            } else {
                                let _ = s.play();
                            }
                        }
                        paused_keys.1.notify_all();
                    } else if key == fltk::enums::Key::from_char('f')
                        || key == fltk::enums::Key::from_char('F')
                    {
                        if is_hyprland() {
                            hyprctl_toggle_fullscreen();
                        } else {
                            w.fullscreen(!w.fullscreen_active());
                        }
                    }
                }
                false
            });
            gl_ready = true;
        } else {
            // 后续源：调整窗口尺寸与标题，复用 GL 上下文
            let win_x = ((screen_w as i32 - disp_w as i32) / 2).max(0);
            let win_y = ((screen_h as i32 - disp_h as i32) / 2 - 20).max(0);
            win.resize(win_x, win_y, disp_w as i32, disp_h as i32);
            win.set_label(title.as_str());
            win.redraw();
        }

        // 该源的主时钟（每个源独立计时，从 0 开始）
        let (master_clock, samples_played_arc) = MasterClock::new(sample_rate, channels, has_audio);

        // 该源的音频输出流与环形缓冲
        let (audio_stream_opt, mut audio_producer) =
            if let (Some(device), Some(config)) = (audio_device.as_ref(), stream_config.as_ref()) {
                // 容量为 2 秒的无锁环形缓冲区
                let ring_buf = HeapRb::<f32>::new((sample_rate * channels * 2) as usize);
                let (producer, mut consumer) = ring_buf.split();
                let samples_played_cb = samples_played_arc.clone();
                let stream = device.build_output_stream(
                    *config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        // 批量弹出音频样本，比逐样本 pop 开销更小
                        let filled = consumer.pop_slice(data);
                        for sample in &mut data[filled..] {
                            *sample = 0.0; // 缓冲区空时输出静音，防止爆音
                        }
                        // 只把实际交付的样本计入主时钟：起播缓冲期时钟不空转，
                        // 视频从 pts≈0 起同步；中途下溢时时钟暂停，视频等待
                        // 而非丢帧，保持音画对齐（无声卡时退回系统时钟）
                        samples_played_cb.fetch_add(filled as u64, Ordering::Relaxed);
                    },
                    |err| eprintln!("Audio Stream Error: {}", err),
                    None,
                );
                match stream {
                    // 播放启动失败时按无音频路径走（producer 一并丢弃），
                    // 否则环形缓冲永不排空会让视频线程死等
                    Ok(s) => match s.play() {
                        Ok(()) => (Some(s), Some(producer)),
                        Err(e) => {
                            eprintln!("Audio output failed to start: {}", e);
                            (None, None)
                        }
                    },
                    Err(e) => {
                        eprintln!("Audio output init failed: {}", e);
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };
        *stream_handle.borrow_mut() = audio_stream_opt;

        // 换源时复位暂停状态
        *paused.0.lock().unwrap() = false;
        paused.1.notify_all();

        // 本源的通道与线程。
        // 有界 sync_channel：UI 线程卡顿（如合成器阻塞 swap）时视频线程在 send
        // 上被背压阻塞，杜绝帧在无界队列里以每帧数 MB 的速度无限堆积；
        // 发送后用 app::awake() 手动唤醒事件循环（见 video.rs）。
        let (tx, rx) = std::sync::mpsc::sync_channel::<Message>(2);
        let (recycle_tx, recycle_rx) = mpsc::channel::<YuvFrame>();
        *recycle_handle.borrow_mut() = Some(recycle_tx);

        // 有界通道：demux 线程分发包给两个解码线程；容量即预缓冲深度，
        // 消费跟不上时 demux 阻塞，自动把网络下载节流到播放速度。
        let (video_tx, video_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(100);
        let (audio_tx, audio_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(200);

        // 解复用线程：统一读取媒体 packet 并分发
        demux::spawn_demux_thread(ictx, video_index, audio_index, video_tx, audio_tx);

        // 音频是否已全部播完（含环形缓冲排空）。视频线程据此决定何时结束播放，
        // 避免视频先播完就把还没播出的音频尾部截断。
        let audio_done = Arc::new(AtomicU64::new(0));
        // 视频线程在"等音频播完"时阻塞于此，由音频线程播完时唤醒，避免忙轮询
        let sync = Arc::new((Mutex::new(()), Condvar::new()));

        // 无音频输出或无音频流时，直接标记音频完成
        if audio_producer.is_none() || audio_index.is_none() {
            audio_done.store(1, Ordering::Relaxed);
            sync.1.notify_all();
        }

        // 音频解码线程：从 demux 通道收 packet 解码、重采样后喂入环形缓冲
        if let (Some(producer), Some(audio_parameters)) = (audio_producer.take(), audio_params)
            && audio_index.is_some()
        {
            audio::spawn_audio_thread(
                audio_rx,
                audio_parameters,
                producer,
                sample_rate,
                audio_done.clone(),
                sync.clone(),
                paused.clone(),
            );
        }

        // 视频解码与同步线程
        video::spawn_video_thread(
            tx,
            recycle_rx,
            master_clock,
            paused.clone(),
            audio_done.clone(),
            sync.clone(),
            video_rx,
            video_params,
            video_time_base,
        );

        played_any = true;

        // 收到 End 结束本源，继续下一个源
        let mut finished = false;
        while app.wait() && !finished {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Message::Frame(new_frame) => {
                        if let Some(old_frame) = frame_store.borrow_mut().replace(new_frame)
                            && let Some(tx) = recycle_handle.borrow().as_ref()
                        {
                            let _ = tx.send(old_frame);
                        }
                        win.redraw();
                    }
                    Message::End => {
                        // 丢弃本源残留帧与通道，避免旧数据串到下一源
                        frame_store.borrow_mut().take();
                        *recycle_handle.borrow_mut() = None;
                        *stream_handle.borrow_mut() = None;
                        finished = true;
                        break;
                    }
                }
            }
        }

        // 收尾本源的 yt-dlp 子进程（自然结束则回收，提前退出则终止）
        util::reap_child(&mut ytdlp_child);

        // 用户关闭了窗口：终止整个播放列表
        if !win.visible() {
            break;
        }
    }

    if !played_any {
        return Err(anyhow::anyhow!("No media source could be opened"));
    }
    Ok(())
}

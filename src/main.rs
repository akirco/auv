mod audio;
mod cli;
mod clock;
mod demux;
mod frame;
mod render;
mod screenshot;
mod ui;
mod util;
mod video;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use fltk::{app, prelude::*, window::GlWindow};
use log::{error, warn};
use ringbuf::HeapRb;
use ringbuf::traits::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};

use cli::{parse_cli, CliAction, USAGE};
use clock::MasterClock;
use demux::{extract_streams, StreamInfo};
use frame::{Message, YuvFrame};
use ui::{calc_display_size, format_time, PlayerState};

fn main() -> Result<()> {
    // 日志分级（RUST_LOG 控制）：默认 info；worker 线程与主循环共用同一 logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // 统一的 panic 钩子：线程 panic 不再让进程裸崩溃（已去 panic=abort），
    // 打印现场与 backtrace 提示，便于定位工作线程问题
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        eprintln!("[auv panic] thread '{name}': {info}");
        if std::env::var("RUST_BACKTRACE").map(|v| v != "0").unwrap_or(false) {
            let bt = std::backtrace::Backtrace::capture();
            eprintln!("backtrace:\n{bt}");
        } else {
            eprintln!("(set RUST_BACKTRACE=1 for a backtrace)");
        }
    }));

    let args: Vec<String> = std::env::args().collect();
    let sources = match parse_cli(&args[1..])? {
        CliAction::Play(sources) => sources,
        CliAction::Help => {
            println!("{}", USAGE);
            return Ok(());
        }
        CliAction::Version => {
            println!("auv {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
    };
    if sources.is_empty() {
        eprintln!("{}", USAGE);
        return Err(anyhow::anyhow!("no media source given"));
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

    // 播放器状态：跨源共享的句柄集中管理（绘制/按键回调与 UI 循环共用）
    let state = PlayerState::new();
    let paused = state.paused.clone();
    let dur_handle = state.dur_handle.clone();
    let volume = state.volume.clone();
    let mute = state.mute.clone();

    // 音频输出设备配置只探测一次，跨源不变
    let host = cpal::default_host();
    let audio_device = host.default_output_device();
    let (sample_rate, channels, stream_config) =
        match audio_device.as_ref().and_then(audio::prepare_audio_config) {
            Some(cfg) => (cfg.sample_rate, cfg.channels as u32, Some(cfg)),
            None => (0, 0, None),
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
                warn!("Skipping {}: {}", source, e);
                continue;
            }
        };
        let StreamInfo {
            video_index,
            audio_index,
            video_params,
            audio_params,
            video_time_base,
            disp_w,
            disp_h,
        } = match extract_streams(&ictx) {
            Ok(info) => info,
            Err(e) => {
                warn!("Skipping {}: {}", source, e);
                continue;
            }
        };
        let (disp_w, disp_h) = (disp_w.max(1), disp_h.max(1));
        let win_size = calc_display_size(disp_w, disp_h);
        let title_base = format!("AUV (A/V Sync Engine) - {}", source);

        if !gl_ready {
            // 首个源：确定窗口尺寸并显示，初始化 GL 上下文，注册绘制/按键回调
            let win_x = ((screen_w as i32 - win_size.0 as i32) / 2).max(0);
            let win_y = ((screen_h as i32 - win_size.1 as i32) / 2 - 20).max(0);
            win.resize(win_x, win_y, win_size.0 as i32, win_size.1 as i32);
            win.set_label(title_base.as_str());
            win.show();

            let rs = ui::setup_gl(&mut win);
            ui::register_window_callbacks(&mut win, &app, &state, rs);
            gl_ready = true;
        } else {
            // 后续源：调整窗口尺寸与标题，复用 GL 上下文
            let win_x = ((screen_w as i32 - win_size.0 as i32) / 2).max(0);
            let win_y = ((screen_h as i32 - win_size.1 as i32) / 2 - 20).max(0);
            win.resize(win_x, win_y, win_size.0 as i32, win_size.1 as i32);
            win.set_label(title_base.as_str());
            win.redraw();
        }

        // 该源的主时钟（每个源独立计时，从 0 开始）。
        // audio_dead：音频输出失效标记，出错时置 1，主时钟回退墙钟、
        // 音频线程中止，避免设备挂掉后播放永久卡死；每源独立避免旧流污染新源
        let audio_dead = Arc::new(AtomicU64::new(0));
        let (master_clock, samples_played_arc) =
            MasterClock::new(sample_rate, channels, has_audio, audio_dead.clone());

        // 环形缓冲清空标记：seek 时由音频线程置位，cpal 回调下次运行时清空环缓冲
        // （Producer 端没有 clear，清空只能由持有 consumer 的回调完成）
        let ring_clear = Arc::new(Mutex::new(false));

        // 该源的音频输出流与环形缓冲
        let (audio_stream_opt, mut audio_producer) =
            if let (Some(device), Some(config)) = (audio_device.as_ref(), stream_config.as_ref()) {
                // 容量为 2 秒的无锁环形缓冲区
                let ring_buf = HeapRb::<f32>::new((sample_rate * channels * 2) as usize);
                let (producer, mut consumer) = ring_buf.split();
                let samples_played_cb = samples_played_arc.clone();
                let audio_dead_cb = audio_dead.clone();
                let volume_cb = volume.clone();
                let mute_cb = mute.clone();
                let ring_clear_cb = ring_clear.clone();
                let stream = device.build_output_stream(
                    *config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        // seek 后清空旧音频：置位由音频线程发起，这里实际清空
                        if *ring_clear_cb.lock().unwrap() {
                            consumer.clear();
                            *ring_clear_cb.lock().unwrap() = false;
                        }
                        // 批量弹出音频样本，比逐样本 pop 开销更小
                        let filled = consumer.pop_slice(data);
                        // 音量/静音：按共享百分比缩放（静音系数 0）。
                        // 系数为 1 时跳过循环，常见路径零开销
                        let gain = if mute_cb.load(Ordering::Relaxed) != 0 {
                            0.0
                        } else {
                            (volume_cb.load(Ordering::Relaxed) as f32 / 100.0).clamp(0.0, 2.0)
                        };
                        if gain != 1.0 {
                            for sample in &mut data[..filled] {
                                *sample *= gain;
                            }
                        }
                        for sample in &mut data[filled..] {
                            *sample = 0.0; // 缓冲区空时输出静音，防止爆音
                        }
                        // 只把实际交付的样本计入主时钟：起播缓冲期时钟不空转，
                        // 视频从 pts≈0 起同步；中途下溢时时钟暂停，视频等待
                        // 而非丢帧，保持音画对齐（无声卡时退回系统时钟）
                        samples_played_cb.fetch_add(filled as u64, Ordering::Relaxed);
                    },
                    move |err| {
                        // 设备出错（拔掉、驱动故障等）后置失效标记并只提示一次。
                        // 主时钟随后回退系统时钟，音频线程中止，播放不卡死
                        if audio_dead_cb.swap(1, Ordering::Relaxed) == 0 {
                            error!("Audio Stream Error: {}", err);
                        }
                    },
                    None,
                );
                match stream {
                    // 播放启动失败时按无音频路径走（producer 一并丢弃），
                    // 否则环形缓冲永不排空会让视频线程死等
                    Ok(s) => match s.play() {
                        Ok(()) => (Some(s), Some(producer)),
                        Err(e) => {
                            error!("Audio output failed to start: {}", e);
                            (None, None)
                        }
                    },
                    Err(e) => {
                        error!("Audio output init failed: {}", e);
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };
        *state.stream_handle.borrow_mut() = audio_stream_opt;

        // 本源的 seek 协调器：就绪屏障需要视频线程（恒为 1）与音频线程（按需 +1）。
        // 必须在本源线程启动前建好，各线程共享同一份
        let has_audio_thread =
            audio_producer.is_some() && audio_index.is_some() && audio_params.is_some();
        let seek_ctl = Arc::new(demux::SeekCtl::new(1 + usize::from(has_audio_thread)));
        *state.seek_handle.borrow_mut() = Some(seek_ctl.clone());
        // 记录总时长（微秒 → 秒）供进度显示与 seek 钳制；未知（0/负）按 0 处理
        let dur_us = ictx.duration();
        *state.dur_handle.borrow_mut() = if dur_us > 0 {
            dur_us as f64 / 1_000_000.0
        } else {
            0.0
        };

        // 换源时复位暂停状态
        *paused.0.lock().unwrap() = false;
        paused.1.notify_all();

        // 本源的通道与线程。
        // 有界 sync_channel：UI 线程卡顿（如合成器阻塞 swap）时视频线程在 send
        // 上被背压阻塞，杜绝帧在无界队列里以每帧数 MB 的速度无限堆积；
        // 发送后用 app::awake() 手动唤醒事件循环（见 video.rs）。
        let (tx, rx) = std::sync::mpsc::sync_channel::<Message>(2);
        let (recycle_tx, recycle_rx) = mpsc::channel::<YuvFrame>();
        *state.recycle_handle.borrow_mut() = Some(recycle_tx);

        // 有界通道：demux 线程分发包给两个解码线程；容量即预缓冲深度，
        // 消费跟不上时 demux 阻塞，自动把网络下载节流到播放速度。
        let (video_tx, video_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(100);
        let (audio_tx, audio_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(200);

        // 解复用线程：统一读取媒体 packet 并分发
        demux::spawn_demux_thread(ictx, video_index, audio_index, video_tx, audio_tx, seek_ctl.clone());

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
                channels,
                audio_done.clone(),
                sync.clone(),
                paused.clone(),
                audio_dead.clone(),
                seek_ctl.clone(),
                samples_played_arc.clone(),
                ring_clear.clone(),
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
            seek_ctl.clone(),
        );

        played_any = true;

        // 收到 End 结束本源，继续下一个源
        let mut finished = false;
        // 上次写进窗口标题的整秒进度：只有整秒变化才重设标题，避免每帧刷新
        let mut last_title_sec = u64::MAX;

        while app.wait() && !finished {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Message::Frame(new_frame) => {
                        // 进度显示：窗口标题反映当前播放位置/总时长
                        let cur_sec = new_frame.pts_sec;
                        if cur_sec as u64 != last_title_sec {
                            last_title_sec = cur_sec as u64;
                            let dur = *dur_handle.borrow();
                            let label = if dur > 0.0 {
                                format!(
                                    "{} [{}/{}]",
                                    title_base,
                                    format_time(cur_sec),
                                    format_time(dur)
                                )
                            } else {
                                format!("{} [{}]", title_base, format_time(cur_sec))
                            };
                            win.set_label(&label);
                        }
                        if let Some(old_frame) = state.frame_store.borrow_mut().replace(new_frame)
                            && let Some(tx) = state.recycle_handle.borrow().as_ref()
                        {
                            let _ = tx.send(old_frame);
                        }
                        win.redraw();
                    }
                    Message::End => {
                        // 丢弃本源残留帧与通道，避免旧数据串到下一源
                        state.frame_store.borrow_mut().take();
                        *state.recycle_handle.borrow_mut() = None;
                        *state.stream_handle.borrow_mut() = None;
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
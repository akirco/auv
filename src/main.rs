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

// 挑选音频输出配置：放大 ALSA period（cpal 内部再按 2× 双缓冲），
// 给实时 worker 线程更多容错余量，降低偶发下溢（xrun/EIO）触发率。
// 采样格式取 F32（与环缓冲样本类型一致）、采样率/声道沿用设备默认；
// 优先指到 2048 帧（≈43ms@48kHz）的 period，超出设备支持范围时退而取上限。
// 解析失败时回退设备默认配置。
fn prepare_audio_config(device: &cpal::Device) -> Option<cpal::StreamConfig> {
    let default = device.default_output_config().ok()?;
    let rate = default.sample_rate();
    let ch = default.channels();
    // 找"F32 + 默认声道 + 覆盖默认采样率"的声明范围；找不到就用默认配置
    let ranges: Vec<_> = device.supported_output_configs().ok()?.collect();
    let sc = ranges
        .iter()
        .find(|r| {
            r.sample_format() == cpal::SampleFormat::F32
                && r.channels() == ch
                && r.min_sample_rate() <= rate
                && rate <= r.max_sample_rate()
        })
        .map(|r| r.with_sample_rate(rate))
        .unwrap_or(default);
    let mut cfg = sc.config();
    // 放大 period：优先 2048 帧，不超过设备支持的周期上限（各有 lower/higher bound）
    if let cpal::SupportedBufferSize::Range { min, max } = sc.buffer_size() {
        let want = 2048u32.clamp(*min, *max);
        if want != 2048 {
            eprintln!(
                "Audio: device period range {}-{} frames, using {}",
                min, max, want
            );
        }
        cfg.buffer_size = cpal::BufferSize::Fixed(want);
    }
    Some(cfg)
}

// 单个媒体源的流信息
type StreamInfo = (
    Option<usize>,                         // video_index
    Option<usize>,                         // audio_index
    Option<codec::parameters::Parameters>, // video_params
    Option<codec::parameters::Parameters>, // audio_params
    Option<ffmpeg_next::Rational>,         // video_time_base
    u32,                                   // video_disp_w（SAR 修正后的显示宽度）
    u32,                                   // video_disp_h
);

// 从已打开的 demux 上下文提取视频/音频流信息，供各解码线程使用
fn extract_streams(ictx: &ffmpeg_next::format::context::Input) -> Result<StreamInfo> {
    let video_stream = ictx.streams().best(media::Type::Video);
    let audio_stream = ictx.streams().best(media::Type::Audio);
    let video_params = video_stream.as_ref().map(|s| s.parameters().clone());
    let audio_params = audio_stream.as_ref().map(|s| s.parameters().clone());
    let (disp_w, disp_h) = match video_params.as_ref() {
        Some(p) => {
            let dec = codec::context::Context::from_parameters(p.clone())?
                .decoder()
                .video()?;
            let w = dec.width();
            let h = dec.height();
            let sar = dec.aspect_ratio();
            // 变形宽银幕（sar != 1）按像素宽高比修正显示宽度
            if sar.numerator() > 0 && sar.denominator() > 0 && sar.numerator() != sar.denominator() {
                let dw = ((w as u64 * sar.numerator() as u64) / sar.denominator() as u64).max(1) as u32;
                (dw, h)
            } else {
                (w, h)
            }
        }
        None => return Err(anyhow::anyhow!("No video stream found")),
    };
    Ok((
        video_stream.as_ref().map(|s| s.index()),
        audio_stream.as_ref().map(|s| s.index()),
        video_params,
        audio_params,
        video_stream.as_ref().map(|s| s.time_base()),
        disp_w,
        disp_h,
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
    // seek 协调器句柄（每源一个，按键处理器经句柄取当前源）
    let seek_handle = Rc::new(RefCell::new(None::<Arc<demux::SeekCtl>>));
    // 当前源总时长（秒）；按键处理器据此把 seek 目标钳制到 [0, 时长]
    let dur_handle = Rc::new(RefCell::new(0.0f64));
    // 音量百分比（0..=200）与静音开关（0/1）：跨源保持，在 cpal 回调里应用
    let volume = Arc::new(AtomicU64::new(100));
    let mute = Arc::new(AtomicU64::new(0));

    // 音频输出设备配置只探测一次，跨源不变
    let host = cpal::default_host();
    let audio_device = host.default_output_device();
    let (sample_rate, channels, stream_config) =
        match audio_device.as_ref().and_then(prepare_audio_config) {
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
            disp_w,
            disp_h,
        ) = match extract_streams(&ictx) {
            Ok(info) => info,
            Err(e) => {
                eprintln!("Skipping {}: {}", source, e);
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
            let frame_store_keys = frame_store.clone();
            let seek_keys = seek_handle.clone();
            let dur_keys = dur_handle.clone();
            let volume_keys = volume.clone();
            let mute_keys = mute.clone();
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
                    } else if key == fltk::enums::Key::from_char('s')
                        || key == fltk::enums::Key::from_char('S')
                    {
                        // 把当前显示的帧存为 PNG 截图（保存到当前工作目录）
                        if let Some(frame) = frame_store_keys.borrow().as_ref() {
                            match util::save_screenshot(frame) {
                                Ok(p) => eprintln!("Screenshot saved: {}", p.display()),
                                Err(e) => eprintln!("Screenshot failed: {}", e),
                            }
                        }
                    } else if key == fltk::enums::Key::Left
                        || key == fltk::enums::Key::Right
                        || key == fltk::enums::Key::from_char('[')
                        || key == fltk::enums::Key::from_char(']')
                        || key == fltk::enums::Key::Home
                    {
                        // 相对/绝对 seek：←/→ 退进 10s，[ / ] 退进 5s，Home 回到开头。
                        // 当前位置取当前显示帧的 pts；总时长未知时只保证不为负
                        let ctl = seek_keys.borrow().clone();
                        if let Some(ctl) = ctl {
                            let cur = frame_store_keys
                                .borrow()
                                .as_ref()
                                .map_or(0.0, |f| f.pts_sec);
                            let delta = if key == fltk::enums::Key::Left {
                                -10.0
                            } else if key == fltk::enums::Key::Right {
                                10.0
                            } else if key == fltk::enums::Key::from_char('[') {
                                -5.0
                            } else if key == fltk::enums::Key::from_char(']') {
                                5.0
                            } else {
                                f64::NEG_INFINITY // Home：跳到开头
                            };
                            let mut target = if delta == f64::NEG_INFINITY {
                                0.0
                            } else {
                                cur + delta
                            };
                            let dur = *dur_keys.borrow();
                            target = if dur > 0.0 {
                                target.clamp(0.0, dur)
                            } else {
                                target.max(0.0)
                            };
                            ctl.request(target);
                            // 唤醒暂停中阻塞在条件变量上的解码线程，让它们立即处理 seek
                            paused_keys.1.notify_all();
                        }
                    } else if key == fltk::enums::Key::from_char('+')
                        || key == fltk::enums::Key::from_char('=')
                    {
                        // 音量 +10%（上限 200%）
                        let _ = volume_keys.fetch_update(
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                            |v| Some((v + 10).min(200)),
                        );
                        eprintln!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
                    } else if key == fltk::enums::Key::from_char('-')
                        || key == fltk::enums::Key::from_char('_')
                    {
                        // 音量 -10%（下限 0%）
                        let _ = volume_keys.fetch_update(
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                            |v| Some(v.saturating_sub(10)),
                        );
                        eprintln!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
                    } else if key == fltk::enums::Key::from_char('m')
                        || key == fltk::enums::Key::from_char('M')
                    {
                        let was = mute_keys.fetch_xor(1, Ordering::Relaxed);
                        eprintln!("Mute: {}", if was == 0 { "ON" } else { "OFF" });
                    }
                }
                false
            });
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
                            eprintln!("Audio Stream Error: {}", err);
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

        // 本源的 seek 协调器：就绪屏障需要视频线程（恒为 1）与音频线程（按需 +1）。
        // 必须在本源线程启动前建好，各线程共享同一份
        let has_audio_thread =
            audio_producer.is_some() && audio_index.is_some() && audio_params.is_some();
        let seek_ctl = Arc::new(demux::SeekCtl::new(1 + usize::from(has_audio_thread)));
        *seek_handle.borrow_mut() = Some(seek_ctl.clone());
        // 记录总时长（微秒 → 秒）供进度显示与 seek 钳制；未知（0/负）按 0 处理
        let dur_us = ictx.duration();
        *dur_handle.borrow_mut() = if dur_us > 0 {
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
        *recycle_handle.borrow_mut() = Some(recycle_tx);

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
                                    util::format_time(cur_sec),
                                    util::format_time(dur)
                                )
                            } else {
                                format!("{} [{}]", title_base, util::format_time(cur_sec))
                            };
                            win.set_label(&label);
                        }
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

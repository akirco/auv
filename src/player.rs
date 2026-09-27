// 源的播放编排：把「一个媒体源从打开到收尾」的全部逻辑集中在这里。
// 包括：打开输入与探测流信息、确定/调整窗口、建主时钟与音频输出流、
// 启动 demux/audio/video 三个解码线程、驱动 UI 消息泵、收尾。
// main 只负责启动装配与播放列表控制，不理解单个源的内部结构。
use anyhow::Result;
use cpal::traits::HostTrait;
use fltk::{app, prelude::*, window::GlWindow};
use log::{error, info, warn};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
    mpsc,
};

use crate::{
    audio,
    clock::MasterClock,
    demux::{self, SeekCtl, StreamInfo, extract_streams},
    frame::{Message, YuvFrame},
    ui::{self, PlayerState, calc_display_size, format_time},
    util, video,
};

// 音频输出设备与配置：进程内只探测一次，跨源复用。
// 具体配置策略（F32 / 默认采样率声道 / 放大缓冲粒度）见 audio.rs。
pub struct AudioOutput {
    device: Option<cpal::Device>,
    sample_rate: u32,
    channels: u32,
    config: Option<cpal::StreamConfig>,
}

impl AudioOutput {
    pub fn detect() -> Self {
        let host = cpal::default_host();
        info!("Audio host: {}", host.id());
        let device = host.default_output_device();
        let (sample_rate, channels, config) =
            match device.as_ref().and_then(audio::prepare_audio_config) {
                Some(cfg) => (cfg.sample_rate, cfg.channels as u32, Some(cfg)),
                None => (0, 0, None),
            };
        Self {
            device,
            sample_rate,
            channels,
            config,
        }
    }
}

// 播放器：持有窗口、跨源共享状态与音频配置，逐个源推进播放列表。
pub struct Player {
    win: GlWindow,
    app: app::App,
    state: PlayerState,
    // 屏幕逻辑尺寸（app::screen_size()），用于窗口居中/缩放
    screen: (f64, f64),
    audio: AudioOutput,
    gl_ready: bool,
    played_any: bool,
}

impl Player {
    pub fn new(win: GlWindow, app: app::App, screen: (f64, f64), audio: AudioOutput) -> Self {
        Self {
            win,
            app,
            state: PlayerState::new(),
            screen,
            audio,
            gl_ready: false,
            played_any: false,
        }
    }

    // 播放一个源；返回 false 表示窗口已被关闭、播放列表应终止。
    // 打不开或没有视频流的源会记警告后跳过，返回 true 继续下一个。
    pub fn play(&mut self, source: &str) -> bool {
        // 每个源独立打开一次 ictx（demux 线程会整体消费它）；
        // 页面链接（B 站等）经 yt-dlp 桥接时需持有子进程直到本源播完
        let (ictx, mut ytdlp_child) = match util::open_input(source) {
            Ok(v) => v,
            Err(e) => {
                warn!("Skipping {}: {}", source, e);
                return true;
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
                return true;
            }
        };
        let (disp_w, disp_h) = (disp_w.max(1), disp_h.max(1));
        let win_size = calc_display_size(disp_w, disp_h);
        let title_base = format!("AUV (A/V Sync Engine) - {}", source);

        self.show_or_resize(win_size, &title_base);

        // 本源的共享句柄：与窗口回调共用同一份 Arc（来自 PlayerState）
        let paused = self.state.paused.clone();
        let dur_handle = self.state.dur_handle.clone();
        let volume = self.state.volume.clone();
        let mute = self.state.mute.clone();

        // 该源的主时钟（每个源独立计时，从 0 开始）。
        // has_audio 必须同时满足「有可用音频输出」与「本源确实有音频轨」：
        // 纯视频源若被当成有音频，采样计数永不推进，主时钟停在 0，视频会卡死首帧。
        // audio_dead：音频输出失效标记，出错时置 1，主时钟回退墙钟、
        // 音频线程中止，避免设备挂掉后播放永久卡死；每源独立避免旧流污染新源
        let audio_dead = Arc::new(AtomicU64::new(0));
        let has_audio = self.audio.config.is_some() && audio_index.is_some();
        let (master_clock, samples_played_arc) = MasterClock::new(
            self.audio.sample_rate,
            self.audio.channels,
            has_audio,
            audio_dead.clone(),
        );

        // 环形缓冲清空标记：seek 时由音频线程置位，cpal 回调下次运行时清空环缓冲
        // （Producer 端没有 clear，清空只能由持有 consumer 的回调完成）
        let ring_clear = Arc::new(Mutex::new(false));

        // 该源的音频输出流与环形缓冲（纯视频源不建流，避免白占一个设备节点）
        let (audio_stream_opt, mut audio_producer) = if let (Some(device), Some(config)) =
            (self.audio.device.as_ref(), self.audio.config.as_ref())
            && audio_index.is_some()
        {
            let build = |cfg: &cpal::StreamConfig| {
                audio::build_output_stream(
                    device,
                    cfg,
                    &samples_played_arc,
                    &audio_dead,
                    &volume,
                    &mute,
                    &ring_clear,
                )
            };
            // 首选放大后的固定缓冲粒度；设备声明的范围可能失真（如 ALSA 经
            // PipeWire 插件时在设定采样率后才把 period 固定为 1024），建流失败
            // 时退回设备默认配置重试。仍失败则按无音频路径走（producer 一并丢弃），
            // 否则环形缓冲永不排空会让视频线程死等
            match build(config).or_else(|e| {
                warn!(
                    "Audio: fixed period rejected ({}), retrying with device default buffer",
                    e
                );
                let fallback = cpal::StreamConfig {
                    buffer_size: cpal::BufferSize::Default,
                    ..*config
                };
                build(&fallback)
            }) {
                Ok((stream, producer)) => (Some(stream), Some(producer)),
                Err(e) => {
                    // 建流彻底失败：置失效标记让主时钟退回墙钟，否则采样计数
                    // 永不推进，视频会卡死首帧
                    audio_dead.store(1, Ordering::Relaxed);
                    error!("Audio output init failed: {}", e);
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
        *self.state.stream_handle.borrow_mut() = audio_stream_opt;

        // 本源的 seek 协调器：就绪屏障需要视频线程（恒为 1）与音频线程（按需 +1）。
        // 必须在本源线程启动前建好，各线程共享同一份
        let has_audio_thread =
            audio_producer.is_some() && audio_index.is_some() && audio_params.is_some();
        let seek_ctl = Arc::new(SeekCtl::new(1 + usize::from(has_audio_thread)));
        *self.state.seek_handle.borrow_mut() = Some(seek_ctl.clone());
        // 记录总时长（微秒 → 秒）供进度显示与 seek 钳制；未知（0/负）按 0 处理
        let dur_us = ictx.duration();
        *self.state.dur_handle.borrow_mut() = if dur_us > 0 {
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
        *self.state.recycle_handle.borrow_mut() = Some(recycle_tx);

        // 有界通道：demux 线程分发包给两个解码线程；容量即预缓冲深度，
        // 消费跟不上时 demux 阻塞，自动把网络下载节流到播放速度。
        let (video_tx, video_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(100);
        let (audio_tx, audio_rx) = mpsc::sync_channel::<ffmpeg_next::codec::packet::Packet>(200);

        // 解复用线程：统一读取媒体 packet 并分发
        demux::spawn_demux_thread(
            ictx,
            video_index,
            audio_index,
            video_tx,
            audio_tx,
            seek_ctl.clone(),
        );

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
            audio::spawn_audio_thread(audio::AudioThreadCtx {
                audio_rx,
                audio_parameters,
                producer,
                sample_rate: self.audio.sample_rate,
                channels: self.audio.channels,
                audio_done: audio_done.clone(),
                done_cond: sync.clone(),
                paused: paused.clone(),
                audio_dead: audio_dead.clone(),
                ctl: seek_ctl.clone(),
                samples_played: samples_played_arc.clone(),
                ring_clear: ring_clear.clone(),
            });
        }

        // 视频解码与同步线程
        video::spawn_video_thread(video::VideoThreadCtx {
            tx,
            recycle_rx,
            clock: master_clock,
            paused: paused.clone(),
            audio_done: audio_done.clone(),
            sync: sync.clone(),
            video_rx,
            video_parameters: video_params,
            video_time_base,
            ctl: seek_ctl.clone(),
        });

        self.played_any = true;

        // 收到 End 结束本源，继续下一个源
        let mut finished = false;
        // 上次写进窗口标题的整秒进度：只有整秒变化才重设标题，避免每帧刷新
        let mut last_title_sec = u64::MAX;

        while self.app.wait() && !finished {
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
                            self.win.set_label(&label);
                        }
                        if let Some(old_frame) =
                            self.state.frame_store.borrow_mut().replace(new_frame)
                            && let Some(tx) = self.state.recycle_handle.borrow().as_ref()
                        {
                            let _ = tx.send(old_frame);
                        }
                        self.win.redraw();
                    }
                    Message::End => {
                        // 丢弃本源残留帧与通道，避免旧数据串到下一源
                        self.state.frame_store.borrow_mut().take();
                        *self.state.recycle_handle.borrow_mut() = None;
                        *self.state.stream_handle.borrow_mut() = None;
                        finished = true;
                        break;
                    }
                }
            }
        }

        // 收尾本源的 yt-dlp 子进程（自然结束则回收，提前退出则终止）
        util::reap_child(&mut ytdlp_child);

        // 窗口仍可见则继续下一个源；用户已关闭则终止播放列表
        self.win.visible()
    }

    // 全部源结束后调用：一个源都没播成功时返回错误
    pub fn finish(&self) -> Result<()> {
        if self.played_any {
            Ok(())
        } else {
            Err(anyhow::anyhow!("No media source could be opened"))
        }
    }

    // 首个源显示窗口并初始化 GL/回调；后续源只调整尺寸与标题，复用 GL 上下文
    fn show_or_resize(&mut self, (w, h): (u32, u32), title: &str) {
        let win_x = ((self.screen.0 as i32 - w as i32) / 2).max(0);
        let win_y = ((self.screen.1 as i32 - h as i32) / 2 - 20).max(0);
        self.win.resize(win_x, win_y, w as i32, h as i32);
        self.win.set_label(title);

        if self.gl_ready {
            self.win.redraw();
            return;
        }
        self.win.show();
        let rs = ui::setup_gl(&mut self.win);
        ui::register_window_callbacks(&mut self.win, &self.app, &self.state, rs);
        self.gl_ready = true;
    }
}

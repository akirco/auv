// 单个媒体源的运行期：从已打开的输入开始，装配音频/视频/demux 线程，
// 驱动 UI 消息泵，最后统一收尾。Player 只负责窗口与播放列表，
// 不再直接操作本源的通道、线程与共享句柄。
use std::{
    cell::RefCell,
    process::Child,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use fltk::{app, prelude::*, window::GlWindow};
use log::{error, info, warn};

use crate::{
    audio::{self, AudioOutput},
    clock::MasterClock,
    demux::{self, SeekCtl, StreamInfo},
    frame::{Message, YuvFrame},
    pause::PauseGate,
    source::{SourceShared, SourceStats},
    ui::{PlayerState, format_time},
    util, video,
};

// 消息泵退出的原因
#[derive(Clone, Copy, Debug)]
pub enum PumpExit {
    Empty,        // 队列已空，回到事件循环等待
    End,          // 正常播完
    Disconnected, // 所有 sender 已释放（视频线程 panic/异常退出）
}

// 构造 SourceSession 所需的全部输入：由 Player 在 open+extract 之后打包
pub struct SessionInputs<'a> {
    pub ictx: ffmpeg_next::format::context::Input,
    pub ytdlp_child: Option<Child>,
    pub watchdog: Option<util::ReadWatchdog>,
    pub info: StreamInfo,
    pub title: String,
    pub dur: f64,
    pub paused: Arc<PauseGate>,
    pub volume: Arc<AtomicU32>,
    pub mute: Arc<AtomicBool>,
    pub audio: &'a AudioOutput,
    // 该源是否支持定位（seek）：yt-dlp 管道为 false，UI 不应发起 seek
    pub seekable: bool,
}

pub struct SourceSession {
    info: StreamInfo,
    title: String,
    dur: f64,
    paused: Arc<PauseGate>,
    volume: Arc<AtomicU32>,
    mute: Arc<AtomicBool>,
    audio_device: Option<cpal::Device>,
    audio_config: Option<cpal::StreamConfig>,
    audio_rate: u32,
    audio_channels: u32,
    ictx: Option<ffmpeg_next::format::context::Input>,
    ytdlp_child: Option<Child>,
    watchdog: Option<util::ReadWatchdog>,
    // 该源是否支持定位（seek）
    seekable: bool,
    // 运行期句柄（start 之后才有）
    rx: Option<mpsc::Receiver<Message>>,
    shared: Option<Arc<SourceShared>>,
    seek_ctl: Option<Arc<SeekCtl>>,
    // 线程句柄：收尾时先 stop/断通道，再短暂 join，避免主线程先退出时
    // 后台解码器（尤其 CUDA 设备）仍在做 FFmpeg 清理而崩溃
    video_handle: Option<std::thread::JoinHandle<()>>,
    audio_handle: Option<std::thread::JoinHandle<()>>,
    demux_handle: Option<std::thread::JoinHandle<()>>,
    // UI 侧状态句柄（与窗口回调共用同一份 Rc）
    frame_store: Rc<RefCell<Option<YuvFrame>>>,
    recycle_handle: Rc<RefCell<Option<mpsc::Sender<YuvFrame>>>>,
    stream_handle: Rc<RefCell<Option<cpal::Stream>>>,
    seek_handle: Rc<RefCell<Option<Arc<SeekCtl>>>>,
    dur_handle: Rc<RefCell<f64>>,
}

// 有界 join：正常情况线程几毫秒内退出；网络读/驱动清理卡住时不能让 UI
// 永久挂起，超时后 detach（进程退出前至少已给解码器清理留出时间）。
fn join_bounded(handle: Option<std::thread::JoinHandle<()>>, timeout: Duration) {
    let Some(handle) = handle else { return };
    let deadline = Instant::now() + timeout;
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

impl SourceSession {
    pub fn new(inputs: SessionInputs<'_>) -> Self {
        let SessionInputs {
            ictx,
            ytdlp_child,
            watchdog,
            info,
            title,
            dur,
            paused,
            volume,
            mute,
            audio,
            seekable,
        } = inputs;
        Self {
            info,
            title,
            dur,
            paused,
            volume,
            mute,
            audio_device: audio.device.clone(),
            audio_config: audio.config,
            audio_rate: audio.sample_rate,
            audio_channels: audio.channels,
            ictx: Some(ictx),
            ytdlp_child,
            watchdog,
            seekable,
            rx: None,
            shared: None,
            seek_ctl: None,
            video_handle: None,
            audio_handle: None,
            demux_handle: None,
            frame_store: Rc::new(RefCell::new(None)),
            recycle_handle: Rc::new(RefCell::new(None)),
            stream_handle: Rc::new(RefCell::new(None)),
            seek_handle: Rc::new(RefCell::new(None)),
            dur_handle: Rc::new(RefCell::new(0.0)),
        }
    }

    // 用 PlayerState 里共享的 Rc 句柄替换本会话自己的占位句柄，
    // 让窗口回调（绘制/按键）能看到本源的数据
    fn bind_state(&mut self, state: &PlayerState) {
        self.frame_store = state.frame_store.clone();
        self.recycle_handle = state.recycle_handle.clone();
        self.stream_handle = state.stream_handle.clone();
        self.seek_handle = state.seek_handle.clone();
        self.dur_handle = state.dur_handle.clone();
    }

    // 装配并启动本源的全部线程。音频输出失败会降级为无声/墙钟，不返回 Err。
    pub fn start(&mut self, state: &PlayerState) {
        self.bind_state(state);
        let ictx = self.ictx.take().expect("session already started");
        let video_index = self.info.video_index;
        let audio_index = self.info.audio_index;
        let video_params = self.info.video_params.take();
        let audio_params = self.info.audio_params.take();
        let video_time_base = self.info.video_time_base;
        let timeline_origin = self.info.timeline_origin;

        *self.dur_handle.borrow_mut() = self.dur;
        self.paused.reset();

        // 该源的主时钟（每个源独立计时，从 0 开始）。
        // has_audio 必须同时满足「有可用音频输出」与「本源确实有音频轨」：
        // 纯视频源若被当成有音频，采样计数永不推进，主时钟停在 0，视频会卡死首帧。
        // audio_dead：音频输出失效标记，主时钟回退墙钟、音频线程中止
        let audio_dead = Arc::new(AtomicBool::new(false));
        let has_audio = self.audio_config.is_some() && audio_index.is_some();
        let (master_clock, samples_played_arc) = MasterClock::new(
            self.audio_rate,
            self.audio_channels,
            has_audio,
            audio_dead.clone(),
        );

        // 环形缓冲清空标记：seek 时由音频线程置位，cpal 回调下次运行时清空
        let ring_clear = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(SourceStats::new());

        // 该源的音频输出流与环形缓冲（纯视频源不建流，避免白占设备节点）
        let (audio_stream_opt, mut audio_producer) = if let (Some(device), Some(config)) =
            (self.audio_device.as_ref(), self.audio_config.as_ref())
            && audio_index.is_some()
        {
            let build = |cfg: &cpal::StreamConfig| {
                audio::build_output_stream(audio::AudioOutParams {
                    device,
                    config: cfg,
                    samples_played: &samples_played_arc,
                    audio_dead: &audio_dead,
                    volume: &self.volume,
                    mute: &self.mute,
                    ring_clear: &ring_clear,
                    stats: &stats,
                })
            };
            // 先试首选 period；失败依次回退 2048 帧、设备默认缓冲。
            // 这是设备能力层面的自适应，不改变已建流运行中的 buffer。
            let mut candidates = vec![*config];
            if !matches!(config.buffer_size, cpal::BufferSize::Fixed(2048)) {
                candidates.push(cpal::StreamConfig {
                    buffer_size: cpal::BufferSize::Fixed(2048),
                    ..*config
                });
            }
            candidates.push(cpal::StreamConfig {
                buffer_size: cpal::BufferSize::Default,
                ..*config
            });
            let mut built = None;
            let mut last_err = None;
            for (i, cfg) in candidates.iter().enumerate() {
                match build(cfg) {
                    Ok(v) => {
                        built = Some(v);
                        break;
                    }
                    Err(e) => {
                        warn!(
                            "Audio: buffer config {} rejected ({}), trying next",
                            i + 1,
                            e
                        );
                        last_err = Some(e);
                    }
                }
            }
            match built {
                Some((stream, producer, space)) => (Some(stream), Some((producer, space))),
                None => {
                    // 建流彻底失败：置失效标记让主时钟退回墙钟
                    audio_dead.store(true, Ordering::Relaxed);
                    if let Some(e) = last_err {
                        error!("Audio output init failed: {}", e);
                    }
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
        *self.stream_handle.borrow_mut() = audio_stream_opt;

        // 是否真的会启动音频解码线程。没有消费者时不能建音频包通道，
        // 否则 demux 会在音频 send 上永久阻塞（A1）
        let audio_thread = match (audio_producer.take(), audio_params) {
            (Some((producer, space)), Some(parameters)) if audio_index.is_some() => {
                Some((producer, space, parameters))
            }
            _ => None,
        };
        let has_audio_thread = audio_thread.is_some();
        log::debug!(
            "audio: track={} output={} thread={}",
            audio_index.is_some(),
            self.audio_config.is_some(),
            has_audio_thread
        );

        // seek 协调器：视频线程恒 1，音频线程按需 +1
        let seek_ctl = Arc::new(SeekCtl::new(1 + usize::from(has_audio_thread)));
        // 不可 seek 的源（yt-dlp 管道）：UI 不发起 seek，定位失败也不会复位解码器
        seek_ctl.set_seekable(self.seekable);
        *self.seek_handle.borrow_mut() = Some(seek_ctl.clone());

        // 有界通道：UI 卡顿时视频线程在 send 上背压，防止帧无限堆积
        let (tx, rx) = mpsc::sync_channel::<Message>(2);
        let (recycle_tx, recycle_rx) = mpsc::channel::<YuvFrame>();
        *self.recycle_handle.borrow_mut() = Some(recycle_tx);

        let stop = Arc::new(AtomicBool::new(false));
        let (video_tx, video_rx) = mpsc::sync_channel::<demux::TimedPacket>(100);
        let (audio_tx, audio_rx) = if has_audio_thread {
            let (tx, rx) = mpsc::sync_channel::<demux::TimedPacket>(200);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        self.demux_handle = Some(demux::spawn_demux_thread(demux::DemuxCtx {
            ictx,
            video_index,
            audio_index,
            video_tx,
            audio_tx,
            ctl: seek_ctl.clone(),
            timeline_origin,
            stop: stop.clone(),
            watchdog: self.watchdog.take(),
            paused: self.paused.clone(),
        }));

        let audio_done = Arc::new(AtomicBool::new(false));
        let sync = Arc::new((Mutex::new(()), Condvar::new()));
        if !has_audio_thread {
            audio_done.store(true, Ordering::Relaxed);
            sync.1.notify_all();
        }

        let shared = Arc::new(SourceShared::new(
            self.paused.clone(),
            audio_done.clone(),
            audio_dead.clone(),
            sync.clone(),
            seek_ctl.clone(),
            stop.clone(),
            stats.clone(),
        ));

        if let Some((producer, space, audio_parameters)) = audio_thread {
            let audio_rx = audio_rx.expect("audio channel must exist when a thread is spawned");
            self.audio_handle = Some(audio::spawn_audio_thread(audio::AudioThreadCtx {
                audio_rx,
                audio_parameters,
                producer,
                sample_rate: self.audio_rate,
                channels: self.audio_channels,
                shared: shared.clone(),
                samples_played: samples_played_arc.clone(),
                ring_clear: ring_clear.clone(),
                space,
            }));
        }

        self.video_handle = Some(video::spawn_video_thread(video::VideoThreadCtx {
            tx,
            recycle_rx,
            clock: master_clock,
            shared: shared.clone(),
            video_rx,
            video_parameters: video_params,
            video_time_base,
            timeline_origin,
        }));

        self.rx = Some(rx);
        self.seek_ctl = Some(seek_ctl);
        self.shared = Some(shared);
    }

    // 驱动消息泵，直到本源结束或窗口关闭
    pub fn pump(&self, app: &app::App, win: &mut GlWindow) -> PumpExit {
        let rx = self.rx.as_ref().expect("pump before start");
        let seek_ctl = self.seek_ctl.as_ref().expect("pump before start");
        let mut finished = false;
        let mut last_exit = PumpExit::Empty;
        let mut last_title_sec = u64::MAX;

        while !finished && app.wait() {
            let reason = loop {
                match rx.try_recv() {
                    Ok(Message::Frame(new_frame)) => {
                        // 丢弃 seek 之前的残留帧
                        if new_frame.seek_epoch != seek_ctl.epoch() {
                            if let Some(tx) = self.recycle_handle.borrow().as_ref() {
                                let _ = tx.send(new_frame);
                            }
                            continue;
                        }
                        // 进度显示：标题反映当前位置/总时长
                        let cur_sec = new_frame.pts_sec;
                        if cur_sec as u64 != last_title_sec {
                            last_title_sec = cur_sec as u64;
                            let dur = *self.dur_handle.borrow();
                            let label = if dur > 0.0 {
                                format!(
                                    "{} [{}/{}]",
                                    self.title,
                                    format_time(cur_sec),
                                    format_time(dur)
                                )
                            } else {
                                format!("{} [{}]", self.title, format_time(cur_sec))
                            };
                            win.set_label(&label);
                        }
                        if let Some(old_frame) = self.frame_store.borrow_mut().replace(new_frame)
                            && let Some(tx) = self.recycle_handle.borrow().as_ref()
                        {
                            let _ = tx.send(old_frame);
                        }
                        win.redraw();
                    }
                    Ok(Message::End) => break PumpExit::End,
                    Err(mpsc::TryRecvError::Disconnected) => break PumpExit::Disconnected,
                    Err(mpsc::TryRecvError::Empty) => break PumpExit::Empty,
                }
            };

            match reason {
                PumpExit::Empty => {}
                PumpExit::End => {
                    self.frame_store.borrow_mut().take();
                    *self.recycle_handle.borrow_mut() = None;
                    *self.stream_handle.borrow_mut() = None;
                    last_exit = PumpExit::End;
                    finished = true;
                }
                PumpExit::Disconnected => {
                    error!("frame channel closed before End; ending source");
                    self.frame_store.borrow_mut().take();
                    *self.recycle_handle.borrow_mut() = None;
                    *self.stream_handle.borrow_mut() = None;
                    last_exit = PumpExit::Disconnected;
                    finished = true;
                }
            }
        }
        last_exit
    }

    // 统一收尾：置 stop 唤醒所有阻塞线程、丢弃帧通道、回收 yt-dlp、打印统计
    pub fn shutdown(&mut self) {
        if let Some(shared) = self.shared.as_ref() {
            shared.stop.store(true, Ordering::Relaxed);
            shared.audio_done.store(true, Ordering::Relaxed);
            shared.paused.wake();
            shared.done_cond.1.notify_all();
            info!(
                "source stats: {} seeks={}",
                shared.stats.summary(),
                self.seek_ctl.as_ref().map_or(0, |c| c.request_count())
            );
        }
        // 丢弃接收端：阻塞在 send 上的视频线程会立刻拿到 Err
        self.rx = None;
        // 清空 UI 侧句柄，避免下一个源/关闭阶段的按键或绘制读到旧数据
        self.frame_store.borrow_mut().take();
        *self.recycle_handle.borrow_mut() = None;
        *self.stream_handle.borrow_mut() = None;
        *self.seek_handle.borrow_mut() = None;
        *self.dur_handle.borrow_mut() = 0.0;
        // 先等视频/音频线程把解码器、CUDA 设备等资源释放掉，再让进程退出；
        // 否则 release 下主线程可能先 exit，后台 av_buffer_unref 会踩空。
        join_bounded(self.video_handle.take(), Duration::from_secs(3));
        join_bounded(self.audio_handle.take(), Duration::from_secs(2));
        join_bounded(self.demux_handle.take(), Duration::from_secs(1));
        util::reap_child(&mut self.ytdlp_child);
    }
}

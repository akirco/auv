// 播放列表编排：打开媒体源、确定窗口、把单个源的运行期交给 session。
// 单个源的主时钟、音频输出、三个解码线程、消息泵与收尾都在 session.rs，
// 这里只保留打开/窗口/播放列表与关闭状态。
use anyhow::Result;
use fltk::{
    app,
    enums::Event,
    prelude::*,
    window::{GlWindow, Window},
};
use log::warn;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use crate::{
    demux::extract_streams,
    session::{SessionInputs, SourceSession},
    ui::{self, PlayerState, calc_display_size},
    util,
};

pub use crate::audio::AudioOutput;

// 打开一个源的结果：成功 / 失败（跳到下一个源）/ 用户关闭窗口
enum OpenOutcome {
    Opened(util::OpenedInput),
    Failed,
    Closed,
}

// 后台打开线程通过通道返回的结果；错误转成字符串以便跨线程传递
type OpenResult = std::result::Result<util::OpenedInput, String>;

// 预取中的下一个源：后台线程按"临近结尾"的时刻开始打开，结果经通道交回。
// started：线程已进入 open_input（不再处于等待开工的睡眠阶段）；
// cancel：放弃预取（还没开工就退出；已开工的结果没人要时会回收子进程）。
struct Prefetch {
    source: String,
    rx: mpsc::Receiver<OpenResult>,
    started: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

// 预取开工时间：提前 PREFETCH_LEAD 秒打开下一个源；比它短的源立即开始。
// dur 未知（0，直播等）时返回 None：不预取，避免长时间挂着一个网络连接
fn prefetch_delay(dur: f64) -> Option<Duration> {
    const PREFETCH_LEAD: f64 = 10.0;
    if dur <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64((dur - PREFETCH_LEAD).max(0.0)))
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
    // 首个源打开期间的纯色占位窗口。不能复用 GlWindow：Wayland 下先 show
    // 640×480 再 resize 到视频尺寸不会提交新的 xdg_toplevel 尺寸
    splash: Option<Window>,
    // 当前源播放期间提前打开的下一个源（见 start_prefetch）
    prefetch: Option<Prefetch>,
    played_any: bool,
    // 用户在打开/播放期间关闭了窗口：finish() 不应再报"没有可播放的源"
    closed: bool,
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
            splash: None,
            prefetch: None,
            played_any: false,
            closed: false,
        }
    }

    // 显示/更新"正在打开"状态：
    // - 首个源用独立的纯色 splash 窗口，避免 GlWindow 先以 640×480 显示、
    //   之后再 resize（Wayland/libdecor 下后者常常不生效）；
    // - 后续源直接改已经存在的 GlWindow 标题。
    fn show_loading(&mut self, source: &str) {
        // 打开期间不应还能对上一个源 seek：先摘掉旧的协调器
        *self.state.seek_handle.borrow_mut() = None;
        let title = format!("AUV (A/V Sync Engine) - Loading {}", source);
        if self.gl_ready {
            self.win.set_label(&title);
            self.win.redraw();
            return;
        }
        if let Some(s) = self.splash.as_mut() {
            s.set_label(&title);
            s.redraw();
            return;
        }
        let (w, h) = (640, 480);
        let win_x = ((self.screen.0 as i32 - w) / 2).max(0);
        let win_y = ((self.screen.1 as i32 - h) / 2 - 20).max(0);
        let mut splash = Window::new(win_x, win_y, w, h, title.as_str());
        splash.set_xclass("auv");
        splash.end();
        splash.handle(move |_, ev| {
            if ev == Event::Close {
                app::quit();
                true
            } else {
                false
            }
        });
        splash.show();
        self.splash = Some(splash);
    }

    // 关掉打开阶段的 splash，随后 GlWindow 会以真实视频尺寸首次 show()
    fn close_splash(&mut self) {
        if let Some(mut s) = self.splash.take() {
            s.hide();
            WidgetBase::delete(s);
        }
    }

    // 在后台线程打开输入，期间泵 UI 事件。打开/探测是阻塞 I/O（超时 20s），
    // 同步做会让窗口在首个源失败时根本不出现、后续源也无法关闭
    fn open_interactive(&mut self, source: &str) -> OpenOutcome {
        let (tx, rx) = mpsc::channel::<OpenResult>();
        let src = source.to_string();
        std::thread::spawn(move || {
            let res = util::open_input(&src).map_err(|e| format!("{e:#}"));
            let _ = tx.send(res);
            app::awake(); // 唤醒可能正阻塞在 app.wait() 的 UI 线程
        });
        loop {
            match rx.try_recv() {
                Ok(Ok(opened)) => return OpenOutcome::Opened(opened),
                Ok(Err(e)) => {
                    warn!("Skipping {}: {}", source, e);
                    return OpenOutcome::Failed;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    warn!("Opening {} ended unexpectedly", source);
                    return OpenOutcome::Failed;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if !self.app.wait() {
                        self.closed = true;
                        return OpenOutcome::Closed;
                    }
                }
            }
        }
    }

    // 在当前源播放期间提前打开下一个源。开工时刻见 prefetch_delay：
    // 只提前 10s，避免网络源/yt-dlp 在整个片长里挂着一个空闲连接。
    fn start_prefetch(&mut self, next: &str, dur: f64) {
        self.discard_prefetch();
        let Some(delay) = prefetch_delay(dur) else {
            return;
        };
        log::debug!("prefetching {} in {:.1}s", next, delay.as_secs_f64());
        let (tx, rx) = mpsc::channel::<OpenResult>();
        let source = next.to_string();
        let src = source.clone();
        let started = Arc::new(AtomicBool::new(false));
        let cancel = Arc::new(AtomicBool::new(false));
        let (started_thread, cancel_thread) = (started.clone(), cancel.clone());
        std::thread::spawn(move || {
            let deadline = Instant::now() + delay;
            while Instant::now() < deadline {
                if cancel_thread.load(Ordering::Relaxed) {
                    return;
                }
                // 100ms 粒度轮询取消，最迟在源结束时立刻退出
                let nap = Duration::from_millis(100)
                    .min(deadline.saturating_duration_since(Instant::now()));
                std::thread::sleep(nap);
            }
            if cancel_thread.load(Ordering::Relaxed) {
                return;
            }
            started_thread.store(true, Ordering::Relaxed);
            let res = util::open_input(&src).map_err(|e| format!("{e:#}"));
            match tx.send(res) {
                // 唤醒可能正阻塞在 app.wait() 的 UI 线程
                Ok(()) => app::awake(),
                // 主线程已经换源/关窗：没人要这个结果，yt-dlp 子进程必须显式回收
                Err(mpsc::SendError(res)) => {
                    if let Ok(mut opened) = res {
                        util::reap_child(&mut opened.child);
                    }
                }
            }
        });
        self.prefetch = Some(Prefetch {
            source,
            rx,
            started,
            cancel,
        });
    }

    // 放弃预取：取消还没开工的线程；已经有结果就把 yt-dlp 子进程收掉
    fn discard_prefetch(&mut self) {
        let Some(p) = self.prefetch.take() else {
            return;
        };
        p.cancel.store(true, Ordering::Relaxed);
        if let Ok(Ok(mut opened)) = p.rx.try_recv() {
            util::reap_child(&mut opened.child);
        }
    }

    // 取走与 source 匹配的预取；不匹配（播放列表被改/顺序变化）则丢弃
    fn take_prefetch(&mut self, source: &str) -> Option<Prefetch> {
        if self.prefetch.as_ref().is_some_and(|p| p.source == source) {
            self.prefetch.take()
        } else {
            self.discard_prefetch();
            None
        }
    }

    // 使用预取结果：已就绪直接取；正在打开就等它（同样泵 UI 事件）；
    // 还停在"等临近结尾"的睡眠阶段（例如用户提前跳到片尾）则取消，
    // 按普通路径重新打开——此时总等待时间与不预取时相同
    fn await_prefetch(&mut self, p: Prefetch, source: &str) -> OpenOutcome {
        let Prefetch {
            rx,
            started,
            cancel,
            ..
        } = p;
        if !started.load(Ordering::Relaxed) {
            cancel.store(true, Ordering::Relaxed);
            return self.open_interactive(source);
        }
        loop {
            match rx.try_recv() {
                Ok(Ok(opened)) => {
                    log::debug!("using prefetched input for {}", source);
                    return OpenOutcome::Opened(opened);
                }
                // 预取在 10s 前失败了，用户此刻仍想播：重新试一次
                Ok(Err(e)) => {
                    warn!("Prefetch of {} failed: {}; opening now", source, e);
                    return self.open_interactive(source);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    return self.open_interactive(source);
                }
                Err(mpsc::TryRecvError::Empty) => {
                    if !self.app.wait() {
                        self.closed = true;
                        // 线程还在打开：置取消并尽快退出；结果无人接收时
                        // 由线程自己回收 yt-dlp 子进程
                        cancel.store(true, Ordering::Relaxed);
                        return OpenOutcome::Closed;
                    }
                }
            }
        }
    }

    // 播放一个源；返回 false 表示窗口已被关闭、播放列表应终止。
    // 打不开或没有视频流的源会记警告后跳过，返回 true 继续下一个。
    // next 是播放列表里的下一个源：本轮播放期间会提前打开它。
    pub fn play(&mut self, source: &str, next: Option<&str>) -> bool {
        // 先把窗口显示出来（首个源用默认尺寸），再异步打开：
        // 打开/探测是阻塞 I/O，最长可能 20s，期间仍要能重绘与响应关闭
        self.show_loading(source);
        let opened = match self.take_prefetch(source) {
            Some(p) => self.await_prefetch(p, source),
            None => self.open_interactive(source),
        };
        let util::OpenedInput {
            ictx,
            mut child,
            watchdog,
            seekable,
        } = match opened {
            OpenOutcome::Opened(opened) => opened,
            OpenOutcome::Failed => return true,
            OpenOutcome::Closed => return false,
        };
        let info = match extract_streams(&ictx) {
            Ok(info) => info,
            Err(e) => {
                // 页面链接经 yt-dlp 打开但拿不到视频流时，必须回收子进程，
                // 否则会留下僵尸/继续下载的 yt-dlp
                util::reap_child(&mut child);
                warn!("Skipping {}: {}", source, e);
                return true;
            }
        };
        let win_size = calc_display_size(info.disp_w.max(1), info.disp_h.max(1));
        log::debug!(
            "source display={}x{} window={:?} screen={:?}",
            info.disp_w,
            info.disp_h,
            win_size,
            self.screen
        );
        let title_base = format!("AUV (A/V Sync Engine) - {}", source);
        // 关掉 splash 后再首次 show GlWindow：此时窗口还没 map，resize 会
        // 直接成为初始尺寸，不依赖 Wayland 的二次 configure
        self.close_splash();
        self.show_or_resize(win_size, &title_base);

        let dur_us = ictx.duration();
        let dur = if dur_us > 0 {
            dur_us as f64 / 1_000_000.0
        } else {
            0.0
        };
        let mut session = SourceSession::new(SessionInputs {
            ictx,
            ytdlp_child: child,
            watchdog,
            info,
            dur,
            paused: self.state.paused.clone(),
            volume: self.state.volume.clone(),
            mute: self.state.mute.clone(),
            audio: &self.audio,
            seekable,
        });
        session.start(&self.state);
        self.played_any = true;
        // 源已经起播：若列表里还有下一个，按剩余时长安排预取
        if let Some(next) = next {
            self.start_prefetch(next, dur);
        }
        session.pump(&self.app, &mut self.win);
        session.shutdown();

        // 窗口仍可见则继续下一个源；用户已关闭则终止播放列表
        let visible = self.win.visible();
        if !visible {
            self.closed = true;
        }
        visible
    }

    // 全部源结束后调用：一个源都没播成功时返回错误（用户主动关闭不算失败）
    pub fn finish(&mut self) -> Result<()> {
        self.discard_prefetch();
        if self.played_any || self.closed {
            Ok(())
        } else {
            Err(anyhow::anyhow!("No media source could be opened"))
        }
    }

    // 显示/调整播放窗口：首个源初始化 GL/回调；后续源分辨率变化时重建窗口。
    // 不能只 resize：Wayland/libdecor 下窗口 map 后再次 resize 往往不提交新的
    // xdg_toplevel 尺寸，表现为换源后窗口尺寸卡在上一源。重建窗口同时重建
    // GL 上下文与 RenderState，源切换本来就会清空帧，代价可接受。
    fn show_or_resize(&mut self, (w, h): (u32, u32), title: &str) {
        let win_x = ((self.screen.0 as i32 - w as i32) / 2).max(0);
        let win_y = ((self.screen.1 as i32 - h as i32) / 2 - 20).max(0);
        if self.win.shown() && (self.win.w() != w as i32 || self.win.h() != h as i32) {
            let mut win = GlWindow::new(win_x, win_y, w as i32, h as i32, title);
            win.set_mode(fltk::enums::Mode::Opengl3);
            win.set_xclass("auv");
            win.end();
            // 旧窗口必须显式删除：fltk-rs 的 Widget 句柄 drop 只是 Rust 侧
            // 释放，不会销毁原生窗口（否则换源后会留下一个旧尺寸的窗口）
            let old = std::mem::replace(&mut self.win, win);
            WidgetBase::delete(old);
            self.gl_ready = false;
        } else {
            self.win.resize(win_x, win_y, w as i32, h as i32);
            self.win.set_label(title);
        }

        if self.gl_ready {
            self.win.redraw();
            return;
        }
        if !self.win.shown() {
            self.win.show();
        }
        let rs = ui::setup_gl(&mut self.win, self.state.snapshot_request.clone());
        ui::register_window_callbacks(&mut self.win, &self.app, &self.state, rs);
        self.gl_ready = true;
    }
}

// GL 初始化/回调之外的纯逻辑单测
#[cfg(test)]
mod tests {
    use super::prefetch_delay;
    use std::time::Duration;

    // 预取只在临近结尾时开工：长片提前 10s，短片立即；时长未知（直播）不预取
    #[test]
    fn prefetch_delay_schedules_near_end_only() {
        assert_eq!(prefetch_delay(0.0), None);
        assert_eq!(prefetch_delay(-1.0), None);
        assert_eq!(prefetch_delay(4.0), Some(Duration::ZERO));
        assert_eq!(prefetch_delay(10.0), Some(Duration::ZERO));
        assert_eq!(prefetch_delay(60.0), Some(Duration::from_secs(50)));
        assert_eq!(prefetch_delay(3600.0), Some(Duration::from_secs(3590)));
    }
}

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
use std::sync::mpsc;

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

    // 播放一个源；返回 false 表示窗口已被关闭、播放列表应终止。
    // 打不开或没有视频流的源会记警告后跳过，返回 true 继续下一个。
    pub fn play(&mut self, source: &str) -> bool {
        // 先把窗口显示出来（首个源用默认尺寸），再异步打开：
        // 打开/探测是阻塞 I/O，最长可能 20s，期间仍要能重绘与响应关闭
        self.show_loading(source);
        let util::OpenedInput {
            ictx,
            mut child,
            watchdog,
            seekable,
        } = match self.open_interactive(source) {
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
            title: title_base,
            dur,
            paused: self.state.paused.clone(),
            volume: self.state.volume.clone(),
            mute: self.state.mute.clone(),
            audio: &self.audio,
            seekable,
        });
        session.start(&self.state);
        self.played_any = true;
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
    pub fn finish(&self) -> Result<()> {
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

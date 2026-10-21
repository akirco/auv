// UI 侧模块：直接操作窗口/GL 的代码与跨源共享的播放器状态。
// 主程序只负责装配与源循环，窗口回调（绘制/按键）与纯展示换算都归这里。
use cpal::traits::StreamTrait;
use fltk::{app, prelude::*, window::GlWindow};
use log::info;
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
};

use crate::{
    frame::YuvFrame,
    pause::PauseGate,
    render::{self, draw_frame},
};

// 跨源共享的播放器状态：绘制/按键回调与 UI 循环共用的句柄。
// 生命周期长于单个源，源切换时各字段按需更新；成组持有避免散落的
// Rc<RefCell>/Arc 相互传递出错位。
pub struct PlayerState {
    pub stream_handle: Rc<RefCell<Option<cpal::Stream>>>,
    pub recycle_handle: Rc<RefCell<Option<mpsc::Sender<YuvFrame>>>>,
    pub frame_store: Rc<RefCell<Option<YuvFrame>>>,
    pub seek_handle: Rc<RefCell<Option<Arc<crate::demux::SeekCtl>>>>,
    pub dur_handle: Rc<RefCell<f64>>,
    pub paused: Arc<PauseGate>,
    pub volume: Arc<AtomicU32>,
    pub mute: Arc<AtomicBool>,
    // DMA-BUF 帧截图：UI 置位，渲染端在下次绘制后读回 back buffer
    pub snapshot_request: Arc<AtomicBool>,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            stream_handle: Rc::new(RefCell::new(None)),
            recycle_handle: Rc::new(RefCell::new(None)),
            frame_store: Rc::new(RefCell::new(None)),
            seek_handle: Rc::new(RefCell::new(None)),
            dur_handle: Rc::new(RefCell::new(0.0)),
            paused: Arc::new(PauseGate::new()),
            volume: Arc::new(AtomicU32::new(100)),
            mute: Arc::new(AtomicBool::new(false)),
            snapshot_request: Arc::new(AtomicBool::new(false)),
        }
    }
}

// 初始化 GL 上下文并构建渲染状态（首个源显示窗口后调用一次）
pub fn setup_gl(win: &mut GlWindow, snap_request: Arc<AtomicBool>) -> render::RenderState {
    win.make_current();
    // 呈现节奏锁定显示器刷新率，避免无谓的满速 present 抢占 GPU
    win.set_swap_interval(1);
    gl::load_with(|s| win.get_proc_address(s) as *const _);
    unsafe { render::setup_opengl(snap_request) }
}

// 注册绘制与按键回调（窗口首次显示时调用一次）。
// 回调闭包只从 PlayerState clone 句柄，生命周期内复用同一批状态。
pub fn register_window_callbacks(
    win: &mut GlWindow,
    app_handle: &fltk::app::App,
    state: &PlayerState,
    rs: render::RenderState,
) {
    let draw_store = state.frame_store.clone();
    win.draw(move |w| {
        let frame_guard = draw_store.borrow();
        if let Some(frame) = frame_guard.as_ref() {
            draw_frame(w, &rs, frame);
        }
    });

    let app_close = *app_handle;
    let paused_keys = state.paused.clone();
    let stream_handle_keys = state.stream_handle.clone();
    let frame_store_keys = state.frame_store.clone();
    let seek_keys = state.seek_handle.clone();
    let dur_keys = state.dur_handle.clone();
    let volume_keys = state.volume.clone();
    let mute_keys = state.mute.clone();
    let snap_request_keys = state.snapshot_request.clone();
    win.handle(move |w, ev| {
        if ev == fltk::enums::Event::Close {
            app_close.quit();
            return true;
        }
        if ev == fltk::enums::Event::KeyDown {
            let key = app::event_key();
            if key == fltk::enums::Key::from_char(' ') {
                // 翻转暂停并唤醒等待中的解码线程
                //（状态变更与 notify 在 PauseGate 内同一临界区完成，不会丢通知）
                let paused_now = paused_keys.toggle();
                if let Some(s) = stream_handle_keys.borrow().as_ref() {
                    if paused_now {
                        let _ = s.pause();
                    } else {
                        let _ = s.play();
                    }
                }
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
                // 把当前显示的帧交给后台线程转 RGB 并编码 PNG：
                // 转换与压缩是 O(像素数) 的 CPU 工作，在 UI 线程做会冻结窗口。
                // DMA-BUF 零拷贝帧没有 CPU 平面，改为请求渲染端用 glReadPixels
                // 在下次绘制后读回视频区域；CPU 帧仍走原来的全分辨率快照。
                if let Some(frame) = frame_store_keys.borrow().as_ref() {
                    if frame.is_dmabuf() {
                        if crate::screenshot::busy() {
                            info!("Screenshot still in progress, ignoring request");
                        } else {
                            snap_request_keys.store(true, Ordering::Relaxed);
                            w.redraw();
                            info!("Screenshot: capturing DMA-BUF frame via GL readback");
                        }
                    } else {
                        crate::screenshot::spawn_save(|| frame.snapshot());
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
                    // 不可 seek 的源（yt-dlp 管道）：定位必然失败，直接忽略，
                    // 避免无谓的解码器复位（会丢掉参考帧导致画面损坏）
                    if !ctl.seekable() {
                        info!("Seek ignored: this source is not seekable");
                        return true;
                    }
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
                    paused_keys.wake();
                }
            } else if key == fltk::enums::Key::from_char('+')
                || key == fltk::enums::Key::from_char('=')
            {
                // 音量 +10%（上限 200%）。只有 UI 线程会写 volume，直接
                // load/store 即可；不用 RMW，避开 MSRV 与 fetch_update 弃用问题。
                let v = volume_keys.load(Ordering::Relaxed);
                volume_keys.store((v + 10).min(200), Ordering::Relaxed);
                info!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
            } else if key == fltk::enums::Key::from_char('-')
                || key == fltk::enums::Key::from_char('_')
            {
                // 音量 -10%（下限 0%）
                let v = volume_keys.load(Ordering::Relaxed);
                volume_keys.store(v.saturating_sub(10), Ordering::Relaxed);
                info!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
            } else if key == fltk::enums::Key::from_char('m')
                || key == fltk::enums::Key::from_char('M')
            {
                let was_muted = mute_keys.fetch_xor(true, Ordering::Relaxed);
                info!("Mute: {}", if was_muted { "OFF" } else { "ON" });
            }
        }
        false
    });
}

// 按视频显示尺寸与屏幕大小计算窗口尺寸（首帧前确定一次）
pub fn calc_display_size(video_w: u32, video_h: u32) -> (u32, u32) {
    let (screen_w, screen_h) = app::screen_size();
    let max_w = screen_w as u32 - 40;
    let max_h = screen_h as u32 - 100;
    let video_ratio = video_w as f64 / video_h as f64;
    let screen_ratio = max_w as f64 / max_h as f64;
    if video_w > max_w || video_h > max_h {
        if video_ratio > screen_ratio {
            (max_w, (max_w as f64 / video_ratio).round() as u32)
        } else {
            ((max_h as f64 * video_ratio).round() as u32, max_h)
        }
    } else {
        (video_w, video_h)
    }
}

// 秒 → mm:ss（窗口标题进度显示用）
pub fn format_time(sec: f64) -> String {
    let s = sec.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

pub fn is_hyprland() -> bool {
    std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
}

pub fn hyprctl_toggle_fullscreen() {
    // FLTK 的 Wayland 全屏在 Hyprland 下不生效，改由 Hyprland 自身的 IPC 切换。
    // 优先 Lua 语法，失败则回退到经典语法。
    let lua_ok = hyprctl_ok(&["dispatch", "hl.dsp.window.fullscreen(0)"]);
    if !lua_ok {
        let _ = hyprctl_ok(&["dispatch", "fullscreen", "0"]);
    }
}

fn hyprctl_ok(args: &[&str]) -> bool {
    std::process::Command::new("hyprctl")
        .args(args)
        .output()
        .map(|o| {
            let out = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            out.contains("ok") && !out.contains("error")
        })
        .unwrap_or(false)
}

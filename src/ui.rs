// UI 侧模块：直接操作窗口/GL 的代码与跨源共享的播放器状态。
// 主程序只负责装配与源循环，窗口回调（绘制/按键）与纯展示换算都归这里。
use cpal::traits::StreamTrait;
use fltk::{app, prelude::*, window::GlWindow};
use log::{error, info};
use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

use crate::{
    frame::YuvFrame,
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
    pub paused: Arc<(Mutex<bool>, Condvar)>,
    pub volume: Arc<AtomicU64>,
    pub mute: Arc<AtomicU64>,
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            stream_handle: Rc::new(RefCell::new(None)),
            recycle_handle: Rc::new(RefCell::new(None)),
            frame_store: Rc::new(RefCell::new(None)),
            seek_handle: Rc::new(RefCell::new(None)),
            dur_handle: Rc::new(RefCell::new(0.0)),
            paused: Arc::new((Mutex::new(false), Condvar::new())),
            volume: Arc::new(AtomicU64::new(100)),
            mute: Arc::new(AtomicU64::new(0)),
        }
    }
}

// 初始化 GL 上下文并构建渲染状态（首个源显示窗口后调用一次）
pub fn setup_gl(win: &mut GlWindow) -> render::RenderState {
    win.make_current();
    // 呈现节奏锁定显示器刷新率，避免无谓的满速 present 抢占 GPU
    win.set_swap_interval(1);
    gl::load_with(|s| win.get_proc_address(s) as *const _);
    unsafe { render::setup_opengl() }
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
                    match crate::screenshot::save_screenshot(frame) {
                        Ok(p) => info!("Screenshot saved: {}", p.display()),
                        Err(e) => error!("Screenshot failed: {}", e),
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
                let _ = volume_keys.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some((v + 10).min(200))
                });
                info!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
            } else if key == fltk::enums::Key::from_char('-')
                || key == fltk::enums::Key::from_char('_')
            {
                // 音量 -10%（下限 0%）
                let _ = volume_keys.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                    Some(v.saturating_sub(10))
                });
                info!("Volume: {}%", volume_keys.load(Ordering::Relaxed));
            } else if key == fltk::enums::Key::from_char('m')
                || key == fltk::enums::Key::from_char('M')
            {
                let was = mute_keys.fetch_xor(1, Ordering::Relaxed);
                info!("Mute: {}", if was == 0 { "ON" } else { "OFF" });
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

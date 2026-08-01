use anyhow::Result;
use fltk::app;

// 解析命令行媒体源：普通参数直接进列表；"-p/--playlist <file>" 读取文件，
// 每行一个条目（忽略空行与 # 开头），可混放本地文件与网络流。
pub fn parse_sources(args: &[String]) -> Result<Vec<String>> {
    let mut sources = Vec::new();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if arg == "-p" || arg == "--playlist" {
            let path = it
                .next()
                .ok_or_else(|| anyhow::anyhow!("{} requires a file path", arg))?;
            let content = std::fs::read_to_string(path)?;
            for line in content.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with('#') {
                    sources.push(line.to_string());
                }
            }
        } else {
            sources.push(arg.clone());
        }
    }
    Ok(sources)
}

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

pub fn is_hyprland() -> bool {
    std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok()
}

pub fn hyprctl_toggle_fullscreen() {
    // FLTK 的 Wayland 全屏在 Hyprland 下不生效，改由 Hyprland 自身的 IPC 切换。
    // 优先Lua 语法，失败则回退到经典语法。
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

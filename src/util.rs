use anyhow::Result;
use ffmpeg_next::format;
use fltk::app;
use std::process::{Child, Command, Stdio};

// 打开媒体源：先按本地文件/直链（mp4、m3u8 等）尝试；失败且为 http(s) 链接时
// 回退到 yt-dlp 解析，支持 B 站视频页、YouTube 等站点链接——yt-dlp 负责
// API 签名、鉴权与音视频合并，stdout 流经自定义 IO 直接喂给 ffmpeg。
// 返回输入上下文与（若走了 yt-dlp）需存活的子进程句柄。
pub fn open_input(source: &str) -> Result<(format::context::Input, Option<Child>)> {
    match format::input(source) {
        Ok(ictx) => Ok((ictx, None)),
        Err(direct_err) if source.starts_with("http://") || source.starts_with("https://") => {
            // 合并容器强制 mkv：默认 mpegts 会把 AV1 等编成私有流，
            // 导致 ffmpeg 探测不到视频轨
            let mut child = Command::new("yt-dlp")
                .args([
                    "-q",
                    "--no-playlist",
                    "--merge-output-format",
                    "mkv",
                    "-o",
                    "-",
                    source,
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!("direct open failed ({direct_err}); spawning yt-dlp failed: {e} (install yt-dlp to play page links)")
                })?;
            let stdout = child.stdout.take().expect("piped stdout");
            let io = format::context::StreamIo::from_read(stdout)?;
            match format::input_from_stream(io, None, None) {
                Ok(ictx) => Ok((ictx, Some(child))),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("Cannot open {source} (direct: {direct_err}; yt-dlp: {e})")
                }
            }
        }
        Err(e) => Err(e.into()),
    }
}

// 结束本源播放后回收 yt-dlp 子进程：自然结束则 reap，提前退出则终止
pub fn reap_child(child: &mut Option<Child>) {
    if let Some(c) = child.as_mut() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

// CLI 解析结果：播放给定源，或要求打印帮助/版本
pub enum CliAction {
    Play(Vec<String>),
    Help,
    Version,
}

// 解析命令行参数：
// - `-h/--help`、`-V/--version` 触发帮助/版本（出现即生效）；
// - `--` 之后的所有参数一律视为媒体源（允许文件名以 `-` 开头，如 `-p`）；
// - `-p/--playlist <file>` 读取播放列表，每行一个条目（忽略空行与 # 开头），
//   可混放本地文件与网络流。
pub fn parse_cli(args: &[String]) -> Result<CliAction> {
    let mut sources = Vec::new();
    let mut it = args.iter();
    let mut literal = false; // `--` 之后不再解析选项
    while let Some(arg) = it.next() {
        if !literal {
            match arg.as_str() {
                "-h" | "--help" => return Ok(CliAction::Help),
                "-V" | "--version" => return Ok(CliAction::Version),
                "--" => {
                    literal = true;
                    continue;
                }
                "-p" | "--playlist" => {
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
                    continue;
                }
                _ => {}
            }
        }
        sources.push(arg.clone());
    }
    Ok(CliAction::Play(sources))
}

// 把当前 YUV420P 帧转为 RGB 并保存为 PNG 截图（保存到当前工作目录）。
// 转换参数与渲染 shader 一致：读帧的色彩信息（含 full/limited range 归一），
// 行序保持图像正向
pub fn save_screenshot(frame: &crate::frame::YuvFrame) -> Result<std::path::PathBuf> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        anyhow::bail!("screenshot: empty frame");
    }
    let cm = crate::frame::color_matrix(frame.color_space, frame.color_range, frame.height);
    // 字节域换算：range 参数按 255 缩放（y_off/uv_center 是 0..1 域的偏移）
    let y_off = cm.y_off * 255.0;
    let uv_center = cm.uv_center * 255.0;
    let y_stride = frame.y_stride as usize;
    let uv_stride = frame.uv_stride as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        let yrow = &frame.y[row * y_stride..row * y_stride + w];
        let uv_off = (row / 2) * uv_stride;
        let out = &mut rgb[row * w * 3..(row + 1) * w * 3];
        for (x, &y) in yrow.iter().enumerate() {
            let yf = (y as f32 - y_off) * cm.y_gain;
            let u = (frame.u[uv_off + x / 2] as f32 - uv_center) * cm.uv_gain;
            let v = (frame.v[uv_off + x / 2] as f32 - uv_center) * cm.uv_gain;
            out[x * 3] = (yf + cm.rv * v).clamp(0.0, 255.0) as u8;
            out[x * 3 + 1] = (yf + cm.gu * u + cm.gv * v).clamp(0.0, 255.0) as u8;
            out[x * 3 + 2] = (yf + cm.bu * u).clamp(0.0, 255.0) as u8;
        }
    }

    let name = format!(
        "auv_{}.png",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let path = std::env::current_dir()?.join(name);
    let file = std::fs::File::create(&path)?;
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    // 截图路径取极快压缩即可，避免 UI 线程卡顿
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header()?;
    writer.write_image_data(&rgb)?;
    Ok(path)
}

// 秒 → mm:ss（窗口标题进度显示用）
pub fn format_time(sec: f64) -> String {
    let s = sec.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
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

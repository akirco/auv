// 截图：把当前显示的 YUV420P 帧转 RGB 并编码为 PNG（保存到当前工作目录）
use anyhow::Result;

use crate::frame::{color_matrix, YuvFrame};

// 转换参数与渲染 shader 一致：读帧的色彩信息（含 full/limited range 归一），
// 行序保持图像正向
pub fn save_screenshot(frame: &YuvFrame) -> Result<std::path::PathBuf> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        anyhow::bail!("screenshot: empty frame");
    }
    let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
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
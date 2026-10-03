// 截图：把当前显示的 YUV420P 帧转 RGB 并编码为 PNG（保存到当前工作目录）
//
// 转换 + PNG 编码是纯 CPU 的 O(像素数) 工作（4K 下数百毫秒），放在 UI 线程会
// 冻结窗口。因此按键时只克隆一份帧，真正的编码在后台线程完成，日志与主线程
// 共用同一个 logger。为避免连按 S 时无节制地派生线程（每线程一份整帧拷贝），
// 同一时刻只允许一个截图任务在跑。
use anyhow::Result;
use log::{error, info};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::frame::{FrameSnapshot, color_matrix};

// 是否有截图任务正在后台执行
static CAPTURING: AtomicBool = AtomicBool::new(false);

// 退出时（含 panic）都要放开标记，否则一次 panic 会让截图功能永久失效
struct CaptureGuard;

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURING.store(false, Ordering::Relaxed);
    }
}

/// 请求截图。`make` 负责在当前线程（UI 线程）把当前帧的平面拷成快照——必须在
/// UI 线程做，回收后的帧会立刻被解码器重填。已有任务在跑时**不会调用 `make`**，
/// 因此连按 S 不会白白做整帧拷贝；否则快照完成后交给后台线程转换与编码，
/// 函数立即返回，不阻塞事件循环。
pub fn spawn_save(make: impl FnOnce() -> FrameSnapshot) {
    if CAPTURING.swap(true, Ordering::Relaxed) {
        info!("Screenshot still in progress, ignoring request");
        return;
    }
    // 占位成功后立刻建守卫再取快照：即便 make panic，守卫也随栈展开 drop，
    // 占位不会永久卡住；正常路径把守卫移进后台线程，编码结束才释放
    let guard = CaptureGuard;
    let frame = make();
    std::thread::spawn(move || {
        let _guard = guard;
        match save(&frame) {
            Ok(p) => info!("Screenshot saved: {}", p.display()),
            Err(e) => error!("Screenshot failed: {e}"),
        }
    });
}

// 同步执行一次截图（转换 + 写文件）。转换参数与渲染 shader 一致：
// 读帧的色彩信息（含 full/limited range 归一），行序保持图像正向
fn save(frame: &FrameSnapshot) -> Result<std::path::PathBuf> {
    let (w, h, rgb) = yuv_to_rgb(frame)?;

    // 毫秒精度：原先只到秒，连按两次 S 会互相覆盖
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_else(|_| "0.000".to_string());
    let path = std::env::current_dir()?.join(format!("auv_{stamp}.png"));
    let file = std::fs::File::create(&path)?;
    let mut enc = png::Encoder::new(file, w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    // 截图取极快压缩即可（本就在后台线程，不再有 UI 卡顿顾虑）
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header()?;
    writer.write_image_data(&rgb)?;
    Ok(path)
}

// YUV420P → RGB8。
// 保持逐像素的平铺循环：实测它能被 LLVM 自动向量化（4K 约 50ms/帧），
// 而按 2×2 色度块重组的写法（色度取一次复用 4 次、变长子切片）反而更慢
// ——运行时分支与变址打断了向量化，收益抵不过开销。矩阵系数与 range 参数
// 提到循环外的局部变量，避免逐像素从 ColorMatrix 取字段。
fn yuv_to_rgb(frame: &FrameSnapshot) -> Result<(usize, usize, Vec<u8>)> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        anyhow::bail!("screenshot: empty frame");
    }
    let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
    // 字节域换算：range 参数按 255 缩放（y_off/uv_center 是 0..1 域的偏移）
    let y_off = cm.y_off * 255.0;
    let uv_center = cm.uv_center * 255.0;
    let (y_gain, uv_gain) = (cm.y_gain, cm.uv_gain);
    let (rv, gu, gv, bu) = (cm.rv, cm.gu, cm.gv, cm.bu);
    let y_stride = frame.y_stride as usize;
    let uv_stride = frame.uv_stride as usize;
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        let yrow = &frame.y[row * y_stride..row * y_stride + w];
        let uv_off = (row / 2) * uv_stride;
        let out = &mut rgb[row * w * 3..(row + 1) * w * 3];
        for (x, &y) in yrow.iter().enumerate() {
            let yf = (y as f32 - y_off) * y_gain;
            let u = (frame.u[uv_off + x / 2] as f32 - uv_center) * uv_gain;
            let v = (frame.v[uv_off + x / 2] as f32 - uv_center) * uv_gain;
            out[x * 3] = (yf + rv * v).clamp(0.0, 255.0) as u8;
            out[x * 3 + 1] = (yf + gu * u + gv * v).clamp(0.0, 255.0) as u8;
            out[x * 3 + 2] = (yf + bu * u).clamp(0.0, 255.0) as u8;
        }
    }
    Ok((w, h, rgb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffmpeg_next::util::color::{Range, Space};

    // 构造一帧确定性测试数据：平面内容按各自长度取模填充。
    // 色度平面高度取 ceil(h/2)——ffmpeg 对 420 就是这么分配的；
    // stride 故意加宽，用于验证按 stride 而非 width 取行
    fn make_frame(w: i32, h: i32, space: Space, range: Range) -> FrameSnapshot {
        let (ws, hs) = (w as usize, h as usize);
        let y_stride = ws + 13;
        let uv_stride = ws / 2 + 7;
        FrameSnapshot {
            y: (0..y_stride * hs).map(|i| (i % 251) as u8).collect(),
            u: (0..uv_stride * hs.div_ceil(2))
                .map(|i| (i % 253) as u8)
                .collect(),
            v: (0..uv_stride * hs.div_ceil(2))
                .map(|i| (i % 241) as u8)
                .collect(),
            width: w,
            height: h,
            y_stride: y_stride as i32,
            uv_stride: uv_stride as i32,
            color_space: space,
            color_range: range,
        }
    }

    // 空帧直接报错，避免空切片 panic
    #[test]
    fn empty_frame_is_rejected() {
        let frame = make_frame(0, 0, Space::BT709, Range::MPEG);
        assert!(yuv_to_rgb(&frame).is_err());
    }

    // 构造内容完全指定的 2×2 帧（色度 1×1，U=V=128 即"接近中性"），
    // 用于按手算公式校验转换结果
    fn make_solid(y: [u8; 4], space: Space, range: Range) -> FrameSnapshot {
        FrameSnapshot {
            y: y.to_vec(),
            u: vec![128],
            v: vec![128],
            width: 2,
            height: 2,
            y_stride: 2,
            uv_stride: 1,
            color_space: space,
            color_range: range,
        }
    }

    // BT.709 + limited range 的手算 oracle：
    //   y_off=16, y_gain=255/219, uv_center=127.5, uv_gain=255/224
    //   U=V=128 → u=v=0.5*255/224≈0.5692，于是 R+=1.5748v≈0.896、G-=0.6554u≈0.373、
    //   B+=1.8556u≈1.056
    //   Y=16  → yf=0    → (0.896, -0.373, 1.056) 截断并夹取 → (0, 0, 1)
    //   Y=235 → yf=255  → (255.9, 254.6, 256.1) 夹取 → (255, 254, 255)
    #[test]
    fn bt709_limited_matches_hand_computed() {
        let frame = make_solid([16, 16, 235, 235], Space::BT709, Range::MPEG);
        let (_, _, rgb) = yuv_to_rgb(&frame).unwrap();
        assert_eq!(rgb, vec![0, 0, 1, 0, 0, 1, 255, 254, 255, 255, 254, 255]);
    }

    // full range 下 y_off=0, y_gain=1, uv_gain=1 → U=V=128 即 u=v=0.5：
    //   Y=0   → (0.787, -0.328, 0.928) → (0, 0, 0)
    //   Y=255 → (255.8, 254.7, 255.9) → (255, 254, 255)
    #[test]
    fn bt709_full_range_matches_hand_computed() {
        let frame = make_solid([0, 0, 255, 255], Space::BT709, Range::JPEG);
        let (_, _, rgb) = yuv_to_rgb(&frame).unwrap();
        assert_eq!(rgb, vec![0, 0, 0, 0, 0, 0, 255, 254, 255, 255, 254, 255]);
    }

    // 奇数宽高（stride 加宽、色度按 ceil(h/2) 分配）不得 panic，
    // 输出长度必须是 w*h*3
    #[test]
    fn odd_dimensions_do_not_panic() {
        for (w, h) in [(7usize, 5usize), (5, 3), (1, 1), (2, 1), (3, 9)] {
            let frame = make_frame(w as i32, h as i32, Space::BT709, Range::MPEG);
            let (rw, rh, rgb) = yuv_to_rgb(&frame).unwrap();
            assert_eq!((rw, rh), (w, h));
            assert_eq!(rgb.len(), w * h * 3);
        }
    }
}

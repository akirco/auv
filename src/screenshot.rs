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
use ffmpeg_next::util::color::{Primaries, Range, TransferCharacteristic};

// 是否有截图任务正在后台执行
static CAPTURING: AtomicBool = AtomicBool::new(false);

pub fn busy() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

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
pub fn spawn_save(make: impl FnOnce() -> Option<FrameSnapshot>) {
    if CAPTURING.swap(true, Ordering::Relaxed) {
        info!("Screenshot still in progress, ignoring request");
        return;
    }
    // 占位成功后立刻建守卫再取快照：即便 make panic，守卫也随栈展开 drop，
    // 占位不会永久卡住；正常路径把守卫移进后台线程，编码结束才释放
    let guard = CaptureGuard;
    let Some(frame) = make() else {
        info!("Current frame has no CPU planes; screenshot skipped");
        return;
    };
    std::thread::spawn(move || {
        let _guard = guard;
        match save(&frame) {
            Ok(p) => info!("Screenshot saved: {}", p.display()),
            Err(e) => error!("Screenshot failed: {e}"),
        }
    });
}

/// 保存已经由 `glReadPixels` 读回的 RGB 像素（行序自下而上）。
/// DMA-BUF 零拷贝帧没有 CPU 平面，渲染端在 swap 前读回 back buffer，
/// 这里在后台线程翻转行序并编码，避免阻塞 UI。
pub fn spawn_save_rgb(w: i32, h: i32, rgb: Vec<u8>) {
    if w <= 0 || h <= 0 || rgb.len() < w as usize * h as usize * 3 {
        error!("Screenshot failed: invalid GL readback buffer");
        return;
    }
    if CAPTURING.swap(true, Ordering::Relaxed) {
        info!("Screenshot still in progress, ignoring request");
        return;
    }
    let guard = CaptureGuard;
    std::thread::spawn(move || {
        let _guard = guard;
        let mut rgb = rgb;
        flip_rows(&mut rgb, w as usize, h as usize);
        match encode_png(w as u32, h as u32, &rgb) {
            Ok(p) => info!("Screenshot saved: {}", p.display()),
            Err(e) => error!("Screenshot failed: {e}"),
        }
    });
}

// GL 的像素原点在左下，PNG 行序要求自上而下。
// 整行 memcpy（3 次）代替逐字节 swap（4K 下约 1200 万次）
fn flip_rows(rgb: &mut [u8], w: usize, h: usize) {
    let stride = w * 3;
    let mut tmp = vec![0u8; stride];
    for y in 0..h / 2 {
        let (top, bottom) = rgb.split_at_mut((h - 1 - y) * stride);
        let top = &mut top[y * stride..y * stride + stride];
        let bottom = &mut bottom[..stride];
        tmp.copy_from_slice(top);
        top.copy_from_slice(bottom);
        bottom.copy_from_slice(&tmp);
    }
}

fn save(frame: &FrameSnapshot) -> Result<std::path::PathBuf> {
    let (w, h, rgb) = yuv_to_rgb(frame)?;
    encode_png(w as u32, h as u32, &rgb)
}

fn encode_png(w: u32, h: u32, rgb: &[u8]) -> Result<std::path::PathBuf> {
    // 毫秒精度：原先只到秒，连按两次 S 会互相覆盖
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}.{:03}", d.as_secs(), d.subsec_millis()))
        .unwrap_or_else(|_| "0.000".to_string());
    let path = std::env::current_dir()?.join(format!("auv_{stamp}.png"));
    let file = std::fs::File::create(&path)?;
    let mut enc = png::Encoder::new(file, w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    // 截图取极快压缩即可（本就在后台线程，不再有 UI 卡顿顾虑）
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header()?;
    writer.write_image_data(rgb)?;
    Ok(path)
}

// PQ/HLG → SDR 的 CPU 版本：与 GPU shader 同一套公式，供截图使用。
// 这样 HDR 源按 S 截出的图与画面观感一致，不会退化成"把 PQ 当 SDR"的灰暗图。
fn hdr_to_rgb(frame: &FrameSnapshot) -> Result<(usize, usize, Vec<u8>)> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    if w == 0 || h == 0 {
        anyhow::bail!("screenshot: empty frame");
    }
    let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
    let (y_off, y_gain) = (cm.y_off, cm.y_gain);
    let (uv_center, uv_gain) = (cm.uv_center, cm.uv_gain);
    let (rv, gu, gv, bu) = (cm.rv, cm.gu, cm.gv, cm.bu);
    let y_stride = frame.y_stride.max(0) as usize;
    let uv_stride = frame.uv_stride.max(0) as usize;
    let bps = if frame.bit_depth == 10 { 2 } else { 1 };
    let read = |buf: &[u8], off: usize| -> f32 {
        if bps == 2 {
            let lo = *buf.get(off).unwrap_or(&0) as u16;
            let hi = *buf.get(off + 1).unwrap_or(&0) as u16;
            let mut code = (lo | (hi << 8)) as f32;
            if frame.p010 {
                code /= 64.0;
            }
            if frame.color_range == Range::JPEG {
                code / 1023.0
            } else {
                code / 1020.0
            }
        } else {
            *buf.get(off).unwrap_or(&0) as f32 / 255.0
        }
    };
    let pq = |e: f32| -> f32 {
        const M1: f32 = 0.15930176;
        const M2: f32 = 78.84375;
        const C1: f32 = 0.8359375;
        const C2: f32 = 18.851562;
        const C3: f32 = 18.6875;
        let e1 = e.clamp(0.0, 1.0).powf(1.0 / M2);
        let num = (e1 - C1).max(0.0);
        (num / (C2 - C3 * e1).max(1e-6)).powf(1.0 / M1)
    };
    let hlg = |e: f32| -> f32 {
        const A: f32 = 0.17883277;
        const B: f32 = 0.28466892;
        const C: f32 = 0.5599107;
        if e <= 0.5 {
            e * e / 3.0
        } else {
            (((e - C) / A).exp() + B) / 12.0
        }
    };
    let hable = |x: f32| -> f32 {
        const A: f32 = 0.15;
        const B: f32 = 0.50;
        const C: f32 = 0.10;
        const D: f32 = 0.20;
        const E: f32 = 0.02;
        const F: f32 = 0.30;
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    };
    let white = hable(11.2);
    let bt2020 = frame.primaries == Primaries::BT2020;
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        let uv_off = (row / 2) * uv_stride;
        for x in 0..w {
            let y = (read(&frame.y, row * y_stride + x * bps) - y_off) * y_gain;
            let u = (read(&frame.u, uv_off + (x / 2) * bps) - uv_center) * uv_gain;
            let v = (read(&frame.v, uv_off + (x / 2) * bps) - uv_center) * uv_gain;
            let mut r = y + rv * v;
            let mut g = y + gu * u + gv * v;
            let mut b = y + bu * u;
            if matches!(frame.transfer, TransferCharacteristic::SMPTE2084) {
                r = pq(r);
                g = pq(g);
                b = pq(b);
            } else {
                r = hlg(r).powf(1.2);
                g = hlg(g).powf(1.2);
                b = hlg(b).powf(1.2);
            }
            if bt2020 {
                let (r0, g0, b0) = (r, g, b);
                r = 1.6605 * r0 - 0.5876 * g0 - 0.0728 * b0;
                g = -0.1246 * r0 + 1.1329 * g0 - 0.0083 * b0;
                b = -0.0182 * r0 - 0.1006 * g0 + 1.1187 * b0;
            }
            // 与 shader 一致：按亮度做一次 Hable，RGB 等比缩放
            let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).max(1e-6);
            let scale = (hable(lum.clamp(0.0, 1.0)) / white).clamp(0.0, 1.0) / lum;
            let gamma = |v: f32| v.clamp(0.0, 1.0).powf(1.0 / 2.2);
            let o = (row * w + x) * 3;
            rgb[o] = (gamma(r * scale) * 255.0) as u8;
            rgb[o + 1] = (gamma(g * scale) * 255.0) as u8;
            rgb[o + 2] = (gamma(b * scale) * 255.0) as u8;
        }
    }
    Ok((w, h, rgb))
}

// 10bit（16bit 容器低 10 位）→ 8bit，仅用于 CPU 截图兜底
fn ten_to_eight(f: &FrameSnapshot) -> FrameSnapshot {
    let w = f.width.max(0) as usize;
    let h = f.height.max(0) as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let ys = f.y_stride.max(0) as usize;
    let uvs = f.uv_stride.max(0) as usize;
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    // P010 的 10bit 在 16bit 容器的高位（code<<6），先还原到 10bit 再降到 8bit
    let shift = if f.p010 { 6 + 2 } else { 2 };
    let sample = |buf: &[u8], off: usize| -> u8 {
        let lo = *buf.get(off).unwrap_or(&0) as u16;
        let hi = *buf.get(off + 1).unwrap_or(&0) as u16;
        ((lo | (hi << 8)) >> shift) as u8
    };
    for row in 0..h {
        for x in 0..w {
            y[row * w + x] = sample(&f.y, row * ys + x * 2);
        }
    }
    for row in 0..ch {
        for x in 0..cw {
            u[row * cw + x] = sample(&f.u, row * uvs + x * 2);
            v[row * cw + x] = sample(&f.v, row * uvs + x * 2);
        }
    }
    FrameSnapshot {
        y,
        u,
        v,
        width: f.width,
        height: f.height,
        y_stride: w as i32,
        uv_stride: cw as i32,
        color_space: f.color_space,
        color_range: f.color_range,
        bit_depth: 8,
        p010: false,
        transfer: f.transfer,
        primaries: f.primaries,
    }
}

// YUV420P → RGB8。
// 保持逐像素的平铺循环：实测它能被 LLVM 自动向量化（4K 约 50ms/帧），
// 而按 2×2 色度块重组的写法（色度取一次复用 4 次、变长子切片）反而更慢
// ——运行时分支与变址打断了向量化，收益抵不过开销。矩阵系数与 range 参数
// 提到循环外的局部变量，避免逐像素从 ColorMatrix 取字段。
fn yuv_to_rgb(frame: &FrameSnapshot) -> Result<(usize, usize, Vec<u8>)> {
    if matches!(
        frame.transfer,
        TransferCharacteristic::SMPTE2084 | TransferCharacteristic::ARIB_STD_B67
    ) {
        return hdr_to_rgb(frame);
    }
    if frame.bit_depth == 10 {
        // 画面路径已在 GPU 做 PQ/HLG tone map；截图仍走 CPU 时先把 10bit
        // 样本降到 8bit 再复用现有矩阵（HDR 截图的颜色仍不如画面准确，
        // 见 PLAN C2 的遗留项）。低 10 位直接右移 2 位。
        return yuv_to_rgb(&ten_to_eight(frame));
    }
    if frame.bit_depth != 8 {
        anyhow::bail!("screenshot: unsupported bit depth {}", frame.bit_depth);
    }
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
            bit_depth: 8,
            p010: false,
            transfer: ffmpeg_next::util::color::TransferCharacteristic::Unspecified,
            primaries: ffmpeg_next::util::color::Primaries::Unspecified,
        }
    }

    // 10bit 快照先降到 8bit 再复用矩阵；只要求不 panic 且输出尺寸正确
    fn make_10bit_frame(w: i32, h: i32) -> FrameSnapshot {
        use ffmpeg_next::util::color::{Primaries, Space, TransferCharacteristic};
        let (ws, hs) = (w as usize, h as usize);
        let y_stride = ws * 2;
        let uv_stride = (ws / 2) * 2;
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
            color_space: Space::BT2020NCL,
            color_range: Range::MPEG,
            bit_depth: 10,
            p010: false,
            transfer: TransferCharacteristic::SMPTE2084,
            primaries: Primaries::BT2020,
        }
    }

    #[test]
    fn ten_bit_snapshot_falls_back_to_8bit_without_panic() {
        for (w, h) in [(8, 4), (7, 5), (2, 2)] {
            let f = make_10bit_frame(w, h);
            let (rw, rh, rgb) = yuv_to_rgb(&f).expect("10bit screenshot should convert");
            assert_eq!((rw, rh), (w as usize, h as usize));
            assert_eq!(rgb.len(), (w * h * 3) as usize);
        }
    }

    // 10bit PQ 中灰（code 512, U=V=512）应得到可辨识的 SDR 输出，
    // 而不是全黑/全白；同时覆盖 HDR CPU 公式的主路径。
    #[test]
    fn ten_bit_pq_mid_gray_tone_maps_to_visible_gray() {
        let mut f = make_10bit_frame(2, 2);
        // 10bit 512 = 0x0200（little endian）
        for v in f.y.iter_mut() {
            *v = 0;
        }
        for i in (0..f.y.len()).step_by(2) {
            f.y[i + 1] = 0x02;
        }
        for i in (0..f.u.len()).step_by(2) {
            f.u[i + 1] = 0x02;
            f.v[i + 1] = 0x02;
        }
        let (_, _, rgb) = yuv_to_rgb(&f).unwrap();
        assert!(
            rgb.iter().all(|&c| c > 10 && c < 250),
            "unexpected: {rgb:?}"
        );
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
            bit_depth: 8,
            p010: false,
            transfer: ffmpeg_next::util::color::TransferCharacteristic::Unspecified,
            primaries: ffmpeg_next::util::color::Primaries::Unspecified,
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

    // GL 读回的行序是自下而上：翻转必须整行搬，奇偶高度都不能丢/重行
    #[test]
    fn flip_rows_reverses_row_order() {
        for h in [1usize, 2, 3, 4, 5] {
            let (w, stride) = (2usize, 6usize);
            // 每行填行号，翻转后第 y 行应等于 h-1-y
            let mut rgb: Vec<u8> = (0..h).flat_map(|y| vec![y as u8; stride]).collect();
            flip_rows(&mut rgb, w, h);
            for y in 0..h {
                assert_eq!(
                    &rgb[y * stride..(y + 1) * stride],
                    &vec![(h - 1 - y) as u8; stride][..],
                    "h={h} row={y}"
                );
            }
        }
    }
}

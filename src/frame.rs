// 解码线程 ⇄ UI 之间传递的共享类型
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{
    Rational,
    format::Pixel,
    util::color::{Primaries, Range, Space, TransferCharacteristic},
};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dmabuf::{DmaBufFrame, DmaBufLayout};

static FRAME_GEN: AtomicU64 = AtomicU64::new(0);

// 进程内唯一递增的帧代号：渲染端据此识别"新帧"，跳过重复纹理上传
pub fn next_frame_gen() -> u64 {
    FRAME_GEN.fetch_add(1, Ordering::Relaxed)
}

// SAR（像素宽高比）修正后的显示尺寸：非方形像素（变形宽银幕等，sar != 1）
// 按像素比修正宽度，避免内容被拉伸；无效/方形 SAR 原样返回。
// demux（决定窗口尺寸）与 video（决定渲染视口）共用同一份实现，
// 避免两处各写一遍、整数类型还不一致。
pub fn sar_corrected(w: u32, h: u32, sar: Rational) -> (u32, u32) {
    if sar.numerator() > 0 && sar.denominator() > 0 && sar.numerator() != sar.denominator() {
        let dw = ((w as u64 * sar.numerator() as u64) / sar.denominator() as u64).max(1) as u32;
        (dw, h)
    } else {
        (w, h)
    }
}

// 读取解码帧某平面的行距（i32，可能为负）。
// 不能用 frame::Video::stride()：它返回 usize，底层 i32 linesize 的负值
// （垂直翻转的行序）到这层已被转成巨大 usize 而丢失符号。usize→i64→i32
// 的转换对二补码精确，能把负 linesize 正确还原。
pub fn linesize(f: &Video, plane: usize) -> i32 {
    f.stride(plane) as i64 as i32
}

// 色度布局：Planar=Y/U/V 三平面；SemiPlanar=NV12/P010 的 Y + 交错 UV 两平面
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChromaLayout {
    Planar,
    SemiPlanar,
}

// 像素数据的来源：CPU 可读平面（软解/swscale 输出），或 GPU 可直接采样的
// DMA-BUF（VAAPI 零拷贝）。两者共用同一套色彩/显示元数据与 shader。
// 平面长度是否够读出全部行：stride 必须放得下一行，且缓冲要覆盖到最后一行。
// 不满足时不能按 stride 切片——那会越界读（UB）或直接 panic。
// snapshot 跑在 UI 线程的绘制回调里，panic 穿过 FLTK 的 C 边界会 abort 进程，
// 所以宁可放弃这次截图。
fn planes_cover_rows(plane_len: usize, stride: usize, rows: usize, row_bytes: usize) -> bool {
    if stride < row_bytes || rows == 0 {
        return false;
    }
    plane_len >= (rows - 1) * stride + row_bytes
}

enum FrameBuf {
    Cpu(Box<Video>),
    DmaBuf(Box<DmaBufFrame>),
}

// 视频 YUV 帧：直接持有解码器/缩放器给出的 AVFrame 所有权，零拷贝。
// 不再自带 Vec 平面——像素已在解码器缓冲里，再拷一份是纯浪费。
// 帧沿"解码 → 渲染/截图 → 回收"流转，回收后复用同一块缓冲。
//
// 不变式（由 video.rs 的构造路径保证）：
// - CPU 帧恒为 YUV420P/NV12/P010、stride 恒为正（负 stride 经 swscale 摆正）；
// - 平面长度 = stride * 行数，行数对色度取 ceil(height/2)。
pub struct YuvFrame {
    // 像素数据所有权（渲染只读；渲染完由 UI 侧回收给解码线程复用）
    buf: FrameBuf,
    pub width: i32,
    pub height: i32,
    pub y_stride: i32,
    pub uv_stride: i32,
    pub pts_sec: f64,   // 视频帧的时间戳（秒）
    pub frame_gen: u64, // 帧代号（跨源唯一递增）
    // 产出该帧时的 seek 代数。seek 后 UI 通道里可能还压着 seek 之前的帧，
    // 消息泵按此丢弃过期帧，避免 seek 之后先闪一帧旧位置画面
    pub seek_epoch: u64,
    // 流的色彩信息（帧级 side data）：渲染/截图据此选择转换矩阵与范围
    pub color_space: Space,
    pub color_range: Range,
    // 色度布局：Planar（YUV420P/10LE）或 SemiPlanar（NV12/P010LE）
    pub layout: ChromaLayout,
    // 每分量位数：8（YUV420P/NV12）或 10（YUV420P10LE/P010LE）
    pub bit_depth: u8,
    // HDR 传递函数/色域：渲染端据此在 shader 里做 PQ/HLG → SDR tone map
    pub transfer: TransferCharacteristic,
    pub primaries: Primaries,
    // SAR 修正后的显示尺寸（渲染视口、窗口宽高比用）
    pub disp_w: i32,
    pub disp_h: i32,
}

impl YuvFrame {
    // 用一个已就绪的 420P 正 stride 帧构造（video.rs 的 SDR 零拷贝路径）
    pub fn new_shared(
        buf: Box<Video>,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        Self::new_with_depth(
            buf,
            ChromaLayout::Planar,
            8,
            pts_sec,
            color_space,
            color_range,
            TransferCharacteristic::Unspecified,
            Primaries::Unspecified,
            disp_w,
            disp_h,
            seek_epoch,
        )
    }

    // HDR（PQ/HLG）帧：8-bit 或 10-bit 420P，带传递函数/色域，渲染端做 tone map
    #[allow(clippy::too_many_arguments)]
    pub fn new_hdr(
        buf: Box<Video>,
        bit_depth: u8,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        transfer: TransferCharacteristic,
        primaries: Primaries,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        Self::new_with_depth(
            buf,
            ChromaLayout::Planar,
            bit_depth,
            pts_sec,
            color_space,
            color_range,
            transfer,
            primaries,
            disp_w,
            disp_h,
            seek_epoch,
        )
    }

    // 半平面 NV12/P010（硬解下载后的常见格式）：Y 平面 + 交错 UV
    #[allow(clippy::too_many_arguments)]
    pub fn new_semi(
        buf: Box<Video>,
        bit_depth: u8,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        transfer: TransferCharacteristic,
        primaries: Primaries,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        Self::new_with_depth(
            buf,
            ChromaLayout::SemiPlanar,
            bit_depth,
            pts_sec,
            color_space,
            color_range,
            transfer,
            primaries,
            disp_w,
            disp_h,
            seek_epoch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_depth(
        buf: Box<Video>,
        layout: ChromaLayout,
        bit_depth: u8,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        transfer: TransferCharacteristic,
        primaries: Primaries,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        assert!(bit_depth == 8 || bit_depth == 10, "只支持 8/10bit");
        let expect = match (layout, bit_depth) {
            (ChromaLayout::Planar, 8) => Pixel::YUV420P,
            (ChromaLayout::Planar, 10) => Pixel::YUV420P10LE,
            (ChromaLayout::SemiPlanar, 8) => Pixel::NV12,
            (ChromaLayout::SemiPlanar, 10) => Pixel::P010LE,
            _ => panic!("unsupported bit depth {bit_depth}"),
        };
        debug_assert_eq!(buf.format(), expect);
        let y_stride = linesize(&buf, 0);
        let uv_stride = linesize(&buf, 1);
        debug_assert!(y_stride > 0 && uv_stride > 0, "stride 必须为正");
        Self {
            width: buf.width() as i32,
            height: buf.height() as i32,
            buf: FrameBuf::Cpu(buf),
            y_stride,
            uv_stride,
            pts_sec,
            frame_gen: next_frame_gen(),
            seek_epoch,
            color_space,
            color_range,
            layout,
            bit_depth,
            transfer,
            primaries,
            disp_w,
            disp_h,
        }
    }

    // 硬解零拷贝帧：像素在 DMA-BUF 里，渲染端用 EGL 直接导入；不经过 CPU。
    #[allow(clippy::too_many_arguments)]
    pub fn new_dmabuf(
        dma: DmaBufFrame,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        transfer: TransferCharacteristic,
        primaries: Primaries,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        let layout = match dma.layout {
            DmaBufLayout::Planar => ChromaLayout::Planar,
            DmaBufLayout::SemiPlanar => ChromaLayout::SemiPlanar,
        };
        let bit_depth = dma.bit_depth;
        let (width, height) = (
            dma.planes[0].map_or(0, |p| p.width),
            dma.planes[0].map_or(0, |p| p.height),
        );
        let y_stride = dma.planes[0].map_or(0, |p| p.pitch) as i32;
        let uv_stride = dma.planes[1].map_or(0, |p| p.pitch) as i32;
        Self {
            buf: FrameBuf::DmaBuf(Box::new(dma)),
            width,
            height,
            y_stride,
            uv_stride,
            pts_sec,
            frame_gen: next_frame_gen(),
            seek_epoch,
            color_space,
            color_range,
            layout,
            bit_depth,
            transfer,
            primaries,
            disp_w,
            disp_h,
        }
    }

    // 每采样字节数（10bit 为 2）
    #[inline]
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth == 10 { 2 } else { 1 }
    }

    // DMA-BUF 帧（硬解零拷贝）；CPU 帧返回 None
    #[inline]
    pub fn dmabuf(&self) -> Option<&DmaBufFrame> {
        match &self.buf {
            FrameBuf::Cpu(_) => None,
            FrameBuf::DmaBuf(d) => Some(d),
        }
    }

    #[inline]
    pub fn is_dmabuf(&self) -> bool {
        matches!(self.buf, FrameBuf::DmaBuf(_))
    }

    // Y 平面（含行尾 padding，渲染按 stride 上传）。DMA-BUF 帧没有 CPU 平面，
    // 只能由 EGL 导入路径读取；返回空切片避免误用。
    #[inline]
    pub fn y(&self) -> &[u8] {
        match &self.buf {
            FrameBuf::Cpu(b) => b.data(0),
            FrameBuf::DmaBuf(_) => &[],
        }
    }
    // U 平面
    #[inline]
    pub fn u(&self) -> &[u8] {
        match &self.buf {
            FrameBuf::Cpu(b) => b.data(1),
            FrameBuf::DmaBuf(_) => &[],
        }
    }
    // V 平面
    #[inline]
    pub fn v(&self) -> &[u8] {
        match &self.buf {
            FrameBuf::Cpu(b) => b.data(2),
            FrameBuf::DmaBuf(_) => &[],
        }
    }

    // 交错 UV 平面（NV12/P010）
    #[inline]
    pub fn uv(&self) -> &[u8] {
        match &self.buf {
            FrameBuf::Cpu(b) => b.data(1),
            FrameBuf::DmaBuf(_) => &[],
        }
    }

    #[inline]
    pub fn is_semi_planar(&self) -> bool {
        self.layout == ChromaLayout::SemiPlanar
    }

    // 拆出底层 AVFrame 供解码线程回收复用（像素缓冲也随之保留）。
    // DMA-BUF 帧的像素属于硬解 surface，不能当软件目标帧复用，直接 drop。
    pub fn into_buf(self) -> Option<Box<Video>> {
        match self.buf {
            FrameBuf::Cpu(buf) => Some(buf),
            FrameBuf::DmaBuf(_) => None,
        }
    }

    // 拷出一份与解码器无关的平面快照，供后台线程做截图。
    // 必须在 UI 线程上拷贝：回收后的帧会立刻被解码器重填，
    // 后台线程再读会与解码竞争。拷贝本身只是几次 memcpy（几 MB），
    // 远比在 UI 线程做完整编码可忽略。
    // DMA-BUF 帧没有 CPU 平面，返回 None；调用方改用 GL readback 截图。
    //
    // 平面长度/stride 与尺寸对不上时返回 None 而不是 panic：snapshot 跑在
    // UI 线程的绘制回调里，panic 穿过 FLTK 的 C 边界会直接 abort 整个进程。
    // 放弃一次截图远好过把播放器搞崩。
    pub fn snapshot(&self) -> Option<FrameSnapshot> {
        let FrameBuf::Cpu(_) = &self.buf else {
            return None;
        };
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 {
            return None;
        }
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let bps = self.bytes_per_sample();
        // 负 stride 不能直接转 usize（会变成巨大值让下面的边界计算失真/溢出）
        if self.y_stride <= 0 || self.uv_stride <= 0 {
            return None;
        }
        let y_stride = self.y_stride as usize;
        let uv_stride = self.uv_stride as usize;
        let y = self.y();
        // 消费方会按 stride 逐行读到最后一行：长度不够就不能给快照。
        // row_bytes 必须是**字节数**：10bit 一行是 w*2，漏乘 bps 会让
        // 截断的 10bit 平面通过校验，随后在消费端越界
        if !planes_cover_rows(y.len(), y_stride, h, w * bps) {
            return None;
        }
        let (u, v, uv_stride_out) = if self.is_semi_planar() {
            // 半平面在 UI 线程拆成 U/V 两份，后台截图代码仍按平面处理。
            // 按行取 uv 平面一次，再逐对拆交错样本（8/10bit 分开写，常量步长
            // 便于向量化；逐像素调 self.uv() + 1 字节 copy 在 4K 下要十几 ms）
            let row_src = cw * 2 * bps;
            let row_dst = cw * bps;
            let uv = self.uv();
            if !planes_cover_rows(uv.len(), uv_stride, ch, row_src) {
                return None;
            }
            let mut up = vec![0u8; cw * ch * bps];
            let mut vp = vec![0u8; cw * ch * bps];
            for row in 0..ch {
                let src = &uv[row * uv_stride..row * uv_stride + row_src];
                let up_row = &mut up[row * row_dst..(row + 1) * row_dst];
                let vp_row = &mut vp[row * row_dst..(row + 1) * row_dst];
                if bps == 1 {
                    let (pairs, _) = src.as_chunks::<2>();
                    for ((u, v), px) in up_row.iter_mut().zip(vp_row.iter_mut()).zip(pairs) {
                        *u = px[0];
                        *v = px[1];
                    }
                } else {
                    let (pairs, _) = src.as_chunks::<4>();
                    let (up2, _) = up_row.as_chunks_mut::<2>();
                    let (vp2, _) = vp_row.as_chunks_mut::<2>();
                    for ((u, v), px) in up2.iter_mut().zip(vp2.iter_mut()).zip(pairs) {
                        *u = [px[0], px[1]];
                        *v = [px[2], px[3]];
                    }
                }
            }
            (up, vp, (cw * bps) as i32)
        } else {
            let u = self.u();
            let v = self.v();
            if !planes_cover_rows(u.len(), uv_stride, ch, cw * bps)
                || !planes_cover_rows(v.len(), uv_stride, ch, cw * bps)
            {
                return None;
            }
            (u.to_vec(), v.to_vec(), self.uv_stride)
        };
        Some(FrameSnapshot {
            y: y.to_vec(),
            u,
            v,
            width: self.width,
            height: self.height,
            y_stride: self.y_stride,
            uv_stride: uv_stride_out,
            color_space: self.color_space,
            color_range: self.color_range,
            bit_depth: self.bit_depth,
            // P010 的 10bit 样本 MSB 对齐（code<<6），截图换算时要先右移
            p010: self.is_semi_planar() && self.bit_depth == 10,
            transfer: self.transfer,
            primaries: self.primaries,
        })
    }
}

// 自持平面的帧快照：给后台截图线程用，与解码器的帧生命周期解耦
pub struct FrameSnapshot {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub y_stride: i32,
    pub uv_stride: i32,
    pub color_space: Space,
    pub color_range: Range,
    pub bit_depth: u8,
    // 10bit 样本是否为 P010 的 MSB 对齐（否则按低 10 位处理）
    pub p010: bool,
    pub transfer: TransferCharacteristic,
    pub primaries: Primaries,
}

pub enum Message {
    Frame(YuvFrame),
    End,
}

// YUV→RGB 的完整转换参数（归一化到 0..1 域，与渲染 shader 一致）：
//   Y'  = (Ytex - y_off) * y_gain
//   U'  = (Utex - uv_center) * uv_gain
//   V'  = (Vtex - uv_center) * uv_gain
//   R = Y' + rv * V';  G = Y' + gu * U' + gv * V';  B = Y' + bu * U'
pub struct ColorMatrix {
    pub rv: f32,
    pub gu: f32,
    pub gv: f32,
    pub bu: f32,
    pub y_off: f32,
    pub y_gain: f32,
    pub uv_center: f32,
    pub uv_gain: f32,
}

// 按流的色彩空间与范围计算转换参数；未标注时按分辨率回退（HD+ 用 BT.709）。
// range 未标注按 limited (MPEG) 处理，与 FFmpeg 默认一致。
pub fn color_matrix(space: Space, range: Range, height: i32) -> ColorMatrix {
    let (rv, gu, gv, bu) = match space {
        Space::BT709 => (1.5748, -0.1873, -0.4681, 1.8556),
        Space::BT2020NCL | Space::BT2020CL => (1.4746, -0.1645531, -0.5713531, 1.8814),
        Space::SMPTE170M | Space::BT470BG => (1.402, -0.344136, -0.714136, 1.772),
        _ => {
            if height >= 720 {
                (1.5748, -0.1873, -0.4681, 1.8556)
            } else {
                (1.402, -0.344136, -0.714136, 1.772)
            }
        }
    };
    let (y_off, y_gain, uv_center, uv_gain) = match range {
        // limited (MPEG)：Y∈[16,235]，Cb/Cr∈[16,240] 映射到 [0,1] / [-0.5,0.5]
        Range::MPEG | Range::Unspecified => (16.0 / 255.0, 255.0 / 219.0, 0.5, 255.0 / 224.0),
        // full (JPEG)：直接使用
        Range::JPEG => (0.0, 1.0, 0.5, 1.0),
    };
    ColorMatrix {
        rv,
        gu,
        gv,
        bu,
        y_off,
        y_gain,
        uv_center,
        uv_gain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 造一个 420P 帧填入可预测的字节，验证 YuvFrame 的访问器/长度/快照
    fn make_420p(w: u32, h: u32) -> ffmpeg_next::util::frame::video::Video {
        let mut f = ffmpeg_next::util::frame::video::Video::empty();
        unsafe { f.alloc(Pixel::YUV420P, w, h) };
        // data_mut 每次调用都新借用，须一次持有一个平面的可变切片
        for (plane, modu) in [(0usize, 251usize), (1, 253), (2, 241)] {
            let buf = f.data_mut(plane);
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (i % modu) as u8;
            }
        }
        f
    }

    // 访问器返回的平面长度必须与 stride×行数 一致（色度为 ceil(h/2) 行）
    #[test]
    fn plane_lengths_match_stride_times_rows() {
        for (w, h) in [(64u32, 32u32), (65, 33), (2, 2), (6, 4)] {
            let f = YuvFrame::new_shared(
                Box::new(make_420p(w, h)),
                0.0,
                Space::BT709,
                Range::MPEG,
                w as i32,
                h as i32,
                0,
            );
            assert!((f.width, f.height) == (w as i32, h as i32));
            assert!(f.y_stride > 0 && f.uv_stride > 0, "stride 必须为正");
            let y_rows = h as usize;
            let uv_rows = (h as usize).div_ceil(2);
            assert_eq!(f.y().len(), f.y_stride as usize * y_rows);
            assert_eq!(f.u().len(), f.uv_stride as usize * uv_rows);
            assert_eq!(f.v().len(), f.uv_stride as usize * uv_rows);
        }
    }

    // 平面长度校验：stride 放不下一行、或缓冲覆盖不到最后一行时必须拒绝，
    // 否则 snapshot 会越界切片（panic/UB）。snapshot 跑在 UI 线程的绘制回调里，
    // panic 穿过 FLTK 的 C 边界会直接 abort 整个进程
    #[test]
    fn plane_row_coverage_check() {
        // 正常：stride=64、行宽=64、4 行 → 需要 3*64+64 = 256 字节
        assert!(planes_cover_rows(256, 64, 4, 64));
        // 缓冲不足最后一行
        assert!(!planes_cover_rows(255, 64, 4, 64));
        // stride 比一行还小（解码器给出损坏的 linesize）
        assert!(!planes_cover_rows(1024, 32, 4, 64));
        // 0 行
        assert!(!planes_cover_rows(0, 64, 0, 64));
        // 10bit：一行是 w*2 字节。只按 8bit 的长度传 row_bytes 会漏判，
        // 因此调用方必须传字节数（w*bps / cw*bps）
        assert!(!planes_cover_rows(3 * 128 + 64, 128, 4, 64 * 2));
        assert!(planes_cover_rows(3 * 128 + 128, 128, 4, 64 * 2));
    }

    // SAR 修正：方形像素原样；变形宽银幕按比例加宽；无效 SAR（分母 0）原样不 panic
    #[test]
    fn sar_correction_cases() {
        assert_eq!(sar_corrected(1920, 1080, Rational::new(1, 1)), (1920, 1080));
        // 2:1 像素比（变形宽银幕）→ 显示宽度翻倍
        assert_eq!(sar_corrected(1920, 816, Rational::new(2, 1)), (3840, 816));
        assert_eq!(sar_corrected(640, 480, Rational::new(1, 0)), (640, 480));
    }

    // 10bit（YUV420P10LE）平面长度按字节算，且 snapshot 记录位深/Space 元数据
    #[test]
    fn ten_bit_plane_lengths_and_metadata() {
        let w = 64u32;
        let h = 33u32;
        let mut f = ffmpeg_next::util::frame::video::Video::empty();
        unsafe { f.alloc(Pixel::YUV420P10LE, w, h) };
        let frame = YuvFrame::new_hdr(
            Box::new(f),
            10,
            0.0,
            Space::BT2020NCL,
            Range::MPEG,
            TransferCharacteristic::SMPTE2084,
            Primaries::BT2020,
            w as i32,
            h as i32,
            0,
        );
        assert_eq!(frame.bytes_per_sample(), 2);
        let uv_rows = (h as usize).div_ceil(2);
        assert_eq!(frame.y().len(), frame.y_stride as usize * h as usize);
        assert_eq!(frame.u().len(), frame.uv_stride as usize * uv_rows);
        assert_eq!(frame.v().len(), frame.uv_stride as usize * uv_rows);
        let snap = frame.snapshot().unwrap();
        assert_eq!(snap.bit_depth, 10);
        assert_eq!(snap.transfer, TransferCharacteristic::SMPTE2084);
        assert_eq!(snap.primaries, Primaries::BT2020);
    }

    // NV12/P010 半平面：UV 交织，snapshot 必须拆成独立 U/V 平面
    #[test]
    fn semi_planar_snapshot_splits_uv() {
        for (pix, depth, bps) in [(Pixel::NV12, 8u8, 1usize), (Pixel::P010LE, 10, 2)] {
            let (w, h) = (8u32, 4u32);
            let mut f = ffmpeg_next::util::frame::video::Video::empty();
            unsafe { f.alloc(pix, w, h) };
            let frame = YuvFrame::new_semi(
                Box::new(f),
                depth,
                0.0,
                Space::BT709,
                Range::MPEG,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
                w as i32,
                h as i32,
                0,
            );
            assert!(frame.is_semi_planar());
            assert_eq!(frame.bytes_per_sample(), bps);
            let uv_rows = (h as usize).div_ceil(2);
            assert_eq!(frame.uv().len(), frame.uv_stride as usize * uv_rows);
            let snap = frame.snapshot().unwrap();
            assert_eq!(snap.bit_depth, depth);
            // 拆分后每像素 bps 字节，stride = cw*bps
            assert_eq!(snap.u.len(), (w as usize / 2) * bps * uv_rows);
            assert_eq!(snap.v.len(), snap.u.len());
        }
    }

    // snapshot 必须是深拷贝：与底层帧解耦（后台线程据此安全读取）
    #[test]
    fn snapshot_is_independent_copy() {
        let f = YuvFrame::new_shared(
            Box::new(make_420p(8, 4)),
            1.5,
            Space::BT2020NCL,
            Range::JPEG,
            8,
            4,
            0,
        );
        let snap = f.snapshot().unwrap();
        assert_eq!(snap.y, f.y());
        assert_eq!(snap.u, f.u());
        assert_eq!(snap.v, f.v());
        assert_eq!((snap.width, snap.height), (8, 4));
        // 帧被回收（销毁）后快照仍持有数据
        drop(f);
        assert!(!snap.y.is_empty());
    }

    // 半平面拆分的字节内容：U/V 必须逐对拆开并按 stride 取行。
    // 原实现逐像素调 uv()，这里同时锁住"拆对了"与"没读串行"
    #[test]
    fn semi_planar_snapshot_deinterleaves_uv() {
        for (pix, depth, bps) in [(Pixel::NV12, 8u8, 1usize), (Pixel::P010LE, 10, 2)] {
            let (w, h) = (6u32, 4u32);
            let mut f = ffmpeg_next::util::frame::video::Video::empty();
            unsafe { f.alloc(pix, w, h) };
            let stride = f.stride(1);
            {
                let uv = f.data_mut(1);
                for (i, b) in uv.iter_mut().enumerate() {
                    *b = (i % 97) as u8;
                }
            }
            let uv_src = f.data(1).to_vec();
            let frame = YuvFrame::new_semi(
                Box::new(f),
                depth,
                0.0,
                Space::BT709,
                Range::MPEG,
                TransferCharacteristic::Unspecified,
                Primaries::Unspecified,
                w as i32,
                h as i32,
                0,
            );
            let snap = frame.snapshot().unwrap();
            let cw = (w as usize).div_ceil(2);
            let ch = (h as usize).div_ceil(2);
            assert_eq!(snap.u.len(), cw * ch * bps);
            for row in 0..ch {
                for x in 0..cw {
                    let src = row * stride + x * 2 * bps;
                    let dst = (row * cw + x) * bps;
                    assert_eq!(
                        &snap.u[dst..dst + bps],
                        &uv_src[src..src + bps],
                        "{pix:?} U row={row} x={x}"
                    );
                    assert_eq!(
                        &snap.v[dst..dst + bps],
                        &uv_src[src + bps..src + 2 * bps],
                        "{pix:?} V row={row} x={x}"
                    );
                }
            }
        }
    }
}

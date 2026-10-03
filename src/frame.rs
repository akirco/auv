// 解码线程 ⇄ UI 之间传递的共享类型
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{
    Rational,
    format::Pixel,
    util::color::{Range, Space},
};
use std::sync::atomic::{AtomicU64, Ordering};

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

// 视频 YUV 帧：直接持有解码器/缩放器给出的 AVFrame 所有权，零拷贝。
// 不再自带 Vec 平面——像素已在解码器缓冲里，再拷一份是纯浪费。
// 帧沿"解码 → 渲染/截图 → 回收"流转，回收后复用同一块缓冲。
//
// 不变式（由 video.rs 的构造路径保证）：
// - buf 恒为 YUV420P、y_stride/uv_stride 恒为正（负 stride 经 swscale 摆正）；
// - 平面长度 = stride * 行数，行数对色度取 ceil(height/2)。
pub struct YuvFrame {
    // 像素平面所有权（渲染只读；渲染完由 UI 侧回收给解码线程复用）
    buf: Box<Video>,
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
    // SAR 修正后的显示尺寸（渲染视口、窗口宽高比用）
    pub disp_w: i32,
    pub disp_h: i32,
}

impl YuvFrame {
    // 用一个已就绪的 420P 正 stride 帧构造（video.rs 的零拷贝路径）
    pub fn new_shared(
        buf: Box<Video>,
        pts_sec: f64,
        color_space: Space,
        color_range: Range,
        disp_w: i32,
        disp_h: i32,
        seek_epoch: u64,
    ) -> Self {
        debug_assert_eq!(buf.format(), Pixel::YUV420P);
        let y_stride = linesize(&buf, 0);
        let uv_stride = linesize(&buf, 1);
        debug_assert!(y_stride > 0 && uv_stride > 0, "stride 必须为正");
        Self {
            width: buf.width() as i32,
            height: buf.height() as i32,
            buf,
            y_stride,
            uv_stride,
            pts_sec,
            frame_gen: next_frame_gen(),
            seek_epoch,
            color_space,
            color_range,
            disp_w,
            disp_h,
        }
    }

    // Y 平面（含行尾 padding，渲染按 stride 上传）
    #[inline]
    pub fn y(&self) -> &[u8] {
        self.buf.data(0)
    }
    // U 平面
    #[inline]
    pub fn u(&self) -> &[u8] {
        self.buf.data(1)
    }
    // V 平面
    #[inline]
    pub fn v(&self) -> &[u8] {
        self.buf.data(2)
    }

    // 拆出底层 AVFrame 供解码线程回收复用（像素缓冲也随之保留）
    pub fn into_buf(self) -> Box<Video> {
        self.buf
    }

    // 拷出一份与解码器无关的平面快照，供后台线程做截图。
    // 必须在 UI 线程上拷贝：回收后的帧会立刻被解码器重填，
    // 后台线程再读会与解码竞争。拷贝本身只是几次 memcpy（几 MB），
    // 远比在 UI 线程做完整编码可忽略。
    pub fn snapshot(&self) -> FrameSnapshot {
        FrameSnapshot {
            y: self.y().to_vec(),
            u: self.u().to_vec(),
            v: self.v().to_vec(),
            width: self.width,
            height: self.height,
            y_stride: self.y_stride,
            uv_stride: self.uv_stride,
            color_space: self.color_space,
            color_range: self.color_range,
        }
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

    // SAR 修正：方形像素原样；变形宽银幕按比例加宽；无效 SAR（分母 0）原样不 panic
    #[test]
    fn sar_correction_cases() {
        assert_eq!(sar_corrected(1920, 1080, Rational::new(1, 1)), (1920, 1080));
        // 2:1 像素比（变形宽银幕）→ 显示宽度翻倍
        assert_eq!(sar_corrected(1920, 816, Rational::new(2, 1)), (3840, 816));
        assert_eq!(sar_corrected(640, 480, Rational::new(1, 0)), (640, 480));
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
        let snap = f.snapshot();
        assert_eq!(snap.y, f.y());
        assert_eq!(snap.u, f.u());
        assert_eq!(snap.v, f.v());
        assert_eq!((snap.width, snap.height), (8, 4));
        // 帧被回收（销毁）后快照仍持有数据
        drop(f);
        assert!(!snap.y.is_empty());
    }
}

// 解码线程 ⇄ UI 之间传递的共享类型
use ffmpeg_next::util::color::{Range, Space};
use std::sync::atomic::{AtomicU64, Ordering};

static FRAME_GEN: AtomicU64 = AtomicU64::new(0);

// 进程内唯一递增的帧代号：渲染端据此识别"新帧"，跳过重复纹理上传
pub fn next_frame_gen() -> u64 {
    FRAME_GEN.fetch_add(1, Ordering::Relaxed)
}

// 视频 YUV 帧结构
pub struct YuvFrame {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub width: i32,
    pub height: i32,
    pub y_stride: i32,
    pub uv_stride: i32,
    pub pts_sec: f64, // 视频帧的时间戳（秒）
    pub frame_gen: u64, // 帧代号（跨源唯一递增）
    // 流的色彩信息（帧级 side data）：渲染/截图据此选择转换矩阵与范围
    pub color_space: Space,
    pub color_range: Range,
    // SAR 修正后的显示尺寸（渲染视口、窗口宽高比用）
    pub disp_w: i32,
    pub disp_h: i32,
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
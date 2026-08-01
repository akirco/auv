// 解码线程 ⇄ UI 之间传递的共享类型
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
}

pub enum Message {
    Frame(YuvFrame),
    End,
}

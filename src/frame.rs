// 解码线程 ⇄ UI 之间传递的共享类型
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
}

pub enum Message {
    Frame(YuvFrame),
    End,
}

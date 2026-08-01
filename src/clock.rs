use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// Audio Master Clock
#[derive(Clone)]
pub struct MasterClock {
    samples_played: Arc<AtomicU64>,
    sample_rate: u32,
    channels: u32,
    start_instant: Instant,
    has_audio: bool,
}

impl MasterClock {
    /// 创建主时钟，同时返回用于音频回调的共享采样计数器 Arc
    pub fn new(sample_rate: u32, channels: u32, has_audio: bool) -> (Self, Arc<AtomicU64>) {
        let samples_played = Arc::new(AtomicU64::new(0));
        let clock = Self {
            samples_played: samples_played.clone(),
            sample_rate,
            channels,
            start_instant: Instant::now(),
            has_audio,
        };
        (clock, samples_played)
    }

    // 获取声卡当前播放到的绝对时间（秒）
    pub fn get_time_sec(&self) -> f64 {
        if self.has_audio && self.sample_rate > 0 {
            let samples = self.samples_played.load(Ordering::Relaxed) as f64;
            samples / (self.sample_rate as f64 * self.channels as f64)
        } else {
            // 如果视频没有音频轨，退回到系统时钟
            self.start_instant.elapsed().as_secs_f64()
        }
    }
}

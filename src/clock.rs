use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

// Audio Master Clock
pub struct MasterClock {
    samples_played: Arc<AtomicU64>,
    sample_rate: u32,
    channels: u32,
    start_instant: Cell<Instant>,
    // keep get_time_sec 的基准偏移：seek 后把基准跳到目标点，各处时间线对齐
    base_sec: Cell<f64>,
    has_audio: bool,
    // 音频输出失效（设备出错等）时置 true：主时钟退回系统时钟继续播放
    audio_dead: Arc<AtomicBool>,
    // 无音频模式下累计的暂停时长（纳秒）：暂停期间墙钟仍在走，
    // 恢复后扣除这部分，避免视频帧被误判为迟到而全部丢弃
    paused_nanos: AtomicU64,
    // 平滑时钟：回调按缓冲块（约 43ms）增加 samples_played，直接用它会
    // 让视频在每次回调后成批放行。这里记住上次观察到的样本数与时刻，
    // 在两次回调之间按墙钟外推——上限一个缓冲周期，避免欠载时无限跑快
    last_samples: Cell<u64>,
    last_sample_instant: Cell<Instant>,
}

impl MasterClock {
    /// 创建主时钟，同时返回用于音频回调的共享采样计数器 Arc
    pub fn new(
        sample_rate: u32,
        channels: u32,
        has_audio: bool,
        audio_dead: Arc<AtomicBool>,
    ) -> (Self, Arc<AtomicU64>) {
        let samples_played = Arc::new(AtomicU64::new(0));
        let clock = Self {
            samples_played: samples_played.clone(),
            sample_rate,
            channels,
            start_instant: Cell::new(Instant::now()),
            base_sec: Cell::new(0.0),
            has_audio,
            audio_dead,
            paused_nanos: AtomicU64::new(0),
            last_samples: Cell::new(0),
            last_sample_instant: Cell::new(Instant::now()),
        };
        (clock, samples_played)
    }

    // 累加一次暂停时长（由视频线程在暂停等待结束后调用）。
    // 有音频时采样计数随流暂停而冻结，本值不参与计算；
    // 无音频（墙钟路径）时暂停期间时间仍往前走，需扣除这部分。
    pub fn add_pause_duration(&self, d: Duration) {
        self.paused_nanos
            .fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }

    // seek 后复位：采样计数/暂停累计归零，基准时间设为目标点。
    // 访问的都是内部可变状态（Cell/原子），无需 &mut，视频线程闭包内可直接调用
    pub fn seek_to(&self, target_sec: f64) {
        self.samples_played.store(0, Ordering::Relaxed);
        self.paused_nanos.store(0, Ordering::Relaxed);
        self.base_sec.set(target_sec);
        self.start_instant.set(Instant::now());
        self.last_samples.set(0);
        self.last_sample_instant.set(Instant::now());
    }

    // 把"当前时刻"对齐到 t：seek 后首个解码帧的 pts 与之对齐，修正解码器
    // 选中关键帧与请求点的偏差（保持已累计的采样/墙钟量，只平移基准）
    pub fn rebase(&self, t: f64) {
        self.base_sec
            .set(self.base_sec.get() + t - self.get_time_sec());
    }

    // 已交付给声卡的样本数（帧数×声道）。音频收尾看门狗用它判断"是否还有进展"：
    // get_time_sec 在无音频时会退回墙钟，不能用于这个判断
    pub fn audio_samples(&self) -> u64 {
        self.samples_played.load(Ordering::Relaxed)
    }

    // 获取声卡当前播放到的绝对时间（秒）
    pub fn get_time_sec(&self) -> f64 {
        let audio_ok =
            self.has_audio && self.sample_rate > 0 && !self.audio_dead.load(Ordering::Relaxed);
        if audio_ok {
            let raw = self.samples_played.load(Ordering::Relaxed);
            if raw != self.last_samples.get() {
                self.last_samples.set(raw);
                self.last_sample_instant.set(Instant::now());
            }
            // 两次回调之间按墙钟外推，最多外推 50ms（覆盖 2048 帧@48k 的
            // 一个缓冲周期）；欠载/暂停时不会无限跑快
            let elapsed = self.last_sample_instant.get().elapsed().as_secs_f64();
            let denom = self.sample_rate as f64 * self.channels as f64;
            let extra = (elapsed * denom).min(denom * 0.050);
            self.base_sec.get() + (raw as f64 + extra) / denom
        } else {
            // 音频轨缺失、音频输出失效或声卡不可用时，退回系统时钟，
            // 并扣除暂停期间的墙钟漂移
            let paused = self.paused_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
            let wall = (self.start_instant.get().elapsed().as_secs_f64() - paused).max(0.0);
            self.base_sec.get() + wall
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 收尾看门狗依赖 audio_samples 的原始计数：它与 get_time_sec 不同，
    // 在无音频/音频失效时不会退回墙钟；seek 后必须归零
    #[test]
    fn audio_samples_track_and_reset_on_seek() {
        let dead = Arc::new(AtomicBool::new(false));
        let (clock, samples) = MasterClock::new(48_000, 2, true, dead);
        samples.fetch_add(96_000, Ordering::Relaxed); // 1 秒立体声
        assert_eq!(clock.audio_samples(), 96_000);
        assert!((clock.get_time_sec() - 1.0).abs() < 0.01);

        clock.seek_to(10.0);
        assert_eq!(clock.audio_samples(), 0);
        assert!((clock.get_time_sec() - 10.0).abs() < 0.01);
    }

    // 平滑时钟：两次回调之间按墙钟外推，但不能超过一个缓冲周期（50ms）；
    // seek 后必须重新从目标点开始，不能带着之前的 extrapolation
    #[test]
    fn clock_extrapolates_between_callbacks_with_cap() {
        let dead = Arc::new(AtomicBool::new(false));
        let (clock, samples) = MasterClock::new(48_000, 2, true, dead);
        samples.fetch_add(96_000, Ordering::Relaxed); // 1 秒
        let t0 = clock.get_time_sec();
        std::thread::sleep(Duration::from_millis(20));
        let t1 = clock.get_time_sec();
        assert!(t1 > t0, "两次回调之间应向前外推");
        assert!(t1 - t0 <= 0.051, "外推不得超过一个缓冲周期: {}", t1 - t0);
        clock.seek_to(5.0);
        assert!((clock.get_time_sec() - 5.0).abs() < 0.01);
    }
}

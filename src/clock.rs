use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
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
    // 音频输出失效（设备出错等）时置 1：主时钟退回系统时钟继续播放
    audio_dead: Arc<AtomicU64>,
    // 无音频模式下累计的暂停时长（纳秒）：暂停期间墙钟仍在走，
    // 恢复后扣除这部分，避免视频帧被误判为迟到而全部丢弃
    paused_nanos: AtomicU64,
}

impl MasterClock {
    /// 创建主时钟，同时返回用于音频回调的共享采样计数器 Arc
    pub fn new(
        sample_rate: u32,
        channels: u32,
        has_audio: bool,
        audio_dead: Arc<AtomicU64>,
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
    }

    // 把"当前时刻"对齐到 t：seek 后首个解码帧的 pts 与之对齐，修正解码器
    // 选中关键帧与请求点的偏差（保持已累计的采样/墙钟量，只平移基准）
    pub fn rebase(&self, t: f64) {
        self.base_sec
            .set(self.base_sec.get() + t - self.get_time_sec());
    }

    // 获取声卡当前播放到的绝对时间（秒）
    pub fn get_time_sec(&self) -> f64 {
        let audio_ok =
            self.has_audio && self.sample_rate > 0 && self.audio_dead.load(Ordering::Relaxed) == 0;
        if audio_ok {
            let samples = self.samples_played.load(Ordering::Relaxed) as f64;
            self.base_sec.get() + samples / (self.sample_rate as f64 * self.channels as f64)
        } else {
            // 音频轨缺失、音频输出失效或声卡不可用时，退回系统时钟，
            // 并扣除暂停期间的墙钟漂移
            let paused = self.paused_nanos.load(Ordering::Relaxed) as f64 / 1_000_000_000.0;
            let wall = (self.start_instant.get().elapsed().as_secs_f64() - paused).max(0.0);
            self.base_sec.get() + wall
        }
    }
}

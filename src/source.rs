// 一个媒体源内多个线程共享的句柄集合。
// 这些 Arc 原本分别重复出现在 VideoThreadCtx / AudioThreadCtx 里，
// 收成一份后各线程 ctx 只保留自己独有的字段，构造处也不易错位。
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use crate::{demux::SeekCtl, pause::PauseGate};

// 一个源的运行期计数：只在源结束时汇总一次 info! 输出，
// 用于区分"解码慢 / 丢帧 / 音频欠载 / seek 频繁"这几类问题。
#[derive(Default)]
pub struct SourceStats {
    pub decoded: AtomicU64,
    pub displayed: AtomicU64,
    // 因严重迟到被丢弃的空闲帧（不含 catch-up 自己跳过的）
    pub late_dropped: AtomicU64,
    // 因持续迟到而发起的"追帧 seek"次数
    pub catchups: AtomicU64,
    // 解码器返回的无效数据/硬件错误次数（EAGAIN 背压不计）。流损坏、驱动故障
    // 时它增长，而画面只表现为"停住"，没有这个计数就完全看不出来
    pub decode_errors: AtomicU64,
    // 走 EGL/DMABUF 零拷贝上屏的硬件帧数
    pub dmabuf_frames: AtomicU64,
    // cpal 报告的 xrun / 欠载
    pub xruns: AtomicU64,
    pub underruns: AtomicU64,
    // demux 读错误次数（网络中断/磁盘错误等）。正常播完不算错误。
    // 没有它就无法把"文件读坏了"和"播放结束"区分开
    pub read_errors: AtomicU64,
}

impl SourceStats {
    pub fn new() -> Self {
        Self::default()
    }

    fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }

    pub fn summary(&self) -> String {
        format!(
            "decoded={} displayed={} late_dropped={} catchups={} dmabuf={} xruns={} underruns={} decode_errors={} read_errors={}",
            Self::get(&self.decoded),
            Self::get(&self.displayed),
            Self::get(&self.late_dropped),
            Self::get(&self.catchups),
            Self::get(&self.dmabuf_frames),
            Self::get(&self.xruns),
            Self::get(&self.underruns),
            Self::get(&self.decode_errors),
            Self::get(&self.read_errors),
        )
    }
}

pub struct SourceShared {
    // 暂停门（UI 翻转，解码线程等待）
    pub paused: Arc<PauseGate>,
    // 音频是否已播完（含环形缓冲排空）；视频线程据此决定何时结束
    pub audio_done: Arc<AtomicBool>,
    // 音频输出/解码失效标记；置位后主时钟退回墙钟
    pub audio_dead: Arc<AtomicBool>,
    // 视频线程等待"音频播完"时挂起于此，由音频线程唤醒
    pub done_cond: Arc<(Mutex<()>, Condvar)>,
    // 跨线程 seek 协调器
    pub ctl: Arc<SeekCtl>,
    // 本源已进入收尾：所有阻塞等待（解码/环缓冲/暂停/seek）都应尽快退出。
    // cpal 回调被 drop 后环缓冲不再排空，没有这个标志音频线程会永久驻留
    pub stop: Arc<AtomicBool>,
    // 运行期计数（视频线程 + 音频回调 + 收尾日志共用）
    pub stats: Arc<SourceStats>,
}

impl SourceShared {
    pub fn new(
        paused: Arc<PauseGate>,
        audio_done: Arc<AtomicBool>,
        audio_dead: Arc<AtomicBool>,
        done_cond: Arc<(Mutex<()>, Condvar)>,
        ctl: Arc<SeekCtl>,
        stop: Arc<AtomicBool>,
        stats: Arc<SourceStats>,
    ) -> Self {
        Self {
            paused,
            audio_done,
            audio_dead,
            done_cond,
            ctl,
            stop,
            stats,
        }
    }
}

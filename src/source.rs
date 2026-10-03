// 一个媒体源内多个线程共享的句柄集合。
// 这些 Arc 原本分别重复出现在 VideoThreadCtx / AudioThreadCtx 里，
// 收成一份后各线程 ctx 只保留自己独有的字段，构造处也不易错位。
use std::sync::{Arc, Condvar, Mutex, atomic::AtomicBool};

use crate::{demux::SeekCtl, pause::PauseGate};

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
}

impl SourceShared {
    pub fn new(
        paused: Arc<PauseGate>,
        audio_done: Arc<AtomicBool>,
        audio_dead: Arc<AtomicBool>,
        done_cond: Arc<(Mutex<()>, Condvar)>,
        ctl: Arc<SeekCtl>,
    ) -> Self {
        Self {
            paused,
            audio_done,
            audio_dead,
            done_cond,
            ctl,
        }
    }
}

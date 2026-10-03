// 暂停门：跨线程共享的"暂停中"状态与唤醒原语。
// UI 线程翻转状态并唤醒，解码线程在暂停期间挂起等待。
// 集中成一个具名类型，避免 Arc<(Mutex<bool>, Condvar)> 元组散落各处、
// 也让"notify 可能丢失"的论证只需写一遍。
use std::{
    sync::{Condvar, Mutex},
    time::Duration,
};

// 暂停等待的轮询间隔：seek 的 notify 可能恰好在进入等待前发生，
// 定时醒来复查可兜住这种丢失；50ms 对交互延迟足够，空闲时也只极少唤醒。
const POLL_INTERVAL: Duration = Duration::from_millis(50);

pub struct PauseGate {
    lock: Mutex<bool>,
    cond: Condvar,
}

impl PauseGate {
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(false),
            cond: Condvar::new(),
        }
    }

    // 当前是否暂停中
    pub fn is_paused(&self) -> bool {
        *self.lock.lock().unwrap()
    }

    // 翻转暂停状态并唤醒等待者，返回翻转后的值（UI 线程用）。
    // 状态变更与 notify 在同一临界区完成，等待者不会错过通知
    pub fn toggle(&self) -> bool {
        let mut g = self.lock.lock().unwrap();
        *g = !*g;
        let now = *g;
        self.cond.notify_all();
        now
    }

    // 复位为非暂停并唤醒（换源时用）
    pub fn reset(&self) {
        let mut g = self.lock.lock().unwrap();
        *g = false;
        self.cond.notify_all();
    }

    // 单纯唤醒等待者（不改状态）：seek 请求需要让暂停中的解码线程立即醒来处理
    pub fn wake(&self) {
        self.cond.notify_all();
    }

    // 若处于暂停则挂起等待。`interrupted` 在等待期间（持锁）被调用，
    // 返回 true 表示应中断等待（如出现了新的 seek）；此时本函数返回 true。
    // 轮询式检查避免 notify 恰好在进入等待前发生的丢失。
    pub fn wait_while_paused(&self, interrupted: impl Fn() -> bool) -> bool {
        let mut guard = self.lock.lock().unwrap();
        while *guard {
            if interrupted() {
                return true;
            }
            let (g, _) = self.cond.wait_timeout(guard, POLL_INTERVAL).unwrap();
            guard = g;
        }
        false
    }
}

impl Default for PauseGate {
    fn default() -> Self {
        Self::new()
    }
}

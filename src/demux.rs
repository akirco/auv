use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::format;
use log::{info, warn};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Condvar, Mutex};

// 跨线程 seek 协调（每源一个）。
// 角色分工：
// - 主线程  request()：写入目标并递增代数（epoch），随后 notify 唤醒暂停中的解码线程；
// - 解码线程 snapshot_new()：发现代数变化 → 本地复位（flush 解码器、排空通道、
//   清空环形缓冲、复位主时钟）→ report_ready() 回报就绪；
// - demux   snapshot_new()：发现代数变化 → wait_ready() 等所有解码线程就绪 →
//   take_target() 取目标并 seek 文件 → 重建 packet 迭代器继续分发。
// 就绪屏障按代数记录：中途又来新 seek 时，新代数覆盖旧计数、旧目标随即失效
// （take_target 校验世代），demux 下一轮自然取到最新目标，不会永久等待。
pub struct SeekCtl {
    // (目标, 代数) 在同一把锁下更新，保证两者读数一致
    state: Mutex<SeekState>,
    // 就绪屏障：(已回报就绪的代数, 该代下已回报的线程数)
    ready: Mutex<(u64, usize)>,
    ready_cond: Condvar,
    ready_max: usize,
}

struct SeekState {
    target: Option<f64>, // 待处理的 seek 目标（秒）
    epoch: u64,          // 单调递增的 seek 代数
}

impl SeekCtl {
    /// ready_max：需要就绪的解码线程数（视频线程 1 + 有音频输出时的音频线程 1）
    pub fn new(ready_max: usize) -> Self {
        Self {
            state: Mutex::new(SeekState {
                target: None,
                epoch: 0,
            }),
            ready: Mutex::new((0, 0)),
            ready_cond: Condvar::new(),
            ready_max,
        }
    }

    /// 主线程请求 seek 到 target_sec（秒）。同一时刻只保留最新目标
    pub fn request(&self, target_sec: f64) {
        let mut g = self.state.lock().unwrap();
        g.target = Some(target_sec);
        g.epoch += 1;
    }

    /// 当前 seek 代数（只读）。解码线程在暂停等待里用它检测是否有新 seek
    pub fn epoch(&self) -> u64 {
        self.state.lock().unwrap().epoch
    }

    /// 轮询：local_epoch 为本线程已处理的代数；有新代数时返回 (新代数, 目标)。
    /// 目标与代数在同一锁内读取，保证配对一致
    pub fn snapshot_new(&self, local_epoch: u64) -> Option<(u64, Option<f64>)> {
        let g = self.state.lock().unwrap();
        if g.epoch != local_epoch {
            Some((g.epoch, g.target))
        } else {
            None
        }
    }

    /// 解码线程在本地复位完成后回报就绪。新代数会重置计数再累加
    pub fn report_ready(&self, epoch: u64) {
        let mut g = self.ready.lock().unwrap();
        if g.0 != epoch {
            g.0 = epoch;
            g.1 = 0;
        }
        g.1 += 1;
        self.ready_cond.notify_all();
    }

    /// demux 等待所有解码线程就绪到"至少该代数"。若期间出现更新代数的完整就绪
    /// （如连续快速 seek），同样满足条件，demux 不会永久等待
    pub fn wait_ready(&self, epoch: u64) {
        let mut g = self.ready.lock().unwrap();
        while !(g.0 >= epoch && g.1 >= self.ready_max) {
            g = self.ready_cond.wait(g).unwrap();
        }
    }

    /// demux 取走目标：仅当目标仍属于世代 epoch（未被新 seek 覆盖）时才有效
    pub fn take_target(&self, epoch: u64) -> Option<f64> {
        let mut g = self.state.lock().unwrap();
        if g.epoch == epoch {
            g.target.take()
        } else {
            None
        }
    }
}

// 解复用线程：统一读取媒体 packet，按流类型分发给视频/音频解码线程。
// 单一 demuxer 避免视频、音频线程各自打开网络流（m3u8 等）重复下载，带宽减半。
// 通道有容量上限，消费跟不上时 send 会阻塞，自动把下载/解包节流到播放速度。
// seek 时协调三个线程：解码线程先完成本地复位，demux 才移动读位置，保证
// seek 后的新 packet 不会与旧 packet 混入解码管线（时序由就绪屏障保证）。
pub fn spawn_demux_thread(
    mut ictx: format::context::Input,
    video_index: Option<usize>,
    audio_index: Option<usize>,
    video_tx: SyncSender<Packet>,
    audio_tx: SyncSender<Packet>,
    ctl: Arc<SeekCtl>,
) {
    std::thread::spawn(move || {
        let mut seen = 0u64;
        'source: loop {
            // 用带标签的块把 packets 迭代器的只读借用限制在块内：
            // 块结束后才能 &mut ictx 做 seek。块结果 Some(epoch) 表示要 seek
            let seek_epoch = 'read: {
                let mut iter = ictx.packets();
                loop {
                    if let Some((e, _t)) = ctl.snapshot_new(seen) {
                        break 'read Some(e);
                    }
                    match iter.next() {
                        Some((stream, packet)) => {
                            let idx = stream.index();
                            let result = if Some(idx) == video_index {
                                video_tx.send(packet)
                            } else if Some(idx) == audio_index {
                                audio_tx.send(packet)
                            } else {
                                continue;
                            };
                            if result.is_err() {
                                break 'source; // 解码线程已退出，停止分发
                            }
                        }
                        None => break 'read None, // EOF 或读取错误（迭代器内部已处理）
                    }
                }
            };
            let Some(e) = seek_epoch else {
                break 'source;
            };
            seen = e;
            // 等两个解码线程全部复位完成（flush/排空），再移动读位置发新包
            ctl.wait_ready(e);
            if let Some(target_sec) = ctl.take_target(e) {
                // avformat_seek_file(stream=-1) 使用微秒（AV_TIME_BASE）；
                // 允许 ±0.25s 容差便于选中目标附近的关键帧
                let ts = (target_sec * 1_000_000.0) as i64;
                let lo = ts.saturating_sub(250_000);
                let hi = ts.saturating_add(250_000);
                let ok = ictx.seek(ts, lo..hi).is_ok();
                if ok {
                    info!("Seek to {:.2}s", target_sec);
                } else {
                    warn!("Seek to {:.2}s failed", target_sec);
                }
            }
            // 继续 'source：重建 packet 迭代器（读位置已更新）
        }
        // 循环结束（EOF/错误）后 sender 随线程退出而 drop，
        // 解码线程的 recv 返回 Err，触发各自的 flush 收尾。
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // 基本流程：请求 → 解码线程各报就绪 → demux 取目标（仅一次）
    #[test]
    fn request_ready_take_target() {
        let ctl = SeekCtl::new(2);
        assert!(ctl.snapshot_new(0).is_none());

        ctl.request(12.5);
        let (e, t) = ctl.snapshot_new(0).expect("应看到新代数");
        assert_eq!(e, 1);
        assert_eq!(t, Some(12.5));

        ctl.report_ready(e);
        ctl.report_ready(e);
        ctl.wait_ready(e); // 两个线程都就绪，应立即可通过
        assert_eq!(ctl.take_target(e), Some(12.5));
        assert_eq!(ctl.take_target(e), None); // 只能取一次
    }

    // 连续快速 seek：旧代数等待不能死锁，旧目标失效、新目标有效
    #[test]
    fn rapid_seek_supersedes_old_target() {
        let ctl = SeekCtl::new(2);

        ctl.request(1.0);
        let (e1, _) = ctl.snapshot_new(0).unwrap();
        ctl.report_ready(e1); // 只报了一个线程就又有新 seek

        ctl.request(2.0);
        let (e2, _) = ctl.snapshot_new(e1).unwrap();
        assert_eq!(e2, e1 + 1);

        // 两个线程改为对新代数就绪（report_ready 换代会重置计数）
        ctl.report_ready(e2);
        ctl.report_ready(e2);

        // demux 等旧代数 e1 也必须能通过（e2 就绪已覆盖 e1）
        ctl.wait_ready(e1);
        assert_eq!(ctl.take_target(e1), None); // 旧目标已被覆盖
        assert_eq!(ctl.take_target(e2), Some(2.0));
    }

    // 无 seek 时轮询返回 None，不会误触发复位
    #[test]
    fn no_seek_snapshot_is_none() {
        let ctl = SeekCtl::new(1);
        assert!(ctl.snapshot_new(0).is_none());
        assert_eq!(ctl.epoch(), 0);
        assert!(ctl.take_target(0).is_none());
    }
}
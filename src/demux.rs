use anyhow::Result;
use ffmpeg_next::{codec, codec::packet::Packet, format, media};
use log::{info, warn};

use crate::pause::PauseGate;
use crate::util::ReadWatchdog;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::SyncSender,
};

// 带 seek 代数的 packet：demux 在读取时打上当前代数，解码线程在本地复位后
// 丢弃"seek 前已读出、但在复位后才送达"的旧 packet。没有这个标记时，
// demux 阻塞在 send 上的旧包会在解码线程 reset 之后被当成新时间轴数据解码。
pub struct TimedPacket {
    pub epoch: u64,
    pub packet: Packet,
}

// 一次 seek 的处理结果：demux 移动读位置之后发布，解码线程据此决定是否本地复位。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekOutcome {
    // 定位成功：读位置已变，解码端需要本地复位（flush 解码器、排空通道、
    // 把主时钟拨到目标点），否则新旧时间轴的帧会混在一起。
    Reset,
    // 定位失败：读位置没变（如 yt-dlp 管道这类不可 seek 的输入）。
    // 解码端必须保持当前状态继续播放——此时若照常 flush，参考帧会被清空，
    // 后续每一帧都找不到参考帧（HEVC 会刷 "Could not find ref with POC"），
    // 画面直接坏掉直到下一个关键帧。
    Failed,
}

// demux 发布、解码线程消费的一次 seek 处理结果
#[derive(Clone, Copy, Debug)]
pub struct SeekDecision {
    pub epoch: u64,
    pub outcome: SeekOutcome,
    // 本次实际定位到的目标（秒）；Failed 时为 None
    pub target: Option<f64>,
}

// 跨线程 seek 协调（每源一个）。
// 角色分工：
// - 主线程  request()：写入目标并递增代数（epoch），随后 notify 唤醒暂停中的解码线程；
// - demux   snapshot_new()：发现代数变化 → **先移动读位置**（seek 文件）→
//     publish_decision() 发布处理结果 → wait_ready() 等所有解码线程就绪；
// - 解码线程 snapshot_decision()：看到处理结果 → 按结果本地复位（Reset：
//   flush 解码器、排空通道、清空环形缓冲、复位主时钟）或保持现状（Failed）
//   → report_ready() 回报就绪。
//
// 顺序至关重要：定位失败时读位置没有变化，解码器若已 flush 就会丢掉全部参考帧，
// 后续帧全部解不出来。因此必须"先定位、后复位"，定位失败就不复位。
//
// 就绪屏障按代数记录：中途又来新 seek 时，新代数覆盖旧计数，demux 下一轮
// 自然取到最新目标，不会永久等待。
pub struct SeekCtl {
    // (目标, 代数) 在同一把锁下更新，保证两者读数一致
    state: Mutex<SeekState>,
    // state.epoch 的原子镜像：热路径只关心"有没有新 seek"——demux 每读一个包
    // 调一次 snapshot_new、视频同步的 250µs 逼近循环每轮调一次 epoch()，
    // 原来每次都要抢 state 的 Mutex。改读原子后无 seek 时完全无锁。
    // 镜像在 state 锁内发布（见 request），因此它与 state.epoch 始终一致，
    // 读到的值就是权威值；snapshot_new 命中镜像后仍取锁复核，读者随后发生的
    // 新 seek 只会让它看到更新的代数与配对目标。
    epoch_hint: AtomicU64,
    // state.decided 的原子镜像：解码线程轮询处理结果的快路径
    decision_hint: AtomicU64,
    // 就绪屏障：当前代数、已回报的线程数、还需要多少线程
    ready: Mutex<ReadyState>,
    ready_cond: Condvar,
    // seek 请求计数（含 UI 与视频线程的追帧 seek），用于源结束时的统计
    requests: AtomicU64,
    // 该源是否支持定位（seek）：yt-dlp 管道是单向流，matroska 的 Cues 索引
    // 在文件末尾、管道根本读不到，定位必然失败。这种源直接禁用 UI 的 seek。
    seekable: AtomicBool,
}

struct ReadyState {
    epoch: u64,
    count: usize,
    needed: usize,
}

struct SeekState {
    target: Option<f64>, // 最新一次请求的 seek 目标（秒）
    epoch: u64,          // 单调递增的 seek 代数（权威值，镜像是它的副本）
    // 已发布处理结果的最高代数（0 = 还没有）；对应结果与目标一并保存，
    // 避免解码线程读到"更新的目标配旧的代数"
    decided: u64,
    decided_outcome: SeekOutcome,
    decided_target: Option<f64>,
}

impl SeekCtl {
    /// ready_max：需要就绪的解码线程数（视频线程 1 + 有音频输出时的音频线程 1）
    pub fn new(ready_max: usize) -> Self {
        Self {
            state: Mutex::new(SeekState {
                target: None,
                epoch: 0,
                decided: 0,
                decided_outcome: SeekOutcome::Failed,
                decided_target: None,
            }),
            epoch_hint: AtomicU64::new(0),
            decision_hint: AtomicU64::new(0),
            ready: Mutex::new(ReadyState {
                epoch: 0,
                count: 0,
                needed: ready_max,
            }),
            ready_cond: Condvar::new(),
            requests: AtomicU64::new(0),
            seekable: AtomicBool::new(true),
        }
    }

    /// 主线程请求 seek 到 target_sec（秒）。同一时刻只保留最新目标。
    /// 目标、代数与原子镜像在同一临界区内一起更新：读者要么三者都看不到
    /// （无新 seek），要么看到已配好的新代数与新目标，不会读到中间态。
    pub fn request(&self, target_sec: f64) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        let mut g = self.state.lock().unwrap();
        g.target = Some(target_sec);
        g.epoch += 1;
        // 必须在锁内发布：否则 state 会短暂领先镜像，解码线程从锁内读到的
        // 代数将与 epoch() 不一致，原子快路径此后每轮都判为"有变化"而白跑一次锁
        self.epoch_hint.store(g.epoch, Ordering::Release);
    }

    /// 当前 seek 代数（只读，无锁）。解码线程在暂停等待与同步逼近循环里
    /// 用它检测是否有新 seek；返回值与 state.epoch 一致，无需再取锁
    pub fn epoch(&self) -> u64 {
        self.epoch_hint.load(Ordering::Acquire)
    }

    /// 累计收到的 seek 请求数（统计用）
    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    /// 轮询：local_epoch 为本线程已处理的代数；有新代数时返回 (新代数, 目标)。
    /// 快路径先读原子镜像（无 seek 时不加锁），有变化时取锁复核，
    /// 目标与代数在同一锁内读取，保证配对一致
    pub fn snapshot_new(&self, local_epoch: u64) -> Option<(u64, Option<f64>)> {
        // 快路径：镜像与本地代数一致即无新 seek，不加锁
        if self.epoch() == local_epoch {
            return None;
        }
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
        if g.epoch != epoch {
            g.epoch = epoch;
            g.count = 0;
        }
        g.count += 1;
        self.ready_cond.notify_all();
    }

    /// demux 移动读位置之后发布该代数的处理结果：解码线程据此决定本地复位
    /// 还是保持现状。过期决定（代数小于已发布的）直接忽略。
    pub fn publish_decision(&self, epoch: u64, outcome: SeekOutcome, target: Option<f64>) {
        let mut g = self.state.lock().unwrap();
        if epoch < g.decided {
            return;
        }
        g.decided = epoch;
        g.decided_outcome = outcome;
        g.decided_target = target;
        self.decision_hint.store(epoch, Ordering::Release);
    }

    /// 解码线程轮询处理结果：local_epoch 为本线程已处理的代数，
    /// 有新结果时返回 (代数, 结果, 目标)。快路径只读原子镜像，无结果时完全无锁。
    pub fn snapshot_decision(&self, local_epoch: u64) -> Option<SeekDecision> {
        if self.decision_hint.load(Ordering::Acquire) <= local_epoch {
            return None;
        }
        let g = self.state.lock().unwrap();
        if g.decided > local_epoch {
            Some(SeekDecision {
                epoch: g.decided,
                outcome: g.decided_outcome,
                target: g.decided_target,
            })
        } else {
            None
        }
    }

    /// 该源是否支持定位（seek）。不可 seek 的源（yt-dlp 管道）上 UI 不应发起 seek
    pub fn seekable(&self) -> bool {
        self.seekable.load(Ordering::Relaxed)
    }

    /// 设置该源是否支持定位（seek）
    pub fn set_seekable(&self, seekable: bool) {
        self.seekable.store(seekable, Ordering::Relaxed);
    }

    /// 某个解码线程已永久退出（初始化失败 / 解码错误 / panic）：把它从就绪
    /// 屏障里移除。否则下一次 seek 时 demux 会永远等一个再也不会 report_ready
    /// 的线程，整个源卡死。幂等性不保证，调用方只在非正常退出路径各调一次。
    pub fn abandon_decoder(&self) {
        let mut g = self.ready.lock().unwrap();
        g.needed = g.needed.saturating_sub(1);
        self.ready_cond.notify_all();
    }

    /// demux 等待所有解码线程就绪到"至少该代数"。若期间出现更新代数的完整就绪
    /// （如连续快速 seek），同样满足条件，demux 不会永久等待
    pub fn wait_ready(&self, epoch: u64) {
        let mut g = self.ready.lock().unwrap();
        while !(g.epoch >= epoch && g.count >= g.needed) {
            g = self.ready_cond.wait(g).unwrap();
        }
    }
}

// 解复用线程的完整输入：收成结构体，避免 8 个散参数（也便于后续
// SourceSession 直接持有/传递）
pub struct DemuxCtx {
    pub ictx: format::context::Input,
    pub video_index: Option<usize>,
    pub audio_index: Option<usize>,
    pub video_tx: SyncSender<TimedPacket>,
    pub audio_tx: Option<SyncSender<TimedPacket>>,
    pub ctl: Arc<SeekCtl>,
    pub timeline_origin: f64,
    pub stop: Arc<AtomicBool>,
    pub watchdog: Option<ReadWatchdog>,
    // 暂停门：发布 seek 处理结果后要唤醒暂停中的解码线程，
    // 否则它们会一直挂在暂停等待里看不到新结果
    pub paused: Arc<PauseGate>,
}

// 解复用线程：统一读取媒体 packet，按流类型分发给视频/音频解码线程。
// 单一 demuxer 避免视频、音频线程各自打开网络流（m3u8 等）重复下载，带宽减半。
// 通道有容量上限，消费跟不上时 send 会阻塞，自动把下载/解包节流到播放速度。
// seek 时协调三个线程：demux 先移动读位置，再把处理结果（成功/失败）发布给
// 解码线程，解码线程据此决定是否本地复位，全部就绪后 demux 才继续发包。
// 顺序反过来会有致命问题：定位失败（不可 seek 的流）时读位置没变，
// 解码器却已 flush，参考帧被清空，之后每帧都解不出来。
//
// 返回 JoinHandle：demux 内的 seek/迭代错误是就地 warn 处理、不上抛的，
// 但 panic（如 unwrap 失配）仍可能发生，调用方保留句柄以便将来 join。
pub fn spawn_demux_thread(ctx: DemuxCtx) -> std::thread::JoinHandle<()> {
    let DemuxCtx {
        mut ictx,
        video_index,
        audio_index,
        video_tx,
        mut audio_tx,
        ctl,
        timeline_origin,
        stop,
        watchdog,
        paused,
    } = ctx;
    std::thread::spawn(move || {
        let mut seen = 0u64;
        'source: loop {
            // 本源已收尾（播完/关窗/视频线程异常）：尽快退出，不再读网络/文件
            if stop.load(Ordering::Relaxed) {
                break 'source;
            }
            // 用带标签的块把 packets 迭代器的只读借用限制在块内：
            // 块结束后才能 &mut ictx 做 seek。块结果 Some((epoch, target)) 表示要 seek
            let seek_req = 'read: {
                let mut iter = ictx.packets();
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break 'source;
                    }
                    if let Some((e, t)) = ctl.snapshot_new(seen) {
                        break 'read Some((e, t));
                    }
                    if let Some(w) = watchdog.as_ref() {
                        w.refresh();
                    }
                    match iter.next() {
                        Some((stream, packet)) => {
                            let idx = stream.index();
                            if Some(idx) == video_index {
                                // 视频是必需的：视频消费端退出后整个源没有继续意义
                                if video_tx
                                    .send(TimedPacket {
                                        epoch: seen,
                                        packet,
                                    })
                                    .is_err()
                                {
                                    break 'source;
                                }
                            } else if Some(idx) == audio_index {
                                // 没有音频输出/没有音频线程（None）时直接丢弃音频包，
                                // 绝不能让 demux 阻塞在一个无人消费的通道上
                                let failed = matches!(
                                    audio_tx.as_ref().map(|tx| tx.send(TimedPacket {
                                        epoch: seen,
                                        packet
                                    })),
                                    Some(Err(_))
                                );
                                if failed {
                                    // 音频解码线程已退出：只停音频，继续送视频。
                                    // 不能像以前那样 break 整个源，否则音频故障会
                                    // 连带把画面一起停掉
                                    warn!("audio decoder stopped; dropping audio stream");
                                    audio_tx = None;
                                }
                            }
                        }
                        None => break 'read None, // EOF 或读取错误（迭代器内部已处理）
                    }
                }
            };
            let Some((e, target)) = seek_req else {
                break 'source;
            };
            seen = e;
            // 1) 先移动读位置。avformat_seek_file(stream=-1) 使用微秒
            //    （AV_TIME_BASE）的**绝对**时间戳：容器 start_time 不为 0
            //    （TS/HLS 常见）时必须加回原点，否则 seek 会整体偏移 start_time
            let outcome = match target {
                Some(target_sec) => {
                    if let Some(w) = watchdog.as_ref() {
                        w.refresh();
                    }
                    let ts = seek_timestamp_us(target_sec, timeline_origin);
                    let lo = ts.saturating_sub(250_000);
                    let hi = ts.saturating_add(250_000);
                    if ictx.seek(ts, lo..hi).is_ok() {
                        info!("Seek to {:.2}s", target_sec);
                        SeekOutcome::Reset
                    } else {
                        warn!("Seek to {:.2}s failed", target_sec);
                        SeekOutcome::Failed
                    }
                }
                // 没有目标：不动读位置，也不让解码器复位
                None => SeekOutcome::Failed,
            };
            // 2) 发布处理结果：解码线程据此决定本地复位（Reset）还是保持现状（Failed）
            //    失败时不带目标：读位置没变，解码端不该按新目标重置主时钟
            let decided_target = if outcome == SeekOutcome::Reset {
                target
            } else {
                None
            };
            ctl.publish_decision(e, outcome, decided_target);
            // 唤醒暂停中的解码线程，让它们立即看到处理结果
            paused.wake();
            // 3) 等解码线程处理完（复位完成，或确认无需复位），再继续发包，
            //    保证 seek 后的新 packet 不会与旧 packet 混入解码管线
            ctl.wait_ready(e);
            // 继续 'source：重建 packet 迭代器（读位置已更新）
        }
        // 循环结束（EOF/错误）后 sender 随线程退出而 drop，
        // 解码线程的 recv 返回 Err，触发各自的 flush 收尾。
        info!("demux thread finished");
    })
}

// 单个媒体源的流信息（各解码线程初始化用；struct 化避免字段错位）
pub struct StreamInfo {
    pub video_index: Option<usize>,
    pub audio_index: Option<usize>,
    pub video_params: Option<codec::parameters::Parameters>,
    pub audio_params: Option<codec::parameters::Parameters>,
    pub video_time_base: Option<ffmpeg_next::Rational>,
    pub disp_w: u32, // SAR 修正后的显示宽度
    pub disp_h: u32,
    // 时间轴原点（秒）：各流最早的 start_time。TS/HLS 常见非 0（如 1.4s），
    // 视频 PTS 与 seek 都必须减去/加回它，否则起播、结尾与 seek 全部错位
    pub timeline_origin: f64,
}

// 取各流最早的 start_time 作为时间轴原点（秒）。
// 忽略 AV_NOPTS_VALUE、非法 time_base；没有可用值时返回 0（MP4 常见）。
pub fn timeline_origin(starts: &[(i64, ffmpeg_next::Rational)]) -> f64 {
    let mut origin = f64::INFINITY;
    for &(start, tb) in starts {
        if start == ffmpeg_next::ffi::AV_NOPTS_VALUE || tb.denominator() <= 0 {
            continue;
        }
        let sec = start as f64 * tb.numerator() as f64 / tb.denominator() as f64;
        if sec < origin {
            origin = sec;
        }
    }
    if origin.is_finite() { origin } else { 0.0 }
}

// UI 的 seek 目标是 0 基的"内容时间"，而 avformat_seek_file 要 AV_TIME_BASE
// 单位的绝对时间戳（实测 TS/HLS：seek(12s) 落在 12.167，而不是 start+12）。
// 这里统一把两者换算，避免两处各写一遍（也方便单测）。
pub fn seek_timestamp_us(target_sec: f64, origin_sec: f64) -> i64 {
    ((target_sec + origin_sec) * 1_000_000.0).round() as i64
}

// 直接从 AVCodecParameters 读宽高与 SAR，避免只为拿显示尺寸而打开一个临时
// 视频解码器（会建线程池、初始化编解码器，4K 源上开销可感知）。
// codecpar 宽高为 0 时返回 None，由调用方退回解码器探测。
fn codecpar_display_size(p: &codec::parameters::Parameters) -> Option<(u32, u32)> {
    unsafe {
        // SAFETY: Parameters 持有有效的 AVCodecParameters；as_ptr 只在本次调用
        // 内使用，且 width/height/sample_aspect_ratio 是只读标量字段。
        let par = &*p.as_ptr();
        if par.width <= 0 || par.height <= 0 {
            return None;
        }
        let sar = ffmpeg_next::Rational::from(par.sample_aspect_ratio);
        Some(crate::frame::sar_corrected(
            par.width as u32,
            par.height as u32,
            sar,
        ))
    }
}

// 从已打开的 demux 上下文提取视频/音频流信息，供各解码线程使用
pub fn extract_streams(ictx: &format::context::Input) -> Result<StreamInfo> {
    let video_stream = ictx.streams().best(media::Type::Video);
    let audio_stream = ictx.streams().best(media::Type::Audio);
    let video_params = video_stream.as_ref().map(|s| s.parameters().clone());
    let audio_params = audio_stream.as_ref().map(|s| s.parameters().clone());
    let starts: Vec<_> = [video_stream.as_ref(), audio_stream.as_ref()]
        .into_iter()
        .flatten()
        .map(|s| (s.start_time(), s.time_base()))
        .collect();
    let timeline_origin = timeline_origin(&starts);
    let (disp_w, disp_h) = match video_params.as_ref() {
        Some(p) => match codecpar_display_size(p) {
            Some(v) => v,
            None => {
                // 少数流在 codecpar 里的宽高为 0（要解码首帧才知道），
                // 这时才退回临时解码器，正常路径不再为拿宽高 open 一次
                let dec = codec::context::Context::from_parameters(p.clone())?
                    .decoder()
                    .video()?;
                crate::frame::sar_corrected(dec.width(), dec.height(), dec.aspect_ratio())
            }
        },
        None => return Err(anyhow::anyhow!("No video stream found")),
    };
    Ok(StreamInfo {
        video_index: video_stream.as_ref().map(|s| s.index()),
        audio_index: audio_stream.as_ref().map(|s| s.index()),
        video_params,
        audio_params,
        video_time_base: video_stream.as_ref().map(|s| s.time_base()),
        disp_w,
        disp_h,
        timeline_origin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // 基本流程：请求 → demux 发布处理结果 → 解码线程各报就绪
    #[test]
    fn request_publish_decision_report_ready() {
        let ctl = SeekCtl::new(2);
        assert!(ctl.snapshot_new(0).is_none());
        assert!(ctl.snapshot_decision(0).is_none(), "未发布结果时不应有决定");

        ctl.request(12.5);
        let (e, t) = ctl.snapshot_new(0).expect("应看到新代数");
        assert_eq!(e, 1);
        assert_eq!(t, Some(12.5));

        // demux 定位成功后发布结果，解码线程据此复位
        ctl.publish_decision(e, SeekOutcome::Reset, t);
        let decision = ctl.snapshot_decision(0).expect("应看到处理结果");
        assert_eq!(decision.epoch, e);
        assert_eq!(decision.outcome, SeekOutcome::Reset);
        assert_eq!(decision.target, Some(12.5));

        ctl.report_ready(e);
        ctl.report_ready(e);
        ctl.wait_ready(e); // 两个线程都就绪，应立即可通过
        assert!(
            ctl.snapshot_decision(e).is_none(),
            "已处理过的代数不应重复下发"
        );
    }

    // 定位失败：必须下发 Failed 且不带目标，解码线程据此保持现状（不 flush）
    #[test]
    fn failed_decision_keeps_target_empty() {
        let ctl = SeekCtl::new(1);
        ctl.request(30.0);
        let (e, _) = ctl.snapshot_new(0).unwrap();
        ctl.publish_decision(e, SeekOutcome::Failed, None);

        let decision = ctl.snapshot_decision(0).expect("应看到处理结果");
        assert_eq!(decision.outcome, SeekOutcome::Failed);
        assert_eq!(decision.target, None);
        // 失败也要回报就绪，否则 demux 会卡在屏障上
        ctl.report_ready(e);
        ctl.wait_ready(e);
    }

    // 过期决定被忽略：不能把旧代数的结果套到新请求上
    #[test]
    fn stale_decision_is_ignored() {
        let ctl = SeekCtl::new(1);
        ctl.request(1.0);
        let (e1, _) = ctl.snapshot_new(0).unwrap();
        ctl.request(2.0);
        let (e2, _) = ctl.snapshot_new(e1).unwrap();

        ctl.publish_decision(e2, SeekOutcome::Reset, Some(2.0));
        // 迟到的旧决定不能覆盖新决定
        ctl.publish_decision(e1, SeekOutcome::Failed, None);

        let decision = ctl.snapshot_decision(0).expect("应看到处理结果");
        assert_eq!(decision.epoch, e2);
        assert_eq!(decision.outcome, SeekOutcome::Reset);
        assert_eq!(decision.target, Some(2.0));
    }

    // 连续快速 seek：旧代数等待不能死锁，新代数的结果覆盖旧结果
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
        ctl.publish_decision(e2, SeekOutcome::Reset, Some(2.0));
        let decision = ctl.snapshot_decision(e1).expect("应看到处理结果");
        assert_eq!(decision.epoch, e2, "旧代数应直接看到最新决定");
    }

    // 无 seek 时轮询返回 None，不会误触发复位
    #[test]
    fn no_seek_snapshot_is_none() {
        let ctl = SeekCtl::new(1);
        assert!(ctl.snapshot_new(0).is_none());
        assert_eq!(ctl.epoch(), 0);
        assert!(ctl.snapshot_decision(0).is_none());
    }

    // 不可 seek 的源：标志可读可写，UI 据此禁用 seek
    #[test]
    fn seekable_flag_roundtrip() {
        let ctl = SeekCtl::new(1);
        assert!(ctl.seekable(), "默认可 seek");
        ctl.set_seekable(false);
        assert!(!ctl.seekable());
    }

    // 原子镜像与权威状态保持一致：epoch() 无锁可读且不落后于 state
    #[test]
    fn epoch_mirror_tracks_state() {
        let ctl = SeekCtl::new(1);
        assert_eq!(ctl.epoch(), 0);

        ctl.request(3.0);
        assert_eq!(ctl.epoch(), 1, "镜像应已发布新代数");
        let (e, _) = ctl.snapshot_new(0).unwrap();
        assert_eq!(e, ctl.epoch());

        ctl.request(4.0);
        assert_eq!(ctl.epoch(), 2);
        // 已处理到代数 2 后，快路径应报告"无新 seek"
        assert!(ctl.snapshot_new(2).is_none());
    }

    // 原子快路径 + 锁内复核：并发连续 seek 时读到的 (代数, 目标) 必须配对，
    // 不能出现"新代数配旧目标"（镜像先于 state 发布会破坏该不变量）
    #[test]
    fn snapshot_pairing_stays_consistent_under_concurrency() {
        const N: u64 = 20_000;
        let ctl = Arc::new(SeekCtl::new(1));
        let writer = {
            let ctl = ctl.clone();
            std::thread::spawn(move || {
                // 目标值恒等于其代数，便于校验配对
                for e in 1..=N {
                    ctl.request(e as f64);
                }
            })
        };

        let mut seen = 0u64;
        let mut observations = 0u64;
        while seen < N {
            match ctl.snapshot_new(seen) {
                None => {
                    // 快路径返回"无新 seek"：该时刻镜像与本地代数相等。
                    // 断言时可能又来了新 seek（下一轮轮询就会看到），故只校验不倒退
                    let now = ctl.epoch();
                    assert!(now >= seen, "代数倒退: {now} < {seen}");
                }
                Some((e, t)) => {
                    assert!(e > seen, "代数必须单调前进: {e} <= {seen}");
                    assert_eq!(t, Some(e as f64), "代数 {e} 配到了错位的目标 {t:?}");
                    seen = e;
                    observations += 1;
                }
            }
        }
        writer.join().unwrap();
        assert!(observations > 0, "应至少观察到一次新代数");
        assert_eq!(ctl.epoch(), N);
    }

    // 线程永久退出后，seek 屏障必须能放行，否则 demux 会永远等一个不再
    // report_ready 的线程（音频初始化失败 / 视频解码错误后的真实卡死路径）
    #[test]
    fn abandoned_decoder_unblocks_seek_barrier() {
        let ctl = SeekCtl::new(2);
        ctl.request(1.0);
        let (e, _) = ctl.snapshot_new(0).unwrap();
        ctl.report_ready(e); // 只剩一个线程就绪
        ctl.abandon_decoder(); // 另一个永久退出
        ctl.wait_ready(e); // 若未移除，这里会永久阻塞
    }

    // 时间轴原点取各流最早的 start_time；NOPTS / 非法 time_base 必须忽略。
    // 数值取自实测的 MPEG-TS：start_time 1.4s（126000 @ 1/90000）
    #[test]
    fn timeline_origin_uses_earliest_valid_start() {
        let tb = ffmpeg_next::Rational::new(1, 90_000);
        assert_eq!(timeline_origin(&[]), 0.0);
        assert!((timeline_origin(&[(126_000, tb)]) - 1.4).abs() < 1e-9);
        assert!(
            (timeline_origin(&[
                (1_026_000, tb),                        // 11.4s
                (1_024_051, tb),                        // 11.3768s（更早）
                (ffmpeg_next::ffi::AV_NOPTS_VALUE, tb), // 忽略
                (0, ffmpeg_next::Rational::new(1, 0)),  // 非法 time_base：忽略
            ]) - 1_024_051.0 / 90_000.0)
                .abs()
                < 1e-9
        );
    }

    // seek 的 0 基目标必须加回容器原点，才是 avformat_seek_file 要的绝对时间戳
    #[test]
    fn seek_timestamp_adds_origin() {
        assert_eq!(seek_timestamp_us(2.0, 0.0), 2_000_000);
        assert_eq!(seek_timestamp_us(2.0, 11.376_778), 13_376_778);
    }
}

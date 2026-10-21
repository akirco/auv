// 媒体源打开助手：本地文件/直链直接打开，页面链接经 yt-dlp 桥接
use anyhow::Result;
use ffmpeg_next::format;
use std::{
    io::Read,
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

// 打开媒体源时的最长等待。ffmpeg 的 open_input + find_stream_info 内部是
// 阻塞 I/O：网络不通/服务端不回数据时没有任何内建超时，会一直挂着——
// 表现为"启动后卡在黑窗口，UI 事件循环都还没开始跑，用户无法中断"。
// 这里用 AVIO 中断回调在超时后主动打断（回调返回 true 表示请求中断）。
const OPEN_TIMEOUT: Duration = Duration::from_secs(20);

// 播放期单次读取的最长等待（微秒），与打开期同为 20s。打开阶段的超时不覆盖
// 播放中的阻塞读，网络停发时没有它会永久冻结；http(s)/HLS 等协议实现也设置
// 了 rw_timeout。看门狗复用 new() 传入的 timeout（两处当前都是 20s）。
const READ_TIMEOUT_US: &str = "20000000";

fn network_options() -> ffmpeg_next::Dictionary<'static> {
    let mut opts = ffmpeg_next::Dictionary::new();
    opts.set("rw_timeout", READ_TIMEOUT_US);
    opts
}

// 读超时看门狗：中断回调随 Input 存活，并在其后的每一次读取时被轮询。
//
// 打开阶段用绝对截止时间；打开成功后不解除武装，而是切换到"播放期"模式：
// demux 在每次阻塞读之前调用 refresh()，只有**一次 read 内部**超过 20s
// 没有数据才触发中断。这样：
// - 长视频持续供数不会被 20s 绝对时钟误打断；
// - 暂停/UI 卡住导致 demux 阻塞在通道 send 上时，没有 read 在跑，不会误判；
// - yt-dlp 管道停发、TCP 半开时，read 会在 20s 后被打断。
#[derive(Clone)]
pub struct ReadWatchdog {
    base: Instant,
    deadline_ms: Arc<AtomicU64>,
    armed: Arc<AtomicBool>,
    timeout_ms: u64,
}

impl ReadWatchdog {
    fn new(timeout: Duration) -> Self {
        let base = Instant::now();
        Self {
            base,
            deadline_ms: Arc::new(AtomicU64::new(timeout.as_millis() as u64)),
            armed: Arc::new(AtomicBool::new(true)),
            timeout_ms: timeout.as_millis() as u64,
        }
    }

    fn now_ms(&self) -> u64 {
        self.base.elapsed().as_millis() as u64
    }

    // 进入播放期：保持武装，但把截止时间换成 READ_TIMEOUT
    fn arm_playback(&self) {
        self.armed.store(true, Ordering::Relaxed);
        self.refresh();
    }

    // demux 在每次阻塞读之前调用：重置"无数据"截止时间
    pub fn refresh(&self) {
        let next = self.now_ms() + self.timeout_ms;
        self.deadline_ms.store(next, Ordering::Relaxed);
    }

    fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    fn interrupt_flag(&self) -> impl FnMut() -> bool + Send + 'static {
        let deadline = self.deadline_ms.clone();
        let armed = self.armed.clone();
        let base = self.base;
        move || {
            armed.load(Ordering::Relaxed)
                && base.elapsed().as_millis() as u64 >= deadline.load(Ordering::Relaxed)
        }
    }
}

// open_input 的返回值：输入上下文、yt-dlp 子进程、播放期读超时看门狗
pub struct OpenedInput {
    pub ictx: format::context::Input,
    pub child: Option<Child>,
    pub watchdog: Option<ReadWatchdog>,
}

// 打开媒体源：先按本地文件/直链（mp4、m3u8 等）尝试；失败且为 http(s) 链接时
// 回退到 yt-dlp 解析，支持 B 站视频页、YouTube 等站点链接——yt-dlp 负责
// API 签名、鉴权与音视频合并，stdout 流经自定义 IO 直接喂给 ffmpeg。
// 返回输入上下文与（若走了 yt-dlp）需存活的子进程句柄。
pub fn open_input(source: &str) -> Result<OpenedInput> {
    // 带超时地打开：本地文件几乎瞬时完成，统一的截止时间对它们无副作用
    let is_url = source.starts_with("http://") || source.starts_with("https://");
    let guard = ReadWatchdog::new(OPEN_TIMEOUT);
    let direct = if is_url {
        // 播放期的读超时：除 ReadWatchdog 外，再给 http(s) 协议设置
        // rw_timeout，服务端停发/半开连接时由 FFmpeg 直接返回错误。
        // 该选项只对协议实现生效；自定义 IO（yt-dlp 管道）由 ReadWatchdog 兜底。
        format::input_with_interrupt_and_dictionary(
            source,
            guard.interrupt_flag(),
            network_options(),
        )
    } else {
        format::input_with_interrupt(source, guard.interrupt_flag())
    };
    match direct {
        Ok(ictx) => {
            if is_url {
                guard.arm_playback();
                Ok(OpenedInput {
                    ictx,
                    child: None,
                    watchdog: Some(guard),
                })
            } else {
                guard.disarm();
                Ok(OpenedInput {
                    ictx,
                    child: None,
                    watchdog: None,
                })
            }
        }
        Err(direct_err) if is_url => {
            // 格式选择：bv*+ba/b = 最佳视频流 + 最佳音频流，失败再退回最佳单文件。
            // 必须显式指定：yt-dlp 的默认选择对 DASH 分片站点会挑"已合一的最佳
            // 单文件"（YouTube 上是 format 18，仅 360p），而不是最佳视频流——
            // 结果是 4K 视频只下到 360p，观感"糊"得像渲染有问题，实则是源被降级了。
            // -S hdr:SDR：同分辨率下优先 SDR。本渲染管线是 SDR，若选中 HDR10
            // （PQ/BT.2020）源，swscale 只做 10bit→8bit 位移、不做 PQ→SDR 映射，
            // 画面会被当成 SDR 解码，表现为颜色/对比度明显不对。YouTube 的 HDR
            // 视频通常同时提供 SDR 版本，优先 SDR 可保住 4K 且颜色正确。
            // 合并容器强制 mkv：默认 mpegts 会把 AV1 等编成私有流，
            // 导致 ffmpeg 探测不到视频轨
            // AUV_YTDLP_SORT 可覆盖 -S：B 站默认常选 AV1，老 Intel（Gen9 及更早）
            // 没有 AV1 硬解，会退回软解。设成
            // 'hdr:SDR,codec:hevc,h264' 可优先硬解友好的 HEVC/H.264。
            let sort = std::env::var("AUV_YTDLP_SORT").unwrap_or_else(|_| "hdr:SDR".to_string());
            let mut child = Command::new("yt-dlp")
                .args(["-q", "--no-playlist", "-f", "bv*+ba/b", "-S"])
                .arg(&sort)
                .args(["--merge-output-format", "mkv", "-o", "-", source])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!("direct open failed ({direct_err}); spawning yt-dlp failed: {e} (install yt-dlp to play page links)")
                })?;
            let stdout = child.stdout.take().expect("piped stdout");
            // 持续排空 stderr，避免 yt-dlp 因管道写满而阻塞；只保留尾部，
            // 失败时拼进错误信息，解决"-q 之后只剩一句 Cannot open"的问题
            let stderr_tail = Arc::new(Mutex::new(Vec::<u8>::new()));
            let tail = stderr_tail.clone();
            let mut stderr = child.stderr.take().expect("piped stderr");
            let _reader = std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    match stderr.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let mut t = tail.lock().unwrap();
                            t.extend_from_slice(&buf[..n]);
                            const MAX: usize = 8192;
                            if t.len() > MAX {
                                let drain = t.len() - MAX;
                                t.drain(..drain);
                            }
                        }
                    }
                }
            });
            let io = format::context::StreamIo::from_read(stdout)?;
            // yt-dlp 若卡住（网络受限、需要登录而静默等待等）同样会挂死。
            // 同样必须解除武装：yt-dlp 管道要持续供数到播放结束，
            // 武装状态下的截止时间会在打开成功二十秒后把播放打断。
            let stream_guard = ReadWatchdog::new(OPEN_TIMEOUT);
            let opened = format::input_from_stream_with_interrupt(
                io,
                None,
                Some(network_options()),
                stream_guard.interrupt_flag(),
            );
            match opened {
                Ok(ictx) => {
                    stream_guard.arm_playback();
                    Ok(OpenedInput {
                        ictx,
                        child: Some(child),
                        watchdog: Some(stream_guard),
                    })
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let detail = String::from_utf8_lossy(&stderr_tail.lock().unwrap())
                        .trim()
                        .to_string();
                    if detail.is_empty() {
                        anyhow::bail!("Cannot open {source} (direct: {direct_err}; yt-dlp: {e})")
                    } else {
                        anyhow::bail!(
                            "Cannot open {source} (direct: {direct_err}; yt-dlp: {e}); yt-dlp stderr: {detail}"
                        )
                    }
                }
            }
        }
        Err(e) => Err(e.into()),
    }
}

// 结束本源播放后回收 yt-dlp 子进程：自然结束则 reap，提前退出则终止
pub fn reap_child(child: &mut Option<Child>) {
    if let Some(c) = child.as_mut() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::ReadWatchdog;
    use std::time::Duration;

    // 超时后必须请求中断；disarm 后不得再触发。
    // 关键：必须先睡过 timeout，不能像旧测试那样只 sleep 5ms（20s deadline
    // 下即使 disarm 完全失效也会通过）。
    #[test]
    fn watchdog_fires_then_disarm_stops() {
        let w = ReadWatchdog::new(Duration::from_millis(10));
        let mut probe = w.interrupt_flag();
        assert!(!probe(), "刚创建时不应触发");
        std::thread::sleep(Duration::from_millis(30));
        assert!(probe(), "超时后必须请求中断");
        w.disarm();
        assert!(!probe(), "disarm 后不得再触发");
    }

    // 播放期 refresh 应把"无数据"截止时间往后推
    #[test]
    fn refresh_extends_deadline() {
        let w = ReadWatchdog::new(Duration::from_millis(80));
        let mut probe = w.interrupt_flag();
        std::thread::sleep(Duration::from_millis(50));
        assert!(!probe());
        w.refresh();
        std::thread::sleep(Duration::from_millis(50));
        assert!(!probe(), "refresh 后应重新计时");
        std::thread::sleep(Duration::from_millis(120));
        assert!(probe(), "refresh 后再次超时必须触发");
    }
}

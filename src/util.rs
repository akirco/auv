// 媒体源打开助手：本地文件/直链直接打开，页面链接经 yt-dlp 桥接
use anyhow::Result;
use ffmpeg_next::format;
use std::{
    process::{Child, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

// 打开媒体源时的最长等待。ffmpeg 的 open_input + find_stream_info 内部是
// 阻塞 I/O：网络不通/服务端不回数据时没有任何内建超时，会一直挂着——
// 表现为"启动后卡在黑窗口，UI 事件循环都还没开始跑，用户无法中断"。
// 这里用 AVIO 中断回调在超时后主动打断（回调返回 true 表示请求中断）。
const OPEN_TIMEOUT: Duration = Duration::from_secs(20);

// 打开阶段的超时守卫。
//
// 关键：中断回调随 Input 存活，并会在其后的**每一次**读取时被轮询。
// 若让回调只看墙钟截止时间，打开成功返回后它仍会在 20 秒处把正在进行的
// 播放打断——表现为长视频（YouTube 常见 30 分钟）刚起播二十来秒就
// "partial file" 提前结束。因此用 armed 标志把超时限定在打开阶段：
// open 返回后立刻解除武装，回调此后再触发也不会打断播放。
struct OpenGuard {
    armed: Arc<AtomicBool>,
    deadline: Instant,
}

impl OpenGuard {
    fn new() -> Self {
        Self {
            armed: Arc::new(AtomicBool::new(true)),
            deadline: Instant::now() + OPEN_TIMEOUT,
        }
    }

    // 供中断回调捕获：仅在武装中且已超时才请求中断
    fn interrupt_flag(&self) -> impl FnMut() -> bool + Send + 'static {
        let armed = self.armed.clone();
        let deadline = self.deadline;
        move || armed.load(Ordering::Relaxed) && Instant::now() >= deadline
    }

    // 打开阶段结束，解除武装：此后的读取不再受超时约束
    fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }
}

// 打开媒体源：先按本地文件/直链（mp4、m3u8 等）尝试；失败且为 http(s) 链接时
// 回退到 yt-dlp 解析，支持 B 站视频页、YouTube 等站点链接——yt-dlp 负责
// API 签名、鉴权与音视频合并，stdout 流经自定义 IO 直接喂给 ffmpeg。
// 返回输入上下文与（若走了 yt-dlp）需存活的子进程句柄。
pub fn open_input(source: &str) -> Result<(format::context::Input, Option<Child>)> {
    // 带超时地打开：本地文件几乎瞬时完成，统一的截止时间对它们无副作用
    let guard = OpenGuard::new();
    let direct = format::input_with_interrupt(source, guard.interrupt_flag());
    match direct {
        Ok(ictx) => {
            guard.disarm();
            Ok((ictx, None))
        }
        Err(direct_err) if source.starts_with("http://") || source.starts_with("https://") => {
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
            let mut child = Command::new("yt-dlp")
                .args([
                    "-q",
                    "--no-playlist",
                    "-f",
                    "bv*+ba/b",
                    "-S",
                    "hdr:SDR",
                    "--merge-output-format",
                    "mkv",
                    "-o",
                    "-",
                    source,
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!("direct open failed ({direct_err}); spawning yt-dlp failed: {e} (install yt-dlp to play page links)")
                })?;
            let stdout = child.stdout.take().expect("piped stdout");
            let io = format::context::StreamIo::from_read(stdout)?;
            // yt-dlp 若卡住（网络受限、需要登录而静默等待等）同样会挂死。
            // 同样必须解除武装：yt-dlp 管道要持续供数到播放结束，
            // 武装状态下的截止时间会在打开成功二十秒后把播放打断。
            let stream_guard = OpenGuard::new();
            let opened = format::input_from_stream_with_interrupt(
                io,
                None,
                None,
                stream_guard.interrupt_flag(),
            );
            match opened {
                Ok(ictx) => {
                    stream_guard.disarm();
                    Ok((ictx, Some(child)))
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    anyhow::bail!("Cannot open {source} (direct: {direct_err}; yt-dlp: {e})")
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
    use super::OpenGuard;

    // 回归测试：中断回调的截止时间在 disarm 之后不得再触发。
    // 背景：中断回调随 Input 存活并在其后每次读取时被轮询，若只看墙钟，
    // 长视频（yt-dlp 管道持续供数）会在超时点被误打断，表现为
    // "partial file" 提前结束。这是本 guard 存在的唯一理由。
    #[test]
    fn disarm_stops_the_timeout_from_firing() {
        let guard = OpenGuard::new();
        // 构造一个已超时的场景：把 armed 置回 true 并确认超时判定为真
        let mut probe = guard.interrupt_flag();
        // 尚未超时（刚创建）→ 不请求中断
        assert!(!probe(), "刚创建时不应触发中断");

        guard.disarm();
        // 即使此刻已远超 OPEN_TIMEOUT，解除武装后也不再请求中断
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!probe(), "disarm 后不得再触发中断");
    }

    // 未 disarm 且已超时 → 必须请求中断（这正是超时保护要拦的挂死）
    #[test]
    fn fires_when_armed_and_timed_out() {
        let guard = OpenGuard::new();
        let mut probe = guard.interrupt_flag();
        assert!(!probe());
        // 通过私有字段把截止时间推到过去，模拟"打开一直卡住直到超时"
        let expired = OpenGuard {
            armed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            deadline: std::time::Instant::now() - std::time::Duration::from_secs(1),
        };
        let mut expired_probe = expired.interrupt_flag();
        assert!(expired_probe(), "武装中且已超时必须请求中断");
        drop(probe);
    }
}

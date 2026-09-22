// 媒体源打开助手：本地文件/直链直接打开，页面链接经 yt-dlp 桥接
use anyhow::Result;
use ffmpeg_next::format;
use std::process::{Child, Command, Stdio};

// 打开媒体源：先按本地文件/直链（mp4、m3u8 等）尝试；失败且为 http(s) 链接时
// 回退到 yt-dlp 解析，支持 B 站视频页、YouTube 等站点链接——yt-dlp 负责
// API 签名、鉴权与音视频合并，stdout 流经自定义 IO 直接喂给 ffmpeg。
// 返回输入上下文与（若走了 yt-dlp）需存活的子进程句柄。
pub fn open_input(source: &str) -> Result<(format::context::Input, Option<Child>)> {
    match format::input(source) {
        Ok(ictx) => Ok((ictx, None)),
        Err(direct_err) if source.starts_with("http://") || source.starts_with("https://") => {
            // 合并容器强制 mkv：默认 mpegts 会把 AV1 等编成私有流，
            // 导致 ffmpeg 探测不到视频轨
            let mut child = Command::new("yt-dlp")
                .args([
                    "-q",
                    "--no-playlist",
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
            match format::input_from_stream(io, None, None) {
                Ok(ictx) => Ok((ictx, Some(child))),
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

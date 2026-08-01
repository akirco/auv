use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::format;
use std::sync::mpsc::SyncSender;

// 解复用线程：统一读取媒体 packet，按流类型分发给视频/音频解码线程。
// 单一 demuxer 避免视频、音频线程各自打开网络流（m3u8 等）重复下载，带宽减半。
// 通道有容量上限，消费跟不上时 send 会阻塞，自动把下载/解包节流到播放速度。
pub fn spawn_demux_thread(
    mut ictx: format::context::Input,
    video_index: Option<usize>,
    audio_index: Option<usize>,
    video_tx: SyncSender<Packet>,
    audio_tx: SyncSender<Packet>,
) {
    std::thread::spawn(move || {
        for (stream, packet) in ictx.packets() {
            let idx = stream.index();
            let result = if Some(idx) == video_index {
                video_tx.send(packet)
            } else if Some(idx) == audio_index {
                audio_tx.send(packet)
            } else {
                continue;
            };
            if result.is_err() {
                break; // 解码线程已退出，停止分发
            }
        }
        // 循环结束（EOF 或网络错误）后 sender 随线程退出而 drop，
        // 解码线程的 recv 返回 Err，触发各自的 flush 收尾。
    });
}

use anyhow::Result;
use ffmpeg_next::codec;
use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::channel_layout::ChannelLayout;
use ffmpeg_next::software::resampling;
use ringbuf::HeapProd;
use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::demux::SeekCtl;

// 音频解码线程：从 demux 通道收 packet 解码、重采样后喂入环形缓冲。
// 解码与视频分离，视频同步休眠不会再阻塞音频，避免缓冲下溢产生静音断裂。
// seek 时：flush 解码器、排空通道、由 cpal 回调清空环缓冲、重建重采样器丢弃
// 滞留样本、复位采样计数（主时钟基准），然后回报就绪；demux 待所有解码线程
// 就绪后才移动读位置。播完尾部后驻留响应 seek，直到 demux 通道断开。
#[allow(clippy::too_many_arguments)]
pub fn spawn_audio_thread(
    audio_rx: Receiver<Packet>,
    audio_parameters: codec::parameters::Parameters,
    mut producer: HeapProd<f32>,
    sample_rate: u32,
    channels: u32,
    audio_done: Arc<AtomicU64>,
    done_cond: Arc<(Mutex<()>, Condvar)>,
    paused: Arc<(Mutex<bool>, Condvar)>,
    audio_dead: Arc<AtomicU64>,
    ctl: Arc<SeekCtl>,
    samples_played: Arc<AtomicU64>,
    ring_clear: Arc<Mutex<bool>>,
) {
    std::thread::spawn(move || -> Result<()> {
        let mut decoder = codec::context::Context::from_parameters(audio_parameters)?
            .decoder()
            .audio()?;
        let src_rate = decoder.rate() as f64;
        let f32_packed = ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed);

        // 重采样输出布局必须与设备实际通道数一致，否则环形缓冲每时刻写入与
        // 消费的样本数不匹配，导致播放变速 + 持续下溢补静音。
        // 常见配置 1/2/4/6/8 声道直接映射，其余罕见配置回退立体声并告警。
        let out_layout = match channels {
            1 => ChannelLayout::MONO,
            4 => ChannelLayout::QUAD,
            6 => ChannelLayout::_5POINT1,
            8 => ChannelLayout::_7POINT1,
            _ => ChannelLayout::STEREO,
        };
        if out_layout.channels() != channels as i32 {
            eprintln!(
                "Audio: device has {} channels, no direct layout, falling back to stereo",
                channels
            );
        }
        // 重采样器参数固定，seek 时重建整个上下文以丢弃滞留的旧样本
        let in_format = decoder.format();
        let in_layout = decoder.channel_layout();
        let in_rate = decoder.rate();
        let build_resampler = || -> std::result::Result<resampling::Context, ffmpeg_next::Error> {
            resampling::Context::get(
                in_format,
                in_layout,
                in_rate,
                f32_packed,
                out_layout,
                sample_rate,
            )
        };
        let mut resampler = build_resampler()?;

        // 已处理的 seek 代数（Cell：暂停闭包与主循环都要读写，避免借用冲突）
        let local_epoch = std::cell::Cell::new(0u64);

        // 暂停时在条件变量上挂起，恢复时由键盘处理器 notify_all 唤醒，
        // 避免暂停期间继续解码预填缓冲或忙等空转。
        // 返回 true 表示被 seek 请求打断（seek 也会 notify 唤醒所有暂停线程），
        // 调用方应回到主循环顶部执行 seek 复位
        let wait_if_paused = |paused: &Arc<(Mutex<bool>, Condvar)>| -> bool {
            let (lock, cond) = &**paused;
            let mut guard = lock.lock().unwrap();
            while *guard {
                // 先查 seek 再等待：被 seek 打断时立即返回；
                // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
                if ctl.epoch() != local_epoch.get() {
                    return true;
                }
                let (g, _) = cond
                    .wait_timeout(guard, Duration::from_millis(50))
                    .unwrap();
                guard = g;
            }
            false
        };

        // resampler.run 只会为输出帧分配和输入等长的缓冲，
        // 重采样后样本数会变多(如 44.1k->48k)，超出的部分会滞留在重采样器内部。
        // 因此这里按输出比例预留空间，并按 nb_samples*channels 读取实际样本数。
        // 返回 false 表示音频输出已失效，调用方应中止解码。
        let push_samples = |resampled: &ffmpeg_next::util::frame::Audio,
                            producer: &mut HeapProd<f32>| -> bool {
            let n = resampled.samples();
            if n > 0 {
                let nch = resampled.channels() as usize;
                let bytes = resampled.data(0);
                let total = n * nch;
                let valid = &bytes[..total * std::mem::size_of::<f32>()];
                let samples: &[f32] =
                    unsafe { std::slice::from_raw_parts(valid.as_ptr() as *const f32, total) };
                // 批量推入环形缓冲，缓冲满时休眠等待，避免逐样本 push 的调用开销。
                // 音频失效标记只在需要睡眠（push 返回 0）时检查——成功推进是常态，
                // 避免每次迭代的热路径上做原子读；失效后环缓冲不会再排空，
                // push 必然返回 0，届时检查必然触发中止
                let mut offset = 0;
                while offset < samples.len() {
                    let pushed = producer.push_slice(&samples[offset..]);
                    if pushed == 0 {
                        if audio_dead.load(Ordering::Relaxed) != 0 {
                            return false; // 设备已失效，环缓冲不会再排空，放弃剩余数据
                        }
                        let is_paused = *paused.0.lock().unwrap();
                        if is_paused {
                            // seek 打断暂停：放弃本次推入，剩余样本随复位清除
                            if wait_if_paused(&paused) {
                                break;
                            }
                        } else {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                    } else {
                        offset += pushed;
                    }
                }
            }
            true
        };

        // 音频 seek 的本地复位（flush/排空/清环缓冲/重建重采样器/复位采样计数）。
        // decoder/resampler 作为参数传入，避免闭包长期借用它们与主循环冲突
        let handle_audio_seek = |
            epoch: u64,
            decoder: &mut codec::decoder::audio::Audio,
            resampler: &mut resampling::Context,
        | -> Result<()> {
            decoder.flush();
            while audio_rx.try_recv().is_ok() {}
            // 请求 cpal 回调清空环形缓冲（seek 前的旧音频不必播完）。
            // 回调运行正常时 ~10ms 内清空；暂停中回调不运行，则最多等 100ms
            // 后继续——标记保持置位，恢复播放后回调第一次运行仍会清空，
            // 因此旧样本绝不会被播出（新推入的少量样本可能一并被清，属可接受）
            *ring_clear.lock().unwrap() = true;
            for _ in 0..20 {
                if !*ring_clear.lock().unwrap() || audio_dead.load(Ordering::Relaxed) != 0 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            samples_played.store(0, Ordering::Relaxed);
            *resampler = build_resampler()?;
            ctl.report_ready(epoch);
            Ok(())
        };

        // 复用一个重采样输出帧：容量按需增长，避免每个解码帧都分配释放。
        // swr_convert_frame 会把 nb_samples 改写为实际写入样本数，
        // 所以每次调用前要重置回容量值。
        let mut resampled = ffmpeg_next::util::frame::Audio::empty();
        let mut resampled_cap = 0usize;

        let mut eof = false;
        let mut dead = false;
        while !eof && !dead {
            // 先响应 seek（暂停被唤醒后也会回到这里）：
            // 就地复位并回报就绪，demux 才移动读位置并发送新包
            if let Some((e, _t)) = ctl.snapshot_new(local_epoch.get()) {
                local_epoch.set(e);
                handle_audio_seek(e, &mut decoder, &mut resampler)?;
                continue;
            }

            if wait_if_paused(&paused) {
                continue; // 暂停被 seek 打断，回到顶部执行复位
            }
            if audio_dead.load(Ordering::Relaxed) != 0 {
                dead = true;
                break;
            }
            // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
            // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
            match audio_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(packet) => {
                    let _ = decoder.send_packet(&packet);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = decoder.send_eof();
                    eof = true;
                }
            }
            let mut decoded = ffmpeg_next::util::frame::Audio::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let out_cap =
                    (decoded.samples() as f64 * sample_rate as f64 / src_rate).ceil() as usize + 1;
                // 注意：ffmpeg-next 的 Audio::alloc 对已持有缓冲的帧会静默失败
                // （av_frame_get_buffer 返回 EINVAL 被忽略），必须丢弃旧帧换新帧，
                // 否则 nb_samples 与实际缓冲脱节导致越界写（堆损坏）
                if out_cap > resampled_cap {
                    let mut grown = ffmpeg_next::util::frame::Audio::empty();
                    unsafe { grown.alloc(f32_packed, out_cap, out_layout) };
                    resampled = grown;
                    resampled_cap = out_cap;
                }
                resampled.set_samples(resampled_cap);
                if resampler.run(&decoded, &mut resampled).is_ok()
                    && !push_samples(&resampled, &mut producer)
                {
                    dead = true;
                    break;
                }
            }
        }

        // 冲刷重采样器内部残留的尾部样本（设备已失效时跳过）
        loop {
            if dead {
                break;
            }
            if resampled_cap == 0 {
                unsafe { resampled.alloc(f32_packed, 8192, out_layout) };
                resampled_cap = 8192;
            }
            resampled.set_samples(resampled_cap);
            match resampler.flush(&mut resampled) {
                Ok(_) => {
                    if resampled.samples() == 0 || !push_samples(&resampled, &mut producer) {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // 等环形缓冲里的音频全部播完再标记结束，让视频线程等音频播完才退出；
        // 设备已失效时环缓冲永远不会排空，直接结束
        loop {
            if producer.occupied_len() == 0 || audio_dead.load(Ordering::Relaxed) != 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        audio_done.store(1, Ordering::Relaxed);
        done_cond.1.notify_all();

        // 驻留期：本源临近结束但 demux 可能还活着（用户 seek 会重新发包）。
        // 维持就绪屏障不悬挂、丢弃再发的包；demux 退出（通道断开）即收尾。
        // 注意：设备失效（audio_dead）时不能退出——否则 seek 的就绪屏障凑不齐
        // 会卡死 demux，这里继续驻留直至通道断开
        loop {
            if let Some((e, _t)) = ctl.snapshot_new(local_epoch.get()) {
                local_epoch.set(e);
                handle_audio_seek(e, &mut decoder, &mut resampler)?;
                continue;
            }
            match audio_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(_p) => {} // seek 后再发的包：音频已播完，直接丢弃
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    });
}
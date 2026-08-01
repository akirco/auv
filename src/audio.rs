use anyhow::Result;
use ffmpeg_next::codec;
use ffmpeg_next::codec::packet::Packet;
use ffmpeg_next::software::resampling;
use ringbuf::HeapProd;
use ringbuf::traits::{Observer, Producer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// 音频解码线程：从 demux 通道收 packet 解码、重采样后喂入环形缓冲。
// 解码与视频分离，视频同步休眠不会再阻塞音频，避免缓冲下溢产生静音断裂。
pub fn spawn_audio_thread(
    audio_rx: Receiver<Packet>,
    audio_parameters: codec::parameters::Parameters,
    mut producer: HeapProd<f32>,
    sample_rate: u32,
    audio_done: Arc<AtomicU64>,
    done_cond: Arc<(Mutex<()>, Condvar)>,
) {
    std::thread::spawn(move || -> Result<()> {
        let mut decoder = codec::context::Context::from_parameters(audio_parameters)?
            .decoder()
            .audio()?;
        let src_rate = decoder.rate() as f64;
        let mut resampler = resampling::Context::get(
            decoder.format(),
            decoder.channel_layout(),
            decoder.rate(),
            ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed),
            ffmpeg_next::channel_layout::ChannelLayout::STEREO,
            sample_rate,
        )?;

        // resampler.run 只会为输出帧分配和输入等长的缓冲，
        // 重采样后样本数会变多(如 44.1k->48k)，超出的部分会滞留在重采样器内部。
        // 因此这里按输出比例预留空间，并按 nb_samples*channels 读取实际样本数。
        let push_samples = |resampled: &ffmpeg_next::util::frame::Audio,
                            producer: &mut HeapProd<f32>| {
            let n = resampled.samples();
            if n > 0 {
                let channels = resampled.channels() as usize;
                let bytes = resampled.data(0);
                let total = n * channels;
                let valid = &bytes[..total * std::mem::size_of::<f32>()];
                let samples: &[f32] =
                    unsafe { std::slice::from_raw_parts(valid.as_ptr() as *const f32, total) };
                // 批量推入环形缓冲，缓冲满时休眠等待，避免逐样本 push 的调用开销
                let mut offset = 0;
                while offset < samples.len() {
                    let pushed = producer.push_slice(&samples[offset..]);
                    if pushed == 0 {
                        std::thread::sleep(Duration::from_millis(1));
                    } else {
                        offset += pushed;
                    }
                }
            }
        };

        let mut eof = false;
        while !eof {
            match audio_rx.recv() {
                Ok(packet) => {
                    let _ = decoder.send_packet(&packet);
                }
                Err(_) => {
                    let _ = decoder.send_eof();
                    eof = true;
                }
            }
            let mut decoded = ffmpeg_next::util::frame::Audio::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let out_cap =
                    (decoded.samples() as f64 * sample_rate as f64 / src_rate).ceil() as usize + 1;
                let mut resampled = ffmpeg_next::util::frame::Audio::empty();
                unsafe {
                    resampled.alloc(
                        ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                        out_cap,
                        ffmpeg_next::channel_layout::ChannelLayout::STEREO,
                    );
                }
                if resampler.run(&decoded, &mut resampled).is_ok() {
                    push_samples(&resampled, &mut producer);
                }
            }
        }

        // 冲刷重采样器内部残留的尾部样本
        //todo 判断 flush 返回的实际样本数，如果小于 8192，可以将其截断（resized）后再推送，或者直接推 8192 但只读取前 N 个样本（取决于 push_samples 的实现是否依赖 samples() 字段）。
        loop {
            let mut resampled = ffmpeg_next::util::frame::Audio::empty();
            unsafe {
                resampled.alloc(
                    ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed),
                    8192,
                    ffmpeg_next::channel_layout::ChannelLayout::STEREO,
                );
            }
            match resampler.flush(&mut resampled) {
                Ok(_) => {
                    if resampled.samples() == 0 {
                        break;
                    }
                    push_samples(&resampled, &mut producer);
                }
                Err(_) => break,
            }
        }

        // 等环形缓冲里的音频全部播完再标记结束，让视频线程等音频播完才退出
        while producer.occupied_len() > 0 {
            std::thread::sleep(Duration::from_millis(5));
        }
        audio_done.store(1, Ordering::Relaxed);
        done_cond.1.notify_all();
        Ok(())
    });
}

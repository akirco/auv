use anyhow::Result;
use cpal::traits::DeviceTrait;
use ffmpeg_next::{
    channel_layout::ChannelLayout, codec, codec::packet::Packet, software::resampling,
};
use log::warn;
use ringbuf::{
    HeapProd,
    traits::{Observer, Producer},
};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::Receiver,
    },
    time::Duration,
};

use crate::demux::SeekCtl;

// 音频解码线程：从 demux 通道收 packet 解码、重采样后喂入环形缓冲。
// 解码与视频分离，视频同步休眠不会再阻塞音频，避免缓冲下溢产生静音断裂。
// seek 时：flush 解码器、排空通道、由 cpal 回调清空环缓冲、重建重采样器丢弃
// 滞留样本、复位采样计数（主时钟基准），然后回报就绪；demux 待所有解码线程
// 就绪后才移动读位置。播完尾部后驻留响应 seek，直到 demux 通道断开。
pub fn spawn_audio_thread(ctx: AudioThreadCtx) {
    std::thread::spawn(move || -> Result<()> { AudioThread::new(ctx)?.run() });
}

// 音频线程的完整输入：由主循环一次性打包传入，
// 避免散参数顺序/个数错误（原为 12 个参数的 spawn）
pub struct AudioThreadCtx {
    pub audio_rx: Receiver<Packet>,
    pub audio_parameters: codec::parameters::Parameters,
    pub producer: HeapProd<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    pub audio_done: Arc<AtomicU64>,
    pub done_cond: Arc<(Mutex<()>, Condvar)>,
    pub paused: Arc<(Mutex<bool>, Condvar)>,
    pub audio_dead: Arc<AtomicU64>,
    pub ctl: Arc<SeekCtl>,
    pub samples_played: Arc<AtomicU64>,
    pub ring_clear: Arc<Mutex<bool>>,
}

// 音频解码线程的局部状态：原为 thread::spawn 内的一组闭包，
// 收成结构体后提炼为可独立调用的普通方法，便于阅读与复查。
// 字段与闭包捕获一一对应，职责不变。
struct AudioThread {
    decoder: codec::decoder::audio::Audio,
    resampler: resampling::Context,
    // 重采样输出布局与输入参数（seek 重建 resampler 时复用）
    f32_packed: ffmpeg_next::format::Sample,
    out_layout: ChannelLayout,
    in_format: ffmpeg_next::format::Sample,
    in_layout: ChannelLayout,
    in_rate: u32,
    src_rate: f64,
    sample_rate: u32,
    producer: HeapProd<f32>,
    audio_rx: Receiver<Packet>,
    audio_done: Arc<AtomicU64>,
    done_cond: Arc<(Mutex<()>, Condvar)>,
    paused: Arc<(Mutex<bool>, Condvar)>,
    audio_dead: Arc<AtomicU64>,
    ctl: Arc<SeekCtl>,
    samples_played: Arc<AtomicU64>,
    ring_clear: Arc<Mutex<bool>>,
    // 已处理的 seek 代数
    local_epoch: u64,
    // 复用一个重采样输出帧：容量按需增长，避免每个解码帧都分配释放。
    // swr_convert_frame 会把 nb_samples 改写为实际写入样本数，
    // 所以每次调用前要重置回容量值。
    resampled: ffmpeg_next::util::frame::Audio,
    resampled_cap: usize,
}

impl AudioThread {
    fn new(ctx: AudioThreadCtx) -> Result<Self> {
        let AudioThreadCtx {
            audio_rx,
            audio_parameters,
            producer,
            sample_rate,
            channels,
            audio_done,
            done_cond,
            paused,
            audio_dead,
            ctl,
            samples_played,
            ring_clear,
        } = ctx;
        let decoder = codec::context::Context::from_parameters(audio_parameters)?
            .decoder()
            .audio()?;
        let src_rate = decoder.rate() as f64;
        let f32_packed =
            ffmpeg_next::format::Sample::F32(ffmpeg_next::format::sample::Type::Packed);

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
            warn!(
                "Audio: device has {} channels, no direct layout, falling back to stereo",
                channels
            );
        }
        // 重采样器参数固定，seek 时重建整个上下文以丢弃滞留的旧样本
        let in_format = decoder.format();
        let in_layout = decoder.channel_layout();
        let in_rate = decoder.rate();
        let resampler = Self::build_resampler(
            in_format,
            in_layout,
            in_rate,
            f32_packed,
            out_layout,
            sample_rate,
        )?;

        Ok(Self {
            decoder,
            resampler,
            f32_packed,
            out_layout,
            in_format,
            in_layout,
            in_rate,
            src_rate,
            sample_rate,
            producer,
            audio_rx,
            audio_done,
            done_cond,
            paused,
            audio_dead,
            ctl,
            samples_played,
            ring_clear,
            local_epoch: 0,
            resampled: ffmpeg_next::util::frame::Audio::empty(),
            resampled_cap: 0,
        })
    }

    // 固定参数的 resampler 工厂：seek 复位时也要重建（丢弃滞留样本），
    // 因此从 new 与 handle_seek 两处共用，参数显式传入避免借用纠缠
    fn build_resampler(
        in_format: ffmpeg_next::format::Sample,
        in_layout: ChannelLayout,
        in_rate: u32,
        f32_packed: ffmpeg_next::format::Sample,
        out_layout: ChannelLayout,
        sample_rate: u32,
    ) -> std::result::Result<resampling::Context, ffmpeg_next::Error> {
        resampling::Context::get(
            in_format,
            in_layout,
            in_rate,
            f32_packed,
            out_layout,
            sample_rate,
        )
    }

    // 暂停时在条件变量上挂起，恢复时由键盘处理器 notify_all 唤醒，
    // 避免暂停期间继续解码预填缓冲或忙等空转。
    // 返回 true 表示被 seek 请求打断（seek 也会 notify 唤醒所有暂停线程），
    // 调用方应回到主循环顶部执行 seek 复位
    fn wait_if_paused(&self) -> bool {
        let (lock, cond) = &*self.paused;
        let mut guard = lock.lock().unwrap();
        while *guard {
            // 先查 seek 再等待：被 seek 打断时立即返回；
            // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
            if self.ctl.epoch() != self.local_epoch {
                return true;
            }
            let (g, _) = cond.wait_timeout(guard, Duration::from_millis(50)).unwrap();
            guard = g;
        }
        false
    }

    // resampler.run 只会为输出帧分配和输入等长的缓冲，
    // 重采样后样本数会变多(如 44.1k->48k)，超出的部分会滞留在重采样器内部。
    // 因此这里按输出比例预留空间，并按 nb_samples*channels 读取实际样本数。
    // 返回 false 表示音频输出已失效，调用方应中止解码。
    fn push_samples(&mut self) -> bool {
        let n = self.resampled.samples();
        if n > 0 {
            let nch = self.resampled.channels() as usize;
            let bytes = self.resampled.data(0);
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
                let pushed = self.producer.push_slice(&samples[offset..]);
                if pushed == 0 {
                    if self.audio_dead.load(Ordering::Relaxed) != 0 {
                        return false; // 设备已失效，环缓冲不会再排空，放弃剩余数据
                    }
                    let is_paused = *self.paused.0.lock().unwrap();
                    if is_paused {
                        // seek 打断暂停：放弃本次推入，剩余样本随复位清除
                        if self.wait_if_paused() {
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
    }

    // 音频 seek 的本地复位（flush/排空/清环缓冲/重建重采样器/复位采样计数）
    fn handle_seek(&mut self, epoch: u64) -> Result<()> {
        self.decoder.flush();
        while self.audio_rx.try_recv().is_ok() {}
        // 请求 cpal 回调清空环形缓冲（seek 前的旧音频不必播完）。
        // 回调运行正常时 ~10ms 内清空；暂停中回调不运行，则最多等 100ms
        // 后继续——标记保持置位，恢复播放后回调第一次运行仍会清空，
        // 因此旧样本绝不会被播出（新推入的少量样本可能一并被清，属可接受）
        *self.ring_clear.lock().unwrap() = true;
        for _ in 0..20 {
            if !*self.ring_clear.lock().unwrap() || self.audio_dead.load(Ordering::Relaxed) != 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.samples_played.store(0, Ordering::Relaxed);
        self.resampler = Self::build_resampler(
            self.in_format,
            self.in_layout,
            self.in_rate,
            self.f32_packed,
            self.out_layout,
            self.sample_rate,
        )?;
        self.ctl.report_ready(epoch);
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        let mut eof = false;
        let mut dead = false;
        while !eof && !dead {
            // 先响应 seek（暂停被唤醒后也会回到这里）：
            // 就地复位并回报就绪，demux 才移动读位置并发送新包
            if let Some((e, _t)) = self.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                self.handle_seek(e)?;
                continue;
            }

            if self.wait_if_paused() {
                continue; // 暂停被 seek 打断，回到顶部执行复位
            }
            if self.audio_dead.load(Ordering::Relaxed) != 0 {
                dead = true;
                break;
            }
            // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
            // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
            match self.audio_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(packet) => {
                    let _ = self.decoder.send_packet(&packet);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = self.decoder.send_eof();
                    eof = true;
                }
            }
            let mut decoded = ffmpeg_next::util::frame::Audio::empty();
            while self.decoder.receive_frame(&mut decoded).is_ok() {
                let out_cap = (decoded.samples() as f64 * self.sample_rate as f64 / self.src_rate)
                    .ceil() as usize
                    + 1;
                // 注意：ffmpeg-next 的 Audio::alloc 对已持有缓冲的帧会静默失败
                // （av_frame_get_buffer 返回 EINVAL 被忽略），必须丢弃旧帧换新帧，
                // 否则 nb_samples 与实际缓冲脱节导致越界写（堆损坏）
                if out_cap > self.resampled_cap {
                    let mut grown = ffmpeg_next::util::frame::Audio::empty();
                    unsafe { grown.alloc(self.f32_packed, out_cap, self.out_layout) };
                    self.resampled = grown;
                    self.resampled_cap = out_cap;
                }
                self.resampled.set_samples(self.resampled_cap);
                if self.resampler.run(&decoded, &mut self.resampled).is_ok() && !self.push_samples()
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
            if self.resampled_cap == 0 {
                unsafe { self.resampled.alloc(self.f32_packed, 8192, self.out_layout) };
                self.resampled_cap = 8192;
            }
            self.resampled.set_samples(self.resampled_cap);
            match self.resampler.flush(&mut self.resampled) {
                Ok(_) => {
                    if self.resampled.samples() == 0 || !self.push_samples() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // 等环形缓冲里的音频全部播完再标记结束，让视频线程等音频播完才退出；
        // 设备已失效时环缓冲永远不会排空，直接结束
        loop {
            if self.producer.occupied_len() == 0 || self.audio_dead.load(Ordering::Relaxed) != 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.audio_done.store(1, Ordering::Relaxed);
        self.done_cond.1.notify_all();

        // 驻留期：本源临近结束但 demux 可能还活着（用户 seek 会重新发包）。
        // 维持就绪屏障不悬挂、丢弃再发的包；demux 退出（通道断开）即收尾。
        // 注意：设备失效（audio_dead）时不能退出——否则 seek 的就绪屏障凑不齐
        // 会卡死 demux，这里继续驻留直至通道断开
        loop {
            if let Some((e, _t)) = self.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                self.handle_seek(e)?;
                continue;
            }
            match self.audio_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(_p) => {} // seek 后再发的包：音频已播完，直接丢弃
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        Ok(())
    }
}

// 挑选音频输出配置：放大 ALSA period（cpal 内部再按 2× 双缓冲），
// 给实时 worker 线程更多容错余量，降低偶发下溢（xrun/EIO）触发率。
// 采样格式取 F32（与环缓冲样本类型一致）、采样率/声道沿用设备默认；
// 优先指到 2048 帧（≈43ms@48kHz）的 period，超出设备支持范围时退而取上限。
// 解析失败时回退设备默认配置。
pub fn prepare_audio_config(device: &cpal::Device) -> Option<cpal::StreamConfig> {
    let default = device.default_output_config().ok()?;
    let rate = default.sample_rate();
    let ch = default.channels();
    // 找"F32 + 默认声道 + 覆盖默认采样率"的声明范围；找不到就用默认配置
    let ranges: Vec<_> = device.supported_output_configs().ok()?.collect();
    let sc = ranges
        .iter()
        .find(|r| {
            r.sample_format() == cpal::SampleFormat::F32
                && r.channels() == ch
                && r.min_sample_rate() <= rate
                && rate <= r.max_sample_rate()
        })
        .map(|r| r.with_sample_rate(rate))
        .unwrap_or(default);
    let mut cfg = sc.config();
    // 放大 period：优先 2048 帧，不超过设备支持的周期上限（各有 lower/higher bound）
    if let cpal::SupportedBufferSize::Range { min, max } = sc.buffer_size() {
        let want = 2048u32.clamp(*min, *max);
        if want != 2048 {
            warn!(
                "Audio: device period range {}-{} frames, using {}",
                min, max, want
            );
        }
        cfg.buffer_size = cpal::BufferSize::Fixed(want);
    }
    Some(cfg)
}

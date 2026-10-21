use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ffmpeg_next::{channel_layout::ChannelLayout, codec, software::resampling};
use log::{error, info, warn};
use ringbuf::{
    HeapProd, HeapRb,
    traits::{Consumer, Observer, Producer, Split},
};
use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc::Receiver,
    },
    time::Duration,
};

use crate::{
    demux::{SeekCtl, TimedPacket},
    source::{SourceShared, SourceStats},
};

// 音频输出设备与配置：进程内只探测一次，跨源复用。
// 具体配置策略（F32 / 默认采样率声道 / 放大缓冲粒度）见 prepare_audio_config。
pub struct AudioOutput {
    pub device: Option<cpal::Device>,
    pub sample_rate: u32,
    pub channels: u32,
    pub config: Option<cpal::StreamConfig>,
}

impl AudioOutput {
    pub fn detect() -> Self {
        let host = cpal::default_host();
        info!("Audio host: {}", host.id());
        let device = host.default_output_device();
        let (sample_rate, channels, config) = match device.as_ref().and_then(prepare_audio_config) {
            Some(cfg) => (cfg.sample_rate, cfg.channels as u32, Some(cfg)),
            None => (0, 0, None),
        };
        Self {
            device,
            sample_rate,
            channels,
            config,
        }
    }
}

// 环形缓冲“有空间”的通知：回调只 notify（不需要拿锁），生产者仅在环满时
// 带超时等待，避免原来每 1ms 轮询一次。
pub struct AudioSpace {
    lock: Mutex<()>,
    cond: Condvar,
}

impl AudioSpace {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            lock: Mutex::new(()),
            cond: Condvar::new(),
        })
    }
}

// 音频解码线程：从 demux 通道收 packet 解码、重采样后喂入环形缓冲。
// 解码与视频分离，视频同步休眠不会再阻塞音频，避免缓冲下溢产生静音断裂。
// seek 时：flush 解码器、排空通道、由 cpal 回调清空环缓冲、重建重采样器丢弃
// 滞留样本、复位采样计数（主时钟基准），然后回报就绪；demux 待所有解码线程
// 就绪后才移动读位置。播完尾部后驻留响应 seek，直到 demux 通道断开。
pub fn spawn_audio_thread(ctx: AudioThreadCtx) -> std::thread::JoinHandle<()> {
    // 退出守卫：无论正常结束、返回 Err 还是 panic 展开，Drop 都会执行：
    //  - 置 audio_done + notify，放行视频线程的 wait_audio_done（否则要干等 5 秒）；
    //  - 非正常收尾时置 audio_dead，让主时钟退回墙钟。否则采样计数不再推进，
    //    视频线程的同步逼近循环会永久空转（画面卡在首帧）
    struct ExitGuard {
        audio_done: Arc<AtomicBool>,
        audio_dead: Arc<AtomicBool>,
        done_cond: Arc<(Mutex<()>, Condvar)>,
        ctl: Arc<SeekCtl>,
        normal: bool,
    }
    impl Drop for ExitGuard {
        fn drop(&mut self) {
            if !self.normal {
                self.audio_dead.store(true, Ordering::Relaxed);
                // 本线程不会再 report_ready：把它从 seek 屏障移除，否则下一次
                // seek 时 demux 会永久等待（A3）
                self.ctl.abandon_decoder();
            }
            self.audio_done.store(true, Ordering::Relaxed);
            self.done_cond.1.notify_all();
        }
    }

    let (audio_done, done_cond, audio_dead, ctl) = (
        ctx.shared.audio_done.clone(),
        ctx.shared.done_cond.clone(),
        ctx.shared.audio_dead.clone(),
        ctx.shared.ctl.clone(),
    );
    std::thread::spawn(move || {
        let mut guard = ExitGuard {
            audio_done,
            audio_dead,
            done_cond,
            ctl,
            normal: false,
        };
        // 工作线程的错误不会自动上抛，join 句柄又被丢弃：必须在这里就地记录
        if let Err(e) = AudioThread::new(ctx).and_then(|mut t| t.run()) {
            error!("audio thread exited with error: {e:#}");
        } else {
            guard.normal = true;
        }
        // guard 在此 Drop：置位、唤醒
    })
}

// 音频线程的完整输入：由主循环一次性打包传入，
// 避免散参数顺序/个数错误（原为 12 个参数的 spawn）
pub struct AudioThreadCtx {
    pub audio_rx: Receiver<TimedPacket>,
    pub audio_parameters: codec::parameters::Parameters,
    pub producer: HeapProd<f32>,
    pub sample_rate: u32,
    pub channels: u32,
    pub shared: Arc<SourceShared>,
    pub samples_played: Arc<AtomicU64>,
    pub ring_clear: Arc<AtomicBool>,
    pub space: Arc<AudioSpace>,
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
    audio_rx: Receiver<TimedPacket>,
    shared: Arc<SourceShared>,
    samples_played: Arc<AtomicU64>,
    ring_clear: Arc<AtomicBool>,
    space: Arc<AudioSpace>,
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
            shared,
            samples_played,
            ring_clear,
            space,
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
            shared,
            samples_played,
            ring_clear,
            space,
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
        // 先查 seek 再等待：被 seek 打断时立即返回；
        // 轮询式检查也避免 seek 的 notify 恰好发生在进入等待前而丢失
        self.shared.paused.wait_while_paused(|| {
            self.shared.stop.load(Ordering::Relaxed) || self.shared.ctl.epoch() != self.local_epoch
        })
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
                    if self.shared.stop.load(Ordering::Relaxed) {
                        return false; // 本源已收尾：不再等环缓冲排空
                    }
                    if self.shared.audio_dead.load(Ordering::Relaxed) {
                        return false; // 设备已失效，环缓冲不会再排空，放弃剩余数据
                    }
                    let is_paused = self.shared.paused.is_paused();
                    if is_paused {
                        // seek 打断暂停：放弃本次推入，剩余样本随复位清除
                        if self.wait_if_paused() {
                            break;
                        }
                    } else {
                        // 环满：等回调腾出空间。回调只 notify、不持锁，
                        // 实时线程不会被这里的互斥锁阻塞；1ms 超时仅作兜底
                        let guard = self.space.lock.lock().unwrap_or_else(|e| e.into_inner());
                        if self.producer.occupied_len() >= samples.len() {
                            let _ = self
                                .space
                                .cond
                                .wait_timeout(guard, Duration::from_millis(1));
                        }
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
        self.ring_clear.store(true, Ordering::Release);
        // 回调运行正常时 ~10ms 内清空；暂停中回调不运行，直接跳过等待
        // （恢复后第一次回调仍会看到标记并清空），避免每次暂停 seek 固定等 100ms
        if !self.shared.paused.is_paused() {
            for _ in 0..20 {
                if !self.ring_clear.load(Ordering::Acquire)
                    || self.shared.audio_dead.load(Ordering::Relaxed)
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
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
        self.shared.ctl.report_ready(epoch);
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        let mut eof = false;
        let mut dead = false;
        // 复用的解码输出帧（见下方 receive_frame 处注释）
        let mut decoded = ffmpeg_next::util::frame::Audio::empty();
        while !eof && !dead {
            // 本源收尾：不再解码/等待，直接进入收尾路径
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            // 先响应 seek（暂停被唤醒后也会回到这里）：
            // 就地复位并回报就绪，demux 才移动读位置并发送新包
            if let Some((e, _t)) = self.shared.ctl.snapshot_new(self.local_epoch) {
                self.local_epoch = e;
                self.handle_seek(e)?;
                continue;
            }

            if self.wait_if_paused() {
                continue; // 暂停被 seek 打断，回到顶部执行复位
            }
            if self.shared.audio_dead.load(Ordering::Relaxed) {
                dead = true;
                break;
            }
            // 有界超时接收：留出每 10ms 检查一次 seek 的机会，避免 seek 时
            // demux 阻塞在就绪屏障上而本线程阻塞在 recv 造成的死锁
            match self.audio_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(tp) => {
                    // 与视频同理：丢弃复位后才送达的 seek 前旧包
                    if tp.epoch != self.local_epoch {
                        continue;
                    }
                    let _ = self.decoder.send_packet(&tp.packet);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = self.decoder.send_eof();
                    eof = true;
                }
            }
            // 复用的解码输出帧：receive_frame 内部先 unref 再重填，跨 packet
            // 复用安全（原先每个 packet 新建一个 AVFrame）。它只作为重采样器
            // 的输入，run 期间数据即被拷走，不跨迭代持有引用。
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

        // 冲刷重采样器内部残留的尾部样本（设备已失效/本源已收尾时跳过）
        loop {
            if dead || self.shared.stop.load(Ordering::Relaxed) {
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
            if self.producer.occupied_len() == 0
                || self.shared.audio_dead.load(Ordering::Relaxed)
                || self.shared.stop.load(Ordering::Relaxed)
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.shared.audio_done.store(true, Ordering::Relaxed);
        self.shared.done_cond.1.notify_all();

        // 驻留期：本源临近结束但 demux 可能还活着（用户 seek 会重新发包）。
        // 维持就绪屏障不悬挂、丢弃再发的包；demux 退出（通道断开）即收尾。
        // 注意：设备失效（audio_dead）时不能退出——否则 seek 的就绪屏障凑不齐
        // 会卡死 demux，这里继续驻留直至通道断开
        loop {
            if self.shared.stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some((e, _t)) = self.shared.ctl.snapshot_new(self.local_epoch) {
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

// 挑选音频输出配置：优先 1024 帧（≈21ms@48kHz）以降低起播/seek 延迟；
// 设备拒绝时由 session 依次回退 2048 帧、设备默认缓冲。
// 采样格式取 F32（与环缓冲样本类型一致）、采样率/声道沿用设备默认。
// `BufferSize::Fixed` 的含义随后端而变：
//   - 原生 PipeWire 后端：解释为节点 quantum/latency（node.latency=<n>/<rate>），
//     由 PipeWire 负责整体缓冲，不存在「固定 2×period」的小缓冲问题；
//   - ALSA 后端（--no-default-features）：是 period，cpal 再按 2× 双缓冲分配，
//     且其上报的周期范围未必可靠（见 build_output_stream），建流失败由调用方回退。
// 超出声明范围时退而取边界；解析失败时回退设备默认配置。
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
    // 优先 1024 帧，不超过设备支持的周期范围（各有 lower/higher bound）
    if let cpal::SupportedBufferSize::Range { min, max } = sc.buffer_size() {
        let want = 1024u32.clamp(*min, *max);
        if want != 1024 {
            warn!(
                "Audio: device period range {}-{} frames, using {}",
                min, max, want
            );
        }
        cfg.buffer_size = cpal::BufferSize::Fixed(want);
    }
    Some(cfg)
}

// 构建音频输出流所需的全部输入：收成一个结构体，避免 7 个散参数与
// `#[allow(clippy::too_many_arguments)]`
pub struct AudioOutParams<'a> {
    pub device: &'a cpal::Device,
    pub config: &'a cpal::StreamConfig,
    pub samples_played: &'a Arc<AtomicU64>,
    pub audio_dead: &'a Arc<AtomicBool>,
    pub volume: &'a Arc<AtomicU32>,
    pub mute: &'a Arc<AtomicBool>,
    pub ring_clear: &'a Arc<AtomicBool>,
    pub stats: &'a Arc<SourceStats>,
}

// 构建音频输出流并启动播放，返回流句柄与环形缓冲 producer。
// 回调里弹出样本、按共享音量/静音缩放、不足补静音，并把实际交付样本数计入主时钟。
// 固定缓冲粒度不被设备接受时返回 Err，由调用方依次回退 2048 帧/设备默认配置。
// （ALSA 后端经 PipeWire-ALSA 插件时，枚举阶段未锁采样率，上报范围可能很宽；
//  建流后 period 才被固定，Fixed(1024/2048) 仍可能被拒，故需要多级兜底。）
pub fn build_output_stream(
    p: AudioOutParams,
) -> Result<(cpal::Stream, HeapProd<f32>, Arc<AudioSpace>)> {
    let AudioOutParams {
        device,
        config,
        samples_played,
        audio_dead,
        volume,
        mute,
        ring_clear,
        stats,
    } = p;
    // 无锁环形缓冲容量（毫秒）。取 500ms 而非原先的 2s，主要是为了收尾延迟：
    // 播完后 video.rs::wait_audio_done 要等 occupied_len()==0，即把环里剩下的
    // 音频全部播完，2s 的环意味着每个源结尾都要多等 2 秒。
    // 500ms 仍远大于实时 worker 的调度抖动与一次 xrun 的恢复窗口，抗下溢够用。
    const RING_MILLIS: u64 = 500;
    let ring_samples =
        ((config.sample_rate as u64 * config.channels as u64 * RING_MILLIS) / 1000) as usize;
    let ring_buf = HeapRb::<f32>::new(ring_samples);
    let (producer, mut consumer) = ring_buf.split();
    let space = AudioSpace::new();
    let space_cb = space.clone();
    let samples_played_cb = samples_played.clone();
    let audio_dead_cb = audio_dead.clone();
    let volume_cb = volume.clone();
    let mute_cb = mute.clone();
    let ring_clear_cb = ring_clear.clone();
    let stats_cb = stats.clone();
    let stats_err = stats.clone();
    let stream = device.build_output_stream(
        *config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            // seek 后清空旧音频：置位由音频线程发起，这里实际清空
            if ring_clear_cb.swap(false, Ordering::AcqRel) {
                consumer.clear();
                space_cb.cond.notify_one();
            }
            // 批量弹出音频样本，比逐样本 pop 开销更小
            let filled = consumer.pop_slice(data);
            if filled > 0 {
                space_cb.cond.notify_one();
            }
            if filled < data.len() {
                stats_cb.underruns.fetch_add(1, Ordering::Relaxed);
            }
            // 音量/静音：按共享百分比缩放（静音系数 0）。
            // 系数为 1 时跳过循环，常见路径零开销
            let gain = if mute_cb.load(Ordering::Relaxed) {
                0.0
            } else {
                (volume_cb.load(Ordering::Relaxed) as f32 / 100.0).clamp(0.0, 2.0)
            };
            if gain != 1.0 {
                for sample in &mut data[..filled] {
                    *sample *= gain;
                }
            }
            for sample in &mut data[filled..] {
                *sample = 0.0; // 缓冲区空时输出静音，防止爆音
            }
            // 只把实际交付的样本计入主时钟：起播缓冲期时钟不空转，
            // 视频从 pts≈0 起同步；中途下溢时时钟暂停，视频等待
            // 而非丢帧，保持音画对齐（无声卡时退回系统时钟）
            samples_played_cb.fetch_add(filled as u64, Ordering::Relaxed);
        },
        move |err| {
            if err.kind() == cpal::ErrorKind::Xrun {
                stats_err.xruns.fetch_add(1, Ordering::Relaxed);
            }
            // 可恢复/非致命错误只告警，不置失效标记，否则启动阶段一次欠载
            // （xrun，cpal 会自行 prepare() 恢复）或默认设备切换就会永久静音
            match err.kind() {
                cpal::ErrorKind::Xrun
                | cpal::ErrorKind::DeviceChanged
                | cpal::ErrorKind::RealtimeDenied => {
                    warn!("Audio Stream (recoverable, {:?}): {}", err.kind(), err);
                    return;
                }
                _ => {}
            }
            // 其余错误（拔掉、驱动故障等）视为致命：置失效标记并只提示一次。
            // 主时钟随后回退系统时钟，音频线程中止，播放不卡死
            if !audio_dead_cb.swap(true, Ordering::Relaxed) {
                error!("Audio Stream Error: {}", err);
            }
        },
        None,
    )?;
    stream.play()?;
    Ok((stream, producer, space))
}

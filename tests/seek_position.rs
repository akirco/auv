//! seek 定位落点的回归测试。
//!
//! `avformat_seek_file` 的 min_ts/max_ts 不是"容差"，而是**关键帧搜索窗口**：
//! 给了有界下界（如 ts-250ms），FFmpeg 会向前找第一个 `pts >= min_ts` 的关键帧，
//! 定位点变成"下一个关键帧"而不是"上一个关键帧"——误差最大一整个 GOP；
//! 文件末尾附近向后已无关键帧时定位还会直接失败，seek 完全无效。
//!
//! 本测试自建一段固定 GOP 的素材（mpeg4 编码，1s 一个关键帧），
//! 验证定位落在 **<= 目标** 的最近关键帧上。
use std::path::{Path, PathBuf};

use ffmpeg_next::{
    Rational,
    codec::{Id, context::Context, encoder},
    format, frame,
    packet::Packet,
};

const W: u32 = 64;
const H: u32 = 64;
const FPS: i32 = 30;
/// 关键帧间隔（帧）：1s
const GOP: i32 = 30;
/// 总帧数：3s
const FRAMES: i32 = 90;

/// 生成固定 GOP 的测试素材。返回 false 表示本机没有可用的 mpeg4 编码器
/// （测试应跳过而不是失败）。
fn make_fixture(path: &Path) -> bool {
    let Some(codec) = encoder::find(Id::MPEG4) else {
        return false;
    };
    let ctx = Context::new_with_codec(codec);
    let mut enc = match ctx.encoder().video() {
        Ok(e) => e,
        Err(_) => return false,
    };
    enc.set_width(W);
    enc.set_height(H);
    enc.set_format(format::Pixel::YUV420P);
    enc.set_time_base(Rational(1, FPS));
    enc.set_frame_rate(Some(Rational(1, FPS)));
    // 关键帧间隔：默认 12 帧太小，测不出"跳到下一个关键帧"的误差
    unsafe {
        (*enc.as_mut_ptr()).gop_size = GOP;
    }
    let mut enc = match enc.open() {
        Ok(e) => e,
        Err(_) => return false,
    };
    // 编码器可能自选时间基（如 mpeg4 常用 1/15360），帧的 pts 要按它算
    let enc_tb = enc.time_base();

    let Ok(mut octx) = format::output(path) else {
        return false;
    };
    {
        let Ok(mut ost) = octx.add_stream(Id::MPEG4) else {
            return false;
        };
        ost.set_parameters(&enc);
        ost.set_time_base(Rational(1, FPS));
    }
    if octx.write_header().is_err() {
        return false;
    }
    // 复用器可能改写流的时间基（如 mp4 会按帧率重算 timescale）：
    // 写包前必须按**实际**流时间基换算 pts，否则整条时间线被压缩
    let ost_tb = octx
        .streams_mut()
        .next()
        .map(|s| s.time_base())
        .unwrap_or(Rational(1, FPS));

    // 每帧在编码器时间基下的 tick 数
    let ticks_per_sec = enc_tb.denominator() as i64 / enc_tb.numerator() as i64;
    let ticks_per_frame = (ticks_per_sec / FPS as i64).max(1);

    for i in 0..FRAMES {
        let mut f = frame::Video::new(format::Pixel::YUV420P, W, H);
        f.set_pts(Some(i as i64 * ticks_per_frame));
        // 平面内容任意填充：这里只关心关键帧位置，不关心画面
        for plane in 0..3 {
            let data = f.data_mut(plane);
            for b in data.iter_mut() {
                *b = 16;
            }
        }
        if enc.send_frame(&f).is_err() {
            return false;
        }
        let mut pkt = Packet::empty();
        while enc.receive_packet(&mut pkt).is_ok() {
            pkt.set_stream(0);
            pkt.rescale_ts(enc_tb, ost_tb);
            if pkt.write_interleaved(&mut octx).is_err() {
                return false;
            }
        }
    }
    if enc.send_eof().is_err() {
        return false;
    }
    let mut pkt = Packet::empty();
    while enc.receive_packet(&mut pkt).is_ok() {
        pkt.set_stream(0);
        pkt.rescale_ts(enc_tb, ost_tb);
        if pkt.write_interleaved(&mut octx).is_err() {
            return false;
        }
    }
    octx.write_trailer().is_ok()
}

/// seek 之后读到的第一个包的 pts（秒）
fn first_pts_after_seek(path: &Path, ts_us: i64, range: std::ops::Range<i64>) -> Option<f64> {
    let mut ictx = format::input(path).ok()?;
    ictx.seek(ts_us, range).ok()?;
    let mut iter = ictx.packets();
    for (stream, pkt) in iter.by_ref() {
        if stream.index() != 0 {
            continue;
        }
        let tb = stream.time_base();
        let pts = pkt.pts()?;
        return Some(pts as f64 * tb.numerator() as f64 / tb.denominator() as f64);
    }
    None
}

#[test]
fn seek_lands_on_the_keyframe_at_or_before_the_target() {
    ffmpeg_next::init().unwrap();
    let dir = std::env::temp_dir().join("auv-seek-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path: PathBuf = dir.join("gop1s.mp4");
    if !make_fixture(&path) {
        eprintln!("skip: no usable mpeg4 encoder");
        return;
    }

    // 关键帧在 0s / 1s / 2s；目标取各关键帧之间与末尾附近
    for target in [0.5f64, 1.5, 2.4, 2.9] {
        let ts_us = (target * 1_000_000.0) as i64;
        // 复现 demux.rs 使用的搜索范围
        let landed = first_pts_after_seek(&path, ts_us, auv::demux::seek_range_us(ts_us))
            .unwrap_or_else(|| panic!("seek to {target}s failed"));
        assert!(
            landed <= target + 0.001,
            "seek {target}s 落点 {landed}s 超前：必须落在 <= 目标的最近关键帧上"
        );
        // 也不应回退太多：最近关键帧在 1s 间隔内
        assert!(
            target - landed < 1.001,
            "seek {target}s 落点 {landed}s 落后过多"
        );
    }
    let _ = std::fs::remove_file(&path);
}

// 色彩矩阵的纯函数测试：不同空间/范围/分辨率回退
use auv::frame::color_matrix;
use ffmpeg_next::util::color::{Range, Space};

#[test]
fn bt709_limited_and_full_range() {
    let limited = color_matrix(Space::BT709, Range::MPEG, 1080);
    assert!((limited.y_off - 16.0 / 255.0).abs() < 1e-6);
    assert!((limited.y_gain - 255.0 / 219.0).abs() < 1e-6);
    assert!((limited.uv_gain - 255.0 / 224.0).abs() < 1e-6);
    assert!((limited.rv - 1.5748).abs() < 1e-6);

    let full = color_matrix(Space::BT709, Range::JPEG, 1080);
    assert_eq!(full.y_off, 0.0);
    assert_eq!(full.y_gain, 1.0);
    assert_eq!(full.uv_gain, 1.0);
}

#[test]
fn bt601_coefficients_and_unknown_fallback() {
    let bt601 = color_matrix(Space::SMPTE170M, Range::MPEG, 480);
    assert!((bt601.rv - 1.402).abs() < 1e-6);
    assert!((bt601.gu + 0.344136).abs() < 1e-6);

    // 未标注空间时按高度回退：HD+ 用 709，SD 用 601
    let hd = color_matrix(Space::Unspecified, Range::Unspecified, 1080);
    let sd = color_matrix(Space::Unspecified, Range::Unspecified, 480);
    assert!((hd.rv - 1.5748).abs() < 1e-6);
    assert!((sd.rv - 1.402).abs() < 1e-6);
}

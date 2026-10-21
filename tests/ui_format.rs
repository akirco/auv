use auv::ui::format_time;

#[test]
fn format_time_is_mm_ss_and_clamps_negative() {
    assert_eq!(format_time(0.0), "00:00");
    assert_eq!(format_time(5.9), "00:05");
    assert_eq!(format_time(65.0), "01:05");
    assert_eq!(format_time(3600.0), "60:00");
    assert_eq!(format_time(-3.0), "00:00");
}

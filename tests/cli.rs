// 集成测试：CLI 解析 + 真实二进制入口（--help/--version 不需要图形环境）
use auv::cli::{CliAction, parse_cli};

fn parse(args: &[&str]) -> CliAction {
    parse_cli(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
}

#[test]
fn help_and_version_win_over_sources() {
    assert!(matches!(parse(&["a.mp4", "-h", "b.mp4"]), CliAction::Help));
    assert!(matches!(parse(&["--version"]), CliAction::Version));
}

#[test]
fn double_dash_treats_options_as_sources() {
    match parse(&["--", "-p", "--help"]) {
        CliAction::Play(v) => assert_eq!(v, vec!["-p", "--help"]),
        _ => panic!("expected play"),
    }
}

#[test]
fn playlist_skips_comments_and_blank_lines() {
    let path = std::env::temp_dir().join(format!(
        "auv_cli_playlist_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::write(&path, "# comment\n\none.mp4\nhttps://example.com/x.m3u8\n").unwrap();
    match parse(&["-p", path.to_str().unwrap(), "tail.mp4"]) {
        CliAction::Play(v) => {
            assert_eq!(v, vec!["one.mp4", "https://example.com/x.m3u8", "tail.mp4"])
        }
        _ => panic!("expected play"),
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn missing_playlist_path_is_an_error() {
    let args = vec!["-p".to_string()];
    assert!(parse_cli(&args).is_err());
}

#[test]
fn binary_version_and_help_do_not_need_a_display() {
    let exe = env!("CARGO_BIN_EXE_auv");
    let out = std::process::Command::new(exe)
        .arg("--version")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).starts_with("auv "));

    let out = std::process::Command::new(exe)
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage: auv"));
}

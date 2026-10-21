use anyhow::Result;
use fltk::{app, prelude::*, window::GlWindow};

use auv::cli::{CliAction, USAGE, parse_cli};
use auv::player::{AudioOutput, Player};

fn main() -> Result<()> {
    // 日志分级（RUST_LOG 控制）：默认 info；worker 线程与主循环共用同一 logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // 统一的 panic 钩子：线程 panic 不再让进程裸崩溃（已去 panic=abort），
    // 打印现场与 backtrace 提示，便于定位工作线程问题
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        eprintln!("[auv panic] thread '{name}': {info}");
        if std::env::var("RUST_BACKTRACE")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            let bt = std::backtrace::Backtrace::capture();
            eprintln!("backtrace:\n{bt}");
        } else {
            eprintln!("(set RUST_BACKTRACE=1 for a backtrace)");
        }
    }));

    let args: Vec<String> = std::env::args().collect();
    let sources = match parse_cli(&args[1..])? {
        CliAction::Play(sources) => sources,
        CliAction::Help => {
            println!("{}", USAGE);
            return Ok(());
        }
        CliAction::Version => {
            println!("auv {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
    };
    if sources.is_empty() {
        eprintln!("{}", USAGE);
        return Err(anyhow::anyhow!("no media source given"));
    }

    ffmpeg_next::init()?;
    // 压掉 ffmpeg 的 INFO 日志（如 HLS 每个分片的 "Opening ... for reading"），
    // 只保留真正的错误输出
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Error);

    let app = app::App::default();
    let screen = app::screen_size();

    // 预创建窗口（尺寸在首个成功打开的源处确定后再调整并显示）
    let mut win = GlWindow::new(0, 0, 640, 480, "");
    win.set_mode(fltk::enums::Mode::Opengl3);
    // Wayland app_id / X11 class：Hyprland 等合成器可用它给视频窗口下
    // noblur/norounding/opaque 规则，减少合成开销
    win.set_xclass("auv");
    win.end();

    // 播放器负责后续所有源的打开、线程装配与 UI 消息泵
    log::debug!("detecting audio output");
    let audio = AudioOutput::detect();
    log::debug!("audio output ready");
    let mut player = Player::new(win, app, screen, audio);
    for (i, source) in sources.iter().enumerate() {
        // play 返回 false 表示用户关闭了窗口，终止播放列表；
        // next 用于在当前源播放期间提前打开下一个源
        let next = sources.get(i + 1).map(String::as_str);
        if !player.play(source, next) {
            break;
        }
    }
    player.finish()
}

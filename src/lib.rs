//! AUV 的内部库表面：把无 UI 依赖的模块（cli/clock/frame/demux/pause）
//! 与播放器各层暴露出来，便于集成测试与后续复用。二进制入口在 main.rs。

pub mod audio;
pub mod cli;
pub mod clock;
pub mod demux;
pub mod dmabuf;
pub mod egl;
pub mod frame;
pub mod pause;
pub mod player;
pub mod render;
pub mod screenshot;
pub mod session;
pub mod source;
pub mod ui;
pub mod util;
pub mod video;

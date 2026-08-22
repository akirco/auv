# AUV

AUV（A/V Sync Engine）是一个使用 Rust 编写的轻量桌面媒体播放器，基于 FLTK + OpenGL 提供窗口和视频渲染，使用 FFmpeg 解码媒体，使用 CPAL 输出音频。

## 功能

- 播放本地媒体文件和 HTTP/HTTPS 直链
- 支持通过 `yt-dlp` 播放 Bilibili、YouTube 等网站页面
- 支持一次传入多个媒体源，按命令行顺序连续播放
- 支持使用播放列表文件批量加载媒体源
- 独立的音频解码、视频解码和解复用线程
- 音视频时钟同步，视频帧使用有界通道传递，避免播放过程中无限积压
- 支持 YUV420P 及其他常见像素格式，并通过 OpenGL 渲染
- 根据视频尺寸自动调整窗口并保持宽高比
- 音频输出不可用时自动回退为无音频时钟模式

## 运行环境

建议使用 Linux、Rust stable 和支持 OpenGL 3 的图形环境。

构建前需要安装以下系统依赖：

- Rust toolchain（`rust` 包含 `cargo`）
- FFmpeg 开发库（头文件和链接库）
- ALSA 开发库
- `pkgconf`（提供 `pkg-config` 命令）
- FLTK 构建所需的 C/C++ 编译工具和 CMake
- Wayland 开发库（项目启用了 FLTK 的 Wayland 支持）

在 Arch Linux 上可以安装：

```bash
sudo pacman -S --needed base-devel cmake pkgconf ffmpeg alsa-lib wayland rust
```

要播放网页链接，还需要安装 `yt-dlp`，并确保它位于 `PATH` 中：

```bash
sudo pacman -S --needed yt-dlp
```

也可以按照 Arch 软件包仓库或 `yt-dlp` 官方文档安装较新的版本。

## 构建

```bash
cargo build --release
```

生成的可执行文件位于 `target/release/auv`。

开发时可以直接运行：

```bash
cargo run -- path/to/video.mp4
```

## 使用方法

播放一个或多个媒体源：

```bash
auv video.mp4 another.mkv
auv 'https://example.com/video.m3u8'
auv 'https://www.bilibili.com/video/...'
```

使用播放列表：

```bash
auv --playlist playlist.txt
```

播放列表每行一个文件路径或 URL。空行和以 `#` 开头的行会被忽略，也可以与普通参数混合使用：

```text
# morning playlist
/home/user/videos/intro.mp4
https://example.com/live.m3u8
```

完整命令格式：

```text
auv <file_or_url>... [-p|--playlist <list.txt>]
```

## 播放控制

| 按键 | 操作 |
| --- | --- |
| `Space` | 暂停或恢复播放 |
| `F` | 切换全屏 |
| 关闭窗口 | 退出播放 |

在 Hyprland 下，全屏切换通过 `hyprctl` 执行，因此需要确保 `hyprctl` 可用。

## 网页链接

AUV 会先尝试让 FFmpeg 直接打开 HTTP/HTTPS 地址。直接打开失败后，会自动调用：

```text
yt-dlp -q --no-playlist --merge-output-format mkv -o - <URL>
```

因此网页链接播放依赖 `yt-dlp` 对对应站点的支持。需要登录、Cookie 或浏览器鉴权时，请在 `~/.config/yt-dlp/config` 中配置 `yt-dlp` 选项，例如 `--cookies` 或 `--cookies-from-browser`。

## 注意事项

- 项目当前要求输入源包含视频流；只有音频的文件会被跳过。
- 音频默认重采样为立体声浮点格式，并输出到系统默认音频设备。
- 播放多个源时，窗口会根据当前媒体的分辨率重新调整大小。
- 网络流的可播放性和稳定性取决于 FFmpeg、网络环境以及 `yt-dlp` 对目标站点的支持。


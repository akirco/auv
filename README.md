# AUV

AUV（A/V Sync Engine）是一个使用 Rust 编写的轻量桌面媒体播放器，基于 FLTK + OpenGL 提供窗口和视频渲染，使用 FFmpeg 解码媒体，使用 CPAL 输出音频。

## 功能

- 播放本地媒体文件和 HTTP/HTTPS 直链
- 支持通过 `yt-dlp` 播放 Bilibili、YouTube 等网站页面
- 支持一次传入多个媒体源，按命令行顺序连续播放（临近结尾时提前打开下一个源，缩短切源等待）
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
- OpenGL/GLU/EGL 开发库（`libgl1-mesa-dev`、`libglu1-mesa-dev`、`libegl1-mesa-dev`，FLTK 的 Wayland 后端需要 `glu`）
- Cairo/Pango 开发库（`libcairo2-dev`、`libpango1.0-dev`，FLTK 的 Wayland 图形驱动需要）
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

`dev` profile 保留了 `opt-level = 1`（播放路径不至于慢到掩盖真实性能问题），
debuginfo 设为 `line-tables-only` 以缩短编译时间。

如需进一步优化本机性能（YUV→RGB 标量循环、音频重采样可吃到 SIMD），
可在项目下创建 `.cargo/config.toml`：

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

注意这会让产物绑定当前 CPU，无法跨机分发。

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
auv [options] <file_or_url>...
  -h, --help               显示帮助并退出
  -V, --version            打印版本并退出
  -p, --playlist <file>    从播放列表读取源（每行一个，忽略空行与 # 注释）
  --                       之后的参数一律视为文件（允许文件名以 - 开头）

日志分级：设置环境变量 RUST_LOG=error|warn|info|debug 控制输出（默认 info）。
```

## 播放控制

| 按键 | 操作 |
| --- | --- |
| `Space` | 暂停或恢复播放 |
| `F` | 切换全屏 |
| `S` | 将当前帧保存为 PNG 截图（保存到当前工作目录，文件名 `auv_<时间戳>.<毫秒>.png`；编码在后台线程进行，不阻塞播放） |
| `←` / `→` | 后退 / 前进 10 秒 |
| `[` / `]` | 后退 / 前进 5 秒 |
| `Home` | 跳到开头 |
| `+` / `=` | 音量 +10%（上限 200%） |
| `-` / `_` | 音量 -10% |
| `M` | 切换静音 |
| 关闭窗口 | 退出播放 |

进度显示：当前时间/总时长（mm:ss）以叠层显示在窗口右上角，每秒刷新；窗口标题只保留媒体名。seek 相关功能要求媒体源可定位（本地文件与多数可缓存网络流支持；纯直播流可能不可 seek）。

在 Hyprland 下，全屏切换通过 `hyprctl` 执行，因此需要确保 `hyprctl` 可用。

### 窗口尺寸与平铺

在 Wayland（Hyprland / Sway 等）下，窗口尺寸由**合成器**决定：AUV 不再把视频分辨率
作为窗口尺寸约束交给合成器（FLTK 默认会把窗口尺寸同时作为 min/max 上报，
固定尺寸的窗口合成器无法平铺），视频按比例居中显示，窗口四周留黑边。
因此可以直接用合成器规则控制窗口，例如：

```lua
hl.window_rule({
	name = "auv-video",
	match = { class = "^(auv)$" },
	tile = true,
	no_blur = true,
	opaque = true,
})
```

在 X11 下没有合成器接管尺寸，窗口仍按视频分辨率（不超过屏幕）创建。

## 网页链接

AUV 会先尝试让 FFmpeg 直接打开 HTTP/HTTPS 地址。直接打开失败后，会自动调用：

```text
yt-dlp -q --no-playlist -f 'bv*+ba/b' -S 'hdr:SDR' --merge-output-format mkv -o - <URL>
```

其中 `-f 'bv*+ba/b'` 表示"最佳视频流 + 最佳音频流，失败再退回最佳单文件"。
**这一步不能省**：yt-dlp 的默认格式选择对 DASH 分片站点（YouTube 等）会挑
"已合一的最佳单文件"而非最佳视频流，实测会把 4K 视频降级到 360p，
观感像是渲染发糊，实际是下载源被降级了。

`-S 'hdr:SDR'` 表示同分辨率下优先选 SDR 版本。AUV 的渲染管线是 SDR；选中
HDR10（PQ/BT.2020）源时若不处理，10bit→8bit 转换不会做 PQ→SDR 映射，画面会被
当成 SDR 解码，颜色与对比度会明显不对。YouTube/B 站的 HDR 视频一般同时提供
SDR 版本，优先 SDR 可保住 4K 且颜色正确（原生 SDR 通常也比 tone map 更准）。

B 站默认常选 **AV1**。老 Intel 核显（Gen9 及更早，如 UHD 620）没有 AV1
硬解，VAAPI 解码器会打开但首帧退回软解，日志显示
`hardware decode unavailable for this stream`，此时 `dmabuf=0`（没有零拷贝）。
可以用 `AUV_YTDLP_SORT` 覆盖默认排序，优先硬解友好的 HEVC/H.264：

```bash
AUV_YTDLP_SORT='hdr:SDR,codec:hevc,h264' auv 'https://www.bilibili.com/video/BV...'
```

支持 AV1 硬解的新显卡可以保持默认，或改成 `'hdr:SDR,codec:av1'`。

若某个视频**只有 HDR 版本**，AUV 会把 10bit 帧（YUV420P10LE）直接上传为
`GL_R16` 纹理，在片元着色器里完成 PQ/HDR10 或 HLG → 线性光 → BT.2020→BT.709
→ Hable tone map → gamma 的转换，CPU 不再跑 `zscale`/`tonemap` 滤镜图。
非 10bit 或 GPU 路径无法处理时会退回原来的 CPU `zscale` + `tonemap`（需要
FFmpeg 带 `zscale`/`tonemap`，缺失时退回普通转换并告警）。

调试/兜底可用 `AUV_CPU_TONEMAP=1` 强制走 CPU tone map；截图也使用同一套
CPU 公式，HDR 截图与画面观感一致（HLG 的 OOTF 使用了近似）。

因此网页链接播放依赖 `yt-dlp` 对对应站点的支持。需要登录、Cookie 或浏览器鉴权时，请在 `~/.config/yt-dlp/config` 中配置 `yt-dlp` 选项，例如 `--cookies` 或 `--cookies-from-browser`。

## 注意事项

- 项目当前要求输入源包含视频流；只有音频的文件会被跳过。
- 播放 HDR 源时，若为 PQ/HDR10 或 HLG，会优先在 GPU shader 里 tone map 成 SDR BT.709；设置 `AUV_CPU_TONEMAP=1` 可强制走 CPU 滤镜图。
- 音频默认重采样为立体声浮点格式，并输出到系统默认音频设备。
- 播放多个源时，窗口会根据当前媒体的分辨率重新调整大小。
- 网络流的可播放性和稳定性取决于 FFmpeg、网络环境以及 `yt-dlp` 对目标站点的支持。
- 打开媒体源有 20 秒超时（FFmpeg 的打开与探测是阻塞 I/O，网络不通时若无超时会一直挂死）。超时后该源被跳过，播放列表继续。
- 音频输出缓冲按设备支持范围放大到 2048 帧（≈43ms@48kHz），以减少偶发的 ALSA 下溢（`Audio Stream Error: ...I/O error`）；设备不支持时自动退而取上限或回退默认配置。音频输出仍偶发失效时，按设计降级为无声并用系统时钟同步，播放不中断。


## 开发与测试

```bash
cargo +nightly fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

`src/lib.rs` 暴露了 `cli/clock/frame/demux/pause` 等无 UI 依赖的模块，集成测试在 `tests/`；CI 配置见 `.github/workflows/ci.yml`（Ubuntu + FFmpeg/ALSA/PipeWire/Wayland 依赖）。

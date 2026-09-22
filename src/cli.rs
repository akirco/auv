// 命令行解析：参数 → 播放源列表 / 帮助 / 版本
use anyhow::Result;

// 命令行用法说明（--help / 无参数时打印）
pub const USAGE: &str = "\
Usage: auv [options] <file_or_url>...
  -h, --help               show this help and exit
  -V, --version            print version and exit
  -p, --playlist <file>    load sources from a playlist file (one per line, '#' comments)
  --                       treat all following arguments as file names (e.g. files starting with '-')

Page links (bilibili, youtube, ...) require yt-dlp installed;
yt-dlp options like --cookies go into ~/.config/yt-dlp/config";

// CLI 解析结果：播放给定源，或要求打印帮助/版本
pub enum CliAction {
    Play(Vec<String>),
    Help,
    Version,
}

// 解析命令行参数：
// - `-h/--help`、`-V/--version` 触发帮助/版本（出现即生效）；
// - `--` 之后的所有参数一律视为媒体源（允许文件名以 `-` 开头，如 `-p`）；
// - `-p/--playlist <file>` 读取播放列表，每行一个条目（忽略空行与 # 开头），
//   可混放本地文件与网络流。
pub fn parse_cli(args: &[String]) -> Result<CliAction> {
    let mut sources = Vec::new();
    let mut it = args.iter();
    let mut literal = false; // `--` 之后不再解析选项
    while let Some(arg) = it.next() {
        if !literal {
            match arg.as_str() {
                "-h" | "--help" => return Ok(CliAction::Help),
                "-V" | "--version" => return Ok(CliAction::Version),
                "--" => {
                    literal = true;
                    continue;
                }
                "-p" | "--playlist" => {
                    let path = it
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("{} requires a file path", arg))?;
                    let content = std::fs::read_to_string(path)?;
                    for line in content.lines() {
                        let line = line.trim();
                        if !line.is_empty() && !line.starts_with('#') {
                            sources.push(line.to_string());
                        }
                    }
                    continue;
                }
                _ => {}
            }
        }
        sources.push(arg.clone());
    }
    Ok(CliAction::Play(sources))
}

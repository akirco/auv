use std::{
    cell::{Cell, RefCell},
    ffi::CString,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use ffmpeg_next::util::color::{Primaries, Range, Space, TransferCharacteristic};
use fltk::{prelude::*, window::GlWindow};
use log::{error, warn};

use crate::{
    dmabuf::DmaBufFrame,
    egl::{self, EGLImage},
    frame::{YuvFrame, color_matrix},
};

// 上次设置 uniform 对应的色彩特征：空间、范围、高度、位深、传递函数、色域
type ColorKey = (
    Space,
    Range,
    i32,
    u8,
    TransferCharacteristic,
    Primaries,
    bool,
);

// 渲染状态：PBO 三缓冲 id（每帧一个缓冲，三个平面连续存放）与已分配容量、
// 纹理尺寸跟踪、当前缓冲索引、色彩 uniform 位置、每缓冲的异步上传同步对象
pub struct RenderState {
    // 视频路径的 program 与 VAO（draw_frame 每帧绑定：文本 OSD 会切换自己的）
    program: u32,
    vao: u32,
    pbo: [u32; 3],
    // 三个采样单元对应的纹理对象（DMABUF 导入时直接替换其存储）
    textures: [u32; 3],
    pbo_cap: Cell<[isize; 3]>,
    tex_size: Cell<(i32, i32, u8, bool)>,
    pbo_cur: Cell<usize>,
    cmat_loc: i32,
    y_params_loc: i32,
    uv_params_loc: i32,
    hdr_mode_loc: i32,
    hdr_primaries_loc: i32,
    semi_planar_loc: i32,
    last_gen: Cell<u64>, // 已上传的帧代号，窗口 damage 重绘时跳过重复上传
    // 上次 uniform 对应的色彩特征（空间/范围/高度）。不能用 frame_gen：
    // 它每帧都变，会让"只在色彩变化时重设"永远命中。同一流内该三元组恒定。
    cmat_key: Cell<Option<ColorKey>>,
    // 每帧缓冲一个 fence：记录"该缓冲上次 TexSubImage2D 的 GPU 完成信号"，
    // 回绕复用时先等待，从而允许 MAP_UNSYNCHRONIZED 立即映射而无需驱动隐式同步
    fences: [Cell<gl::types::GLsync>; 3],
    // 与 fences 对应：该槽位是否已丢失完成信号（FenceSync 创建失败或 sync 失效）。
    // 此时即便 fence 为 null 也不能按"空闲"放行 MAP_UNSYNCHRONIZED，必须退回
    // 驱动隐式同步，直到重新成功创建 fence
    fence_lost: Cell<[bool; 3]>,
    // EGL/DMABUF 导入器：仅 Wayland/EGL 上下文可用，否则为 None 走 PBO 路径
    egl: Option<egl::EglInterop>,
    // 已提交绘制、GPU 可能仍在采样的 DMA-BUF 帧。fence 触发前不能释放，
    // 否则硬解 surface 会被解码器复写，出现撕裂/花屏。
    dmabuf_retained: RefCell<Vec<RetainedDmaBuf>>,
    // 截图请求：DMA-BUF 帧没有 CPU 平面，改在绘制后用 glReadPixels 取回
    snap_request: Arc<AtomicBool>,
    // 时间 OSD：最小纯色 program + 独立 VAO/VBO；顶点只在文字/窗口尺寸变化时重建
    // 时间叠层的 program：编译失败时为 None（只影响 OSD，不影响视频）
    text_program: Option<u32>,
    text_vao: u32,
    text_vbo: u32,
    text_color_loc: i32,
    text_cache: RefCell<TextCache>,
    // 是否已经释放过 GL/EGL 资源：GL 对象必须在上下文销毁前删除，
    // 因此由 Player 在删窗口前显式调用 cleanup()；Drop 只做兜底
    cleaned: Cell<bool>,
}

// 已上传到 text_vbo 的 OSD 顶点：文字或窗口尺寸变化时才重建
struct TextCache {
    text: String,
    w: i32,
    h: i32,
    verts: usize,
}

// 保留区（已提交绘制、GPU 可能仍在采样的 DMA-BUF 帧）最多同时存在多少帧。
// 保留区满时不再导入新帧（见 draw_frame），而不是强制释放最老的一帧——
// 强制释放会把仍被 GPU 采样的 surface 提前还给解码器，造成撕裂/花屏。
const MAX_RETAINED_DMABUF: usize = 3;

// 一个正在被 GPU 采样的 DMA-BUF 帧、其 EGLImage 与完成信号。
// EGLImage 在当前帧绘制期间一直被纹理引用；fence 触发后销毁时，
// EGL 实现会按“仍被 client API 使用”延迟释放底层存储。
struct RetainedDmaBuf {
    frame: DmaBufFrame,
    images: Vec<EGLImage>,
    fence: gl::types::GLsync,
}

// 初始化 GL 上下文：编译着色器、创建 VAO/纹理/PBO 三缓冲。
// program、纹理绑定、VAO 都是进程生命周期内不变的状态，只设置一次，
// draw 时只切换采样单元并更新纹理数据。
///
/// # Safety
/// 调用方必须已让当前线程持有有效的 OpenGL 上下文（通常是窗口的
/// `make_current()` 之后），并在同一线程上使用返回的 `RenderState`。
/// 着色器编译/链接失败时返回 None：此时不能继续绘制（黑屏且无从排查）。
pub unsafe fn setup_opengl(snap_request: Arc<AtomicBool>) -> Option<RenderState> {
    unsafe {
        let shader_program = build_program(
            include_str!("shaders/basic.vert.glsl"),
            include_str!("shaders/basic.frag.glsl"),
        )?;

        gl::UseProgram(shader_program);
        let cmat_loc = uniform_loc(shader_program, "cmat");
        let y_params_loc = uniform_loc(shader_program, "yParams");
        let uv_params_loc = uniform_loc(shader_program, "uvParams");
        let hdr_mode_loc = uniform_loc(shader_program, "hdrMode");
        let hdr_primaries_loc = uniform_loc(shader_program, "hdrPrimaries");
        let semi_planar_loc = uniform_loc(shader_program, "semiPlanar");
        // 初始默认 full range，避免首帧前出现全黑/灰屏
        gl::Uniform2f(y_params_loc, 1.0, 0.0);
        gl::Uniform2f(uv_params_loc, 1.0, -0.5);
        gl::Uniform1i(hdr_mode_loc, 0);
        gl::Uniform1i(hdr_primaries_loc, 0);
        gl::Uniform1i(semi_planar_loc, 0);
        gl::Uniform1i(uniform_loc(shader_program, "texY"), 0);
        gl::Uniform1i(uniform_loc(shader_program, "texU"), 1);
        gl::Uniform1i(uniform_loc(shader_program, "texV"), 2);
        let vertices: [f32; 16] = [
            1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0,
        ];
        let indices: [u32; 6] = [0, 1, 3, 1, 2, 3];

        let (mut vao, mut vbo, mut ebo) = (0, 0, 0);
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);
        gl::GenBuffers(1, &mut ebo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * std::mem::size_of::<f32>()) as isize,
            vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );
        gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo);
        gl::BufferData(
            gl::ELEMENT_ARRAY_BUFFER,
            (indices.len() * std::mem::size_of::<u32>()) as isize,
            indices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            4 * std::mem::size_of::<f32>() as i32,
            std::ptr::null(),
        );
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(
            1,
            2,
            gl::FLOAT,
            gl::FALSE,
            4 * std::mem::size_of::<f32>() as i32,
            (2 * std::mem::size_of::<f32>()) as *const _,
        );
        gl::EnableVertexAttribArray(1);

        let mut textures = [0; 3];
        gl::GenTextures(3, textures.as_mut_ptr());
        for &tex in &textures {
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
        }
        // 解码器行对齐可能不是 4 字节，禁用像素打包对齐避免读取越界
        gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);

        // 一次性绑定：VAO + 三张纹理到采样单元 0/1/2
        gl::BindVertexArray(vao);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, textures[0]);
        gl::ActiveTexture(gl::TEXTURE1);
        gl::BindTexture(gl::TEXTURE_2D, textures[1]);
        gl::ActiveTexture(gl::TEXTURE2);
        gl::BindTexture(gl::TEXTURE_2D, textures[2]);

        // 时间 OSD：独立的最小 program + VAO/VBO（顶点为 NDC 坐标，纯色着色）。
        // 与视频共用上下文，但属性布局不同，必须是另一个 VAO。
        // OSD 着色器失败不影响视频：只是不画时间叠层
        let text_program = build_program(
            include_str!("shaders/overlay.vert.glsl"),
            include_str!("shaders/overlay.frag.glsl"),
        );
        let text_color_loc = text_program.map_or(-1, |p| uniform_loc(p, "uColor"));
        if let Some(p) = text_program {
            gl::UseProgram(p);
        }
        let mut text_vao = 0;
        let mut text_vbo = 0;
        gl::GenVertexArrays(1, &mut text_vao);
        gl::GenBuffers(1, &mut text_vbo);
        gl::BindVertexArray(text_vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, text_vbo);
        gl::VertexAttribPointer(
            0,
            2,
            gl::FLOAT,
            gl::FALSE,
            2 * std::mem::size_of::<f32>() as i32,
            std::ptr::null(),
        );
        gl::EnableVertexAttribArray(0);
        // 恢复视频路径的 program/VAO 绑定（draw_frame 每帧也会重新绑定）
        gl::UseProgram(shader_program);
        gl::BindVertexArray(vao);

        // PBO 三缓冲：每帧一个缓冲，Y/U/V 平面按 16B 对齐连续存放；轮转写入，
        // 隐藏 GPU 异步上传的等待。存储首次使用（或换分辨率）时按需分配。
        let mut pbo = [0u32; 3];
        gl::GenBuffers(3, pbo.as_mut_ptr());

        // EGL/DMABUF：仅当当前上下文是 EGL（Wayland）且驱动支持导入时启用
        let egl = egl::EglInterop::new();
        if egl.is_none() {
            // 窗口重建后新的上下文不支持导入时，必须让视频线程退回下载路径，
            // 否则它会继续送 DMA-BUF 帧而渲染端无法显示
            egl::disable("current GL context cannot import DMA-BUF");
        }

        let state = RenderState {
            program: shader_program,
            vao,
            pbo,
            textures,
            pbo_cap: Cell::new([0; 3]),
            tex_size: Cell::new((-1, -1, 0, false)),
            pbo_cur: Cell::new(0),
            cmat_loc,
            y_params_loc,
            uv_params_loc,
            hdr_mode_loc,
            hdr_primaries_loc,
            semi_planar_loc,
            last_gen: Cell::new(u64::MAX),
            cmat_key: Cell::new(None),
            fences: std::array::from_fn(|_| Cell::new(std::ptr::null())),
            fence_lost: Cell::new([false; 3]),
            egl,
            dmabuf_retained: RefCell::new(Vec::new()),
            snap_request,
            text_program,
            text_vao,
            text_vbo,
            text_color_loc,
            text_cache: RefCell::new(TextCache {
                text: String::new(),
                w: 0,
                h: 0,
                verts: 0,
            }),
            cleaned: Cell::new(false),
        };
        check_gl_error("setup_opengl");
        Some(state)
    }
}

// 绘制一帧：按视频显示宽高比（含 SAR 修正）居中留黑边。
// 纹理存储只在尺寸变化时重新分配；此后每帧把数据写进 PBO（CPU 快速拷贝），
// 再 TexSubImage2D 让 GPU 异步 DMA 到纹理，三缓冲 + fence 避免 CPU 阻塞等待上传。
pub fn draw_frame(w: &mut GlWindow, state: &RenderState, frame: &YuvFrame) {
    // cleanup() 之后 GL 对象已删除（源结束/换窗口），但队列里可能还有重绘：
    // 再拿已删除的 program/VAO/纹理解引用是未定义行为
    if state.cleaned.get() {
        return;
    }
    unsafe {
        // 文本 OSD 会切换 program/VAO，这里显式恢复视频路径的绑定
        gl::UseProgram(state.program);
        gl::BindVertexArray(state.vao);
        // HiDPI/Wayland 下 GL 视口以物理像素计，用 pixel_* 而非逻辑尺寸
        let win_w = w.pixel_w() as f32;
        let win_h = w.pixel_h() as f32;
        let vw = frame.disp_w as f32;
        let vh = frame.disp_h as f32;
        let (vp_w, vp_h) = if vw / vh > win_w / win_h {
            (win_w, win_w * vh / vw)
        } else {
            (win_h * vw / vh, win_h)
        };
        let vp_x = ((win_w - vp_w) / 2.0) as i32;
        let vp_y = ((win_h - vp_h) / 2.0) as i32;

        // 视频铺满窗口时两个三角形会覆盖每个像素，清屏是多余的；
        // 4K60 下这一笔 clear 是 ~2GB/s 的无谓写带宽（黑边场景仍然清）
        let fills_window =
            vp_x == 0 && vp_y == 0 && vp_w as i32 == win_w as i32 && vp_h as i32 == win_h as i32;
        if !fills_window {
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        gl::Viewport(vp_x, vp_y, vp_w as i32, vp_h as i32);

        // 按流的色彩空间/范围选择转换参数（含 full/limited range 归一），
        // 不再按分辨率猜测；未标注时保持 HD+ → BT.709 的回退。
        // 同一流内色彩信息恒定，按 (空间, 范围, 高度) 记住上次设置值，
        // 避免每帧重复计算/重设 uniform，也避免 FLTK 无关重绘时重复设置
        let cmat_key = (
            frame.color_space,
            frame.color_range,
            frame.height,
            frame.bit_depth,
            frame.transfer,
            frame.primaries,
            frame.is_semi_planar(),
        );
        if state.cmat_key.get() != Some(cmat_key) {
            let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
            gl::Uniform4f(state.cmat_loc, cm.rv, cm.gu, cm.gv, cm.bu);
            // 10bit 样本在 R16 里是 0..65535，先换算成 8bit 码值再套既有
            // range/矩阵；limited 10bit 的 64..940 正好对应 8bit 16..235。
            // P010/NV12 半平面的 10bit 样本是 MSB 对齐（code << 6），
            // 与 YUV420P10LE 的 LSB 对齐不同，除数要再乘 64。
            let sample_scale = if frame.bit_depth == 10 {
                let ten_bit = if frame.color_range == Range::JPEG {
                    1023.0
                } else {
                    1020.0
                };
                let divisor = if frame.is_semi_planar() {
                    ten_bit * 64.0
                } else {
                    ten_bit
                };
                65535.0 / divisor
            } else {
                1.0
            };
            // 把 sampleScale 与 range 合并成 raw*scale+bias，片元里每像素
            // 少一次乘/减；常量在流特征变化时才重算
            gl::Uniform2f(
                state.y_params_loc,
                sample_scale * cm.y_gain,
                -cm.y_off * cm.y_gain,
            );
            gl::Uniform2f(
                state.uv_params_loc,
                sample_scale * cm.uv_gain,
                -cm.uv_center * cm.uv_gain,
            );
            let hdr_mode = match frame.transfer {
                TransferCharacteristic::SMPTE2084 => 1,
                TransferCharacteristic::ARIB_STD_B67 => 2,
                _ => 0,
            };
            gl::Uniform1i(state.hdr_mode_loc, hdr_mode);
            gl::Uniform1i(
                state.hdr_primaries_loc,
                i32::from(frame.primaries == Primaries::BT2020),
            );
            gl::Uniform1i(state.semi_planar_loc, i32::from(frame.is_semi_planar()));
            state.cmat_key.set(Some(cmat_key));
        }

        // 只有新帧才重新分配纹理与上传数据；FLTK 在鼠标移动等无关事件
        // 触发窗口 damage 重绘时会再次调用本回调，重复整帧上传纯属浪费
        let fresh = frame.frame_gen != state.last_gen.get();
        let mut imported: Option<(DmaBufFrame, Vec<EGLImage>)> = None;

        if fresh {
            if let Some(dma) = frame.dmabuf() {
                // DMA-BUF 零拷贝：直接把各平面导入成纹理，不经过 PBO/客户端内存。
                // 导入失败时全局退回下载路径，下一帧会走下面的 CPU 上传分支。
                //
                // 保留区满时**不导入**：里面的帧 GPU 可能还在采样，强制释放会让
                // 解码器复用仍被读取的 surface（撕裂/花屏）。此时本帧跳过，
                // 屏幕保持上一帧，等 fence 触发后再导入下一帧。
                if state.dmabuf_slot_available() {
                    match state.import_dmabuf(dma) {
                        Some(ok) => {
                            state.last_gen.set(frame.frame_gen);
                            imported = Some(ok);
                        }
                        None => {
                            // 导入失败：import_dmabuf 内部已全局关闭零拷贝路径，
                            // 视频线程从下一帧起改走 CPU 上传。本帧作废。
                            state.last_gen.set(frame.frame_gen);
                        }
                    }
                }
                // 保留区满：跳过本帧，不推进 last_gen——下一帧到达时会再试
            } else {
                let cur = state.pbo_cur.get();

                // 三个平面必须同进同出：只有当本轮要用的 3 个 PBO 槽位都已空闲
                // （上一轮上传在 GPU 侧完成）才上传本帧。任一平面仍忙就整体跳过，
                // 纹理保持上一帧内容——否则会出现 Y 是新帧、U/V 是旧帧的色度错位。
                // 跳过时不更新 last_gen、不推进 pbo_cur，下次重绘会用同一帧重试。
                let readiness = slots_ready(state, cur);
                if !matches!(readiness, SlotReadiness::Busy) {
                    state.last_gen.set(frame.frame_gen);
                    // 只有确认三平面都空闲才敢用 MAP_UNSYNCHRONIZED；有 fence 失效
                    // 导致无法确认时退回驱动隐式同步（正确性优先于异步性）
                    let unsync = matches!(readiness, SlotReadiness::Ready);

                    let need_alloc = state.tex_size.get()
                        != (
                            frame.width,
                            frame.height,
                            frame.bit_depth,
                            frame.is_semi_planar(),
                        );

                    if need_alloc {
                        // 分配纹理存储（不传数据）；先解绑 PBO，避免 TexImage2D 从 PBO 读垃圾
                        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
                        gl::ActiveTexture(gl::TEXTURE0);
                        gl::TexImage2D(
                            gl::TEXTURE_2D,
                            0,
                            texture_internal_format(frame.bit_depth, 1),
                            frame.width,
                            frame.height,
                            0,
                            gl::RED,
                            texture_type(frame.bit_depth),
                            std::ptr::null(),
                        );
                        // 色度平面为 ceil(w/2)×ceil(h/2)（420 的分配方式），奇数尺寸不能整除
                        let semi = frame.is_semi_planar();
                        let uv_channels = if semi { 2 } else { 1 };
                        gl::ActiveTexture(gl::TEXTURE1);
                        gl::TexImage2D(
                            gl::TEXTURE_2D,
                            0,
                            texture_internal_format(frame.bit_depth, uv_channels),
                            chroma_dim(frame.width),
                            chroma_dim(frame.height),
                            0,
                            if semi { gl::RG } else { gl::RED },
                            texture_type(frame.bit_depth),
                            std::ptr::null(),
                        );
                        if !semi {
                            gl::ActiveTexture(gl::TEXTURE2);
                            gl::TexImage2D(
                                gl::TEXTURE_2D,
                                0,
                                texture_internal_format(frame.bit_depth, 1),
                                chroma_dim(frame.width),
                                chroma_dim(frame.height),
                                0,
                                gl::RED,
                                texture_type(frame.bit_depth),
                                std::ptr::null(),
                            );
                        }
                        state
                            .tex_size
                            .set((frame.width, frame.height, frame.bit_depth, semi));
                    }

                    // 三个平面写入同一个 PBO（对齐连续存放），只 map/unmap 一次
                    upload_frame(state, cur, frame, unsync);
                    state.pbo_cur.set((cur + 1) % 3);

                    gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
                    check_gl_error("upload_frame");
                } else {
                    // 槽位仍忙：纹理保持上一帧。必须主动再排一次重绘，否则只有
                    // 下一帧到来才会重试，这一帧就被静默丢掉了。重试本身受
                    // wait_fence 的 16ms 上限节流，最坏约 60 次/s
                    w.redraw();
                }
            }
        }

        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
        check_gl_error("draw_frame");

        // DMA-BUF 帧的 surface 要活到 GPU 采样完成：提交 fence 后保留，
        // 后续绘制时按 fence 状态回收（见 release_finished_dmabuf）
        if let Some((dma, images)) = imported {
            let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
            state.dmabuf_retained.borrow_mut().push(RetainedDmaBuf {
                frame: dma,
                images,
                fence,
            });
        }
        state.release_finished_dmabuf();

        // 截图：DMA-BUF 帧没有 CPU 平面，从当前 back buffer 读回视频区域
        state.capture_if_requested(vp_x, vp_y, vp_w as i32, vp_h as i32);
    }
}

// 右上角时间 OSD：文字画成一组 NDC 矩形（3×5 点阵），带半透明底衬。
// 放在视频之后、swap 之前；截图读回在 draw_frame 内部完成，因此不会把 OSD 拍进去。
pub fn draw_time_overlay(w: &GlWindow, state: &RenderState, text: &str) {
    if text.is_empty() || state.cleaned.get() {
        return;
    }
    let (win_w, win_h) = (w.pixel_w(), w.pixel_h());
    if win_w <= 0 || win_h <= 0 {
        return;
    }
    unsafe {
        let mut cache = state.text_cache.borrow_mut();
        // 文字每秒才变一次、窗口尺寸很少变：只有变化时才重建并上传顶点
        if cache.text != text || cache.w != win_w || cache.h != win_h {
            let verts = time_verts(text, win_w, win_h);
            gl::BindVertexArray(state.text_vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, state.text_vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (verts.len() * std::mem::size_of::<f32>()) as isize,
                verts.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
            cache.text.clear();
            cache.text.push_str(text);
            cache.w = win_w;
            cache.h = win_h;
            cache.verts = verts.len() / 2;
        }

        // 视口恢复成整个窗口（draw_frame 把它设成了视频区域）
        gl::Viewport(0, 0, win_w, win_h);
        let Some(text_program) = state.text_program else {
            return; // OSD 着色器编译失败：不画时间叠层，视频照常
        };
        gl::UseProgram(text_program);
        gl::BindVertexArray(state.text_vao);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        // 前 6 个顶点是背景，其余是文字点阵
        gl::Uniform4f(state.text_color_loc, 0.0, 0.0, 0.0, 0.55);
        gl::DrawArrays(gl::TRIANGLES, 0, 6);
        let glyphs = cache.verts.saturating_sub(6);
        if glyphs > 0 {
            gl::Uniform4f(state.text_color_loc, 1.0, 1.0, 1.0, 0.95);
            gl::DrawArrays(gl::TRIANGLES, 6, glyphs as i32);
        }
        gl::Disable(gl::BLEND);
    }
}

// 3×5 点阵字形：每行 3 bit，最高位在左。只覆盖时间显示需要的字符，
// 其余字符（空格等）只占位不画点
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        _ => return None,
    })
}

// 生成 OSD 顶点：先是背景矩形（6 顶点），再是每个点亮像素的矩形（各 6 顶点）。
// 坐标为 NDC；点阵像素边长随窗口高度缩放（1080p 下字形约 25px 高）
fn time_verts(text: &str, win_w: i32, win_h: i32) -> Vec<f32> {
    let chars = text.chars().count() as i32;
    // 字号随窗口高度缩放，同时保证整块 OSD（含底衬、边距）不超出窗口宽度
    let s = ((win_h as f32 / 200.0).round() as i32).clamp(2, 8);
    let s = s.min((win_w / (4 * chars + 1)).max(1));
    let margin = 3 * s;
    let advance = 4 * s;
    let width = (chars * advance - s).max(s);
    let x0 = win_w - margin - width;
    // 像素坐标系 y 向下：顶部就是 margin（glyph 从这里向下排 5 行点阵）
    let y0 = margin;
    let pad = 2 * s;
    let mut v = Vec::with_capacity(6 + text.len() * 15 * 6 * 2);
    rect(
        &mut v,
        x0 - pad,
        y0 - pad,
        x0 + width + pad,
        y0 + 5 * s + pad,
        win_w,
        win_h,
    );
    for (i, c) in text.chars().enumerate() {
        let Some(g) = glyph(c) else { continue };
        let gx = x0 + i as i32 * advance;
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if *bits & (0b100u8 >> col) != 0 {
                    let px = gx + col * s;
                    let py = y0 + row as i32 * s;
                    rect(&mut v, px, py, px + s, py + s, win_w, win_h);
                }
            }
        }
    }
    v
}

// 像素矩形（左上原点、y 向下）→ NDC 两个三角形
fn rect(v: &mut Vec<f32>, x0: i32, y0: i32, x1: i32, y1: i32, w: i32, h: i32) {
    let nx = |x: i32| x as f32 * 2.0 / w as f32 - 1.0;
    let ny = |y: i32| 1.0 - y as f32 * 2.0 / h as f32;
    let (l, r, t, b) = (nx(x0), nx(x1), ny(y0), ny(y1));
    v.extend_from_slice(&[l, t, r, t, l, b, r, t, r, b, l, b]);
}

impl RenderState {
    // 把一帧 DMA-BUF 的所有平面导入成纹理。全部成功才真正绑定；
    // 任一步失败则销毁已建 EGLImage 并让视频线程退回下载路径。
    fn import_dmabuf(&self, frame: &DmaBufFrame) -> Option<(DmaBufFrame, Vec<EGLImage>)> {
        let Some(egl) = self.egl.as_ref() else {
            // 没有 EGL 导入器却收到了 DMA-BUF 帧：必须全局关闭零拷贝路径。
            // 不关的话视频线程会一直产出这种帧、每帧都在这里被丢掉 → 画面永久冻结
            egl::disable("no EGL interop for DMA-BUF frame");
            return None;
        };
        // 先 retain：EGLImage 创建与绘制期间 surface 不能被解码器回收
        let Some(retained) = frame.retain() else {
            egl::disable("av_frame_ref(DRM_PRIME) failed");
            return None;
        };
        let mut images: Vec<(usize, EGLImage)> = Vec::with_capacity(3);
        for (i, plane) in frame.planes.iter().enumerate() {
            let Some(plane) = plane else { continue };
            let Some(image) = egl.create_image(plane) else {
                for (_, img) in images.drain(..) {
                    unsafe { egl.destroy_image(img) };
                }
                egl::disable("eglCreateImageKHR failed");
                return None;
            };
            images.push((i, image));
        }
        if images.is_empty() {
            egl::disable("DMA-BUF frame has no importable planes");
            return None;
        }
        for (i, image) in &images {
            // 平面序号同时对应纹理与采样单元（Y=0, U/UV=1, V=2）
            if !unsafe { egl.bind_image(*i as u32, self.textures[*i], *image) } {
                // 已经绑定到纹理的 EGLImage 不能立刻销毁（纹理仍引用其存储）；
                // 全局退回 CPU 路径后这些是最后一次，直接泄漏也比 UAF 安全
                warn_once_bind_failure();
                std::mem::forget(images);
                // 已绑定的纹理仍引用这些存储，surface 也不能立刻回收
                std::mem::forget(retained);
                // 部分纹理的存储已被 EGLImage 替换：必须让下一帧 CPU 路径重新
                // TexImage2D，否则会往 EGLImage 存储里 TexSubImage2D（GL 错误，
                // 画面持续错乱）。这条路径之后零拷贝已全局关闭，下一帧必然走 CPU
                self.tex_size.set((-1, -1, 0, false));
                egl::disable("glEGLImageTargetTexture2DOES failed");
                return None;
            }
        }
        // 纹理存储已由 EGLImage 定义；下次 CPU 帧必须重新 TexImage2D
        self.tex_size.set((-1, -1, 0, false));
        // bind_image 已把每个平面绑定到自己的采样单元（unit = 平面序号）；
        // 半平面只导入 unit 0/1，unit 2 从 setup 起就一直绑定 textures[2]，
        // 无需再整轮重绑
        Some((retained, images.into_iter().map(|(_, img)| img).collect()))
    }

    // 回收已完成采样的 DMA-BUF 帧（fence 触发后）。
    //
    // 两条硬约束：
    // 1) fence 没触发就不能回收——解码器会立刻复用仍被 GPU 采样的 surface，
    //    画面出现撕裂/花屏。堆积到上限时改为**不再导入新帧**（见 draw_frame），
    //    绝不强制释放最老的一帧。
    // 2) 列表最后一个（最近一次导入）不能回收：纹理的存储仍指向它的 EGLImage，
    //    提前销毁会让后续绘制采样已释放的存储（未定义行为）。它要等下一次
    //    导入替换纹理存储后才可回收。
    fn release_finished_dmabuf(&self) {
        let mut retained = self.dmabuf_retained.borrow_mut();
        let mut i = 0;
        // 条件里每轮重新取 len：remove(i) 会让后面的元素前移。用固定的
        // end 会越界（i >= len → panic），或把列表最后一个误当成普通条目
        // 释放——而最后一个仍被纹理引用，销毁其 EGLImage 就是 UAF
        while i + 1 < retained.len() {
            let fence = retained[i].fence;
            let done = if fence.is_null() {
                true
            } else {
                let s = unsafe { gl::ClientWaitSync(fence, 0, 0) };
                match s {
                    gl::ALREADY_SIGNALED | gl::CONDITION_SATISFIED => true,
                    gl::TIMEOUT_EXPIRED => false,
                    // WAIT_FAILED 表示 sync 已失效（上下文丢失/对象无效），
                    // 无法再确认完成状态。若一直按"未完成"处理，保留区会永久
                    // 占满、不再导入新帧 → 画面永久冻结。因此按已完成处理
                    // （与 wait_fence 的 Invalid 分支一致），代价是极小概率撕裂
                    _ => {
                        warn_once_invalid_fence();
                        true
                    }
                }
            };
            if done {
                let r = retained.remove(i);
                self.free_retained(r);
            } else {
                i += 1;
            }
        }
    }

    // 保留区是否还有空槽：没有的话不能再导入新帧
    fn dmabuf_slot_available(&self) -> bool {
        self.dmabuf_retained.borrow().len() < MAX_RETAINED_DMABUF
    }

    fn free_retained(&self, r: RetainedDmaBuf) {
        if let Some(egl) = &self.egl {
            for img in &r.images {
                unsafe { egl.destroy_image(*img) };
            }
        }
        if !r.fence.is_null() {
            unsafe { gl::DeleteSync(r.fence) };
        }
        drop(r.frame);
    }

    // 截图请求：在 swap 之前读回刚绘制完的 back buffer（视频视口区域）。
    // 仅 DMA-BUF 帧使用；CPU 帧仍在 UI 线程做 YUV 快照（分辨率更高）。
    fn capture_if_requested(&self, x: i32, y: i32, w: i32, h: i32) {
        if !self.snap_request.swap(false, Ordering::Relaxed) {
            return;
        }
        if w <= 0 || h <= 0 {
            warn!("Screenshot skipped: invalid viewport {w}x{h}");
            return;
        }
        let mut rgb = vec![0u8; w as usize * h as usize * 3];
        unsafe {
            gl::ReadBuffer(gl::BACK);
            gl::PixelStorei(gl::PACK_ALIGNMENT, 1);
            gl::ReadPixels(
                x,
                y,
                w,
                h,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                rgb.as_mut_ptr() as *mut _,
            );
        }
        // glReadPixels 没有返回值，失败只体现在 GL 错误队列里。
        // 不检查就写文件 → 读回失败时会存出一张全黑 PNG，还提示"已保存"
        let err = unsafe { gl::GetError() };
        if err != gl::NO_ERROR {
            error!("glReadPixels failed (0x{err:x}); screenshot discarded");
            return;
        }
        crate::screenshot::spawn_save_rgb(w, h, rgb);
    }

    // 在 GL 上下文仍然有效时释放全部 GL/EGL 资源。
    //
    // 必须在窗口（连同 GL 上下文）销毁之前调用：上下文没了之后再调
    // glDelete*/eglDestroyImageKHR 是未定义行为（驱动可能已卸载该上下文）。
    // 幂等：重复调用只生效一次。
    pub fn cleanup(&self) {
        if self.cleaned.get() {
            return;
        }
        self.cleaned.set(true);
        unsafe {
            // fence 属于各自的 GL 上下文，先删
            for f in self.fences.iter() {
                let f = f.get();
                if !f.is_null() {
                    gl::DeleteSync(f);
                }
            }
            // 纹理先删：删掉之后 EGLImage 的存储才不再被引用
            gl::DeleteTextures(3, self.textures.as_ptr());
            gl::DeleteBuffers(3, self.pbo.as_ptr());
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteVertexArrays(1, &self.text_vao);
            gl::DeleteBuffers(1, &self.text_vbo);
            if let Some(p) = self.text_program {
                gl::DeleteProgram(p);
            }
            gl::DeleteProgram(self.program);
            // EGLImage 最后销毁（此时纹理已不再引用其存储），
            // 同时把保留帧交还解码器（DRM fd 随之释放）
            if let Some(egl) = &self.egl {
                for r in self.dmabuf_retained.borrow_mut().drain(..) {
                    for img in &r.images {
                        egl.destroy_image(*img);
                    }
                    if !r.fence.is_null() {
                        gl::DeleteSync(r.fence);
                    }
                    drop(r.frame);
                }
            } else {
                self.dmabuf_retained.borrow_mut().clear();
            }
        }
    }

    // 只释放不依赖 GL 上下文的部分（保留帧 → DRM fd 交还解码器）。
    //
    // 窗口已被用户关闭时用这个：此时上下文无法再激活（FLTK 的 make_current
    // 对未显示的窗口会弹警告框并终止进程），调任何 GL/EGL 函数都没有意义。
    // GL 对象随上下文销毁由驱动回收，EGLImage 走进程退出时的 EGL 清理。
    pub fn release_without_context(&self) {
        self.dmabuf_retained.borrow_mut().clear();
    }
}

impl Drop for RenderState {
    fn drop(&mut self) {
        if self.cleaned.get() {
            return; // 已由 cleanup() 释放（上下文销毁前）
        }
        // 兜底路径：没有显式清理就走到这里，说明 GL 上下文可能已经没了。
        // 此时调 GL/EGL 是未定义行为，只能放弃——GL 对象会随上下文销毁
        // 由驱动回收，DRM fd 则随保留帧的 AVFrame drop 释放
        self.release_without_context();
        warn!("RenderState dropped without cleanup(); GL resources left to the driver");
    }
}

// glEGLImageTargetTexture2DOES 失败只提示一次，避免每帧刷屏
fn warn_once_bind_failure() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        error!("glEGLImageTargetTexture2DOES failed; disabling DMA-BUF zero-copy");
    }
}

// DMA-BUF 完成 fence 失效（WAIT_FAILED）只提示一次：之后每帧都会走到这里
fn warn_once_invalid_fence() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        warn!(
            "DMA-BUF completion fence is invalid (WAIT_FAILED); \
             releasing retained frames to avoid a permanent freeze"
        );
    }
}

// GL 错误检查。release 下同样启用：驱动异常（纹理格式不匹配、UNPACK_ROW_LENGTH
// 取负、上下文丢失）只表现为"画面不对/不动"，日志却干干净净，根本无从排查。
// GetError 无错误时只是读一次客户端状态，代价可忽略。
// 只报前几条：同一个错误每帧重复会刷爆日志，第一次的信息量最大。
fn check_gl_error(context: &str) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static REPORTED: AtomicUsize = AtomicUsize::new(0);
    let e = unsafe { gl::GetError() };
    if e == gl::NO_ERROR {
        return;
    }
    let n = REPORTED.fetch_add(1, Ordering::Relaxed);
    if n < 5 {
        error!("GL error after {context}: 0x{e:x}");
    } else if n == 5 {
        error!("further GL errors suppressed (last: 0x{e:x} after {context})");
    }
}

// 单个 PBO 槽位上一条 fence 的等待结果
enum FenceState {
    // 已 signaled：该缓冲空闲，可安全走 MAP_UNSYNCHRONIZED 快路径
    Signaled,
    // 仍忙（等满超时）：本帧放弃该槽位，下一次重绘再试
    Busy,
    // WAIT_FAILED 等：sync 对象已失效，无法再确认完成状态
    Invalid,
}

// 等待某个 PBO 槽位上一次的异步上传在 GPU 侧完成。
// 三缓冲回绕后已隔 2 帧，正常情况早已 signaled，timeout=0 的检查只是一次
// 状态查询；GPU 积压超深时分块等待（每次 16ms，最多 8 次）。
fn wait_fence(fence: gl::types::GLsync) -> FenceState {
    if fence.is_null() {
        return FenceState::Signaled; // 该槽位还没上传过
    }
    unsafe {
        let mut status = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 0);
        if status == gl::TIMEOUT_EXPIRED {
            // 总等待上限 16ms（约一帧）：GPU 积压过深时宁可跳过本帧上传，
            // 也不要把 UI 线程卡 128ms（会反过来把视频同步推入"迟到"状态）
            for _ in 0..4 {
                status = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 4_000_000);
                if status != gl::TIMEOUT_EXPIRED {
                    break;
                }
            }
        }
        match status {
            gl::ALREADY_SIGNALED | gl::CONDITION_SATISFIED => FenceState::Signaled,
            gl::TIMEOUT_EXPIRED => FenceState::Busy,
            // WAIT_FAILED 等：sync 失效后无法再确认完成状态。不能一直判为
            // "未就绪"——那样该槽位永远放行不了、pbo_cur 永不前进，播放会
            // 永久冻结。交给 slots_ready 删除该 fence 并降级处理
            _ => FenceState::Invalid,
        }
    }
}

// 本轮要用的 PBO 槽位（每帧一个缓冲）的判定
enum SlotReadiness {
    // 空闲：可走 MAP_UNSYNCHRONIZED 快路径
    Ready,
    // 仍忙：本帧整体跳过，纹理保持上一帧
    Busy,
    // 有 fence 失效：放行本帧，但退回驱动隐式同步
    Fallback,
}

fn slots_ready(state: &RenderState, cur: usize) -> SlotReadiness {
    let mut fallback = false;
    if state.fence_lost.get()[cur] {
        fallback = true;
    }
    match wait_fence(state.fences[cur].get()) {
        FenceState::Signaled => {}
        FenceState::Busy => return SlotReadiness::Busy,
        FenceState::Invalid => {
            // 失效的 sync 无法再等待：删除并清空，后续退回驱动隐式同步
            let f = state.fences[cur].get();
            if !f.is_null() {
                unsafe { gl::DeleteSync(f) };
                state.fences[cur].set(std::ptr::null());
            }
            let mut lost = state.fence_lost.get();
            lost[cur] = true;
            state.fence_lost.set(lost);
            fallback = true;
        }
    }
    if fallback {
        SlotReadiness::Fallback
    } else {
        SlotReadiness::Ready
    }
}

// 420 色度平面的边长：ceil(v/2)。i32 的 div_ceil 尚未稳定，这里手写
fn chroma_dim(v: i32) -> i32 {
    (v + 1) / 2
}

// 单个平面的尺寸/行距描述（y/uv 平面共用上传签名，避免散参数）
#[derive(Clone, Copy)]
struct PlaneDims {
    width: i32,
    height: i32,
    // 行距：字节数（AVFrame.linesize 的单位）
    stride: i32,
    bytes_per_sample: usize,
    channels: usize,
}

fn texture_internal_format(bit_depth: u8, channels: usize) -> i32 {
    match (bit_depth, channels) {
        (10, 2) => gl::RG16 as i32,
        (10, _) => gl::R16 as i32,
        (_, 2) => gl::RG8 as i32,
        _ => gl::R8 as i32,
    }
}

fn texture_type(bit_depth: u8) -> u32 {
    texture_type_for_bytes(if bit_depth == 10 { 2 } else { 1 })
}

fn texture_type_for_bytes(bytes_per_sample: usize) -> u32 {
    if bytes_per_sample == 2 {
        gl::UNSIGNED_SHORT
    } else {
        gl::UNSIGNED_BYTE
    }
}

// 一个待上传平面的描述（与 PBO 中的偏移解耦）
#[derive(Clone, Copy)]
struct UploadPlane<'a> {
    unit: u32,
    bytes: &'a [u8],
    dims: PlaneDims,
}

impl<'a> UploadPlane<'a> {
    fn empty() -> Self {
        Self {
            unit: 0,
            bytes: &[],
            dims: PlaneDims {
                width: 0,
                height: 0,
                stride: 0,
                bytes_per_sample: 1,
                channels: 1,
            },
        }
    }
}

// 把整帧的 2/3 个平面连续写进同一个 PBO，只 Map/Unmap 一次，然后用 PBO
// offset 分别 TexSubImage2D。相比每平面各 map 一次，减少驱动调用；
// 一个 fence 覆盖整帧，三缓冲回绕逻辑不变。
unsafe fn upload_frame(state: &RenderState, cur: usize, frame: &YuvFrame, map_unsync: bool) {
    unsafe {
        const ALIGN: isize = 16;
        let chroma_w = chroma_dim(frame.width);
        let chroma_h = chroma_dim(frame.height);
        let bps = frame.bytes_per_sample();
        let mut planes = [UploadPlane::empty(); 3];
        planes[0] = UploadPlane {
            unit: gl::TEXTURE0,
            bytes: frame.y(),
            dims: PlaneDims {
                width: frame.width,
                height: frame.height,
                stride: frame.y_stride,
                bytes_per_sample: bps,
                channels: 1,
            },
        };
        let n = if frame.is_semi_planar() {
            planes[1] = UploadPlane {
                unit: gl::TEXTURE1,
                bytes: frame.uv(),
                dims: PlaneDims {
                    width: chroma_w,
                    height: chroma_h,
                    stride: frame.uv_stride,
                    bytes_per_sample: bps,
                    channels: 2,
                },
            };
            2
        } else {
            planes[1] = UploadPlane {
                unit: gl::TEXTURE1,
                bytes: frame.u(),
                dims: PlaneDims {
                    width: chroma_w,
                    height: chroma_h,
                    stride: frame.uv_stride,
                    bytes_per_sample: bps,
                    channels: 1,
                },
            };
            planes[2] = UploadPlane {
                unit: gl::TEXTURE2,
                bytes: frame.v(),
                dims: PlaneDims {
                    width: chroma_w,
                    height: chroma_h,
                    stride: frame.uv_stride,
                    bytes_per_sample: bps,
                    channels: 1,
                },
            };
            3
        };

        let mut offsets = [0isize; 3];
        let mut total = 0isize;
        for i in 0..n {
            offsets[i] = (total + ALIGN - 1) / ALIGN * ALIGN;
            total = offsets[i] + planes[i].bytes.len() as isize;
        }

        let slot = cur;
        let prev = state.fences[slot].get();
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, state.pbo[cur]);
        if total > state.pbo_cap.get()[slot] {
            gl::BufferData(
                gl::PIXEL_UNPACK_BUFFER,
                total,
                std::ptr::null(),
                gl::STREAM_DRAW,
            );
            let mut caps = state.pbo_cap.get();
            caps[slot] = total;
            state.pbo_cap.set(caps);
        }
        let mut flags = gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT;
        if map_unsync {
            flags |= gl::MAP_UNSYNCHRONIZED_BIT;
        }
        let ptr = gl::MapBufferRange(gl::PIXEL_UNPACK_BUFFER, 0, total, flags);
        let mapped = !ptr.is_null();
        if mapped {
            let base = ptr as *mut u8;
            for i in 0..n {
                std::ptr::copy_nonoverlapping(
                    planes[i].bytes.as_ptr(),
                    base.offset(offsets[i]),
                    planes[i].bytes.len(),
                );
            }
            gl::UnmapBuffer(gl::PIXEL_UNPACK_BUFFER);
        } else {
            // 映射失败兜底：直接从客户端内存上传（不再经过 PBO）
            gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        }

        for i in 0..n {
            let p = &planes[i];
            gl::ActiveTexture(p.unit);
            let bytes_per_pixel = p.dims.bytes_per_sample.max(1) * p.dims.channels.max(1);
            gl::PixelStorei(
                gl::UNPACK_ROW_LENGTH,
                p.dims.stride / bytes_per_pixel as i32,
            );
            let offset = if mapped {
                offsets[i] as usize as *const _
            } else {
                p.bytes.as_ptr() as *const _
            };
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                p.dims.width,
                p.dims.height,
                if p.dims.channels == 2 {
                    gl::RG
                } else {
                    gl::RED
                },
                texture_type_for_bytes(p.dims.bytes_per_sample),
                offset,
            );
        }

        // 解绑：PBO 只用于像素上传，DrawElements 不需要它；
        // 留在这里而不是 draw_frame，可以只在上传过的帧付这次调用
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);

        let done = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if !prev.is_null() {
            gl::DeleteSync(prev);
        }
        let mut lost = state.fence_lost.get();
        if !done.is_null() {
            state.fences[slot].set(done);
            lost[slot] = false;
        } else {
            state.fences[slot].set(std::ptr::null());
            lost[slot] = true;
        }
        state.fence_lost.set(lost);
    }
}

// 编译 + 链接一对 GLSL。编译/链接失败返回 None：调用方必须放弃这个 program，
// 不能继续 UseProgram——链接失败的 program 一切调用都是 GL_INVALID_OPERATION，
// 结果是整块黑屏，而日志里只有一行 error，用户完全看不出发生了什么。
unsafe fn build_program(vs_src: &str, fs_src: &str) -> Option<u32> {
    unsafe {
        let vs_src = CString::new(vs_src).unwrap();
        let fs_src = CString::new(fs_src).unwrap();

        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, &vs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(vs);
        let vs_ok = log_shader_status(vs, "vertex");

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, &fs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(fs);
        let fs_ok = log_shader_status(fs, "fragment");

        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        let link_ok = log_program_status(program);
        if !vs_ok || !fs_ok || !link_ok {
            gl::DeleteProgram(program);
            return None;
        }
        Some(program)
    }
}

// 取 uniform 位置并校验：位置为 -1 表示该 uniform 不存在（被改名或被优化掉），
// 此时 glUniform* 是"合法的空操作"——画面参数悄悄不生效，却没有任何报错。
unsafe fn uniform_loc(program: u32, name: &str) -> i32 {
    unsafe {
        let loc = gl::GetUniformLocation(program, CString::new(name).unwrap().as_ptr());
        if loc < 0 {
            error!("GL uniform '{name}' not found in shader");
        }
        loc
    }
}

// 着色器编译/链接状态校验：失败时输出日志并返回 false
unsafe fn log_shader_status(shader: u32, label: &str) -> bool {
    unsafe {
        let mut ok = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut ok);
        if ok != gl::TRUE as i32 {
            error!(
                "GL {} shader compile failed: {}",
                label,
                info_log(|len, written, buf| gl::GetShaderInfoLog(shader, len, written, buf))
            );
            return false;
        }
        true
    }
}

unsafe fn log_program_status(program: u32) -> bool {
    unsafe {
        let mut ok = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut ok);
        if ok != gl::TRUE as i32 {
            error!(
                "GL program link failed: {}",
                info_log(|len, written, buf| gl::GetProgramInfoLog(program, len, written, buf))
            );
            return false;
        }
        true
    }
}

unsafe fn info_log(mut get: impl FnMut(i32, *mut i32, *mut i8)) -> String {
    let mut buf = [0u8; 1024];
    let mut written = 0;
    get(buf.len() as i32, &mut written, buf.as_mut_ptr() as *mut i8);
    String::from_utf8_lossy(&buf[..written.max(0) as usize]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 一个点亮像素 = 一个矩形 = 6 顶点 = 12 个 f32；背景固定 6 顶点。
    // "1" 的 3×5 点阵亮 8 个点 → 6 + 8×6 = 54 顶点
    #[test]
    fn time_verts_count_matches_lit_pixels() {
        let v = time_verts("1", 100, 100);
        assert_eq!(v.len(), 54 * 2);
        // 空格只占位，不产生点阵矩形
        let space = time_verts(" ", 100, 100);
        assert_eq!(space.len(), 6 * 2);
    }

    // 所有顶点必须落在 NDC 内，且背景在文字之后（前 6 顶点为底衬）
    #[test]
    fn time_verts_stay_in_ndc_and_put_background_first() {
        for (w, h) in [(320, 240), (1920, 1080), (3840, 2160), (97, 61)] {
            let v = time_verts("12:34 / 56:78", w, h);
            assert!(v.len() > 6 * 2);
            assert!(v.iter().all(|c| (-1.0..=1.0).contains(c)), "{w}x{h}");
            // 背景：左上 > 右下（y 轴向上），且比字形区域更靠左上
            let (l, t, r, b) = (v[0], v[1], v[2], v[5]);
            assert!(l < r && b < t, "背景矩形方向: {w}x{h}");
        }
    }
}

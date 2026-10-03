use std::{cell::Cell, ffi::CString};

use fltk::window::GlWindow;
use log::error;

use crate::frame::{YuvFrame, color_matrix};

// 渲染状态：PBO 三缓冲 id（3 个平面 × 3 个缓冲）与已分配容量、纹理尺寸跟踪、
// 当前使用的缓冲索引、色彩矩阵与 range uniform 位置、每缓冲的异步上传同步对象
pub struct RenderState {
    pbo: [[u32; 3]; 3],
    pbo_cap: Cell<[isize; 3]>,
    tex_size: Cell<(i32, i32)>,
    pbo_cur: Cell<usize>,
    cmat_loc: i32,
    range_y_loc: i32,
    range_uv_loc: i32,
    last_gen: Cell<u64>, // 已上传的帧代号，窗口 damage 重绘时跳过重复上传
    // 上传 uniform 时所用的帧代号：色彩空间/范围几乎不变（同一流恒定），
    // 记下上次设置时的 frame_gen，重复重绘时无需重算 ColorMatrix 并重设 uniform
    cmat_gen: Cell<u64>,
    // 每平面 × 每缓冲一个 fence：记录"该缓冲上次 TexSubImage2D 的 GPU 完成信号"，
    // 回绕复用时先等待，从而允许 MAP_UNSYNCHRONIZED 立即映射而无需驱动隐式同步
    fences: [Cell<gl::types::GLsync>; 9],
    // 与 fences 对应：该槽位是否已丢失完成信号（FenceSync 创建失败或 sync 失效）。
    // 此时即便 fence 为 null 也不能按"空闲"放行 MAP_UNSYNCHRONIZED，必须退回
    // 驱动隐式同步，直到重新成功创建 fence
    fence_lost: Cell<[bool; 9]>,
}

// 初始化 GL 上下文：编译着色器、创建 VAO/纹理/PBO 三缓冲。
// program、纹理绑定、VAO 都是进程生命周期内不变的状态，只设置一次，
// draw 时只切换采样单元并更新纹理数据。
pub unsafe fn setup_opengl() -> RenderState {
    unsafe {
        let vs_src = CString::new(include_str!("shaders/basic.vert.glsl")).unwrap();
        let fs_src = CString::new(include_str!("shaders/basic.frag.glsl")).unwrap();

        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, &vs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(vs);
        log_shader_status(vs, "vertex");

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, &fs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(fs);
        log_shader_status(fs, "fragment");

        let shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vs);
        gl::AttachShader(shader_program, fs);
        gl::LinkProgram(shader_program);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);
        log_program_status(shader_program);

        gl::UseProgram(shader_program);
        let cmat_loc =
            gl::GetUniformLocation(shader_program, CString::new("cmat").unwrap().as_ptr());
        let range_y_loc =
            gl::GetUniformLocation(shader_program, CString::new("rangeY").unwrap().as_ptr());
        let range_uv_loc =
            gl::GetUniformLocation(shader_program, CString::new("rangeUV").unwrap().as_ptr());
        // 初始默认 full range，避免首帧前出现全黑/灰屏
        gl::Uniform2f(range_y_loc, 0.0, 1.0);
        gl::Uniform2f(range_uv_loc, 0.5, 1.0);
        gl::Uniform1i(
            gl::GetUniformLocation(shader_program, CString::new("texY").unwrap().as_ptr()),
            0,
        );
        gl::Uniform1i(
            gl::GetUniformLocation(shader_program, CString::new("texU").unwrap().as_ptr()),
            1,
        );
        gl::Uniform1i(
            gl::GetUniformLocation(shader_program, CString::new("texV").unwrap().as_ptr()),
            2,
        );

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

        // PBO 三缓冲：3 个平面各 3 个缓冲，轮转写入，隐藏 GPU 异步上传的等待。
        // 相比双缓冲，回绕间隔更长（2 帧 → 3 帧），几乎总能命中已完成的缓冲。
        // 存储在首次使用（或换分辨率）时才按需分配。
        let mut pbo = [[0; 3]; 3];
        gl::GenBuffers(9, pbo.as_mut_ptr() as *mut u32);

        RenderState {
            pbo,
            pbo_cap: Cell::new([0; 3]),
            tex_size: Cell::new((-1, -1)),
            pbo_cur: Cell::new(0),
            cmat_loc,
            range_y_loc,
            range_uv_loc,
            last_gen: Cell::new(u64::MAX),
            cmat_gen: Cell::new(u64::MAX),
            fences: std::array::from_fn(|_| Cell::new(std::ptr::null())),
            fence_lost: Cell::new([false; 9]),
        }
    }
}

// 绘制一帧：按视频显示宽高比（含 SAR 修正）居中留黑边。
// 纹理存储只在尺寸变化时重新分配；此后每帧把数据写进 PBO（CPU 快速拷贝），
// 再 TexSubImage2D 让 GPU 异步 DMA 到纹理，三缓冲 + fence 避免 CPU 阻塞等待上传。
pub fn draw_frame(w: &GlWindow, state: &RenderState, frame: &YuvFrame) {
    unsafe {
        gl::Clear(gl::COLOR_BUFFER_BIT);

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
        gl::Viewport(vp_x, vp_y, vp_w as i32, vp_h as i32);

        // 按流的色彩空间/范围选择转换参数（含 full/limited range 归一），
        // 不再按分辨率猜测；未标注时保持 HD+ → BT.709 的回退。
        // 同一流内色彩信息恒定，因此按 frame_gen 记住上次设置值，
        // 避免 FLTK 因鼠标移动等无关事件重绘时反复重算并重设 uniform
        if frame.frame_gen != state.cmat_gen.get() {
            let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
            gl::Uniform4f(state.cmat_loc, cm.rv, cm.gu, cm.gv, cm.bu);
            gl::Uniform2f(state.range_y_loc, cm.y_off, cm.y_gain);
            gl::Uniform2f(state.range_uv_loc, cm.uv_center, cm.uv_gain);
            state.cmat_gen.set(frame.frame_gen);
        }

        // 只有新帧才重新分配纹理与上传数据；FLTK 在鼠标移动等无关事件
        // 触发窗口 damage 重绘时会再次调用本回调，重复整帧上传纯属浪费
        let fresh = frame.frame_gen != state.last_gen.get();

        if fresh {
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

                let need_alloc = state.tex_size.get() != (frame.width, frame.height);

                if need_alloc {
                    // 分配纹理存储（不传数据）；先解绑 PBO，避免 TexImage2D 从 PBO 读垃圾
                    gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
                    gl::ActiveTexture(gl::TEXTURE0);
                    gl::TexImage2D(
                        gl::TEXTURE_2D,
                        0,
                        gl::RED as i32,
                        frame.width,
                        frame.height,
                        0,
                        gl::RED,
                        gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                    // 色度平面为 ceil(w/2)×ceil(h/2)（420 的分配方式），奇数尺寸不能整除
                    gl::ActiveTexture(gl::TEXTURE1);
                    gl::TexImage2D(
                        gl::TEXTURE_2D,
                        0,
                        gl::RED as i32,
                        chroma_dim(frame.width),
                        chroma_dim(frame.height),
                        0,
                        gl::RED,
                        gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                    gl::ActiveTexture(gl::TEXTURE2);
                    gl::TexImage2D(
                        gl::TEXTURE_2D,
                        0,
                        gl::RED as i32,
                        chroma_dim(frame.width),
                        chroma_dim(frame.height),
                        0,
                        gl::RED,
                        gl::UNSIGNED_BYTE,
                        std::ptr::null(),
                    );
                    state.tex_size.set((frame.width, frame.height));
                }

                // 色度纹理为 ceil(w/2)×ceil(h/2)（420 的分配方式），奇数尺寸不能整除
                let chroma_w = chroma_dim(frame.width);
                let chroma_h = chroma_dim(frame.height);
                upload_plane(
                    state,
                    cur,
                    gl::TEXTURE0,
                    0,
                    frame.y(),
                    unsync,
                    PlaneDims {
                        width: frame.width,
                        height: frame.height,
                        stride: frame.y_stride,
                    },
                );
                upload_plane(
                    state,
                    cur,
                    gl::TEXTURE1,
                    1,
                    frame.u(),
                    unsync,
                    PlaneDims {
                        width: chroma_w,
                        height: chroma_h,
                        stride: frame.uv_stride,
                    },
                );
                upload_plane(
                    state,
                    cur,
                    gl::TEXTURE2,
                    2,
                    frame.v(),
                    unsync,
                    PlaneDims {
                        width: chroma_w,
                        height: chroma_h,
                        stride: frame.uv_stride,
                    },
                );
                state.pbo_cur.set((cur + 1) % 3);

                gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
            }
        }

        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
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
            for _ in 0..8 {
                status = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 16_000_000);
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

// 本轮要用的 3 个平面槽位（同一缓冲索引）的总体判定
enum SlotReadiness {
    // 三平面都空闲：可走 MAP_UNSYNCHRONIZED 快路径
    Ready,
    // 至少一个平面仍忙：本帧整体跳过，纹理保持上一帧
    Busy,
    // 有 fence 失效：放行本帧，但退回驱动隐式同步
    Fallback,
}

// 三平面必须同进同出，否则会出现 Y 新帧配 U/V 旧帧的色度错位，
// 因此这里一次性判定，draw_frame 据此决定是否上传整帧。
fn slots_ready(state: &RenderState, cur: usize) -> SlotReadiness {
    let mut fallback = false;
    for pbo_idx in 0..3 {
        let slot = pbo_idx * 3 + cur;
        // 已丢失完成信号的槽位：即使没有 fence 可等，也只能按"未确认空闲"
        // 处理（退回驱动隐式同步），不能走 MAP_UNSYNCHRONIZED
        if state.fence_lost.get()[slot] {
            fallback = true;
        }
        match wait_fence(state.fences[slot].get()) {
            FenceState::Signaled => {}
            FenceState::Busy => return SlotReadiness::Busy,
            FenceState::Invalid => {
                // 失效的 sync 无法再等待：删除并清空，标记该槽位已丢失信号，
                // 本帧及后续（直到重新建 fence）都退回驱动隐式同步
                let f = state.fences[slot].get();
                if !f.is_null() {
                    unsafe { gl::DeleteSync(f) };
                    state.fences[slot].set(std::ptr::null());
                }
                let mut lost = state.fence_lost.get();
                lost[slot] = true;
                state.fence_lost.set(lost);
                fallback = true;
            }
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
struct PlaneDims {
    width: i32,
    height: i32,
    stride: i32,
}

// 把一帧单个平面的数据写入 PBO 并触发异步上传。
// 三缓冲：本帧写缓冲 N 时，GPU 正在从之前的缓冲读取，互不等待。
// slots_ready 确认槽位空闲（Ready）时 map_unsync=true，走 MAP_UNSYNCHRONIZED
// 立即映射、跳过驱动隐式同步；有 fence 失效（Fallback）时 map_unsync=false，
// 由驱动隐式同步保证不覆盖仍被读取的内存。
// 存储只在容量不足（首次或换分辨率）时分配一次；之后每帧 MapBufferRange
// 映射后自行拷贝并 INVALIDATE，避免 glBufferData 每帧重新分配存储的开销。
unsafe fn upload_plane(
    state: &RenderState,
    cur: usize,
    unit: u32,
    pbo_idx: usize,
    bytes: &[u8],
    map_unsync: bool,
    dims: PlaneDims,
) {
    unsafe {
        let len = bytes.len() as isize;
        let slot = pbo_idx * 3 + cur;
        // 该槽位上一轮的 fence（Ready 时已确认完成；Fallback 时失效的已被
        // slots_ready 删除并置 null），用完即可丢弃
        let prev = state.fences[slot].get();

        gl::ActiveTexture(unit);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, state.pbo[pbo_idx][cur]);
        if len > state.pbo_cap.get()[pbo_idx] {
            gl::BufferData(
                gl::PIXEL_UNPACK_BUFFER,
                len,
                std::ptr::null(),
                gl::STREAM_DRAW,
            );
            let mut caps = state.pbo_cap.get();
            caps[pbo_idx] = len;
            state.pbo_cap.set(caps);
        }
        let mut flags = gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT;
        if map_unsync {
            flags |= gl::MAP_UNSYNCHRONIZED_BIT;
        }
        let ptr = gl::MapBufferRange(gl::PIXEL_UNPACK_BUFFER, 0, len, flags);
        if ptr.is_null() {
            // 映射失败兜底：退回驱动内拷贝
            gl::BufferData(
                gl::PIXEL_UNPACK_BUFFER,
                len,
                bytes.as_ptr() as *const _,
                gl::STREAM_DRAW,
            );
        } else {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            gl::UnmapBuffer(gl::PIXEL_UNPACK_BUFFER);
        }
        gl::PixelStorei(gl::UNPACK_ROW_LENGTH, dims.stride);
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            dims.width,
            dims.height,
            gl::RED,
            gl::UNSIGNED_BYTE,
            std::ptr::null(),
        );

        // 为本缓冲记录本轮上传的完成信号，替换已等待完的旧 fence
        let done = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if !prev.is_null() {
            gl::DeleteSync(prev);
        }
        let mut lost = state.fence_lost.get();
        if !done.is_null() {
            state.fences[slot].set(done);
            lost[slot] = false; // 重新拿到有效信号，恢复异步快路径
        } else {
            // fence 创建失败：清掉旧 fence 并标记丢失信号。后续该槽位即使
            // fence 为 null 也走驱动隐式同步，避免覆盖可能仍在被读取的内存
            state.fences[slot].set(std::ptr::null());
            lost[slot] = true;
        }
        state.fence_lost.set(lost);
    }
}

// 着色器编译/链接状态校验：失败时输出日志，避免驱动异常时静默黑屏难排查
unsafe fn log_shader_status(shader: u32, label: &str) {
    unsafe {
        let mut ok = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut ok);
        if ok != gl::TRUE as i32 {
            error!(
                "GL {} shader compile failed: {}",
                label,
                info_log(|len, written, buf| gl::GetShaderInfoLog(shader, len, written, buf))
            );
        }
    }
}

unsafe fn log_program_status(program: u32) {
    unsafe {
        let mut ok = 0;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut ok);
        if ok != gl::TRUE as i32 {
            error!(
                "GL program link failed: {}",
                info_log(|len, written, buf| gl::GetProgramInfoLog(program, len, written, buf))
            );
        }
    }
}

unsafe fn info_log(mut get: impl FnMut(i32, *mut i32, *mut i8)) -> String {
    let mut buf = [0u8; 1024];
    let mut written = 0;
    get(buf.len() as i32, &mut written, buf.as_mut_ptr() as *mut i8);
    String::from_utf8_lossy(&buf[..written.max(0) as usize]).into_owned()
}

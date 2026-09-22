use std::cell::Cell;
use std::ffi::CString;

use fltk::window::GlWindow;

use crate::frame::{color_matrix, YuvFrame};

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
    // 每平面 × 每缓冲一个 fence：记录"该缓冲上次 TexSubImage2D 的 GPU 完成信号"，
    // 回绕复用时先等待，从而允许 MAP_UNSYNCHRONIZED 立即映射而无需驱动隐式同步
    fences: [Cell<gl::types::GLsync>; 9],
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
            fences: std::array::from_fn(|_| Cell::new(std::ptr::null())),
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
        // 不再按分辨率猜测；未标注时保持 HD+ → BT.709 的回退
        let cm = color_matrix(frame.color_space, frame.color_range, frame.height);
        gl::Uniform4f(state.cmat_loc, cm.rv, cm.gu, cm.gv, cm.bu);
        gl::Uniform2f(state.range_y_loc, cm.y_off, cm.y_gain);
        gl::Uniform2f(state.range_uv_loc, cm.uv_center, cm.uv_gain);

        // 只有新帧才重新分配纹理与上传数据；FLTK 在鼠标移动等无关事件
        // 触发窗口 damage 重绘时会再次调用本回调，重复整帧上传纯属浪费
        let fresh = frame.frame_gen != state.last_gen.get();

        if fresh {
            state.last_gen.set(frame.frame_gen);

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
                gl::ActiveTexture(gl::TEXTURE1);
                gl::TexImage2D(
                    gl::TEXTURE_2D,
                    0,
                    gl::RED as i32,
                    frame.width / 2,
                    frame.height / 2,
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
                    frame.width / 2,
                    frame.height / 2,
                    0,
                    gl::RED,
                    gl::UNSIGNED_BYTE,
                    std::ptr::null(),
                );
                state.tex_size.set((frame.width, frame.height));
            }

            let cur = state.pbo_cur.get();
            upload_plane(
                state,
                cur,
                gl::TEXTURE0,
                0,
                frame.y.as_ptr(),
                frame.y.len() as isize,
                frame.width,
                frame.height,
                frame.y_stride,
            );
            upload_plane(
                state,
                cur,
                gl::TEXTURE1,
                1,
                frame.u.as_ptr(),
                frame.u.len() as isize,
                frame.width / 2,
                frame.height / 2,
                frame.uv_stride,
            );
            upload_plane(
                state,
                cur,
                gl::TEXTURE2,
                2,
                frame.v.as_ptr(),
                frame.v.len() as isize,
                frame.width / 2,
                frame.height / 2,
                frame.uv_stride,
            );
            state.pbo_cur.set((cur + 1) % 3);

            gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
        }

        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
    }
}

// 把一帧单个平面的数据写入 PBO 并触发异步上传。
// 三缓冲：本帧写缓冲 N 时，GPU 正在从之前的缓冲读取，互不等待。
// 回绕复用同一缓冲前，先 ClientWaitSync 等上一轮的异步上传完成，
// 从而允许 MAP_UNSYNCHRONIZED 立即映射、跳过驱动的隐式同步（避免偶发阻塞）。
// 存储只在容量不足（首次或换分辨率）时分配一次；之后每帧 MapBufferRange
// 映射后自行拷贝并 INVALIDATE，避免 glBufferData 每帧重新分配存储的开销。
#[allow(clippy::too_many_arguments)]
unsafe fn upload_plane(
    state: &RenderState,
    cur: usize,
    unit: u32,
    pbo_idx: usize,
    data: *const u8,
    len: isize,
    width: i32,
    height: i32,
    stride: i32,
) {
    unsafe {
        let slot = pbo_idx * 3 + cur;
        // 等该缓冲上一轮 TexSubImage2D 在 GPU 侧完成（回绕已隔 2 帧，
        // 正常情况已经 signaled，timeout=0 的检查只是一次状态查询）
        let prev = state.fences[slot].get();
        if !prev.is_null() {
            // 正常情况已 signaled，timeout=0 只是一次状态查询
            let mut status = gl::ClientWaitSync(prev, gl::SYNC_FLUSH_COMMANDS_BIT, 0);
            if status == gl::TIMEOUT_EXPIRED {
                // 极少见：GPU 积压超深，分块等待直到完成（每次 16ms，最多 8 次）
                for _ in 0..8 {
                    status = gl::ClientWaitSync(prev, gl::SYNC_FLUSH_COMMANDS_BIT, 16_000_000);
                    if status != gl::TIMEOUT_EXPIRED {
                        break;
                    }
                }
            }
            if status == gl::TIMEOUT_EXPIRED {
                // 依然忙：跳过本帧上传（纹理保持上一帧内容），
                // 避免 MAP_UNSYNCHRONIZED 写入仍在被 DMA 读取的内存
                return;
            }
        }

        gl::ActiveTexture(unit);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, state.pbo[pbo_idx][cur]);
        if len > state.pbo_cap.get()[pbo_idx] {
            gl::BufferData(gl::PIXEL_UNPACK_BUFFER, len, std::ptr::null(), gl::STREAM_DRAW);
            let mut caps = state.pbo_cap.get();
            caps[pbo_idx] = len;
            state.pbo_cap.set(caps);
        }
        let ptr = gl::MapBufferRange(
            gl::PIXEL_UNPACK_BUFFER,
            0,
            len,
            gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT | gl::MAP_UNSYNCHRONIZED_BIT,
        );
        if ptr.is_null() {
            // 映射失败兜底：退回驱动内拷贝
            gl::BufferData(gl::PIXEL_UNPACK_BUFFER, len, data as *const _, gl::STREAM_DRAW);
        } else {
            std::ptr::copy_nonoverlapping(data, ptr as *mut u8, len as usize);
            gl::UnmapBuffer(gl::PIXEL_UNPACK_BUFFER);
        }
        gl::PixelStorei(gl::UNPACK_ROW_LENGTH, stride);
        gl::TexSubImage2D(
            gl::TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl::RED,
            gl::UNSIGNED_BYTE,
            std::ptr::null(),
        );

        // 为本缓冲记录本轮上传的完成信号，替换已等待完的旧 fence
        let done = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
        if !done.is_null() {
            if !prev.is_null() {
                gl::DeleteSync(prev);
            }
            state.fences[slot].set(done);
        }
    }
}

// 着色器编译/链接状态校验：失败时输出日志，避免驱动异常时静默黑屏难排查
unsafe fn log_shader_status(shader: u32, label: &str) {
    unsafe {
        let mut ok = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut ok);
        if ok != gl::TRUE as i32 {
            eprintln!(
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
            eprintln!(
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

use std::cell::Cell;
use std::ffi::CString;

use fltk::prelude::*;
use fltk::window::GlWindow;

use crate::frame::YuvFrame;

// 渲染状态：PBO 双缓冲 id（3 个平面 × 2 个缓冲）、纹理尺寸跟踪、当前使用的缓冲索引
pub struct RenderState {
    pbo: [[u32; 2]; 3],
    tex_size: Cell<(i32, i32)>,
    pbo_cur: Cell<usize>,
}

// 初始化 GL 上下文：编译着色器、创建 VAO/纹理/PBO 双缓冲。
// program、纹理绑定、VAO 都是进程生命周期内不变的状态，只设置一次，
// draw 时只切换采样单元并更新纹理数据。
pub unsafe fn setup_opengl() -> RenderState {
    unsafe {
        let vs_src = CString::new(include_str!("shaders/basic.vert.glsl")).unwrap();
        let fs_src = CString::new(include_str!("shaders/basic.frag.glsl")).unwrap();

        let vs = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(vs, 1, &vs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(vs);

        let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(fs, 1, &fs_src.as_ptr(), std::ptr::null());
        gl::CompileShader(fs);

        let shader_program = gl::CreateProgram();
        gl::AttachShader(shader_program, vs);
        gl::AttachShader(shader_program, fs);
        gl::LinkProgram(shader_program);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        gl::UseProgram(shader_program);
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

        // PBO 双缓冲：3 个平面各 2 个缓冲，交替写入，隐藏 GPU 异步上传的等待
        let mut pbo = [[0; 2]; 3];
        gl::GenBuffers(6, pbo.as_mut_ptr() as *mut u32);

        RenderState {
            pbo,
            tex_size: Cell::new((-1, -1)),
            pbo_cur: Cell::new(0),
        }
    }
}

// 绘制一帧：按视频宽高比居中留黑边。
// 纹理存储只在尺寸变化时重新分配；此后每帧把数据写进 PBO（CPU 快速拷贝），
// 再 TexSubImage2D 让 GPU 异步 DMA 到纹理，双缓冲避免 CPU 阻塞等待上传。
pub fn draw_frame(w: &GlWindow, state: &RenderState, frame: &YuvFrame) {
    unsafe {
        gl::Clear(gl::COLOR_BUFFER_BIT);

        let win_w = w.w() as f32;
        let win_h = w.h() as f32;
        let vw = frame.width as f32;
        let vh = frame.height as f32;
        let (vp_w, vp_h) = if vw / vh > win_w / win_h {
            (win_w, win_w * vh / vw)
        } else {
            (win_h * vw / vh, win_h)
        };
        let vp_x = ((win_w - vp_w) / 2.0) as i32;
        let vp_y = ((win_h - vp_h) / 2.0) as i32;
        gl::Viewport(vp_x, vp_y, vp_w as i32, vp_h as i32);

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
        state.pbo_cur.set(1 - cur);

        gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, 0);
        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
    }
}

// 把一帧单个平面的数据写入 PBO 并触发异步上传。
// 双缓冲：本帧写缓冲 A 时，上一帧 GPU 正在从缓冲 B 读取，互不等待。
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
        gl::ActiveTexture(unit);
        gl::PixelStorei(gl::UNPACK_ROW_LENGTH, stride);
        gl::BindBuffer(gl::PIXEL_UNPACK_BUFFER, state.pbo[pbo_idx][cur]);
        gl::BufferData(
            gl::PIXEL_UNPACK_BUFFER,
            len,
            data as *const _,
            gl::STREAM_DRAW,
        );
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
    }
}

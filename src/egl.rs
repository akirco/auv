// EGL/DMABUF 导入：在 FLTK 的 EGL 上下文里，把 DMA-BUF 平面直接绑定成
// GL 纹理（glEGLImageTargetTexture2DOES），避免下载到内存再 PBO 上传。
//
// 仅在当前线程确实持有 EGL（Wayland 后端）上下文时启用；X11/GLX 或无相关
// 扩展时返回 None，渲染端走原有 PBO 路径。libEGL 用 dlopen 延迟加载，
// 不新增构建期链接依赖。
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
};

use log::{debug, info, warn};

use crate::dmabuf::{DRM_FORMAT_MOD_LINEAR, DmaBufPlane};

pub type EGLDisplay = *mut c_void;
pub type EGLContext = *mut c_void;
pub type EGLImage = *mut c_void;

type EGLint = c_int;
type EGLBoolean = u32;
type EGLenum = u32;
type EGLAttrib = isize;

const EGL_EXTENSIONS: EGLint = 0x3055;
const EGL_NONE: EGLint = 0x3038;
const EGL_WIDTH: EGLint = 0x3057;
const EGL_HEIGHT: EGLint = 0x3056;

const EGL_LINUX_DMA_BUF_EXT: EGLenum = 0x3270;
const EGL_LINUX_DRM_FOURCC_EXT: EGLint = 0x3271;
const EGL_DMA_BUF_PLANE0_FD_EXT: EGLint = 0x3272;
const EGL_DMA_BUF_PLANE0_OFFSET_EXT: EGLint = 0x3273;
const EGL_DMA_BUF_PLANE0_PITCH_EXT: EGLint = 0x3274;
const EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT: EGLint = 0x3443;
const EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT: EGLint = 0x3444;

type FnGetCurrentDisplay = unsafe extern "C" fn() -> EGLDisplay;
type FnGetCurrentContext = unsafe extern "C" fn() -> EGLContext;
type FnQueryString = unsafe extern "C" fn(EGLDisplay, EGLint) -> *const c_char;
type FnGetError = unsafe extern "C" fn() -> EGLint;
type FnGetProcAddress = unsafe extern "C" fn(*const c_char) -> *const c_void;
type FnCreateImageKhr =
    unsafe extern "C" fn(EGLDisplay, EGLContext, EGLenum, *mut c_void, *const EGLint) -> EGLImage;
type FnCreateImageCore = unsafe extern "C" fn(
    EGLDisplay,
    EGLContext,
    EGLenum,
    *mut c_void,
    *const EGLAttrib,
) -> EGLImage;
type FnDestroyImage = unsafe extern "C" fn(EGLDisplay, EGLImage) -> EGLBoolean;
type FnImageTargetTexture = unsafe extern "C" fn(u32, EGLImage);

enum CreateImage {
    Khr(FnCreateImageKhr),
    // EGL 1.5 核心函数（参数是 intptr_t 而非 EGLint）
    Core(FnCreateImageCore),
}

// dlopen 一次，进程生命周期内常驻：函数指针被各处缓存，卸载没有收益
struct EglApi {
    _lib: &'static libloading::Library,
    get_current_display: FnGetCurrentDisplay,
    get_current_context: FnGetCurrentContext,
    query_string: FnQueryString,
    get_error: FnGetError,
    get_proc_address: FnGetProcAddress,
    create_image: CreateImage,
    destroy_image: FnDestroyImage,
}

impl EglApi {
    fn load() -> Option<Self> {
        unsafe {
            let lib = Box::leak(Box::new(libloading::Library::new("libEGL.so.1").ok()?));
            let get_current_display: FnGetCurrentDisplay =
                load_sym(lib, b"eglGetCurrentDisplay\0")?;
            let get_current_context: FnGetCurrentContext =
                load_sym(lib, b"eglGetCurrentContext\0")?;
            let query_string: FnQueryString = load_sym(lib, b"eglQueryString\0")?;
            let get_error: FnGetError = load_sym(lib, b"eglGetError\0")?;
            let get_proc_address: FnGetProcAddress = load_sym(lib, b"eglGetProcAddress\0")?;

            // KHR 扩展函数需要 eglGetProcAddress；EGL 1.5 驱动也可能只导出核心名
            let khr = get_proc_address(c"eglCreateImageKHR".as_ptr());
            let create_image = if !khr.is_null() {
                CreateImage::Khr(std::mem::transmute::<*const c_void, FnCreateImageKhr>(khr))
            } else {
                CreateImage::Core(load_sym::<FnCreateImageCore>(lib, b"eglCreateImage\0")?)
            };
            let destroy = {
                let p = get_proc_address(c"eglDestroyImageKHR".as_ptr());
                if !p.is_null() {
                    std::mem::transmute::<*const c_void, FnDestroyImage>(p)
                } else {
                    load_sym::<FnDestroyImage>(lib, b"eglDestroyImage\0")?
                }
            };
            Some(Self {
                _lib: lib,
                get_current_display,
                get_current_context,
                query_string,
                get_error,
                get_proc_address,
                create_image,
                destroy_image: destroy,
            })
        }
    }
}

// 从 libloading 的 Symbol 拷出函数指针
unsafe fn load_sym<T: Copy + 'static>(lib: &libloading::Library, name: &[u8]) -> Option<T> {
    let sym = unsafe { lib.get::<T>(name) }.ok()?;
    Some(*sym)
}

static EGL_API: OnceLock<Option<EglApi>> = OnceLock::new();

// 视频线程据此决定是否把硬解帧交给渲染端做零拷贝；渲染端导入失败时会清掉，
// 后续帧自动退回下载路径。
static DMABUF_AVAILABLE: AtomicBool = AtomicBool::new(false);

pub fn available() -> bool {
    DMABUF_AVAILABLE.load(Ordering::Relaxed)
}

// 渲染端导入失败（不是格式不支持）时调用：全局退回 CPU 下载路径
pub fn disable(reason: &str) {
    if DMABUF_AVAILABLE.swap(false, Ordering::Relaxed) {
        warn!("EGL/DMABUF disabled, falling back to download: {reason}");
    }
}

// 当前 EGL 上下文上的导入器。每个 GL 上下文一个（窗口重建时重新创建）。
pub struct EglInterop {
    api: &'static EglApi,
    display: EGLDisplay,
    has_modifiers: bool,
    image_target_texture: FnImageTargetTexture,
}

impl EglInterop {
    // 探测当前 EGL/GL 上下文能否导入 DMA-BUF。必须在 GL 上下文为 current 时调用。
    ///
    /// # Safety
    /// 调用线程必须持有有效的当前 EGL/OpenGL 上下文。
    pub unsafe fn new() -> Option<Self> {
        if std::env::var_os("AUV_DMABUF").is_some_and(|v| v == "0") {
            debug!("AUV_DMABUF=0: DMA-BUF zero-copy disabled");
            return None;
        }
        let api = EGL_API.get_or_init(EglApi::load).as_ref()?;
        let display = unsafe { (api.get_current_display)() };
        let context = unsafe { (api.get_current_context)() };
        if display.is_null() || context.is_null() {
            debug!("no current EGL context (X11/GLX?); DMA-BUF zero-copy disabled");
            return None;
        }

        let exts = unsafe { (api.query_string)(display, EGL_EXTENSIONS) };
        if exts.is_null() {
            return None;
        }
        let exts = unsafe { CStr::from_ptr(exts) }.to_string_lossy();
        let has = |name: &str| exts.split_ascii_whitespace().any(|e| e == name);
        if !has("EGL_KHR_image_base") || !has("EGL_EXT_image_dma_buf_import") {
            debug!("EGL DMA-BUF import extensions unavailable");
            return None;
        }
        let has_modifiers = has("EGL_EXT_image_dma_buf_import_modifiers");

        // glEGLImageTargetTexture2DOES 来自 GL_OES_EGL_image（Mesa 桌面 GL 也提供）
        let proc = unsafe { (api.get_proc_address)(c"glEGLImageTargetTexture2DOES".as_ptr()) };
        if proc.is_null() {
            debug!("glEGLImageTargetTexture2DOES unavailable");
            return None;
        }
        let image_target_texture =
            unsafe { std::mem::transmute::<*const c_void, FnImageTargetTexture>(proc) };
        if !unsafe { gl_has_extension("GL_OES_EGL_image") } {
            debug!("GL_OES_EGL_image unavailable");
            return None;
        }

        DMABUF_AVAILABLE.store(true, Ordering::Relaxed);
        info!(
            "EGL/DMABUF zero-copy enabled{}",
            if has_modifiers {
                " (with modifiers)"
            } else {
                " (linear only)"
            }
        );
        Some(Self {
            api,
            display,
            has_modifiers,
            image_target_texture,
        })
    }

    // 把一个 DMA-BUF 平面导入为 EGLImage。失败时返回 None（调用方整体回退）。
    pub fn create_image(&self, plane: &DmaBufPlane) -> Option<EGLImage> {
        if plane.modifier != DRM_FORMAT_MOD_LINEAR && !self.has_modifiers {
            debug!(
                "dmabuf plane needs modifier {:#x} but EGL import modifiers unavailable",
                plane.modifier
            );
            return None;
        }
        let mut attribs: Vec<EGLint> = Vec::with_capacity(16);
        attribs.extend_from_slice(&[
            EGL_WIDTH,
            plane.width,
            EGL_HEIGHT,
            plane.height,
            EGL_LINUX_DRM_FOURCC_EXT,
            plane.fourcc as EGLint,
            EGL_DMA_BUF_PLANE0_FD_EXT,
            plane.fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            plane.offset as EGLint,
            EGL_DMA_BUF_PLANE0_PITCH_EXT,
            plane.pitch as EGLint,
        ]);
        if self.has_modifiers {
            attribs.extend_from_slice(&[
                EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
                plane.modifier as u32 as EGLint,
                EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
                (plane.modifier >> 32) as u32 as EGLint,
            ]);
        }
        attribs.push(EGL_NONE);

        let image = unsafe {
            match self.api.create_image {
                CreateImage::Khr(f) => f(
                    self.display,
                    std::ptr::null_mut(),
                    EGL_LINUX_DMA_BUF_EXT,
                    std::ptr::null_mut(),
                    attribs.as_ptr(),
                ),
                CreateImage::Core(f) => {
                    let attribs64: Vec<EGLAttrib> =
                        attribs.iter().map(|v| *v as EGLAttrib).collect();
                    f(
                        self.display,
                        std::ptr::null_mut(),
                        EGL_LINUX_DMA_BUF_EXT,
                        std::ptr::null_mut(),
                        attribs64.as_ptr(),
                    )
                }
            }
        };
        if image.is_null() {
            let err = unsafe { (self.api.get_error)() };
            debug!(
                "eglCreateImageKHR failed: 0x{err:x} (fourcc={:#x} {}x{} modifier={:#x})",
                plane.fourcc, plane.width, plane.height, plane.modifier
            );
            return None;
        }
        Some(image)
    }

    ///
    /// # Safety
    /// `image` 必须是本 `EglInterop` 所在的 EGLDisplay 上创建且尚未销毁的 EGLImage。
    pub unsafe fn destroy_image(&self, image: EGLImage) {
        if !image.is_null() {
            unsafe { (self.api.destroy_image)(self.display, image) };
        }
    }

    // 把 EGLImage 绑定为 texture 的存储（替换原有存储）。
    ///
    /// `unit` 是目标纹理单元：BindTexture 作用于**当前活动单元**，
    /// 不显式切换就会把纹理绑到调用方遗留的活动单元上（顶掉别的纹理绑定）。
    ///
    /// # Safety
    /// `image` 必须有效，且调用线程持有创建该 image 的 EGL/GL 上下文。
    pub unsafe fn bind_image(&self, unit: u32, texture: u32, image: EGLImage) -> bool {
        unsafe {
            // 清掉之前探测扩展等操作可能留下的错误，避免误判下面的调用
            while gl::GetError() != gl::NO_ERROR {}
            gl::ActiveTexture(gl::TEXTURE0 + unit);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            (self.image_target_texture)(gl::TEXTURE_2D, image);
            let err = gl::GetError();
            if err != gl::NO_ERROR {
                warn!("glEGLImageTargetTexture2DOES failed: 0x{err:x}");
                return false;
            }
        }
        true
    }
}

// 查 GL 扩展：GL 3.0+ 用 glGetStringi，老上下文退回 glGetString(GL_EXTENSIONS)
unsafe fn gl_has_extension(name: &str) -> bool {
    unsafe {
        let mut count = 0;
        gl::GetIntegerv(gl::NUM_EXTENSIONS, &mut count);
        if count > 0 {
            for i in 0..count as u32 {
                let p = gl::GetStringi(gl::EXTENSIONS, i);
                if !p.is_null() && CStr::from_ptr(p as *const c_char).to_bytes() == name.as_bytes()
                {
                    return true;
                }
            }
            return false;
        }
        let p = gl::GetString(gl::EXTENSIONS);
        if p.is_null() {
            return false;
        }
        let exts = CStr::from_ptr(p as *const c_char).to_string_lossy();
        exts.split_ascii_whitespace().any(|e| e == name)
    }
}

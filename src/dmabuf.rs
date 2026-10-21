// DMA-BUF 帧：把硬解（VAAPI）输出的 AVFrame 映射成 DRM_PRIME 描述符，
// 渲染端据此用 EGL 直接导入成 GL 纹理，省掉下载到内存 + PBO 上传。
//
// 生命周期：`DmaBufFrame` 持有 `av_hwframe_map` 得到的 AVFrame（引用计数），
// 它同时保证底层 VAAPI surface 不被解码器回收；渲染端在 GPU fence 触发前
// 会再 retain 一份，所以 surface 至少活到采样完成。
use ffmpeg_next::{ffi, format::Pixel, util::frame::video::Video};
use log::debug;
use std::sync::atomic::{AtomicBool, Ordering};

// 只在进程内打印一次帧布局，避免 RUST_LOG=debug 时每帧刷屏
static LAYOUT_LOGGED: AtomicBool = AtomicBool::new(false);

// DRM fourcc（小端四字符码）。只列本模块用到的格式。
const DRM_FORMAT_R8: u32 = 0x2020_3852; // 'R8  '
const DRM_FORMAT_R16: u32 = 0x2036_3152; // 'R16 '
const DRM_FORMAT_GR88: u32 = 0x3838_5247; // 'GR88'
const DRM_FORMAT_GR1616: u32 = 0x3233_5247; // 'GR32'（16:16）
const DRM_FORMAT_NV12: u32 = 0x3231_564e; // 'NV12'
const DRM_FORMAT_P010: u32 = 0x3031_3050; // 'P010'
const DRM_FORMAT_YUV420: u32 = 0x3231_5559; // 'YU12'

// 线性布局；Intel VAAPI 常见 Y-tile，此时需要 EGL modifier 支持
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

// 一个可导入 EGL 的 DMA-BUF 平面。fd 属于底层 AVFrame，只要帧还活着就有效。
#[derive(Clone, Copy, Debug)]
pub struct DmaBufPlane {
    pub fd: i32,
    pub offset: u32,
    pub pitch: u32,
    pub modifier: u64,
    // 导入成纹理时使用的 DRM fourcc（R8/R16/GR88/GR1616）
    pub fourcc: u32,
    pub width: i32,
    pub height: i32,
}

// 色度布局与 CPU 帧的 ChromaLayout 对应，渲染端按同一套 shader 采样
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DmaBufLayout {
    Planar,
    SemiPlanar,
}

// 一帧 DMA-BUF 视频：Y 在 planes[0]，UV（半平面）或 U 在 planes[1]，V 在 planes[2]
pub struct DmaBufFrame {
    // 拥有 DRM_PRIME AVFrame：offset/fd/modifier 均来自它，且持有 surface 引用
    drm: Video,
    pub planes: [Option<DmaBufPlane>; 3],
    pub layout: DmaBufLayout,
    pub bit_depth: u8,
}

impl DmaBufFrame {
    // 把 VAAPI 硬件帧映射成 DRM_PRIME，并拆出可导入的平面。
    // 不支持的四cc（例如厂商私有格式）返回 None，调用方回退下载。
    ///
    /// # Safety
    /// `hw` 必须是有效的硬件 AVFrame（VAAPI）。
    pub unsafe fn from_vaapi(hw: &Video) -> Option<Self> {
        let mut drm = Video::empty();
        drm.set_format(Pixel::DRM_PRIME);
        let ret = unsafe {
            ffi::av_hwframe_map(
                drm.as_mut_ptr(),
                hw.as_ptr(),
                ffi::AV_HWFRAME_MAP_READ as i32,
            )
        };
        if ret < 0 {
            debug!("av_hwframe_map(DRM_PRIME) failed: {ret}");
            return None;
        }
        let desc = unsafe { (*drm.as_ptr()).data[0] as *const ffi::AVDRMFrameDescriptor };
        if desc.is_null() {
            debug!("DRM_PRIME frame has no AVDRMFrameDescriptor");
            return None;
        }
        let desc = unsafe { &*desc };
        let w = hw.width() as i32;
        let h = hw.height() as i32;
        let (layout, bit_depth, planes, fourcc) = match planes_from_descriptor(desc, w, h) {
            Some(v) => v,
            None => {
                debug!(
                    "unsupported DRM descriptor: {}x{} layers={} objects={}",
                    w, h, desc.nb_layers, desc.nb_objects
                );
                for (i, layer) in desc
                    .layers
                    .iter()
                    .take(desc.nb_layers.max(0) as usize)
                    .enumerate()
                {
                    debug!(
                        "  layer[{i}] fourcc={:#010x} planes={} p0=(obj={} off={} pitch={})",
                        layer.format,
                        layer.nb_planes,
                        layer.planes[0].object_index,
                        layer.planes[0].offset,
                        layer.planes[0].pitch
                    );
                }
                return None;
            }
        };
        if !LAYOUT_LOGGED.swap(true, Ordering::Relaxed) {
            debug!(
                "dmabuf frame layout: {}x{} fourcc={:#010x} layers={} objects={}",
                w, h, fourcc, desc.nb_layers, desc.nb_objects
            );
        }
        Some(Self {
            drm,
            planes,
            layout,
            bit_depth,
        })
    }

    // 增加一份引用（av_frame_ref，仅引用计数），供渲染端在 GPU 完成前保活。
    pub fn retain(&self) -> Option<Self> {
        let mut drm = Video::empty();
        let ret = unsafe { ffi::av_frame_ref(drm.as_mut_ptr(), self.drm.as_ptr()) };
        if ret < 0 {
            debug!("av_frame_ref(DRM_PRIME) failed: {ret}");
            return None;
        }
        Some(Self {
            drm,
            planes: self.planes,
            layout: self.layout,
            bit_depth: self.bit_depth,
        })
    }
}

// 从 AVDRMFrameDescriptor 取出各平面。VAAPI 驱动有两种导出方式：
// - iHD 等：一个 NV12/P010/YUV420 层内含 2/3 个平面；
// - i965 等：每个平面单独成层（R8 + GR88 / R16 + GR1616）。
// 两种都按 AVDRM 文档的平面顺序（layer 索引 → plane 索引）归一化成
// [Y, UV] 或 [Y, U, V]。
fn planes_from_descriptor(
    desc: &ffi::AVDRMFrameDescriptor,
    width: i32,
    height: i32,
) -> Option<(DmaBufLayout, u8, [Option<DmaBufPlane>; 3], u32)> {
    if desc.nb_layers < 1 || desc.nb_objects < 1 || width <= 0 || height <= 0 {
        return None;
    }
    let first_fourcc = desc.layers[0].format;

    // 先按描述符顺序收集原始平面，并确定每个平面应导入成的纹理格式
    struct Raw {
        fourcc: u32,
        fd: i32,
        offset: u32,
        pitch: u32,
        modifier: u64,
    }
    let mut raw: Vec<Raw> = Vec::with_capacity(3);
    for layer in desc.layers.iter().take(desc.nb_layers.max(0) as usize) {
        let seq: &[u32] = match layer.format {
            DRM_FORMAT_NV12 => &[DRM_FORMAT_R8, DRM_FORMAT_GR88],
            DRM_FORMAT_P010 => &[DRM_FORMAT_R16, DRM_FORMAT_GR1616],
            DRM_FORMAT_YUV420 => &[DRM_FORMAT_R8, DRM_FORMAT_R8, DRM_FORMAT_R8],
            // 单平面层：格式本身就是要绑成纹理的格式
            DRM_FORMAT_R8 | DRM_FORMAT_R16 | DRM_FORMAT_GR88 | DRM_FORMAT_GR1616 => &[layer.format],
            _ => return None,
        };
        if layer.nb_planes < 0 || layer.nb_planes as usize != seq.len() {
            return None;
        }
        for (i, fourcc) in seq.iter().enumerate() {
            let plane = &layer.planes[i];
            let obj = plane.object_index;
            if obj < 0 || obj >= desc.nb_objects || plane.offset < 0 || plane.pitch <= 0 {
                return None;
            }
            let object = &desc.objects[obj as usize];
            if object.fd < 0 {
                return None;
            }
            raw.push(Raw {
                fourcc: *fourcc,
                fd: object.fd,
                offset: plane.offset as u32,
                pitch: plane.pitch as u32,
                modifier: object.format_modifier,
            });
        }
    }

    // 归类：2 平面 = 半平面（Y + 交织 UV），3 平面 = 平面 420
    let (layout, bit_depth) = match raw.len() {
        2 => {
            let depth = fourcc_depth(raw[0].fourcc)?;
            if depth != fourcc_depth(raw[1].fourcc)? || fourcc_channels(raw[1].fourcc)? != 2 {
                return None;
            }
            (DmaBufLayout::SemiPlanar, depth)
        }
        3 => {
            let depth = fourcc_depth(raw[0].fourcc)?;
            if raw[1..]
                .iter()
                .any(|p| fourcc_depth(p.fourcc) != Some(depth))
            {
                return None;
            }
            // 三个平面都必须是单通道
            if raw.iter().any(|p| fourcc_channels(p.fourcc) != Some(1)) {
                return None;
            }
            (DmaBufLayout::Planar, depth)
        }
        _ => return None,
    };

    let chroma_w = (width + 1) / 2;
    let chroma_h = (height + 1) / 2;
    let mut planes = [None; 3];
    for (i, p) in raw.into_iter().enumerate() {
        let (w, h) = if i == 0 {
            (width, height)
        } else {
            (chroma_w, chroma_h)
        };
        planes[i] = Some(DmaBufPlane {
            fd: p.fd,
            offset: p.offset,
            pitch: p.pitch,
            modifier: p.modifier,
            fourcc: p.fourcc,
            width: w,
            height: h,
        });
    }

    Some((layout, bit_depth, planes, first_fourcc))
}

// 纹理格式对应的位深：R8/GR88 = 8，R16/GR1616 = 10
fn fourcc_depth(fourcc: u32) -> Option<u8> {
    match fourcc {
        DRM_FORMAT_R8 | DRM_FORMAT_GR88 => Some(8),
        DRM_FORMAT_R16 | DRM_FORMAT_GR1616 => Some(10),
        _ => None,
    }
}

fn fourcc_channels(fourcc: u32) -> Option<usize> {
    match fourcc {
        DRM_FORMAT_R8 | DRM_FORMAT_R16 => Some(1),
        DRM_FORMAT_GR88 | DRM_FORMAT_GR1616 => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJ_SIZE: usize = 4096;

    // 构造一个最小的 DRM 描述符：一个对象，plane 参数可控
    fn desc_with(
        format: u32,
        nb_planes: i32,
        planes: &[(i32, isize, isize)],
        modifier: u64,
    ) -> ffi::AVDRMFrameDescriptor {
        let mut desc: ffi::AVDRMFrameDescriptor = unsafe { std::mem::zeroed() };
        desc.nb_objects = 1;
        desc.objects[0].fd = 7;
        desc.objects[0].size = OBJ_SIZE;
        desc.objects[0].format_modifier = modifier;
        desc.nb_layers = 1;
        desc.layers[0].format = format;
        desc.layers[0].nb_planes = nb_planes;
        for (i, (obj, offset, pitch)) in planes.iter().enumerate() {
            desc.layers[0].planes[i].object_index = *obj;
            desc.layers[0].planes[i].offset = *offset;
            desc.layers[0].planes[i].pitch = *pitch;
        }
        desc
    }

    #[test]
    fn nv12_splits_into_r8_and_gr88() {
        let d = desc_with(
            DRM_FORMAT_NV12,
            2,
            &[(0, 0, 1920), (0, 1920 * 1080, 1920)],
            DRM_FORMAT_MOD_LINEAR,
        );
        let (layout, depth, planes, fourcc) = planes_from_descriptor(&d, 1920, 1080).unwrap();
        assert_eq!(layout, DmaBufLayout::SemiPlanar);
        assert_eq!(depth, 8);
        assert_eq!(fourcc, DRM_FORMAT_NV12);
        let y = planes[0].unwrap();
        assert_eq!(
            (y.fourcc, y.width, y.height, y.fd),
            (DRM_FORMAT_R8, 1920, 1080, 7)
        );
        let uv = planes[1].unwrap();
        assert_eq!(
            (uv.fourcc, uv.width, uv.height),
            (DRM_FORMAT_GR88, 960, 540)
        );
        assert_eq!(uv.offset, 1920 * 1080);
        assert!(planes[2].is_none());
    }

    #[test]
    fn p010_uses_16bit_plane_formats() {
        let d = desc_with(
            DRM_FORMAT_P010,
            2,
            &[(0, 0, 3840), (0, 3840 * 1080, 3840)],
            0x0100_0000_0000_0003, // 非零 modifier
        );
        let (layout, depth, planes, _) = planes_from_descriptor(&d, 1920, 1080).unwrap();
        assert_eq!(layout, DmaBufLayout::SemiPlanar);
        assert_eq!(depth, 10);
        assert_eq!(planes[0].unwrap().fourcc, DRM_FORMAT_R16);
        assert_eq!(planes[1].unwrap().fourcc, DRM_FORMAT_GR1616);
        assert_eq!(planes[0].unwrap().modifier, 0x0100_0000_0000_0003);
    }

    #[test]
    fn yuv420_is_three_planes() {
        let d = desc_with(
            DRM_FORMAT_YUV420,
            3,
            &[
                (0, 0, 640),
                (0, 640 * 480, 320),
                (0, 640 * 480 + 320 * 240, 320),
            ],
            DRM_FORMAT_MOD_LINEAR,
        );
        let (layout, depth, planes, _) = planes_from_descriptor(&d, 640, 480).unwrap();
        assert_eq!(layout, DmaBufLayout::Planar);
        assert_eq!(depth, 8);
        assert_eq!(planes[0].unwrap().width, 640);
        assert_eq!(planes[1].unwrap().width, 320);
        assert_eq!(planes[2].unwrap().fourcc, DRM_FORMAT_R8);
    }

    // i965 VAAPI 把 Y/UV 导出成两个独立的单平面层（R8 + GR88），
    // 与 iHD 的单层 NV12 描述等价，必须都能归一化
    #[test]
    fn nv12_split_across_two_layers() {
        let mut d: ffi::AVDRMFrameDescriptor = unsafe { std::mem::zeroed() };
        d.nb_objects = 1;
        d.objects[0].fd = 9;
        d.objects[0].size = OBJ_SIZE;
        d.objects[0].format_modifier = 0x0100_0000_0000_0003;
        d.nb_layers = 2;
        d.layers[0].format = DRM_FORMAT_R8;
        d.layers[0].nb_planes = 1;
        d.layers[0].planes[0] = ffi::AVDRMPlaneDescriptor {
            object_index: 0,
            offset: 0,
            pitch: 1920,
        };
        d.layers[1].format = DRM_FORMAT_GR88;
        d.layers[1].nb_planes = 1;
        d.layers[1].planes[0] = ffi::AVDRMPlaneDescriptor {
            object_index: 0,
            offset: 1920 * 1080,
            pitch: 1920,
        };

        let (layout, depth, planes, _) = planes_from_descriptor(&d, 1920, 1080).unwrap();
        assert_eq!(layout, DmaBufLayout::SemiPlanar);
        assert_eq!(depth, 8);
        let y = planes[0].unwrap();
        assert_eq!((y.fourcc, y.width, y.height), (DRM_FORMAT_R8, 1920, 1080));
        let uv = planes[1].unwrap();
        assert_eq!(
            (uv.fourcc, uv.width, uv.height, uv.fd, uv.offset),
            (DRM_FORMAT_GR88, 960, 540, 9, 1920 * 1080)
        );
        assert_eq!(uv.modifier, 0x0100_0000_0000_0003);
    }

    // 未知格式 / 平面数不符 / 非法 object_index 都必须回退而不是 panic
    #[test]
    fn rejects_unsupported_or_inconsistent_descriptors() {
        let unknown = desc_with(0x1234_5678, 1, &[(0, 0, 640)], 0);
        assert!(planes_from_descriptor(&unknown, 640, 480).is_none());

        let wrong_count = desc_with(DRM_FORMAT_NV12, 1, &[(0, 0, 640)], 0);
        assert!(planes_from_descriptor(&wrong_count, 640, 480).is_none());

        let bad_object = desc_with(DRM_FORMAT_NV12, 2, &[(5, 0, 640), (0, 0, 640)], 0);
        assert!(planes_from_descriptor(&bad_object, 640, 480).is_none());

        let bad_pitch = desc_with(DRM_FORMAT_NV12, 2, &[(0, 0, 0), (0, 0, 640)], 0);
        assert!(planes_from_descriptor(&bad_pitch, 640, 480).is_none());
    }
}

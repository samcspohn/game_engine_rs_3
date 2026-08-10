//! Frame capture — the composited swapchain image, straight to a PNG.
//!
//! Captures the *swapchain* image and not the camera's colour attachment,
//! because the UI draws into the swapchain after the present-blit: the camera
//! image is the scene without any of the chrome, and still HDR.
//!
//! The copy rides the same `vkQueueSubmit2` as the frame, appended after the
//! primary so it reads the finished image before present. The host then waits
//! for that submission and encodes — a stall of one frame, on the frames a
//! human asked for a picture.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::format::Format;
use vulkano::image::Image;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};

use std::sync::Arc;

/// Requested captures, oldest first. A queue rather than a slot so a burst
/// asks for consecutive frames rather than overwriting itself.
static PENDING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// The last capture's outcome, for whoever asked.
static RESULT: Mutex<Option<Result<PathBuf, String>>> = Mutex::new(None);

/// Where captures go when no path is given. Under `target/` so it is already
/// ignored by git, and a fixed location an agent or a script can read.
pub fn shot_dir() -> PathBuf {
    std::env::var("ENGINE_SHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/shots"))
}

/// Ask for the next frame. Returns the path it will be written to.
pub fn request(path: Option<&str>) -> PathBuf {
    let path = match path {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => shot_dir().join(format!(
            "shot-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        )),
    };
    RESULT.lock().expect("capture result").take();
    PENDING.lock().expect("capture queue").push(path.clone());
    path
}

/// The outcome of the most recent capture, once the renderer has finished it.
pub fn result() -> Option<Result<PathBuf, String>> {
    RESULT.lock().expect("capture result").clone()
}

fn pending() -> Option<PathBuf> {
    let mut q = PENDING.lock().expect("capture queue");
    match q.is_empty() {
        true => None,
        false => Some(q.remove(0)),
    }
}

/// A capture in flight: the command buffer to submit with the frame, and the
/// buffer it will land in.
pub(crate) struct Capture {
    pub(crate) cb: Arc<PrimaryAutoCommandBuffer>,
    buffer: Subbuffer<[u8]>,
    extent: [u32; 3],
    format: Format,
    path: PathBuf,
}

/// Record a copy of `image` if a capture was requested, else `None`.
pub(crate) fn take(
    image: &Arc<Image>,
    memory: &Arc<StandardMemoryAllocator>,
    cbs: &Arc<StandardCommandBufferAllocator>,
    queue_family_index: u32,
) -> Option<Capture> {
    let path = pending()?;
    let extent = image.extent();
    let len = (extent[0] * extent[1] * 4) as u64;

    let buffer = Buffer::new_slice::<u8>(
        memory.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        len,
    )
    .expect("capture buffer");

    let mut b = AutoCommandBufferBuilder::primary(
        cbs.clone(),
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .expect("capture cb builder");
    b.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
        image.clone(),
        buffer.clone(),
    ))
    .expect("copy_image_to_buffer");

    Some(Capture {
        cb: b.build().expect("build capture cb"),
        buffer,
        extent,
        format: image.format(),
        path,
    })
}

impl Capture {
    /// Encode and write. Call once the submission carrying `cb` has retired.
    pub(crate) fn finish(self) {
        let outcome = self.write();
        if let Err(e) = &outcome {
            eprintln!("[capture] {e}");
        }
        *RESULT.lock().expect("capture result") = Some(outcome);
    }

    fn write(&self) -> Result<PathBuf, String> {
        let src = self
            .buffer
            .read()
            .map_err(|e| format!("capture buffer unreadable: {e}"))?;
        let (w, h) = (self.extent[0], self.extent[1]);

        // Whatever the surface handed us, already display-encoded — a capture
        // wants what the user sees, so there is no tonemapping or gamma here,
        // only unpacking. Unknown formats are an error rather than a guess:
        // the wrong unpack writes a plausible-looking image with wrong
        // colours, which is the hardest kind of bug to notice.
        let unpack: fn(&[u8]) -> [u8; 3] = match self.format {
            Format::R8G8B8A8_UNORM | Format::R8G8B8A8_SRGB => |p| [p[0], p[1], p[2]],
            Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB => |p| [p[2], p[1], p[0]],
            // 10 bits per channel packed into a `u32`, alpha in the top 2.
            Format::A2R10G10B10_UNORM_PACK32 => |p| {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                [ten(v >> 20), ten(v >> 10), ten(v)]
            },
            Format::A2B10G10R10_UNORM_PACK32 => |p| {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                [ten(v), ten(v >> 10), ten(v >> 20)]
            },
            f => return Err(format!("unsupported swapchain format {f:?}")),
        };

        let mut rgba = vec![255u8; src.len()];
        for (o, i) in rgba.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            o[..3].copy_from_slice(&unpack(i));
        }

        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        image::save_buffer(&self.path, &rgba, w, h, image::ColorType::Rgba8)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(absolute(&self.path))
    }
}

/// One 10-bit channel to 8, rounded. Truncating would bias every pixel down
/// by up to one level, which shows up the moment two captures are diffed.
fn ten(x: u32) -> u8 {
    (((x & 0x3FF) * 255 + 511) / 1023) as u8
}

fn absolute(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

//! Render-side camera: owns the GPU attachments (color + depth) a camera
//! renders into, plus the resolution policy that decides when those
//! attachments need to be re-created.
//!
//! This is distinct from [`crate::scene::Camera`], which is a pure
//! view/projection math helper. A `RenderCamera` *uses* a `scene::Camera`
//! (or any other source of `view_proj`) at draw time, but owns the GPU
//! resources independently.
//!
//! # Why this exists as its own type
//!
//! Different cameras want different resolution policies:
//!
//! - The **main camera** wants its color image to match the swapchain extent
//!   1:1 so the present-blit is a true 1:1 copy. Its attachments must be
//!   re-created every time the swapchain resizes.
//! - A **shadow map camera** wants a fixed offscreen size (e.g. 2048×2048)
//!   independent of the swapchain. Its attachments survive swapchain resize
//!   untouched.
//! - A **half-res reflection camera** wants to track the swapchain extent but
//!   at half resolution. Its attachments are re-created on swapchain resize
//!   too, but to a different size than the main camera.
//!
//! Today only [`CameraResolution::MatchSwapchain`] is implemented; the other
//! variants are placeholders for the multi-camera roadmap (`todo.txt` §3).
//! What matters now is that **the decision lives on the camera, not on the
//! swapchain code** — the swapchain handler just informs every camera of the
//! new extent and each one decides whether to rebuild its attachments.

use std::sync::Arc;

use vulkano::{
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

/// Pixel format used for camera-owned offscreen color targets.
///
/// HDR-capable (16-bit float per channel) so future tonemapping / bloom /
/// other post-process passes have headroom; the present-blit converts down
/// to whatever sRGB swapchain format the platform offers.
pub const CAMERA_COLOR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Pixel format used for camera-owned depth targets.
pub const CAMERA_DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// How a camera's attachment extent is determined relative to the swapchain.
///
/// This is a *policy* — the camera consults it on every swapchain-resize
/// event to decide whether (and to what size) to re-create its attachments.
#[derive(Clone, Copy, Debug)]
pub enum CameraResolution {
    /// Track the swapchain extent 1:1. Attachments are re-created on every
    /// swapchain resize. Used by the main camera so the present-blit stays a
    /// 1:1 copy.
    MatchSwapchain,
    // Reserved for future variants (`todo.txt` §3):
    //   Fixed { width: u32, height: u32 }
    //       — fixed offscreen size, survives swapchain resize. Shadow maps,
    //         editor thumbnails, render-to-texture for portals/mirrors.
    //   ScaleSwapchain { numerator: u32, denominator: u32 }
    //       — fraction of swapchain extent (e.g. 1/2 for half-res reflections).
}

impl CameraResolution {
    /// Resolve a swapchain extent into the actual extent this camera should
    /// render at, given its policy.
    fn resolve(&self, swapchain_extent: [u32; 2]) -> [u32; 2] {
        match self {
            CameraResolution::MatchSwapchain => swapchain_extent,
        }
    }

    /// Does this policy depend on the swapchain extent? `true` means a
    /// swapchain resize *might* require re-creating the camera's attachments;
    /// `false` means swapchain resizes are irrelevant to this camera.
    fn depends_on_swapchain(&self) -> bool {
        match self {
            CameraResolution::MatchSwapchain => true,
        }
    }
}

/// Render-side camera: owns the offscreen color + depth attachments the
/// camera renders into, plus the resolution policy that drives when they
/// have to be re-created.
///
/// Does **not** own the matrix staging/device buffers, descriptor sets, or
/// command buffers — those live on `FrameSlot` (per-swapchain-image) for
/// now. When multi-camera lands, each `RenderCamera` will grow its own
/// per-frame-in-flight matrix ring; today the single-camera setup uses the
/// FrameSlot's matrices directly.
pub struct RenderCamera {
    resolution:  CameraResolution,
    extent:      [u32; 2],
    color_image: Arc<Image>,
    depth_image: Arc<Image>,
    color_view:  Arc<ImageView>,
    depth_view:  Arc<ImageView>,
}

impl RenderCamera {
    /// Build a camera whose attachments track the swapchain extent.
    pub fn new_match_swapchain(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        swapchain_extent: [u32; 2],
    ) -> Self {
        Self::new(memory_allocator, CameraResolution::MatchSwapchain, swapchain_extent)
    }

    /// Build a camera with the given resolution policy. `swapchain_extent`
    /// is consulted only if the policy depends on it.
    pub fn new(
        memory_allocator: &Arc<StandardMemoryAllocator>,
        resolution:       CameraResolution,
        swapchain_extent: [u32; 2],
    ) -> Self {
        let extent = resolution.resolve(swapchain_extent);
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(memory_allocator, extent);
        RenderCamera {
            resolution,
            extent,
            color_image,
            depth_image,
            color_view,
            depth_view,
        }
    }

    /// Inform the camera that the swapchain has been re-created with a new
    /// extent. Returns `true` if the camera re-created its attachments
    /// (callers must then rebuild any command buffers that reference the
    /// camera's color/depth views).
    ///
    /// For [`CameraResolution::MatchSwapchain`] this re-creates whenever
    /// the extent actually changes; for swapchain-independent policies
    /// (future) this is a no-op.
    pub fn on_swapchain_resize(
        &mut self,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        new_swapchain_extent: [u32; 2],
    ) -> bool {
        if !self.resolution.depends_on_swapchain() {
            return false;
        }
        let new_extent = self.resolution.resolve(new_swapchain_extent);
        if new_extent == self.extent {
            return false;
        }
        let (color_image, color_view, depth_image, depth_view) =
            allocate_attachments(memory_allocator, new_extent);
        self.extent      = new_extent;
        self.color_image = color_image;
        self.color_view  = color_view;
        self.depth_image = depth_image;
        self.depth_view  = depth_view;
        true
    }

    pub fn extent(&self) -> [u32; 2] { self.extent }
    pub fn color_image(&self) -> &Arc<Image>     { &self.color_image }
    /// Held for future post-process passes / debug visualizers; not yet read
    /// by the current single-pass renderer.
    #[allow(dead_code)]
    pub fn depth_image(&self) -> &Arc<Image>     { &self.depth_image }
    pub fn color_view(&self)  -> &Arc<ImageView> { &self.color_view  }
    pub fn depth_view(&self)  -> &Arc<ImageView> { &self.depth_view  }
}

fn allocate_attachments(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    extent:           [u32; 2],
) -> (Arc<Image>, Arc<ImageView>, Arc<Image>, Arc<ImageView>) {
    let [w, h] = extent;
    let color_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format:     CAMERA_COLOR_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen color image");
    let color_view = ImageView::new_default(color_image.clone())
        .expect("Failed to create offscreen color image view");

    let depth_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format:     CAMERA_DEPTH_FORMAT,
            extent:     [w, h, 1],
            usage:      ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("Failed to create offscreen depth image");
    let depth_view = ImageView::new_default(depth_image.clone())
        .expect("Failed to create offscreen depth image view");

    (color_image, color_view, depth_image, depth_view)
}

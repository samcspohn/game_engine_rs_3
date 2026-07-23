//! Per-frame scene description consumed by the renderer.
//!
//! The renderer is intentionally agnostic about *how* transforms are
//! authored or animated — that is the game's (or editor's) responsibility.
//! It only needs three things each frame:
//!
//! 1. The set of `MeshRenderer` components in the scene (each pairs an entity
//!    with a mesh handle); the renderer derives its draw list from these.
//! 2. A way to read each entity's world transform — provided by the
//!    hierarchy itself.
//! 3. A [`Camera`] (or an [`OrbitController`] that produces one) to build
//!    the view + projection matrices.
//!
//! See `lib.rs` for how these plug into [`Window`](crate::Window).

use glam::{Mat4, Quat, Vec2, Vec3};

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

/// A perspective camera that produces view + projection matrices on demand.
///
/// All fields are `pub` so games can poke them directly. Use
/// [`Camera::view_proj`] to get a clip-space matrix ready for use as
/// `proj * view` (Vulkan Y-flipped).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    /// Sensible default: 60° FOV, looking at origin from `(1.5, 1.5, 2.5)`.
    pub fn default_at_origin() -> Self {
        Self {
            eye: Vec3::new(1.5, 1.5, 2.5),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    /// Right-handed view matrix.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// Vulkan-NDC projection (Y axis flipped from glam's GL convention).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        let mut p = Mat4::perspective_rh(
            self.fov_y_radians,
            aspect.max(1e-6),
            self.z_near,
            self.z_far,
        );
        p.y_axis.y *= -1.0;
        p
    }

    /// Convenience: combined `proj * view` for a given viewport aspect.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OrbitController
// ─────────────────────────────────────────────────────────────────────────────

/// Mouse-driven orbit camera controller.
///
/// * Left-button drag    → orbit (yaw + pitch around `target`).
/// * Right-button drag   → pan (translate `target` in screen plane).
/// * Scroll wheel        → zoom (multiply `distance`).
///
/// Pitch is clamped to (-π/2 + ε, π/2 − ε) to avoid the gimbal flip at the
/// poles. Distance is clamped to a sensible non-zero minimum.
///
/// The controller is fully event-driven; call [`feed_window_event`] for every
/// `WindowEvent` you receive, then read the live [`Camera`] via
/// [`OrbitController::camera`].
pub struct OrbitController {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,

    pub orbit_sensitivity: f32, // radians per pixel
    pub pan_sensitivity: f32,   // world units per pixel per unit distance
    pub zoom_sensitivity: f32,  // multiplicative per scroll line

    // Input state (private)
    last_cursor: Option<Vec2>,
    left_down: bool,
    right_down: bool,
}

impl OrbitController {
    /// Build a controller that frames the origin from a comfortable distance.
    pub fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.6,
            pitch: 0.4,
            distance: 3.5,
            up: Vec3::Y,
            fov_y_radians: 60_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
            orbit_sensitivity: 0.005,
            pan_sensitivity: 0.0015,
            zoom_sensitivity: 0.1,
            last_cursor: None,
            left_down: false,
            right_down: false,
        }
    }

    /// Feed a `winit` window event so the controller can update its state.
    /// Unrecognised events are ignored.
    pub fn feed_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = matches!(state, ElementState::Pressed);
                match button {
                    MouseButton::Left => self.left_down = pressed,
                    MouseButton::Right => self.right_down = pressed,
                    _ => {}
                }
                if !pressed && !self.left_down && !self.right_down {
                    // Reset cursor tracking on full release so the next drag
                    // starts cleanly.
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let cur = Vec2::new(position.x as f32, position.y as f32);
                if let Some(prev) = self.last_cursor {
                    let delta = cur - prev;
                    if self.left_down {
                        self.yaw -= delta.x * self.orbit_sensitivity;
                        self.pitch += delta.y * self.orbit_sensitivity;
                        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
                        self.pitch = self.pitch.clamp(-limit, limit);
                    } else if self.right_down {
                        // Pan in the camera's local right/up plane.
                        let (right, cam_up) = self.local_axes();
                        let scale = self.pan_sensitivity * self.distance;
                        self.target += right * delta.x * scale;
                        self.target += cam_up * delta.y * scale;
                    }
                }
                self.last_cursor = Some(cur);
            }
            WindowEvent::CursorLeft { .. } => {
                self.last_cursor = None;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
                };
                // Multiplicative zoom — exponential feel, never reaches 0.
                let factor = (1.0 - self.zoom_sensitivity * lines).max(0.1);
                self.distance = (self.distance * factor).clamp(0.05, 10_000.0);
            }
            _ => {}
        }
    }

    /// Compute camera-local right + up axes (used for panning).
    fn local_axes(&self) -> (Vec3, Vec3) {
        let cp = self.pitch.cos();
        let forward = Vec3::new(cp * self.yaw.sin(), self.pitch.sin(), cp * self.yaw.cos());
        let right = forward.cross(self.up).normalize_or_zero();
        let cam_up = right.cross(forward).normalize_or_zero();
        (right, cam_up)
    }

    /// Compute the camera's eye position from yaw/pitch/distance/target.
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let dir = Vec3::new(cp * self.yaw.sin(), self.pitch.sin(), cp * self.yaw.cos());
        self.target + dir * self.distance
    }

    /// Build a [`Camera`] reflecting the controller's current state.
    pub fn camera(&self) -> Camera {
        Camera {
            eye: self.eye(),
            target: self.target,
            up: self.up,
            fov_y_radians: self.fov_y_radians,
            z_near: self.z_near,
            z_far: self.z_far,
        }
    }
}

impl Default for OrbitController {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compose a TRS model matrix from position / rotation / scale (matches the
/// convention used by `TransformHierarchy::get_global_*`).
///
/// Reserved for CPU-side debug/test paths; the renderer hot path now
/// builds model matrices on the GPU in [`crate::shaders::mvp_build_cs`].
#[allow(dead_code)]
#[inline]
pub(crate) fn model_matrix(position: Vec3, rotation: Quat, scale: Vec3) -> Mat4 {
    Mat4::from_scale_rotation_translation(scale, rotation, position)
}

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
//! 3. A [`CameraComponent`] — attached to whichever entity is "the" camera —
//!    to build the view + projection matrices.
//!
//! See `lib.rs` for how these plug into [`Window`](crate::Window).
//!
//! # Camera as a component
//!
//! [`CameraComponent`] is deliberately dumb: it only turns a *position* and
//! *rotation* into view/projection matrices. It owns no movement logic of its
//! own, so it can be attached to any entity — a player, a detached editor rig,
//! a cutscene rail — and it will always just draw from wherever that entity's
//! transform currently is.
//!
//! Anything that should *move* the camera (or any other entity) is a
//! separate component that mutates the entity's [`Transform`] every frame,
//! reading input from the global [`crate::input`] accumulator.
//! [`OrbitController`] is the engine-provided example (used by the editor's
//! viewport); games are expected to write their own player-movement
//! components the same way.

use parking_lot::Mutex;

use glam::{Mat4, Quat, Vec3};

use engine_core::reflect::Export;
use engine_core::{Component, Entity, Transform, World, WorldId};

use crate::input::{self, MouseButton};

// ─────────────────────────────────────────────────────────────────────────────
// Viewport
// ─────────────────────────────────────────────────────────────────────────────

/// The on-screen box of the [`Viewport`](crate::ui::Viewport) widget showing
/// the scene, `[x, y, w, h]` in px — `None` while nothing shows it, which is
/// every game and the editor's first frame.
static VIEWPORT: Mutex<Option<[f32; 4]>> = Mutex::new(None);

/// Publish the widget's box. The renderer resizes the camera's attachments
/// to `w x h`, so the scene is rendered *at* the size it is shown at rather
/// than scaled into it.
///
/// Crate-internal: [`Viewport::update`](crate::ui::Viewport::update) is the
/// public way to say this, because a size nothing is drawing is a camera
/// rendering into a target nobody samples.
pub(crate) fn set_viewport(rect: Option<[f32; 4]>) {
    *VIEWPORT.lock() = rect;
}

/// Whether a window-space point is over the scene. Everywhere, until a
/// widget claims a box — a game's camera answers to the whole window.
pub fn in_viewport(p: [f32; 2]) -> bool {
    match *VIEWPORT.lock() {
        Some(r) => (0..2).all(|i| p[i] >= r[i] && p[i] < r[i] + r[i + 2]),
        None => true,
    }
}

/// The published box, raw. `None` means no widget has ever claimed the
/// scene — a game — and is the only state that means "the whole window".
///
/// A widget that is on screen but has *no* box right now (closed tab,
/// collapsed pane, first frame) publishes a zero rect, which is a third
/// thing: it owns no pointer, and the camera keeps the size it had rather
/// than putting two full re-allocations on a tab switch to render something
/// nobody can see.
pub(crate) fn viewport_box() -> Option<[f32; 4]> {
    *VIEWPORT.lock()
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraComponent
// ─────────────────────────────────────────────────────────────────────────────

/// A perspective camera. Attach to an entity via [`engine_core::World::add_component`]
/// — attaching publishes it as [`active_camera`], and the renderer reads that
/// entity's *global* position + rotation each frame to build the view matrix.
/// A scene with two cameras must name the one it means with
/// [`set_active_camera`]; attach order decides nothing worth relying on.
///
/// Deliberately holds no position/orientation of its own — the entity's
/// [`Transform`] is the single source of truth for where the camera is and
/// which way it's looking. Move it by attaching a controller component (see
/// the module docs) that mutates the transform, not by poking this struct.
#[derive(Clone, Copy, Debug, Export)]
pub struct CameraComponent {
    #[export]
    pub fov_y_radians: f32,
    #[export]
    pub z_near: f32,
    #[export]
    pub z_far: f32,
}

impl CameraComponent {
    /// Sensible default: 60° FOV, near/far `0.1`/`1000.0`.
    pub fn new() -> Self {
        Self {
            fov_y_radians: 60_f32.to_radians(),
            z_near: 0.1,
            z_far: 10_000.0,
        }
    }

    /// Right-handed view matrix looking down the entity's local `-Z` axis
    /// (i.e. `rotation` applied to `-Z` is "forward", `rotation` applied to
    /// `Y` is "up") from `position`.
    pub fn view(&self, position: Vec3, rotation: Quat) -> Mat4 {
        let forward = rotation * Vec3::NEG_Z;
        let up = rotation * Vec3::Y;
        Mat4::look_to_rh(position, forward, up)
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
    pub fn view_proj(&self, position: Vec3, rotation: Quat, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view(position, rotation)
    }
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for CameraComponent {
    // Pure data — the renderer reads it (+ the entity's transform) directly
    // each frame; it has no per-frame behavior of its own.
    const HAS_UPDATE: bool = false;

    /// So the common case — a game with one camera — never has to say which.
    fn init(&mut self, transform: &Transform) {
        set_active_camera(transform.world(), Entity::new(transform.get_idx()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Active camera
// ─────────────────────────────────────────────────────────────────────────────

/// The entity the renderer draws from, and the world it is in — a bare index
/// names nothing (ADR-0011 §2), and the editor's camera lives in a different
/// world from the document it looks at. `None` only before the first
/// [`CameraComponent`] is attached.
static ACTIVE_CAMERA: Mutex<Option<(WorldId, Entity)>> = Mutex::new(None);

/// Draw from `entity`'s [`CameraComponent`] from now on.
///
/// The editor's answer to owning a camera *and* showing a scene that has one:
/// which of the two is live is a mode, not an attach order.
pub fn set_active_camera(world: WorldId, entity: Entity) {
    *ACTIVE_CAMERA.lock() = Some((world, entity));
}

/// The entity currently drawn from, with its world.
pub fn active_camera() -> Option<(WorldId, Entity)> {
    *ACTIVE_CAMERA.lock()
}

// ─────────────────────────────────────────────────────────────────────────────
// OrbitController
// ─────────────────────────────────────────────────────────────────────────────

/// Mouse-driven orbit camera controller component.
///
/// * Left-button drag    → orbit (yaw + pitch around `target`).
/// * Right-button drag   → pan (translate `target` in screen plane).
/// * Scroll wheel        → zoom (multiply `distance`).
///
/// Pitch is clamped to (-π/2 + ε, π/2 − ε) to avoid the gimbal flip at the
/// poles. Distance is clamped to a sensible non-zero minimum.
///
/// Reads the global [`crate::input`] accumulator every `update` and writes
/// the resulting eye position + look-at rotation into the entity's
/// [`Transform`] — attach a [`CameraComponent`] to the same entity to
/// actually render from it. This is the editor's example "player controller"
/// equivalent: games needing different movement should write their own
/// component following the same pattern (read `input::*`, mutate
/// `transform`).
#[derive(Clone, Export)]
pub struct OrbitController {
    #[export]
    pub target: Vec3,
    #[export]
    pub yaw: f32,
    #[export]
    pub pitch: f32,
    #[export]
    pub distance: f32,
    #[export]
    pub up: Vec3,

    #[export]
    pub orbit_sensitivity: f32, // radians per pixel
    #[export]
    pub pan_sensitivity: f32, // world units per pixel per unit distance
    #[export]
    pub zoom_sensitivity: f32, // multiplicative per scroll line

    /// Whether the button now down was pressed over the viewport. A gesture
    /// belongs to where it began, so a drag that leaves the panel keeps
    /// orbiting instead of stopping at the edge.
    dragging: bool,
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
            orbit_sensitivity: 0.005,
            pan_sensitivity: 0.0015,
            zoom_sensitivity: 0.1,
            dragging: false,
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
}

impl Default for OrbitController {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for OrbitController {
    fn update(&mut self, _dt: f32, transform: &Transform, _w: &World) {
        let inp = input::global();
        let delta = inp.cursor_delta();
        // The UI gets first refusal on the pointer, so clicking a button
        // doesn't also spin the camera and scrolling over a panel doesn't
        // zoom. Hit testing ran before `worlds::sweep_all` precisely so this read
        // is available here. The transform write below still runs — the
        // camera keeps tracking its target while the UI holds the mouse.
        // The viewport is the second half of the same question: a camera that
        // draws into one panel must not answer a drag started in another.
        let mine = !crate::ui::ui().pointer_captured() && in_viewport(inp.cursor_position().into());
        for b in [MouseButton::Left, MouseButton::Right] {
            if inp.mouse_pressed(b) {
                self.dragging = mine;
            }
        }
        if self.dragging {
            if inp.mouse_down(MouseButton::Left) {
                self.yaw -= delta.x * self.orbit_sensitivity;
                self.pitch += delta.y * self.orbit_sensitivity;
                let limit = std::f32::consts::FRAC_PI_2 - 0.01;
                self.pitch = self.pitch.clamp(-limit, limit);
            } else if inp.mouse_down(MouseButton::Right) {
                let (right, cam_up) = self.local_axes();
                let scale = self.pan_sensitivity * self.distance;
                self.target += right * delta.x * scale;
                self.target += cam_up * delta.y * scale;
            }
        }
        let scroll = inp.scroll_delta();
        if mine && scroll != 0.0 {
            let factor = (1.0 - self.zoom_sensitivity * scroll).max(0.1);
            self.distance = (self.distance * factor).clamp(0.05, 10_000.0);
        }

        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        // `Quat::look_to_rh(dir, up)` builds the *view* rotation (world →
        // camera space) — a `Transform`'s rotation is the opposite sense
        // (camera → world, i.e. "which way is local -Z pointing in world
        // space"), so it must be inverted. Skipping the inverse still
        // *looks* plausible at rest but scrambles yaw/pitch into each other
        // as soon as the camera moves, since the two rotations only agree
        // at identity.
        let rotation = Quat::look_to_rh(forward, self.up).inverse();

        let guard = transform.lock();
        guard.set_position(eye);
        guard.set_rotation(rotation);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The three states the box has to tell apart: a widget with a box, a
    /// widget without one, and no widget at all. Only the last means the
    /// camera owns the whole window — a closed viewport panel owns *no*
    /// pointer, which is not the same as owning every pointer.
    #[test]
    fn a_viewport_with_no_box_is_not_the_same_as_no_viewport() {
        set_viewport(Some([10.0, 20.0, 30.0, 40.0]));
        assert_eq!(viewport_box(), Some([10.0, 20.0, 30.0, 40.0]));
        assert!(in_viewport([11.0, 21.0]) && !in_viewport([9.0, 21.0]));
        assert!(!in_viewport([40.0, 60.0]), "the far edge is outside");

        set_viewport(Some([10.0, 20.0, 0.0, 0.0]));
        assert!(!in_viewport([10.0, 20.0]), "a closed panel owns nothing");

        set_viewport(None);
        assert!(in_viewport([0.0, 0.0]), "nothing claimed it: the whole window");
    }
}

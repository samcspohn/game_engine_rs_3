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
//! 3. A [`CameraHandle`] holding a `view_proj`, which is what a
//!    [`CameraComponent`] drives from its entity's pose.
//!
//! See `lib.rs` for how these plug into [`Window`](crate::Window), and
//! `docs/ADR-0011-worlds.md` §4 for who is allowed to write a camera when.

use glam::{Quat, Vec3};

use engine_core::reflect::Export;
use engine_core::{Component, Entity, Transform, World};

use crate::camera::{self, CameraHandle};
use crate::input::{self, MouseButton};

// ─────────────────────────────────────────────────────────────────────────────
// CameraComponent
// ─────────────────────────────────────────────────────────────────────────────

/// A perspective camera, driven by the entity it is attached to.
///
/// Attaching mints a [`CameraHandle`] bound to the world the entity was
/// spawned in, and a post-frame pass feeds that camera the entity's *global*
/// pose once the sweep has settled. Move it by mutating the transform — a
/// controller component, an animation, a parent — never by poking this.
///
/// An editor that owns its cameras outright skips this and writes the handle
/// directly; see [`OrbitController::for_camera`].
#[derive(Clone, Export)]
pub struct CameraComponent {
    #[export]
    pub fov_y_radians: f32,
    #[export]
    pub z_near: f32,
    #[export]
    pub z_far: f32,
    /// Minted by [`Component::init`] — there is no camera to hold before the
    /// component knows which world it landed in.
    camera: Option<CameraHandle>,
}

impl CameraComponent {
    /// Sensible default: 60° FOV, near/far `0.1`/`10000.0`.
    pub fn new() -> Self {
        Self {
            fov_y_radians: 60_f32.to_radians(),
            z_near: 0.1,
            z_far: 10_000.0,
            camera: None,
        }
    }

    /// The camera this drives, once it has been attached to something.
    pub fn camera(&self) -> Option<&CameraHandle> {
        self.camera.as_ref()
    }
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for CameraComponent {
    // The post-frame pass feeds the camera; running here would sample a pose
    // other components in the same sweep may still change.
    const HAS_UPDATE: bool = false;

    fn init(&mut self, transform: &Transform) {
        let camera = CameraHandle::new(transform.world());
        camera.set_projection(self.fov_y_radians, self.z_near, self.z_far);
        camera::bind(
            transform.world(),
            Entity::new(transform.get_idx()),
            camera.clone(),
        );
        self.camera = Some(camera);
    }

    fn deinit(&mut self, transform: &Transform) {
        camera::unbind(transform.world(), Entity::new(transform.get_idx()));
    }
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

    /// Whether the button now down was pressed over this camera's panel. A
    /// gesture belongs to where it began, so a drag that leaves the panel
    /// keeps orbiting instead of stopping at the edge.
    dragging: bool,
    /// The camera whose panel bounds the gestures this answers to. Given
    /// outright by [`for_camera`](Self::for_camera), or adopted from a
    /// [`CameraComponent`] on the same entity.
    camera: Option<CameraHandle>,
    /// Whether this drives `camera`'s matrix itself. A controller paired with
    /// a `CameraComponent` must not: two writers per frame is last-write-wins.
    drives: bool,
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
            camera: None,
            drives: false,
        }
    }

    /// The same, driving `camera` directly — no `CameraComponent`, and the
    /// entity's transform is along for the ride rather than the source.
    ///
    /// The editor's shape: it owns its cameras and points them at documents
    /// its rig is not part of.
    pub fn for_camera(camera: CameraHandle) -> Self {
        Self {
            camera: Some(camera),
            drives: true,
            ..Self::new()
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
    fn update(&mut self, _dt: f32, transform: &Transform, w: &World) {
        // A `CameraComponent` beside this one already owns a camera; adopt it
        // for the panel test rather than asking the app to wire it up twice.
        if self.camera.is_none() {
            self.camera = w
                .get_component::<CameraComponent>(Entity::new(transform.get_idx()))
                .and_then(|c| c.lock().camera.clone());
        }
        let inp = input::global();
        let delta = inp.cursor_delta();
        // The UI gets first refusal on the pointer, so clicking a button
        // doesn't also spin the camera and scrolling over a panel doesn't
        // zoom. Hit testing ran before `worlds::sweep_all` precisely so this
        // read is available here. The transform write below still runs — the
        // camera keeps tracking its target while the UI holds the mouse.
        // The panel is the second half of the same question: a camera shown
        // in one pane must not answer a drag started in another.
        let mine = !crate::ui::ui().pointer_captured()
            && self
                .camera
                .as_ref()
                .is_none_or(|c| c.contains(inp.cursor_position().into()));
        for b in [MouseButton::Left, MouseButton::Right] {
            if inp.mouse_pressed(b) {
                // The gizmo is the third claimant on the pointer, and it has
                // already decided by the time any component runs — see
                // `gizmo::update`. Only the *drag* defers to it: the wheel
                // must still zoom with the cursor over a handle, which is
                // where it sits for most of an edit.
                self.dragging = mine && !crate::gizmo::captures_pointer();
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
        drop(guard);

        // Safe from here and not from a `CameraComponent`: this matrix comes
        // from state this component alone owns, so no other component in the
        // sweep can invalidate it.
        if let Some(camera) = self.camera.as_ref().filter(|_| self.drives) {
            camera.set_from_trs(eye, rotation);
        }
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
pub(crate) fn model_matrix(position: Vec3, rotation: Quat, scale: Vec3) -> glam::Mat4 {
    glam::Mat4::from_scale_rotation_translation(scale, rotation, position)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three states the box has to tell apart: a panel with a box, a
    /// panel without one, and no panel at all. Only the last means the
    /// camera owns the whole window — a closed panel owns *no* pointer,
    /// which is not the same as owning every pointer.
    #[test]
    fn a_camera_with_no_box_is_not_the_same_as_a_camera_nothing_shows() {
        let cam = CameraHandle::new(0);
        assert!(cam.contains([0.0, 0.0]), "nothing shows it: the whole window");

        cam.set_rect(Some([10.0, 20.0, 30.0, 40.0]));
        assert!(cam.contains([11.0, 21.0]) && !cam.contains([9.0, 21.0]));
        assert!(!cam.contains([40.0, 60.0]), "the far edge is outside");

        cam.set_rect(Some([0.0; 4]));
        assert!(!cam.contains([0.0, 0.0]), "a closed panel owns nothing");
    }

    /// Two documents side by side: each camera answers only for its own box,
    /// and each names the world it draws.
    #[test]
    fn two_cameras_split_the_window() {
        let left = CameraHandle::new(7);
        let right = CameraHandle::new(9);
        left.set_rect(Some([0.0, 0.0, 400.0, 600.0]));
        right.set_rect(Some([400.0, 0.0, 400.0, 600.0]));

        assert!(left.contains([100.0, 300.0]) && !right.contains([100.0, 300.0]));
        assert!(right.contains([500.0, 300.0]) && !left.contains([500.0, 300.0]));
        assert_eq!((left.worlds(), right.worlds()), (vec![7], vec![9]), "its own world");
        assert_ne!(left.slot(), right.slot(), "and its own bindless slot");
    }

    /// A camera composites the worlds it lists, in order, into one image —
    /// the gizmos-over-document case. Listing one twice would z-fight it
    /// against itself, so a repeat is dropped.
    #[test]
    fn a_camera_draws_the_worlds_it_is_given_in_order() {
        let cam = CameraHandle::new(3);
        cam.draw_world(5);
        cam.draw_world(3);
        assert_eq!(cam.worlds(), vec![3, 5], "spawned-in world first, no repeat");
    }
}

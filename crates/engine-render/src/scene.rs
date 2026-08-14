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

/// How many viewports a process can show at once. Each costs a camera with
/// its own attachments and Hi-Z pyramid (ADR-0005), plus one reserved
/// bindless slot — so this is a small number on purpose.
pub const MAX_VIEWPORTS: usize = 4;

/// Which viewport. An index into the registry, handed out by
/// [`add_viewport`]; viewport 0 exists from the start, which is the game
/// case — one camera, the whole window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ViewportId(pub usize);

/// The default viewport: what a game gets without asking, and what
/// [`set_active_camera`] names.
pub const MAIN_VIEWPORT: ViewportId = ViewportId(0);

/// One viewport: a box on screen, the world it shows, and the camera it
/// shows it through.
#[derive(Clone, Copy, Default)]
struct Slot {
    /// `None` while no widget has claimed a box — a game, whose camera
    /// answers to the whole window. A zero rect is a third thing: on screen
    /// but with no box right now (closed tab, collapsed pane, first frame).
    rect: Option<[f32; 4]>,
    /// The world drawn here. `None` is the window's first world, which is
    /// what a game means without saying it.
    shows: Option<WorldId>,
    /// The entity holding the [`CameraComponent`], and the world it lives
    /// in — the editor's rig is not the world it is looking at.
    camera: Option<(WorldId, Entity)>,
    /// Whether [`add_viewport`] claimed this slot. Slot 0 exists for the game
    /// that never asks, so the first explicit caller takes it rather than
    /// leaving a camera nobody looks through.
    claimed: bool,
}

static VIEWPORTS: Mutex<Vec<Slot>> = Mutex::new(Vec::new());

/// Run `f` against the registry, which always has at least [`MAIN_VIEWPORT`].
fn with_slots<R>(f: impl FnOnce(&mut Vec<Slot>) -> R) -> R {
    let mut v = VIEWPORTS.lock();
    if v.is_empty() {
        v.push(Slot::default());
    }
    f(&mut v)
}

/// Register a viewport showing `shows` through `camera`, and return its id.
///
/// Two documents side by side is two of these. The camera's own world is
/// carried beside its entity because an index means nothing without it
/// (ADR-0011 §2), and it is routinely a different world from `shows`.
pub fn add_viewport(shows: WorldId, camera: (WorldId, Entity)) -> ViewportId {
    with_slots(|v| {
        let slot = Slot {
            rect: None,
            shows: Some(shows),
            camera: Some(camera),
            claimed: true,
        };
        if !v[MAIN_VIEWPORT.0].claimed {
            v[MAIN_VIEWPORT.0] = slot;
            return MAIN_VIEWPORT;
        }
        assert!(v.len() < MAX_VIEWPORTS, "at most {MAX_VIEWPORTS} viewports");
        v.push(slot);
        ViewportId(v.len() - 1)
    })
}

/// How many viewports exist. At least one.
pub fn viewport_count() -> usize {
    with_slots(|v| v.len())
}

/// Publish a widget's box. The renderer resizes that viewport's camera
/// attachments to `w x h`, so the scene is rendered *at* the size it is shown
/// at rather than scaled into it.
///
/// Crate-internal: [`Viewport::update`](crate::ui::Viewport::update) is the
/// public way to say this, because a size nothing is drawing is a camera
/// rendering into a target nobody samples.
pub(crate) fn set_viewport(id: ViewportId, rect: Option<[f32; 4]>) {
    with_slots(|v| {
        if let Some(slot) = v.get_mut(id.0) {
            slot.rect = rect;
        }
    });
}

/// Which viewport a window-space point is over, if any.
///
/// A viewport that has never published a box owns the whole window — a game.
/// A zero box owns nothing.
pub fn viewport_at(p: [f32; 2]) -> Option<ViewportId> {
    with_slots(|v| {
        v.iter().enumerate().find_map(|(i, s)| match s.rect {
            Some(r) => ((0..2).all(|k| p[k] >= r[k] && p[k] < r[k] + r[k + 2]))
                .then_some(ViewportId(i)),
            None => Some(ViewportId(i)),
        })
    })
}

/// Whether a point is over any viewport at all.
pub fn in_viewport(p: [f32; 2]) -> bool {
    viewport_at(p).is_some()
}

/// The published box, raw. See [`Slot::rect`] for what each state means.
pub(crate) fn viewport_box(id: ViewportId) -> Option<[f32; 4]> {
    with_slots(|v| v.get(id.0).and_then(|s| s.rect))
}

/// The world this viewport draws, and the camera it draws it through.
pub(crate) fn viewport_camera(id: ViewportId) -> (Option<WorldId>, Option<(WorldId, Entity)>) {
    with_slots(|v| v.get(id.0).map_or((None, None), |s| (s.shows, s.camera)))
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
        claim_main_viewport(transform.world(), Entity::new(transform.get_idx()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Active camera
// ─────────────────────────────────────────────────────────────────────────────

/// The entity the renderer draws from, and the world it is in — a bare index
/// names nothing (ADR-0011 §2), and the editor's camera lives in a different
/// world from the document it looks at. `None` only before the first
/// [`CameraComponent`] is attached.


/// Draw from `entity`'s [`CameraComponent`] from now on.
///
/// The editor's answer to owning a camera *and* showing a scene that has one:
/// which of the two is live is a mode, not an attach order.
pub fn set_active_camera(world: WorldId, entity: Entity) {
    with_slots(|v| v[MAIN_VIEWPORT.0].camera = Some((world, entity)));
}

/// Attaching a [`CameraComponent`] says "draw from me" — but only when nobody
/// has said otherwise. A viewport registered through [`add_viewport`] names
/// its camera explicitly, and the next camera attached anywhere in the process
/// must not silently take that viewport over.
fn claim_main_viewport(world: WorldId, entity: Entity) {
    with_slots(|v| {
        if !v[MAIN_VIEWPORT.0].claimed {
            v[MAIN_VIEWPORT.0].camera = Some((world, entity));
        }
    });
}

/// The entity the main viewport draws from, with its world.
pub fn active_camera() -> Option<(WorldId, Entity)> {
    with_slots(|v| v[MAIN_VIEWPORT.0].camera)
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
    /// Which viewport this camera draws into, so a drag in one panel does
    /// not spin the camera in the one beside it. `None` answers to any of
    /// them, which is a game: one camera, the whole window.
    viewport: Option<ViewportId>,
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
            viewport: None,
        }
    }

    /// The same, answering only to drags in `viewport` — what keeps two
    /// documents side by side from orbiting together.
    pub fn for_viewport(id: ViewportId) -> Self {
        Self {
            viewport: Some(id),
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
        let over = viewport_at(inp.cursor_position().into());
        let mine = !crate::ui::ui().pointer_captured()
            && over.is_some()
            && self.viewport.is_none_or(|id| over == Some(id));
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
        set_viewport(MAIN_VIEWPORT, Some([10.0, 20.0, 30.0, 40.0]));
        assert_eq!(viewport_box(MAIN_VIEWPORT), Some([10.0, 20.0, 30.0, 40.0]));
        assert!(in_viewport([11.0, 21.0]) && !in_viewport([9.0, 21.0]));
        assert!(!in_viewport([40.0, 60.0]), "the far edge is outside");

        set_viewport(MAIN_VIEWPORT, Some([10.0, 20.0, 0.0, 0.0]));
        assert!(!in_viewport([10.0, 20.0]), "a closed panel owns nothing");

        set_viewport(MAIN_VIEWPORT, None);
        assert!(in_viewport([0.0, 0.0]), "nothing claimed it: the whole window");
    }

    /// Two documents side by side: the point picks the one it is over, and
    /// each names its own world and its own camera.
    #[test]
    fn two_viewports_split_the_window() {
        let left = add_viewport(0, (2, Entity::new(1)));
        let right = add_viewport(1, (2, Entity::new(2)));
        set_viewport(MAIN_VIEWPORT, Some([0.0; 4]));
        set_viewport(left, Some([0.0, 0.0, 400.0, 600.0]));
        set_viewport(right, Some([400.0, 0.0, 400.0, 600.0]));

        assert_eq!(viewport_at([100.0, 300.0]), Some(left));
        assert_eq!(viewport_at([500.0, 300.0]), Some(right));
        assert_eq!(viewport_camera(right).0, Some(1), "its own world");
        assert_eq!(
            viewport_camera(right).1,
            Some((2, Entity::new(2))),
            "and its own camera, in the world that holds it"
        );

        with_slots(|v| v.truncate(1));
        set_viewport(MAIN_VIEWPORT, None);
    }
}

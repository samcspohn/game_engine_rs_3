//! The TRS gizmo: move, rotate and scale the selected entity by dragging a
//! handle in the viewport.
//!
//! Driven by the renderer between the pointer update and the sweep, not by a
//! component. A component would race `OrbitController` — both answer to the
//! same press, and whichever ran second would already have decided. Running
//! here means [`captures_pointer`] is settled before any component looks.
//!
//! The editor only says *what* to aim at ([`set_target`]); which handle is
//! under the cursor, what a drag means, and the triangles that show it are
//! all here. Geometry goes out through [`crate::overlay`].
//!
//! See `docs/notes/gizmo.md` for the handle layout and the drag maths.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use glam::{Mat3, Quat, Vec2, Vec3, Vec4Swizzles};
use parking_lot::Mutex;

use engine_core::transform::{TransformHierarchy, WorldId};
use engine_core::{worlds, Entity};

use crate::camera::{camera, CameraHandle, MAX_CAMERAS};
use crate::input::{self, MouseButton};
use crate::overlay::{self, OverlayVertex};

/// What a drag does to the transform under the gizmo.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which handle: one of the three axes, one of the three planes (named by
/// the axis they are normal to), or the uniform centre.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Handle {
    Axis(usize),
    Plane(usize),
    Uniform,
}

/// Fraction of the viewport height the gizmo covers, at any distance.
const SCREEN_FRACTION: f32 = 0.16;
/// How close the cursor has to be to a handle's projection, in pixels.
const PICK_PX: f32 = 9.0;
const AXIS_COLORS: [[f32; 4]; 3] = [
    [0.90, 0.25, 0.28, 1.0],
    [0.35, 0.85, 0.32, 1.0],
    [0.25, 0.50, 0.95, 1.0],
];
const HIGHLIGHT: [f32; 4] = [1.0, 0.85, 0.20, 1.0];

/// The entity each camera's gizmo acts on. Indexed by camera slot, so two
/// documents side by side each get their own.
static TARGETS: Mutex<Vec<Option<(WorldId, Entity)>>> = Mutex::new(Vec::new());
static MODE: AtomicU8 = AtomicU8::new(0);
static CAPTURED: AtomicBool = AtomicBool::new(false);
static DRAG: Mutex<Option<Drag>> = Mutex::new(None);

/// A gesture in flight. Every anchor is taken at the press and the transform
/// is written as an absolute function of them, so a drag cannot accumulate
/// drift from its own output.
struct Drag {
    camera_slot: usize,
    handle: Handle,
    origin: Vec3,
    axes: Mat3,
    /// Local delta → world delta for the target's parent chain.
    basis: Mat3,
    start_position: Vec3,
    start_rotation: Quat,
    start_scale: Vec3,
    /// Position along the axis, angle around the ring, or the plane hit —
    /// whichever this handle's maths anchors on.
    param: f32,
    point: Vec3,
}

/// Aim `camera`'s gizmo at `entity` in `world`, or at nothing.
///
/// By id and not by handle: the editor holds the world alive, and a gizmo
/// pointed at a world that has gone away should vanish rather than keep it.
pub fn set_target(camera: &CameraHandle, world: WorldId, entity: Option<Entity>) {
    let mut targets = TARGETS.lock();
    if targets.is_empty() {
        targets.resize(MAX_CAMERAS, None);
    }
    targets[camera.slot()] = entity.map(|e| (world, e));
}

pub fn mode() -> GizmoMode {
    match MODE.load(Ordering::Relaxed) {
        1 => GizmoMode::Rotate,
        2 => GizmoMode::Scale,
        _ => GizmoMode::Translate,
    }
}

pub fn set_mode(mode: GizmoMode) {
    MODE.store(mode as u8, Ordering::Relaxed);
}

/// Whether the gizmo owns the pointer — hovering a handle or mid-drag. What
/// keeps grabbing an arrow from also spinning the camera.
pub fn captures_pointer() -> bool {
    CAPTURED.load(Ordering::Relaxed)
}

/// Hit-test, drag and re-draw every camera's gizmo. Called once per frame
/// from the renderer, after the UI has had the pointer and before the sweep.
pub(crate) fn update() {
    let targets = TARGETS.lock().clone();
    if targets.is_empty() {
        return;
    }
    let inp = input::global();
    let ui_has_pointer = crate::ui::ui().pointer_captured();
    let cursor = inp.cursor_position();
    let mut drag = DRAG.lock();
    if !inp.mouse_down(MouseButton::Left) {
        *drag = None;
    }
    let mut captured = drag.is_some();

    for (slot, target) in targets.iter().enumerate() {
        let Some(cam) = camera(slot) else { continue };
        let Some((world, entity)) = *target else {
            overlay::publish(slot, Vec::new());
            continue;
        };
        let Some(world) = worlds::world(world) else {
            overlay::publish(slot, Vec::new());
            continue;
        };
        let h = world.hierarchy();
        let Some(t) = h.get_transform(entity.id) else {
            overlay::publish(slot, Vec::new());
            continue;
        };
        let mode = mode();
        let (origin, axes, basis, local) = {
            let g = t.lock();
            // Scale handles are the entity's own axes — scale has no meaning
            // in any other frame. Move and turn are in world axes, which is
            // what "along X" means to someone dragging.
            let axes = match mode {
                GizmoMode::Scale => Mat3::from_quat(g.get_global_rotation()),
                _ => Mat3::IDENTITY,
            };
            (
                g.get_global_position(),
                axes,
                parent_basis(h, &g),
                (g.get_position(), g.get_rotation(), g.get_scale()),
            )
        };

        let Some(rect) = cam.rect().filter(|r| r[2] > 0.0 && r[3] > 0.0) else {
            overlay::publish(slot, Vec::new());
            continue;
        };
        let (view_proj, eye) = cam.view_proj();
        // The world height the panel covers at the gizmo's distance, times
        // the slice of it the gizmo is meant to fill.
        let visible = 2.0 * (origin - eye).length() * (cam.fov_y() * 0.5).tan();
        let size = visible * SCREEN_FRACTION;
        let ray = ray_through(view_proj, rect, cursor);

        // A drag belongs to the camera it began in, so a second viewport
        // neither steals it nor answers to it.
        let active = match drag.as_ref() {
            Some(d) if d.camera_slot == slot => Some(d.handle),
            Some(_) => {
                overlay::publish(slot, geometry(mode, origin, axes, size, None));
                continue;
            }
            None => None,
        };
        let hovered = match (active, ray) {
            (Some(handle), _) => Some(handle),
            (None, Some(ray)) if !ui_has_pointer && cam.contains(cursor.into()) => {
                pick(mode, origin, axes, size, view_proj, rect, cursor, ray)
            }
            _ => None,
        };
        captured |= hovered.is_some();

        match (active, hovered, ray) {
            (None, Some(handle), Some(ray)) if inp.mouse_pressed(MouseButton::Left) => {
                let (param, point) = anchor(mode, handle, origin, axes, size, ray);
                *drag = Some(Drag {
                    camera_slot: slot,
                    handle,
                    origin,
                    axes,
                    basis,
                    start_position: local.0,
                    start_rotation: local.1,
                    start_scale: local.2,
                    param,
                    point,
                });
            }
            (Some(_), _, Some(ray)) => {
                let d = drag.as_ref().expect("an active handle is a live drag");
                apply(mode, d, ray, size, &t.lock());
            }
            _ => {}
        }

        overlay::publish(slot, geometry(mode, origin, axes, size, hovered));
    }
    CAPTURED.store(captured, Ordering::Relaxed);
}

/// Local delta → world delta for everything above `t`, composed exactly the
/// way `get_global_position` composes it (scale after rotation, per level).
fn parent_basis(h: &TransformHierarchy, t: &engine_core::transform::TransformGuard<'_>) -> Mat3 {
    let mut chain = Vec::new();
    let mut parent = t.get_parent();
    while let Some(p) = parent {
        let g = h.get_transform_unchecked(p).lock();
        chain.push((g.get_rotation(), g.get_scale()));
        parent = g.get_parent();
    }
    let map = |mut v: Vec3| {
        for (r, s) in &chain {
            v = (*r * v) * *s;
        }
        v
    };
    Mat3::from_cols(map(Vec3::X), map(Vec3::Y), map(Vec3::Z))
}

// ─────────────────────────────────────────────────────────────────────────────
// Picking
// ─────────────────────────────────────────────────────────────────────────────

/// A world-space ray through the cursor, or `None` when the cursor is not
/// over a camera whose matrix can be inverted yet.
fn ray_through(view_proj: glam::Mat4, rect: [f32; 4], cursor: Vec2) -> Option<(Vec3, Vec3)> {
    let inv = view_proj.inverse();
    if !inv.is_finite() {
        return None;
    }
    let ndc = Vec2::new(
        (cursor.x - rect[0]) / rect[2] * 2.0 - 1.0,
        (cursor.y - rect[1]) / rect[3] * 2.0 - 1.0,
    );
    let unproject = |z: f32| {
        let p = inv * glam::Vec4::new(ndc.x, ndc.y, z, 1.0);
        p.xyz() / p.w
    };
    let (near, far) = (unproject(0.0), unproject(1.0));
    Some((near, (far - near).normalize_or_zero()))
}

/// Where `p` lands in window pixels, or `None` behind the eye.
fn project(view_proj: glam::Mat4, rect: [f32; 4], p: Vec3) -> Option<Vec2> {
    let c = view_proj * p.extend(1.0);
    (c.w > 1e-6).then(|| {
        let ndc = c.xy() / c.w;
        Vec2::new(
            rect[0] + (ndc.x * 0.5 + 0.5) * rect[2],
            rect[1] + (ndc.y * 0.5 + 0.5) * rect[3],
        )
    })
}

fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = (p - a).dot(ab) / ab.length_squared().max(1e-9);
    (p - (a + ab * t.clamp(0.0, 1.0))).length()
}

/// The parameter along `axis` of the point on it closest to the ray, or
/// `None` when the two are within a degree of parallel and the answer would
/// be noise.
fn closest_on_axis(ray: (Vec3, Vec3), origin: Vec3, axis: Vec3) -> Option<f32> {
    let w = ray.0 - origin;
    let b = ray.1.dot(axis);
    let denom = 1.0 - b * b;
    (denom.abs() > 1e-3).then(|| (axis.dot(w) - b * ray.1.dot(w)) / denom)
}

fn ray_plane(ray: (Vec3, Vec3), point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denom = ray.1.dot(normal);
    (denom.abs() > 1e-4)
        .then(|| (point - ray.0).dot(normal) / denom)
        .filter(|t| *t > 0.0)
        .map(|t| ray.0 + ray.1 * t)
}

/// The handle under the cursor, most specific first: a plane quad sits
/// between two axes and would otherwise never win.
#[allow(clippy::too_many_arguments)]
fn pick(
    mode: GizmoMode,
    origin: Vec3,
    axes: Mat3,
    size: f32,
    view_proj: glam::Mat4,
    rect: [f32; 4],
    cursor: Vec2,
    ray: (Vec3, Vec3),
) -> Option<Handle> {
    let axis = |i: usize| axes.col(i).normalize_or_zero();

    if mode == GizmoMode::Rotate {
        return (0..3)
            .map(|i| {
                (
                    i,
                    ring_distance(origin, axis(i), size * RING, view_proj, rect, cursor),
                )
            })
            .filter(|(_, d)| *d < PICK_PX)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| Handle::Axis(i));
    }

    if mode == GizmoMode::Scale
        && project(view_proj, rect, origin).is_some_and(|p| (p - cursor).length() < PICK_PX)
    {
        return Some(Handle::Uniform);
    }

    if mode == GizmoMode::Translate {
        for i in 0..3 {
            let (u, v) = (axis((i + 1) % 3), axis((i + 2) % 3));
            let Some(hit) = ray_plane(ray, origin, axis(i)) else {
                continue;
            };
            let local = hit - origin;
            let (a, b) = (local.dot(u) / size, local.dot(v) / size);
            if (PLANE_NEAR..PLANE_FAR).contains(&a) && (PLANE_NEAR..PLANE_FAR).contains(&b) {
                return Some(Handle::Plane(i));
            }
        }
    }

    (0..3)
        .filter_map(|i| {
            let a = project(view_proj, rect, origin)?;
            let b = project(view_proj, rect, origin + axis(i) * size)?;
            Some((i, distance_to_segment(cursor, a, b)))
        })
        .filter(|(_, d)| *d < PICK_PX)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| Handle::Axis(i))
}

/// Closest approach to a ring's projection, sampled as a polyline — the
/// projection of a circle is an ellipse, and 32 chords are within a pixel of
/// one at gizmo size.
fn ring_distance(
    center: Vec3,
    normal: Vec3,
    radius: f32,
    view_proj: glam::Mat4,
    rect: [f32; 4],
    cursor: Vec2,
) -> f32 {
    let (u, v) = basis_around(normal);
    let point = |k: usize| {
        let a = k as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        project(
            view_proj,
            rect,
            center + (u * a.cos() + v * a.sin()) * radius,
        )
    };
    (0..RING_SEGMENTS)
        .filter_map(|k| Some(distance_to_segment(cursor, point(k)?, point(k + 1)?)))
        .fold(f32::MAX, f32::min)
}

// ─────────────────────────────────────────────────────────────────────────────
// Dragging
// ─────────────────────────────────────────────────────────────────────────────

/// What this handle's maths measures, at the moment of the press.
fn anchor(
    mode: GizmoMode,
    handle: Handle,
    origin: Vec3,
    axes: Mat3,
    size: f32,
    ray: (Vec3, Vec3),
) -> (f32, Vec3) {
    match (mode, handle) {
        (GizmoMode::Rotate, Handle::Axis(i)) => (
            ring_angle(origin, axes.col(i).normalize_or_zero(), ray),
            Vec3::ZERO,
        ),
        (_, Handle::Axis(i)) => (
            closest_on_axis(ray, origin, axes.col(i).normalize_or_zero()).unwrap_or(0.0),
            Vec3::ZERO,
        ),
        (_, Handle::Plane(i)) => (
            0.0,
            ray_plane(ray, origin, axes.col(i).normalize_or_zero()).unwrap_or(origin),
        ),
        (_, Handle::Uniform) => (screen_param(origin, ray, size), Vec3::ZERO),
    }
}

/// Where the ray crosses the ring's plane, as an angle around it.
fn ring_angle(center: Vec3, normal: Vec3, ray: (Vec3, Vec3)) -> f32 {
    let (u, v) = basis_around(normal);
    match ray_plane(ray, center, normal) {
        Some(hit) => (hit - center).dot(v).atan2((hit - center).dot(u)),
        None => 0.0,
    }
}

/// A scalar that grows as the cursor moves away from the gizmo's centre,
/// for the uniform-scale handle — measured on the plane facing the ray so
/// it behaves the same from any angle.
fn screen_param(origin: Vec3, ray: (Vec3, Vec3), size: f32) -> f32 {
    match ray_plane(ray, origin, -ray.1) {
        Some(hit) => (hit - origin).length() / size.max(1e-6),
        None => 0.0,
    }
}

/// Write the transform for this frame's cursor, from the press anchors.
fn apply(
    mode: GizmoMode,
    d: &Drag,
    ray: (Vec3, Vec3),
    size: f32,
    t: &engine_core::transform::TransformGuard<'_>,
) {
    let world_to_local = d.basis.inverse();
    let axis = |i: usize| d.axes.col(i).normalize_or_zero();
    match (mode, d.handle) {
        (GizmoMode::Translate, Handle::Axis(i)) => {
            let Some(p) = closest_on_axis(ray, d.origin, axis(i)) else {
                return;
            };
            t.set_position(d.start_position + world_to_local * (axis(i) * (p - d.param)));
        }
        (GizmoMode::Translate, Handle::Plane(i)) => {
            let Some(hit) = ray_plane(ray, d.origin, axis(i)) else {
                return;
            };
            t.set_position(d.start_position + world_to_local * (hit - d.point));
        }
        (GizmoMode::Rotate, Handle::Axis(i)) => {
            let delta = ring_angle(d.origin, axis(i), ray) - d.param;
            t.set_rotation(local_turn(d, Quat::from_axis_angle(axis(i), delta)) * d.start_rotation);
        }
        (GizmoMode::Scale, Handle::Axis(i)) => {
            let Some(p) = closest_on_axis(ray, d.origin, axis(i)) else {
                return;
            };
            let mut scale = d.start_scale;
            scale[i] = (d.start_scale[i] * (1.0 + (p - d.param) / size.max(1e-6))).max(1e-3);
            t.set_scale(scale);
        }
        (GizmoMode::Scale, Handle::Uniform) => {
            let factor = (screen_param(d.origin, ray, size) - d.param + 1.0).max(1e-3);
            t.set_scale(d.start_scale * factor);
        }
        _ => {}
    }
}

/// A world-space turn, expressed in the parent's frame — which is the frame
/// a local rotation composes in.
fn local_turn(d: &Drag, world: Quat) -> Quat {
    let parent = Quat::from_mat3(&orthonormal(d.basis));
    parent.inverse() * world * parent
}

/// The rotation half of a parent basis, with its scale divided out.
fn orthonormal(m: Mat3) -> Mat3 {
    let cols: [Vec3; 3] = std::array::from_fn(|i| m.col(i).normalize_or_zero());
    match cols.iter().all(|c| *c != Vec3::ZERO) {
        true => Mat3::from_cols(cols[0], cols[1], cols[2]),
        false => Mat3::IDENTITY,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry
// ─────────────────────────────────────────────────────────────────────────────

const RING: f32 = 0.9;
const RING_SEGMENTS: usize = 32;
const PLANE_NEAR: f32 = 0.25;
const PLANE_FAR: f32 = 0.55;

/// Any two unit vectors perpendicular to `n` and to each other.
fn basis_around(n: Vec3) -> (Vec3, Vec3) {
    let a = match n.x.abs() > 0.9 {
        true => Vec3::Y,
        false => Vec3::X,
    };
    let u = n.cross(a).normalize_or_zero();
    (u, n.cross(u))
}

/// The whole gizmo for one mode, in world space. Rebuilt every frame: it
/// moves with the camera (constant screen size) as much as with the entity.
fn geometry(
    mode: GizmoMode,
    origin: Vec3,
    axes: Mat3,
    size: f32,
    hovered: Option<Handle>,
) -> Vec<OverlayVertex> {
    let mut out = Vec::new();
    let color = |handle: Handle, base: [f32; 4]| match hovered == Some(handle) {
        true => HIGHLIGHT,
        false => base,
    };
    for i in 0..3 {
        let dir = axes.col(i).normalize_or_zero();
        let c = color(Handle::Axis(i), AXIS_COLORS[i]);
        match mode {
            GizmoMode::Translate => {
                tube(
                    &mut out,
                    origin,
                    origin + dir * size * 0.78,
                    size * 0.012,
                    c,
                );
                cone(
                    &mut out,
                    origin + dir * size * 0.78,
                    dir,
                    size * 0.22,
                    size * 0.055,
                    c,
                );
            }
            GizmoMode::Scale => {
                tube(
                    &mut out,
                    origin,
                    origin + dir * size * 0.88,
                    size * 0.012,
                    c,
                );
                cube(&mut out, origin + dir * size * 0.93, axes, size * 0.055, c);
            }
            GizmoMode::Rotate => {
                ring(&mut out, origin, dir, size * RING, size * 0.016, c);
            }
        }
    }
    if mode == GizmoMode::Translate {
        for i in 0..3 {
            let (u, v) = (
                axes.col((i + 1) % 3).normalize_or_zero() * size,
                axes.col((i + 2) % 3).normalize_or_zero() * size,
            );
            let mut c = color(Handle::Plane(i), AXIS_COLORS[i]);
            c[3] = 0.35;
            let corner = origin + (u + v) * PLANE_NEAR;
            quad(
                &mut out,
                corner,
                corner + u * (PLANE_FAR - PLANE_NEAR),
                corner + (u + v) * (PLANE_FAR - PLANE_NEAR),
                corner + v * (PLANE_FAR - PLANE_NEAR),
                c,
            );
        }
    }
    if mode == GizmoMode::Scale {
        cube(
            &mut out,
            origin,
            axes,
            size * 0.07,
            color(Handle::Uniform, [0.85, 0.85, 0.85, 1.0]),
        );
    }
    out
}

fn tri(out: &mut Vec<OverlayVertex>, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
    out.extend([a, b, c].map(|p| OverlayVertex::new(p, color)));
}

fn quad(out: &mut Vec<OverlayVertex>, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
    tri(out, a, b, c, color);
    tri(out, a, c, d, color);
}

/// A square-section tube from `a` to `b` — a line thick enough to aim at.
fn tube(out: &mut Vec<OverlayVertex>, a: Vec3, b: Vec3, radius: f32, color: [f32; 4]) {
    let dir = (b - a).normalize_or_zero();
    let (u, v) = basis_around(dir);
    let corner = |p: Vec3, k: usize| {
        let (s, c) = (k as f32 / 4.0 * std::f32::consts::TAU).sin_cos();
        p + (u * c + v * s) * radius
    };
    for k in 0..4 {
        quad(
            out,
            corner(a, k),
            corner(a, k + 1),
            corner(b, k + 1),
            corner(b, k),
            color,
        );
    }
}

/// An arrow head: `length` along `dir` from `base`, `radius` at the base.
fn cone(
    out: &mut Vec<OverlayVertex>,
    base: Vec3,
    dir: Vec3,
    length: f32,
    radius: f32,
    color: [f32; 4],
) {
    let (u, v) = basis_around(dir);
    let tip = base + dir * length;
    let rim = |k: usize| {
        let (s, c) = (k as f32 / 12.0 * std::f32::consts::TAU).sin_cos();
        base + (u * c + v * s) * radius
    };
    for k in 0..12 {
        tri(out, rim(k), rim(k + 1), tip, color);
        tri(out, rim(k + 1), rim(k), base, color);
    }
}

fn cube(out: &mut Vec<OverlayVertex>, center: Vec3, axes: Mat3, half: f32, color: [f32; 4]) {
    let a: [Vec3; 3] = std::array::from_fn(|i| axes.col(i).normalize_or_zero() * half);
    for i in 0..3 {
        let (u, v) = (a[(i + 1) % 3], a[(i + 2) % 3]);
        for face in [a[i], -a[i]] {
            quad(
                out,
                center + face - u - v,
                center + face + u - v,
                center + face + u + v,
                center + face - u + v,
                color,
            );
        }
    }
}

/// A square-section torus in the plane normal to `normal`.
fn ring(
    out: &mut Vec<OverlayVertex>,
    center: Vec3,
    normal: Vec3,
    radius: f32,
    thickness: f32,
    color: [f32; 4],
) {
    let (u, v) = basis_around(normal);
    let section = |k: usize, i: usize| {
        let a = k as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        let radial = u * a.cos() + v * a.sin();
        let (s, c) = (i as f32 / 4.0 * std::f32::consts::TAU).sin_cos();
        center + radial * radius + (radial * c + normal * s) * thickness
    };
    for k in 0..RING_SEGMENTS {
        for i in 0..4 {
            quad(
                out,
                section(k, i),
                section(k, i + 1),
                section(k + 1, i + 1),
                section(k + 1, i),
                color,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two bits of maths a drag cannot be written without: where a ray
    /// meets an axis, and where it meets a plane.
    #[test]
    fn a_ray_finds_the_point_on_an_axis_it_passes() {
        let ray = (Vec3::new(2.0, 1.0, 5.0), Vec3::NEG_Z);
        let t = closest_on_axis(ray, Vec3::ZERO, Vec3::X).expect("not parallel");
        assert!((t - 2.0).abs() < 1e-4, "the axis point below the ray");
        assert!(
            closest_on_axis(ray, Vec3::ZERO, Vec3::Z).is_none(),
            "parallel"
        );

        let hit = ray_plane(ray, Vec3::ZERO, Vec3::Z).expect("the ray faces the plane");
        assert!((hit - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-4);
    }

    /// A parented, rotated entity: a world-space drag has to arrive in the
    /// parent's frame, or the handle and the object part ways.
    #[test]
    fn a_world_delta_becomes_a_local_one() {
        let basis = Mat3::from_quat(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        let local = basis.inverse() * Vec3::X;
        assert!((basis * local - Vec3::X).length() < 1e-5);
        assert!(
            (local - Vec3::Z).length() < 1e-5,
            "world X is the parent's Z"
        );
    }
}

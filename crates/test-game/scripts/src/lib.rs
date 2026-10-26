//! test-game's scripts: the components this project defines, as against the
//! ones the engine ships.

use engine::{
    glam::{Quat, Vec3},
    transform::Transform,
    Component, Export, Res, World,
};

/// How a spin goes, shared by every `Rotator` that names it: a resource, so
/// editing the file turns all of them at once.
#[derive(Clone, Export)]
#[export(resource = "spin")]
pub struct Spin {
    #[export]
    pub axis: Vec3,
    /// Radians per second.
    #[export]
    pub speed: f32,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            axis: Vec3::Y,
            speed: std::f32::consts::FRAC_PI_4,
        }
    }
}

/// Spins the entity around its local Y axis at `speed` radians per second —
/// or, given a `spin`, the way that says, at `rate` times its speed.
#[derive(Clone, Export)]
pub struct Rotator {
    #[export]
    pub speed: f32,
    #[export]
    pub spin: Option<Res<Spin>>,
    #[export]
    pub rate: f32,
}

impl Default for Rotator {
    fn default() -> Self {
        // ~45°/sec — matches the previous hard-coded test-game animation.
        Self {
            speed: std::f32::consts::FRAC_PI_4,
            spin: None,
            rate: 1.0,
        }
    }
}

impl Component for Rotator {
    fn update(&mut self, dt: f32, transform: &Transform, _w: &World) {
        let (axis, speed) = match self.spin {
            Some(s) => s.with(|s| (s.axis.normalize_or_zero(), s.speed * self.rate)),
            None => (Vec3::Y, self.speed),
        };
        transform
            .lock()
            .rotate_by(Quat::from_axis_angle(axis, speed * dt));
    }
}

engine::declare_scripts!(Rotator; resources: Spin);

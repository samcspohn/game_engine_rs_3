//! test-game's scripts: the components this project defines, as against the
//! ones the engine ships.

use engine::{glam::Quat, transform::Transform, Component, Export, World};

/// Spins the entity around its local Y axis at `speed` radians per second.
#[derive(Clone, Export)]
pub struct Rotator {
    #[export]
    pub speed: f32,
}

impl Default for Rotator {
    fn default() -> Self {
        // ~45°/sec — matches the previous hard-coded test-game animation.
        Self {
            speed: std::f32::consts::FRAC_PI_4,
        }
    }
}

impl Component for Rotator {
    fn update(&mut self, dt: f32, transform: &Transform, _w: &World) {
        transform
            .lock()
            .rotate_by(Quat::from_rotation_y(self.speed * dt));
    }
}

engine::declare_scripts!(Rotator);

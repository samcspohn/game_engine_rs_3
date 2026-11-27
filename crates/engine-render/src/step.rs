//! The plug-in point for per-world GPU features. See
//! [frame-loop](../../../docs/notes/frame-loop.md#adding-a-step).

use engine_core::component::World;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};

use crate::transform_gpu::WorldTransformGpu;

pub(crate) type Primary = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>;

/// What a frame must redo before it submits. Each level includes the ones
/// below it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub(crate) enum Rebuild {
    #[default]
    None,
    /// Only draw-plan bases moved, and they're rewritten in place.
    Plans,
    /// A secondary that the frame primaries run was re-recorded.
    FrameSlots,
    /// A buffer that a camera binds was reallocated.
    Cameras,
}

/// Points in the frame primary where a step can record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum At {
    /// For commands that read host staging. Runs before `signal_cs` hands
    /// staging back to the host.
    Upload,
    /// After every upload, before any camera culls.
    Compute,
}

/// What a step can see of its world. Per-world data is indexed by transform
/// slot, so the transforms are included.
pub(crate) struct StepCx<'a> {
    pub world: &'a World,
    pub transforms: &'a WorldTransformGpu,
}

pub(crate) trait GpuStep: Send + Sync {
    /// Runs before the host wait. Grow device buffers to fit this frame.
    fn prepare(&mut self, _cx: &StepCx) -> Rebuild {
        Rebuild::None
    }
    /// Runs after the host wait. Write this frame's staging for `slot`.
    fn stage(&mut self, _cx: &StepCx, _slot: usize) {}
    /// Runs only when the frame slots are rebuilt, not every frame.
    fn record(&self, _at: At, _slot: usize, _b: &mut Primary) {}
}

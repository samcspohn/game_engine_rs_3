//! A camera's render target, in the layout.
//!
//! # What it forced
//!
//! **A widget that sizes something outside the UI.** Every other widget
//! takes the box taffy hands it and draws inside it. This one hands that box
//! *back* — to the camera, whose attachments the renderer re-allocates to
//! match. So a divider drag or a window resize re-renders the scene at the
//! new size instead of rescaling the old one, and the panel shows one texel
//! per pixel at its own aspect.
//!
//! That is the whole difference between this and [`UiCore::image`]: an image
//! samples a texture somebody else sized.
//!
//! The box goes onto the camera rather than into a registry keyed by
//! position, because the camera is the thing that owns the target the box
//! describes — and the same box is what bounds the gestures a controller
//! driving that camera answers to.

use super::style::Style;
use super::{camera_target, NodeId, UiCore, UiStyle};
use crate::camera::CameraHandle;

/// A camera, as a node.
///
/// ```ignore
/// let view = Viewport::new(&mut ui, dock.content(scene), fill(), camera);
/// // …once a frame:
/// view.update(&ui);
/// ```
#[derive(Clone)]
pub struct Viewport {
    node: NodeId,
    camera: CameraHandle,
}

impl Viewport {
    /// An image leaf bound to `camera`'s target. `style` gives it its box —
    /// usually "fill the pane", since the camera is then sized to the pane.
    pub fn new(
        ui: &mut UiCore,
        parent: impl Into<NodeId>,
        style: Style,
        camera: CameraHandle,
    ) -> Self {
        Self {
            node: ui.image(parent, camera_target(camera.slot()), style),
            camera,
        }
    }

    pub fn node(&self) -> NodeId {
        self.node
    }

    pub fn camera(&self) -> &CameraHandle {
        &self.camera
    }

    /// Show a different camera in the same box — the editor putting a game's
    /// own camera where the document's was, and back again.
    ///
    /// The node is restyled rather than rebuilt, so whatever the panel has
    /// been dragged to and sized at survives the switch. The camera it stops
    /// showing keeps the size it had until something else publishes a box to
    /// it, which is the same thing a closed tab does.
    pub fn set_camera(&mut self, ui: &mut UiCore, camera: CameraHandle) {
        ui.set_background(self.node, UiStyle::image(camera_target(camera.slot())));
        self.camera = camera;
    }

    /// Publish this frame's box onto the camera: it is resized to match
    /// before the next frame is recorded, and a pointer inside it belongs to
    /// the scene rather than to the UI.
    ///
    /// Every frame rather than on a change — a divider drag, a window resize
    /// and the panel being dragged to another edge all move it, and none of
    /// them are events a camera could subscribe to. The renderer compares
    /// before it re-allocates, so a frame where nothing moved costs the
    /// comparison.
    ///
    /// A box of zero — a closed tab, a collapsed pane, the first frame
    /// before any layout — is published as such rather than withheld: it
    /// owns no pointer, and the camera holds the size it had rather than
    /// churning to nothing and back on a tab switch. Only a camera no panel
    /// ever spoke for means "the whole window".
    pub fn update(&self, ui: &UiCore) {
        self.camera.set_rect(Some(ui.node_rect(self.node)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::{px, Size};

    fn boxed(w: f32, h: f32) -> Style {
        Style {
            size: Size {
                width: px(w),
                height: px(h),
            },
            ..Default::default()
        }
    }

    /// The capability: the widget's box *is* the camera's size, and it says
    /// so every frame. Rendering at the size it is shown at is what removes
    /// the scaling and the skewed projection the hole needed.
    #[test]
    fn the_box_it_is_given_is_the_size_it_asks_for() {
        let mut core = UiCore::new();
        let root = core.root();
        let v = Viewport::new(&mut core, root, boxed(320.0, 200.0), CameraHandle::new(0));
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(v.camera().rect(), Some([0.0, 0.0, 320.0, 200.0]));

        // Resized by the layout — a divider drag, in the editor.
        core.set_node_style(v.node(), boxed(640.0, 100.0));
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(v.camera().rect(), Some([0.0, 0.0, 640.0, 100.0]));

        // Collapsed: it owns no pointer, and the zero box is what tells the
        // renderer to hold the size it has rather than reallocate to nothing.
        core.set_visible(v.node(), false);
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(v.camera().rect(), Some([0.0; 4]));
        assert!(!v.camera().contains([100.0, 50.0]));
    }
}

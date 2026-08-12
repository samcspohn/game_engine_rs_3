//! A camera's render target, in the layout.
//!
//! # What it forced
//!
//! **A widget that sizes something outside the UI.** Every other widget
//! takes the box taffy hands it and draws inside it. This one hands that box
//! *back* — to the renderer, which re-allocates the camera's attachments to
//! match. So a divider drag or a window resize re-renders the scene at the
//! new size instead of rescaling the old one, and the panel shows one texel
//! per pixel at its own aspect.
//!
//! That is the whole difference between this and [`UiCore::image`]: an image
//! samples a texture somebody else sized.
//!
//! # Why it is a global and not a handle
//!
//! `UiCore` owns no Vulkan (that is what makes every widget testable without
//! a GPU), so the widget cannot hold a camera. It publishes a rect the way
//! `input` and `stats` publish theirs, and the renderer reads it once a
//! frame. The cost of that is exactly one viewport per process — a second
//! one wants a camera per widget, which is a bigger change than a second
//! static.

use super::style::Style;
use super::{NodeId, UiCore, CAMERA_TARGET};

/// The scene, as a node.
///
/// ```ignore
/// let view = Viewport::new(&mut ui, dock.content(scene), fill());
/// // …once a frame:
/// view.update(&ui);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Viewport {
    node: NodeId,
}

impl Viewport {
    /// An image leaf bound to the camera's target. `style` gives it its box —
    /// usually "fill the pane", since the camera is then sized to the pane.
    pub fn new(ui: &mut UiCore, parent: impl Into<NodeId>, style: Style) -> Self {
        Self {
            node: ui.image(parent, CAMERA_TARGET, style),
        }
    }

    pub fn node(&self) -> NodeId {
        self.node
    }

    /// Publish this frame's box: the camera is resized to it before the next
    /// frame is recorded, and a pointer inside it belongs to the scene
    /// rather than to the UI (see
    /// [`in_viewport`](crate::scene::in_viewport), which is what keeps a
    /// drag in the console from spinning the camera).
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
    /// churning to nothing and back on a tab switch. Only a viewport that
    /// never spoke at all means "the camera is the whole window".
    pub fn update(&self, ui: &UiCore) {
        crate::scene::set_viewport(Some(ui.node_rect(self.node)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene;
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
    ///
    /// One test, run in sequence, because the rect it publishes is a
    /// process-global — two tests asserting on it would race.
    #[test]
    fn the_box_it_is_given_is_the_size_it_asks_for() {
        let mut core = UiCore::new();
        let root = core.root();
        let v = Viewport::new(&mut core, root, boxed(320.0, 200.0));
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(scene::viewport_box(), Some([0.0, 0.0, 320.0, 200.0]));

        // Resized by the layout — a divider drag, in the editor.
        core.set_node_style(v.node(), boxed(640.0, 100.0));
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(scene::viewport_box(), Some([0.0, 0.0, 640.0, 100.0]));

        // Collapsed: it owns no pointer, and the zero box is what tells the
        // renderer to hold the size it has rather than reallocate to nothing.
        core.set_visible(v.node(), false);
        core.run_layout([800.0, 600.0]);
        v.update(&core);
        assert_eq!(scene::viewport_box(), Some([0.0; 4]));
        assert!(!scene::in_viewport([100.0, 50.0]));

        scene::set_viewport(None);
    }
}

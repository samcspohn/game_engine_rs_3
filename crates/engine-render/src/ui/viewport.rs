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
use super::{NodeId, UiCore, UiRef, UiStyle, camera_target};
use crate::camera::CameraHandle;

/// A camera, as a node. `Copy`: the camera lives on the node, and is handed
/// the node's box every time it moves.
///
/// ```ignore
/// let view = Viewport::new(&mut ui, dock.content(scene), fill(), camera);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Viewport(UiRef<CameraHandle>);

impl Viewport {
    /// An image leaf bound to `camera`'s target. `style` gives it its box —
    /// usually "fill the pane", since the camera is then sized to the pane.
    pub fn new(
        ui: &mut UiCore,
        parent: impl Into<NodeId>,
        style: Style,
        camera: CameraHandle,
    ) -> Self {
        let node = ui.image(parent, camera_target(camera.slot()), style);
        // Note: zero until the first layout says otherwise — `None` would
        // mean the whole window.
        camera.set_rect(Some([0.0; 4]));
        let me = ui.own(node, camera);
        // A closed tab or a collapsed pane is a zero box, published as such:
        // the camera owns no pointer and is not drawn while it lasts.
        ui.on_resize(node, move |ui, r| {
            me.with(ui, |c, _| c.set_rect(Some(r)));
        });
        Self(me)
    }

    pub fn node(self) -> NodeId {
        self.0.node()
    }

    pub fn camera(self, ui: &mut UiCore) -> Option<CameraHandle> {
        self.0.with(ui, |c, _| c.clone())
    }

    /// Show a different camera in the same box — the editor putting a game's
    /// own camera where the document's was, and back again. The one it stops
    /// showing is handed a zero box, as a tab going to the back is.
    pub fn set_camera(self, ui: &mut UiCore, camera: CameraHandle) {
        let n = self.node();
        ui.set_background(n, UiStyle::image(camera_target(camera.slot())));
        camera.set_rect(Some(ui.node_rect(n)));
        self.0.with(ui, |c, _| {
            std::mem::replace(c, camera).set_rect(Some([0.0; 4]))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::style::{Size, px};

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
        let camera = CameraHandle::new(0);
        let v = Viewport::new(&mut core, root, boxed(320.0, 200.0), camera.clone());
        core.run_layout([800.0, 600.0]);
        core.dispatch(0.0);
        assert_eq!(camera.rect(), Some([0.0, 0.0, 320.0, 200.0]));

        // Resized by the layout — a divider drag, in the editor.
        core.set_node_style(v.node(), boxed(640.0, 100.0));
        core.run_layout([800.0, 600.0]);
        core.dispatch(0.0);
        assert_eq!(camera.rect(), Some([0.0, 0.0, 640.0, 100.0]));

        // Collapsed: it owns no pointer, and the zero box is what tells the
        // renderer to hold the size it has rather than reallocate to nothing.
        core.set_visible(v.node(), false);
        core.run_layout([800.0, 600.0]);
        core.dispatch(0.0);
        assert_eq!(camera.rect(), Some([0.0; 4]));
        assert!(!camera.contains([100.0, 50.0]));
    }

    /// Switching away is a tab going to the back: play mode must not leave
    /// the document's camera drawing behind the game's.
    #[test]
    fn the_camera_it_stops_showing_goes_dark() {
        let mut core = UiCore::new();
        let root = core.root();
        let editing = CameraHandle::new(0);
        let v = Viewport::new(&mut core, root, boxed(320.0, 200.0), editing.clone());
        core.run_layout([800.0, 600.0]);
        core.dispatch(0.0);
        assert_eq!(editing.rect(), Some([0.0, 0.0, 320.0, 200.0]));

        v.set_camera(&mut core, CameraHandle::new(0));
        assert_eq!(editing.rect(), Some([0.0; 4]), "no box of its own now");
        assert!(editing.is_dark());
    }
}

//! The console pane: what the editor has to say, one line per message.

use engine::ui::{DockSpace, NodeId, PanelId, Side, UiCore, theme};

/// The pane the editor logs to. It holds its own content node, which a
/// panel keeps for as long as it exists however the user rearranges the dock.
#[derive(Clone, Copy)]
pub(crate) struct Console {
    /// What the View menu brings forward, and what the browser is tabbed
    /// against.
    pub(crate) pane: PanelId,
    content: NodeId,
}

impl Console {
    pub(crate) fn new(ui: &mut UiCore, dock: &mut DockSpace, under: PanelId) -> Self {
        let pane = dock.panel(ui, "Console");
        dock.dock(ui, pane, under, Side::Bottom);
        dock.set_ratio(ui, pane, 0.25);
        Self {
            pane,
            content: dock.content(pane),
        }
    }

    pub(crate) fn line(&self, ui: &mut UiCore, text: &str) {
        let t = theme();
        ui.label(self.content, t.text_px, t.text_dim, text);
    }
}

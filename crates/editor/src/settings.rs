//! What the editor remembers about a project between runs.
//!
//! A second file beside `project.json` rather than a section in it: what a
//! project *is* ships in a bundle and belongs to everyone working on it,
//! while which scenes one person left open belongs to that person's next
//! session. See `docs/notes/editor.md`.

use engine::ui::Layout;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The editor's own settings, at the project root beside
/// [`engine::project::PROJECT`].
pub const EDITOR: &str = "editor.json";

#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Settings {
    /// The scenes open as documents, project-relative and in tab order.
    #[serde(default)]
    pub open: Vec<PathBuf>,
    /// Where the documents, the Console and the Browser sit.
    #[serde(default)]
    pub layout: Option<Layout>,
    /// One arrangement for every document's Scene, Hierarchy and Inspector:
    /// the same three panels in each, so the front one's is all of theirs.
    #[serde(default)]
    pub document: Option<Layout>,
}

/// A missing file is an empty one, and one that will not parse is reported
/// and then treated as missing: a half-typed session should not stop the
/// editor opening the project it belongs to.
pub fn load() -> Settings {
    let Ok(text) = std::fs::read_to_string(EDITOR) else {
        return Settings::default();
    };
    serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("{EDITOR}: {e}");
        Settings::default()
    })
}

pub fn save(settings: &Settings) -> Result<(), String> {
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(EDITOR, text + "\n").map_err(|e| e.to_string())
}

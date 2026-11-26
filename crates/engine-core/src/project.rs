//! Rooting the process at the project.
//!
//! Everything the engine opens is a path relative to the working directory,
//! which is what a scene file stores. So there is nothing to resolve — the
//! only question is which directory that is, and a bundle is the one case
//! that cannot be told from outside: the directory it was built from does
//! not exist on the player's machine.
//!
//! Beside that sits what the project says about itself — its startup scene,
//! and whatever settings join it — which ships in the bundle verbatim, so
//! the editor and the packaged game read one file rather than two that can
//! disagree.
//!
//! See [`docs/notes/packaging.md`](../../../docs/notes/packaging.md).

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// Marks a directory as a packaged bundle, and records what it was built
/// from. Written by the `packager`.
pub const MANIFEST: &str = "game.json";

/// The project's own settings, at its root.
pub const PROJECT: &str = "project.json";

/// What a project says about itself.
///
/// Every field is optional and a missing file is an empty one: a directory
/// with scenes and no settings is still a project, it just has no scene to
/// start in and so nothing to play.
#[derive(Debug, Default, Deserialize, Serialize)]
pub struct Settings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The scene play starts in and a packaged game opens — project-relative,
    /// like every other stored path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_scene: Option<PathBuf>,
}

/// The settings of the project [`enter`] entered, read once.
///
/// A file that will not parse is reported and then treated as absent: a typo
/// in one setting should not stop the editor opening the project it is in.
pub fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let Ok(text) = std::fs::read_to_string(PROJECT) else {
            return Settings::default();
        };
        serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("{PROJECT}: {e}");
            Settings::default()
        })
    })
}

/// Enter the project: a bundle's own directory, else `fallback` — the
/// editor's `--project`, or a game crate's `env!("CARGO_MANIFEST_DIR")`.
/// Answers with the directory entered.
///
/// Call before any thread exists and before any asset is requested; both
/// read relative paths, and this moves what they mean.
pub fn enter(fallback: impl Into<PathBuf>) -> std::io::Result<PathBuf> {
    std::env::set_current_dir(bundle().unwrap_or_else(|| fallback.into()))?;
    std::env::current_dir()
}

/// A path typed on the command line is relative to the shell, not to the
/// project — pin it before [`enter`] moves what relative means. A path that
/// does not exist passes through, to fail loudly where it is used.
pub fn pin(path: &str) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path))
}

fn bundle() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    dir.join(MANIFEST).is_file().then_some(dir)
}

//! Loading a project's script dylib.
//!
//! Editor-only by construction: a shipped game links its scripts crate as an
//! rlib, so nothing on a game's dependency path ever calls `dlopen`.
//!
//! See [`docs/notes/scripts.md`](../../../docs/notes/scripts.md).

use std::path::{Path, PathBuf};

use parking_lot::Mutex;

/// Every library loaded this process, kept forever.
///
/// Nothing droppable is handed back on purpose: a `dlclose` would unmap the
/// `&'static str` each registered type names itself by, and the reads that
/// follow are silent — the length still parses, the bytes are whatever the
/// mapping became.
static LOADED: Mutex<Vec<libloading::Library>> = Mutex::new(Vec::new());

/// Register the component types in `<project>/scripts`, if that crate has been
/// built, and answer with the library's path. `Ok(None)` is a project authored
/// without scripts, which is a whole project — assets, scenes and engine
/// components — not a failure.
///
/// The library is found beside the running editor rather than compiled here:
/// building it on demand is runtime compilation, which this is the seam for
/// and not yet the feature.
pub fn load(project: &Path) -> Result<Option<PathBuf>, String> {
    if !project.join("scripts/Cargo.toml").is_file() {
        return Ok(None);
    }
    let Some(path) = beside_editor(&file_name(project)) else {
        return Err(format!(
            "{} has a scripts crate but lib{}.so is not built",
            project.display(),
            file_name(project)
        ));
    };

    // SAFETY: the library's initialisers run here. It is the project's own
    // code, compiled against this exact engine build — a mismatch fails to
    // resolve `engine_core`'s symbols and lands as an `Err` below rather than
    // loading.
    let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| e.to_string())?;
    let addr = unsafe {
        lib.get::<extern "C" fn() -> usize>(b"engine_register_scripts")
            .map_err(|e| format!("{}: {e}", path.display()))?()
    };
    if addr != engine_core::script::registry_addr() {
        return Err(format!(
            "{} has its own copy of the engine's statics — build both with \
             `-C prefer-dynamic` so they share one libengine_core.so",
            path.display()
        ));
    }
    LOADED.lock().push(lib);
    Ok(Some(path))
}

/// `crates/test-game` → `test_game_scripts`, the package name the convention
/// gives that project's scripts crate.
fn file_name(project: &Path) -> String {
    let stem = project
        .file_name()
        .map(|s| s.to_string_lossy().replace('-', "_"))
        .unwrap_or_default();
    format!("{stem}_scripts")
}

fn beside_editor(name: &str) -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let path = dir.join(format!("lib{name}.so"));
    path.is_file().then_some(path)
}

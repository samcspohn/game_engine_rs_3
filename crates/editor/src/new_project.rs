//! Scaffolding a game project with cargo, and reopening the editor on it.
//!
//! The editor roots itself with one `chdir` before any thread exists, so the
//! open project is not something a running process can swap. Creating one
//! therefore ends by launching the editor again on the new directory.
//!
//! Manifests are written by `cargo new` and `cargo add` wherever cargo has a
//! command for it, so what a generated project declares is whatever the
//! installed cargo considers current. Only what cargo has no command for is
//! written here: the dual crate-type, and the source.
//!
//! See [`docs/notes/scripts.md`](../../../docs/notes/scripts.md).

use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// Both link modes from one crate: a packaged game takes the rlib, the
/// editor the dylib.
const CRATE_TYPE: &str = "
[lib]
crate-type = [\"rlib\", \"dylib\"]
";

/// Scaffold `dir` as a cargo project that depends on this editor's engine,
/// and build its scripts dylib. Slow — `cargo build` — so not for the frame
/// thread.
pub fn create(dir: &Path) -> Result<(), String> {
    let name = dir
        .file_name()
        .ok_or("a project is a directory, and that path names none")?
        .to_string_lossy()
        .into_owned();
    let root = workspace_root()?;
    if !dir.starts_with(&root) {
        return Err(format!(
            "{} is outside {}: a project's scripts have to build as a member \
             of the engine's own workspace, or they link an engine of their \
             own that this editor cannot load",
            dir.display(),
            root.display()
        ));
    }
    let engine = engine_dir();
    let manifest = dir.join("Cargo.toml");
    let scripts = dir.join("scripts");
    let scripts_manifest = scripts.join("Cargo.toml");

    // Enrols the new package in the engine's workspace, which is where the
    // `editor` profile and the engine dependency's version come from.
    cargo(&["new".as_ref(), dir.as_ref()])?;
    cargo(&[
        "new".as_ref(),
        scripts.as_ref(),
        "--lib".as_ref(),
        "--vcs".as_ref(),
        "none".as_ref(),
        "--name".as_ref(),
        format!("{name}-scripts").as_ref(),
    ])?;
    append(&scripts_manifest, CRATE_TYPE)?;
    add(&manifest, &engine)?;
    add(&manifest, &scripts)?;
    add(&scripts_manifest, &engine)?;

    // Templates live beside this file rather than in it, so what is Rust to
    // a reader is not Rust to the compiler.
    let main_rs = include_str!("templates/main.rs.in")
        .replace("{name}", &name)
        .replace("{crate}", &name.replace('-', "_"));
    write(&dir.join("src/main.rs"), &main_rs)?;
    let lib_rs = include_str!("templates/scripts.rs.in").replace("{name}", &name);
    write(&scripts.join("src/lib.rs"), &lib_rs)?;
    let project = include_str!("templates/project.json.in").replace("{name}", &name);
    write(&dir.join("project.json"), &project)?;
    // A camera and nothing else: a scene that can be played the moment it is
    // generated, which an empty one cannot.
    write(
        &dir.join("scenes/main.json"),
        include_str!("templates/scene.json"),
    )?;
    std::fs::create_dir_all(dir.join("assets")).map_err(|e| e.to_string())?;

    build_scripts(dir)
}

/// Compile `<project>/scripts` into the editor's own target directory: both
/// where the engine units it links are already built, and where
/// `scripts::load` looks for the result.
pub fn build_scripts(project: &Path) -> Result<(), String> {
    let out = Command::new("cargo")
        .args(["build", "--profile", "editor"])
        .arg("--manifest-path")
        .arg(project.join("scripts/Cargo.toml"))
        .env("CARGO_TARGET_DIR", target_dir()?)
        .env("RUSTFLAGS", "-C prefer-dynamic")
        .output()
        .map_err(|e| format!("cargo build: {e}"))?;
    match out.status.success() {
        true => Ok(()),
        false => Err(String::from_utf8_lossy(&out.stderr).trim().to_string()),
    }
}

/// Start this editor again on `project` and leave. The child inherits the
/// old project as its working directory, so the path handed over is
/// absolute.
pub fn reopen(project: &Path) -> Result<(), String> {
    Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .arg("--project")
        .arg(project)
        .spawn()
        .map_err(|e| format!("cannot reopen the editor: {e}"))?;
    std::process::exit(0);
}

/// Where a name typed into the dialog lands: beside the open project, since
/// a project directory is not a place inside another one.
///
/// `..` is folded away here rather than by `canonicalize`, which cannot see
/// a directory that does not exist yet.
pub fn resolve(name: &str) -> Result<PathBuf, String> {
    let here = std::env::current_dir().map_err(|e| e.to_string())?;
    let path = match Path::new(name).is_absolute() {
        true => PathBuf::from(name),
        false => here
            .parent()
            .ok_or("no directory beside this project")?
            .join(name),
    };
    Ok(path.components().fold(PathBuf::new(), |mut out, c| {
        match c {
            Component::ParentDir => drop(out.pop()),
            Component::CurDir => (),
            c => out.push(c),
        }
        out
    }))
}

/// The engine's own workspace root, which a generated project has to be a
/// member of. Cargo hashes a package's path *relative to its workspace
/// root*, so the same `engine-core` built from two roots is two units
/// writing one `libengine_core.so` — and the loser is a dylib whose symbols
/// no longer exist.
fn workspace_root() -> Result<PathBuf, String> {
    let manifest = engine_dir().join("Cargo.toml");
    let root = cargo(&[
        "locate-project".as_ref(),
        "--workspace".as_ref(),
        "--message-format".as_ref(),
        "plain".as_ref(),
        "--manifest-path".as_ref(),
        manifest.as_ref(),
    ])?;
    Path::new(root.trim())
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("{root} names no directory"))
}

fn add(manifest: &Path, path: &Path) -> Result<(), String> {
    cargo(&[
        "add".as_ref(),
        "--path".as_ref(),
        path.as_ref(),
        "--manifest-path".as_ref(),
        manifest.as_ref(),
    ])
    .map(drop)
}

/// cargo's own diagnostics are the error: a name it refuses and a directory
/// that already exists both arrive here already worded for the console.
fn cargo(args: &[&OsStr]) -> Result<String, String> {
    let out = Command::new("cargo")
        .args(args)
        .output()
        .map_err(|e| format!("cargo: {e}"))?;
    match out.status.success() {
        true => Ok(String::from_utf8_lossy(&out.stdout).trim().to_string()),
        false => Err(String::from_utf8_lossy(&out.stderr).trim().to_string()),
    }
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

fn append(manifest: &Path, text: &str) -> Result<(), String> {
    let was =
        std::fs::read_to_string(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    write(manifest, &(was + text))
}

/// The engine this editor was built against, as an absolute path baked in at
/// compile time. Moving this repo breaks projects generated before the move.
fn engine_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).with_file_name("engine")
}

/// `target/editor`, two levels up from the running binary: cargo puts a
/// `--profile editor` build at `<target>/editor/editor`.
fn target_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    exe.parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("{} is not in a cargo target directory", exe.display()))
}

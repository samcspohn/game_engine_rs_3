//! CLI game-packaging tool: build the project's binary, stage its content
//! beside it, and mark the directory as a bundle.
//!
//! ```text
//! packager --project crates/test-game --out target/dist
//! ```
//!
//! See [`docs/notes/packaging.md`](../../../docs/notes/packaging.md).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::Parser;
use serde::Serialize;

/// Must match `engine_core::project::MANIFEST` — the engine reads it to
/// decide it is running from a bundle.
const MANIFEST: &str = "game.json";

/// Must match `engine_core::project::PROJECT` — the game reads it for the
/// scene it opens with, so it ships verbatim rather than being restated in
/// the manifest where the two could disagree.
const PROJECT: &str = "project.json";

/// Project directories staged verbatim. Cooking is a size and load-time
/// concern, not a portability one, so there is none.
const CONTENT: [&str; 2] = ["assets", "scenes"];

#[derive(Parser, Debug)]
#[command(
    name = "packager",
    about = "Bundle a game crate into a distributable directory",
    version
)]
struct Args {
    /// Game crate to build and package.
    #[arg(long, value_name = "PATH")]
    project: PathBuf,

    /// Directory the bundle is written to.
    #[arg(long, value_name = "PATH")]
    out: PathBuf,

    /// Cross-compile triple. Defaults to the host's.
    #[arg(long, value_name = "TRIPLE")]
    target: Option<String>,

    /// Which binary, when the crate builds more than one.
    #[arg(long, value_name = "NAME")]
    bin: Option<String>,
}

/// What the bundle was built from. Its *presence* is what tells the engine
/// to root itself here rather than at the working directory; the fields are
/// what a bug report needs. What the *game* is — its startup scene and the
/// settings beside it — is `project.json`, staged as authored.
#[derive(Serialize)]
struct Manifest {
    engine: String,
    target: String,
    bin: String,
}

fn main() {
    if let Err(e) = run(Args::parse()) {
        eprintln!("packaging failed: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let cargo_toml = args.project.join("Cargo.toml");
    if !cargo_toml.is_file() {
        return Err(format!(
            "{} has no Cargo.toml — a bundle needs a game binary to build",
            args.project.display()
        ));
    }
    // A bundle with no scene to open is one that starts black, which is
    // worth failing over rather than shipping.
    let startup = startup_scene(&args.project)?;
    if !args.project.join(&startup).is_file() {
        return Err(format!(
            "startup scene {} not found in {}",
            startup.display(),
            args.project.display()
        ));
    }

    let target = match &args.target {
        Some(t) => t.clone(),
        None => host_triple()?,
    };
    println!("[1/3] building {} for {target}", args.project.display());
    let exe = build(&cargo_toml, args.target.as_deref(), args.bin.as_deref())?;

    println!("[2/3] staging → {}", args.out.display());
    prepare(&args.out)?;
    let bin = exe
        .file_name()
        .ok_or("built artifact has no file name")?
        .to_string_lossy()
        .into_owned();
    copy(&exe, &args.out.join(&bin))?;
    for dir in CONTENT {
        let src = args.project.join(dir);
        if src.is_dir() {
            copy_tree(&src, &args.out.join(dir))?;
        }
    }
    copy(&args.project.join(PROJECT), &args.out.join(PROJECT))?;

    println!("[3/3] writing {MANIFEST}");
    let manifest = Manifest {
        engine: engine_revision(),
        target,
        bin: bin.clone(),
    };
    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("serialise {MANIFEST}: {e}"))?;
    std::fs::write(args.out.join(MANIFEST), json + "\n")
        .map_err(|e| format!("write {MANIFEST}: {e}"))?;

    println!(
        "bundled → {}/{bin} ({})",
        args.out.display(),
        startup.display()
    );
    Ok(())
}

/// The scene the game opens with, read from the project's own settings —
/// the same file the editor's play button reads, so a bundle starts where
/// play does.
fn startup_scene(project: &Path) -> Result<PathBuf, String> {
    let path = project.join(PROJECT);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let settings: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    settings
        .get("startup_scene")
        .and_then(|s| s.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| format!("{} names no startup_scene", path.display()))
}

/// Reads the artifact path out of cargo rather than guessing a target
/// directory, which `--target`, a workspace root and `CARGO_TARGET_DIR` all
/// move independently. `target` is passed through only when it was asked
/// for: naming the host triple explicitly moves the output directory, and
/// would rebuild the world for a bundle of the same bytes.
fn build(cargo_toml: &Path, target: Option<&str>, bin: Option<&str>) -> Result<PathBuf, String> {
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.args([
        "build",
        "--release",
        "--message-format=json-render-diagnostics",
    ])
    .arg("--manifest-path")
    .arg(cargo_toml);
    if let Some(target) = target {
        cmd.args(["--target", target]);
    }
    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    let out = cmd
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("running cargo: {e}"))?;
    if !out.status.success() {
        return Err("cargo build failed".into());
    }

    let mut found = None;
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) else {
            continue;
        };
        let name = msg.pointer("/target/name").and_then(|n| n.as_str());
        if bin.is_none() || bin == name {
            found = Some(PathBuf::from(exe));
        }
    }
    found.ok_or_else(|| match bin {
        Some(bin) => format!("cargo built no binary named {bin}"),
        None => "cargo built no binary — pass --bin to pick one".into(),
    })
}

/// A bundle is replaced wholesale, because an asset the new build no longer
/// references would otherwise ship anyway. Anything else is left alone.
fn prepare(out: &Path) -> Result<(), String> {
    if out.join(MANIFEST).is_file() {
        std::fs::remove_dir_all(out).map_err(|e| format!("clear {}: {e}", out.display()))?;
    } else if out.read_dir().is_ok_and(|mut d| d.next().is_some()) {
        return Err(format!(
            "{} is not empty and is not a bundle — refusing to overwrite",
            out.display()
        ));
    }
    std::fs::create_dir_all(out).map_err(|e| format!("create {}: {e}", out.display()))
}

fn copy_tree(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| format!("create {}: {e}", dst.display()))?;
    for entry in std::fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("read {}: {e}", src.display()))?;
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            copy_tree(&from, &to)?;
        } else {
            copy(&from, &to)?;
        }
    }
    Ok(())
}

fn copy(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::copy(from, to)
        .map(drop)
        .map_err(|e| format!("copy {} → {}: {e}", from.display(), to.display()))
}

fn host_triple() -> Result<String, String> {
    let out = Command::new("rustc")
        .arg("-vV")
        .output()
        .map_err(|e| format!("running rustc: {e}"))?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))
        .ok_or_else(|| "rustc -vV printed no host line".into())
}

/// `-dirty` matters more than the hash: a revision that does not describe
/// the tree the binary came from is worse than no revision.
fn engine_revision() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(rev) = git(&["rev-parse", "--short", "HEAD"]) else {
        return "unknown".into();
    };
    match git(&["status", "--porcelain"]) {
        Some(s) if !s.is_empty() => format!("{rev}-dirty"),
        _ => rev,
    }
}

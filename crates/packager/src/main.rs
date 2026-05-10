//! CLI game-packaging tool.
//!
//! The packager takes a game crate path and an output directory, then performs
//! (or describes) the steps needed to produce a distributable bundle:
//!
//! 1. Compile the game crate in release mode with `cargo build --release`.
//! 2. Cook / optimise the asset tree.
//! 3. Bundle the compiled binary together with the cooked assets.
//!
//! Currently every step prints what it *would* do and is marked `// TODO:`.
//! Replace each stub with real logic as the pipeline matures.
//!
//! # Usage
//! ```text
//! packager --project crates/test-game --out target/dist
//! ```

use std::path::PathBuf;

use clap::Parser;

/// CLI arguments accepted by the packager.
#[derive(Parser, Debug)]
#[command(
    name    = "packager",
    about   = "Bundle a game crate into a distributable package",
    version
)]
struct Args {
    /// Path to the game crate that should be compiled and packaged.
    #[arg(long, value_name = "PATH")]
    project: PathBuf,

    /// Output directory where the finished bundle will be written.
    #[arg(long, value_name = "PATH")]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();

    println!(
        "Packaging project '{}' → '{}'",
        args.project.display(),
        args.out.display()
    );

    // ── Step 1: compile ──────────────────────────────────────────────────────
    println!(
        "  [1/3] Would run: cargo build --release --manifest-path {}/Cargo.toml",
        args.project.display()
    );
    // TODO: spawn `cargo build --release` in the project directory and wait for it.

    // ── Step 2: cook assets ──────────────────────────────────────────────────
    println!("  [2/3] Would cook assets from {}/assets/", args.project.display());
    // TODO: iterate the project's asset directory, compress textures, pack
    //       audio files, etc., writing results into a staging directory.

    // ── Step 3: bundle ───────────────────────────────────────────────────────
    println!(
        "  [3/3] Would copy compiled binary + cooked assets → {}",
        args.out.display()
    );
    // TODO: copy the release binary and the cooked-assets directory into
    //       `args.out`, creating a self-contained distributable folder.

    println!("Done (dry-run — no files were written).");
}

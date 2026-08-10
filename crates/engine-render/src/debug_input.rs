//! A debug channel for driving the running app — input in, UI state out.
//!
//! Off unless `ENGINE_DEBUG_INPUT` is set; a runtime check, so turning it on
//! costs a restart rather than a rebuild.
//!
//! Input is addressed to *this process* and given in window coordinates, so
//! it cannot leak into another window, cannot race the compositor's focus,
//! and knows nothing about monitor layout or window position.
//!
//! Injected input is queued as frame *steps*: a click is `[move, press,
//! release]` over three frames, because `mouse_pressed` is edge-triggered and
//! a double click needs two clicks the pointer clock can tell apart. It goes
//! into `Input` through the same fields `feed_window_event` writes, so
//! nothing downstream can tell the difference.
//!
//! Queries skip the queue and lock the UI store directly. Queue and store are
//! never held at once, so the two locks cannot invert.
//!
//! ```text
//! $ poke dblclick "cube"    → ok 60 76.5
//! $ poke type hull
//! $ poke key Enter
//! $ poke find hull          → 22.0 46 71 28 11 hull
//! ```

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use glam::Vec2;

use crate::input::{Input, Key, Keystroke, Mods, MouseButton};
use crate::ui;

/// Socket name under `$XDG_RUNTIME_DIR` when `ENGINE_DEBUG_INPUT=1`; set the
/// variable to a path instead to run two apps at once.
const DEFAULT_SOCK: &str = "engine-poke.sock";

/// One frame's worth of injected input.
enum Step {
    Move([f32; 2]),
    Button([f32; 2], bool),
    Wheel(f32),
    Keys(Vec<Keystroke>),
}

static QUEUE: Mutex<VecDeque<Step>> = Mutex::new(VecDeque::new());

/// Frames that have run [`pump`]; `wait` uses it to tell "injected" from
/// "consumed".
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Open the socket and serve it on a background thread.
pub(crate) fn start() {
    let Ok(v) = std::env::var("ENGINE_DEBUG_INPUT") else {
        return;
    };
    let path = match v.as_str() {
        "1" | "true" => runtime_dir().join(DEFAULT_SOCK),
        other => std::path::PathBuf::from(other),
    };
    // A stale socket from a crashed run would make `bind` fail forever.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[poke] cannot bind {}: {e}", path.display());
            return;
        }
    };
    println!("[poke] listening on {}", path.display());
    std::thread::Builder::new()
        .name("poke".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                // One connection per command, so a wedged client cannot
                // hold the channel.
                if let Err(e) = serve(stream) {
                    eprintln!("[poke] {e}");
                }
            }
        })
        .expect("spawn poke thread");
}

fn runtime_dir() -> std::path::PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
}

/// Apply one queued step, before the frame reads input.
pub(crate) fn pump(input: &mut Input) {
    FRAMES.fetch_add(1, Ordering::Relaxed);
    let step = QUEUE.lock().expect("poke queue").pop_front();
    let Some(step) = step else {
        return;
    };
    match step {
        Step::Move(p) => input.inject_cursor(Vec2::new(p[0], p[1])),
        Step::Button(p, down) => {
            input.inject_cursor(Vec2::new(p[0], p[1]));
            input.inject_button(MouseButton::Left, down);
        }
        Step::Wheel(l) => input.inject_wheel(l),
        Step::Keys(ks) => {
            for k in ks {
                input.inject_keystroke(k);
            }
        }
    }
}

fn push(steps: impl IntoIterator<Item = Step>) {
    QUEUE.lock().expect("poke queue").extend(steps);
}

// ── The protocol ────────────────────────────────────────────────────────

fn serve(mut stream: UnixStream) -> std::io::Result<()> {
    let mut line = String::new();
    BufReader::new(stream.try_clone()?).read_line(&mut line)?;
    let reply = dispatch(line.trim_end());
    stream.write_all(reply.as_bytes())?;
    if !reply.ends_with('\n') {
        stream.write_all(b"\n")?;
    }
    Ok(())
}

/// `<command> <rest of line>`; the argument is never re-split, so text with
/// spaces needs no quoting rules.
fn dispatch(line: &str) -> String {
    let (cmd, arg) = match line.split_once(' ') {
        Some((c, a)) => (c, a.trim()),
        None => (line, ""),
    };
    match cmd {
        "move" => match point(arg) {
            Ok(p) => {
                push([Step::Move(p)]);
                "ok".into()
            }
            Err(e) => e,
        },
        "click" | "dblclick" => match target(arg) {
            Ok(p) => {
                let mut steps = vec![Step::Move(p)];
                let times = if cmd == "dblclick" { 2 } else { 1 };
                for _ in 0..times {
                    steps.push(Step::Button(p, true));
                    steps.push(Step::Button(p, false));
                }
                push(steps);
                format!("ok {} {}", p[0], p[1])
            }
            Err(e) => e,
        },
        "drag" => {
            // `--to` and not `->`: the shell eats the latter as a redirect.
            let Some((a, b)) = arg.split_once("--to") else {
                return "err drag <target> --to <target>".into();
            };
            match (target(a.trim()), target(b.trim())) {
                (Ok(from), Ok(to)) => {
                    push([
                        Step::Move(from),
                        Step::Button(from, true),
                        // First move crosses the drag threshold, second lets
                        // the target resolve before the release.
                        Step::Move(to),
                        Step::Move(to),
                        Step::Button(to, false),
                    ]);
                    "ok".into()
                }
                (Err(e), _) | (_, Err(e)) => e,
            }
        }
        "wheel" => match arg.parse::<f32>() {
            Ok(l) => {
                push([Step::Wheel(l)]);
                "ok".into()
            }
            Err(_) => "err wheel <lines>".into(),
        },
        "key" => match key(arg) {
            Ok(k) => {
                push([Step::Keys(vec![k])]);
                "ok".into()
            }
            Err(e) => e,
        },
        "type" => {
            if arg.is_empty() {
                return "err type <text>".into();
            }
            push([Step::Keys(vec![Keystroke::Text(arg.to_string())])]);
            "ok".into()
        }
        "wait" => wait(),
        "shot" => shot(arg),
        "find" => find(arg),
        "tree" => find(""),
        "focus" => focus(),
        // So a harness can stop the app without hunting for its pid.
        "quit" => std::process::exit(0),
        "" => "err empty".into(),
        other => format!("err unknown command {other:?}"),
    }
}

/// `x,y`.
fn point(s: &str) -> Result<[f32; 2], String> {
    let (x, y) = s.split_once(',').ok_or("err expected x,y")?;
    match (x.trim().parse::<f32>(), y.trim().parse::<f32>()) {
        (Ok(x), Ok(y)) => Ok([x, y]),
        _ => Err(format!("err cannot parse point {s:?}")),
    }
}

/// `x,y`, or the centre of the first node whose visible text matches.
/// Aiming by text avoids re-deriving a layout the engine just solved.
fn target(s: &str) -> Result<[f32; 2], String> {
    if s.contains(',') && !s.contains(char::is_alphabetic) {
        return point(s);
    }
    let ui = ui::ui();
    let hit = ui
        .text_nodes()
        .into_iter()
        .find(|(_, text, _)| text.contains(s));
    match hit {
        Some((_, _, r)) => Ok([r[0] + r[2] * 0.5, r[1] + r[3] * 0.5]),
        None => Err(format!("err no visible node matching {s:?}")),
    }
}

/// `Enter`, `Escape`, `Left`, `ctrl+a`, `shift+End` — case-insensitive.
fn key(s: &str) -> Result<Keystroke, String> {
    let mut mods = Mods::NONE;
    let mut name = s;
    while let Some((m, rest)) = name.split_once('+') {
        mods = mods
            | match m.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => Mods::CTRL,
                "shift" => Mods::SHIFT,
                "alt" => Mods::ALT,
                other => return Err(format!("err unknown modifier {other:?}")),
            };
        name = rest;
    }
    let k = match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => Key::Enter,
        "tab" => Key::Tab,
        "escape" | "esc" => Key::Escape,
        "backspace" => Key::Backspace,
        "delete" | "del" => Key::Delete,
        "left" => Key::Left,
        "right" => Key::Right,
        "up" => Key::Up,
        "down" => Key::Down,
        "home" => Key::Home,
        "end" => Key::End,
        c if c.chars().count() == 1 => Key::Char(c.chars().next().expect("one char")),
        other => return Err(format!("err unknown key {other:?}")),
    };
    Ok(Keystroke::Key(k, mods))
}

/// Block until every queued step has been applied and consumed.
fn wait() -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if QUEUE.lock().expect("poke queue").is_empty() {
            // Two frames: the last step is in `Input` but not yet read by
            // `update_pointer`.
            let from = FRAMES.load(Ordering::Relaxed);
            while FRAMES.load(Ordering::Relaxed) < from + 2 {
                if Instant::now() > deadline {
                    return "err timeout waiting for a frame".into();
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            return "ok".into();
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    "err timeout waiting for the queue".into()
}

/// Capture the next frame and reply with the file it landed in.
///
/// Blocking, so a caller can read the image on the next line of the script —
/// the renderer writes it during the frame, not eventually.
fn shot(arg: &str) -> String {
    let want = crate::capture::request(Some(arg));
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        match crate::capture::result() {
            Some(Ok(p)) => return format!("ok {}", p.display()),
            Some(Err(e)) => return format!("err {e}"),
            None => std::thread::sleep(Duration::from_millis(2)),
        }
    }
    format!("err timeout capturing {}", want.display())
}

/// Visible text nodes matching `arg`, as `<node> <x> <y> <w> <h> <text>`.
fn find(arg: &str) -> String {
    let ui = ui::ui();
    let mut out = String::new();
    for (n, text, r) in ui.text_nodes() {
        if !text.contains(arg) {
            continue;
        }
        out.push_str(&format!(
            "{}.{} {} {} {} {} {}\n",
            n.index(),
            n.generation(),
            r[0],
            r[1],
            r[2],
            r[3],
            text
        ));
    }
    if out.is_empty() {
        out.push_str("none\n");
    }
    out
}

/// What holds the keyboard, and what it contains if it is a field.
fn focus() -> String {
    let ui = ui::ui();
    match ui.focus() {
        None => "none".into(),
        Some(n) => match ui.field_text(n) {
            Some(t) => format!("{}.{} field {t}", n.index(), n.generation()),
            None => format!("{}.{} node", n.index(), n.generation()),
        },
    }
}

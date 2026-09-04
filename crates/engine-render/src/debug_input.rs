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
//! Movement is **swept, not teleported** — a move or a drag is interpolated
//! into a step per [`STEP_PX`], one per frame. Jumping straight to the
//! destination proves the UI *can* do a thing; sweeping proves a hand could,
//! and is what exercises drag thresholds, hover transitions along the way, and
//! the drop mark being recomputed row by row.
//!
//! A white dot marks the pointer (`ENGINE_DEBUG_INPUT` only), so a capture
//! shows *why* something is highlighted rather than just that it is.
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

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use glam::Vec2;

use crate::input::{Input, Key, KeyCode, Keystroke, Mods, MouseButton};
use crate::ui;

/// Socket name under `$XDG_RUNTIME_DIR` when `ENGINE_DEBUG_INPUT=1`; set the
/// variable to a path instead to run two apps at once.
const DEFAULT_SOCK: &str = "engine-poke.sock";

/// Distance between interpolated positions along a swept move.
const STEP_PX: f32 = 24.0;

/// Ceiling on a sweep, so crossing the screen cannot queue hundreds of frames.
const MAX_STEPS: usize = 48;

/// Pointer marker: a white disc with a dark ring, so it reads on any surface.
const CURSOR_D: f32 = 10.0;

/// One frame's worth of injected input.
enum Step {
    Move([f32; 2]),
    Button([f32; 2], bool),
    /// The secondary button — what opens a context menu.
    Right([f32; 2], bool),
    Wheel(f32),
    Keys(Vec<Keystroke>),
    Key(KeyCode, bool),
}

static QUEUE: Mutex<VecDeque<Step>> = Mutex::new(VecDeque::new());

/// Where the queue *will* leave the pointer — the origin the next sweep
/// interpolates from. Tracked here rather than read from `Input`, which
/// belongs to the main thread.
static LAST_POS: Mutex<[f32; 2]> = Mutex::new([0.0, 0.0]);

/// The marker node, and where it was last drawn.
static CURSOR: Mutex<Option<ui::NodeId>> = Mutex::new(None);
static DRAWN: Mutex<Option<[f32; 2]>> = Mutex::new(None);

/// Set once the socket is bound. `pump` runs every frame either way, so this
/// keeps an ordinary run to one atomic load.
static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
    ENABLED.store(true, Ordering::Relaxed);
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
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    FRAMES.fetch_add(1, Ordering::Relaxed);
    let step = QUEUE.lock().pop_front();
    // The marker tracks whatever the engine believes, injected or not, so a
    // real mouse fighting the queue is visible rather than mysterious.
    if let Some(step) = step {
        apply(input, step);
    }
    show_cursor(input.cursor_position());
}

fn apply(input: &mut Input, step: Step) {
    match step {
        Step::Move(p) => input.inject_cursor(Vec2::new(p[0], p[1])),
        Step::Button(p, down) => {
            input.inject_cursor(Vec2::new(p[0], p[1]));
            input.inject_button(MouseButton::Left, down);
        }
        Step::Right(p, down) => {
            input.inject_cursor(Vec2::new(p[0], p[1]));
            input.inject_button(MouseButton::Right, down);
        }
        Step::Wheel(l) => input.inject_wheel(l),
        Step::Keys(ks) => {
            for k in ks {
                input.inject_keystroke(k);
            }
        }
        Step::Key(code, down) => input.inject_key(code, down),
    }
}

fn push(steps: impl IntoIterator<Item = Step>) {
    QUEUE.lock().extend(steps);
}

/// Interpolated positions from where the queue leaves the pointer to `to`,
/// one per frame, ending exactly on `to`.
fn sweep(to: [f32; 2]) -> Vec<Step> {
    let mut last = LAST_POS.lock();
    let from = *last;
    *last = to;
    drop(last);

    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
    let n = (((dx * dx + dy * dy).sqrt() / STEP_PX).ceil() as usize).clamp(1, MAX_STEPS);
    (1..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            Step::Move([from[0] + dx * t, from[1] + dy * t])
        })
        .collect()
}

// ── The pointer marker ──────────────────────────────────────────────────

/// Draw the marker at `p`. A still pointer costs one comparison; anything
/// else would re-emit paint order every frame and break the idle invariant
/// even with nothing moving.
fn show_cursor(p: Vec2) {
    let p = [p.x, p.y];
    {
        let mut drawn = DRAWN.lock();
        if *drawn == Some(p) {
            return;
        }
        *drawn = Some(p);
    }
    let mut ui = ui::ui();
    let mut slot = CURSOR.lock();
    let n = *slot.get_or_insert_with(|| new_cursor(&mut ui));

    let mut style = ui.node_style(n);
    style.inset.left = ui::style::px(p[0] - CURSOR_D * 0.5);
    style.inset.top = ui::style::px(p[1] - CURSOR_D * 0.5);
    ui.set_node_style(n, style);
    // Panels and the drag ghost are added to the root after this node, and
    // the pointer has to stay on top of both.
    ui.raise(n);
}

fn new_cursor(ui: &mut ui::UiCore) -> ui::NodeId {
    use ui::style::{px, LengthPercentageAuto, Position, Rect, Size, Style, TaffyAuto};
    let root = ui.root();
    // No `set_events`, so it is inert to the hit walk it exists to explain.
    let n = ui.node(
        root,
        Style {
            position: Position::Absolute,
            inset: Rect {
                left: px(0.0),
                top: px(0.0),
                right: LengthPercentageAuto::AUTO,
                bottom: LengthPercentageAuto::AUTO,
            },
            size: Size {
                width: px(CURSOR_D),
                height: px(CURSOR_D),
            },
            ..Default::default()
        },
    );
    ui.set_background(
        n,
        ui::UiStyle::fill(ui::rgb(255, 255, 255))
            .border(ui::rgba(0, 0, 0, 200), 1.0)
            .radius(CURSOR_D * 0.5),
    );
    n
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
        "move" => match target(arg) {
            Ok(p) => {
                push(sweep(p));
                format!("ok {} {}", p[0], p[1])
            }
            Err(e) => e,
        },
        "click" | "dblclick" | "rclick" => match target(arg) {
            Ok(p) => {
                // Swept, so the pointer enters the target the way a hand
                // would — hover styling and all.
                let mut steps = sweep(p);
                let times = if cmd == "dblclick" { 2 } else { 1 };
                let button = match cmd {
                    "rclick" => Step::Right,
                    _ => Step::Button,
                };
                for _ in 0..times {
                    steps.push(button(p, true));
                    steps.push(button(p, false));
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
                    let mut steps = sweep(from);
                    steps.push(Step::Button(from, true));
                    // The sweep is the point: it crosses the drag threshold
                    // once, then every row between, so the drop mark is
                    // recomputed the whole way rather than appearing at the end.
                    steps.extend(sweep(to));
                    // One still frame so the mark settles before the release.
                    steps.push(Step::Move(to));
                    steps.push(Step::Button(to, false));
                    push(steps);
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
            // The keystroke a text field reads *and* the physical press a
            // tool shortcut reads — one `poke key` is one key, whichever
            // half of the input the app happens to be looking at. Released
            // on the next frame, so `key_down` is true for exactly one.
            Ok((k, code)) => {
                let physical = code
                    .into_iter()
                    .flat_map(|c| [Step::Key(c, true), Step::Key(c, false)]);
                push([Step::Keys(vec![k])].into_iter().chain(physical));
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
        "rec" => rec(arg),
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
/// The physical code comes back beside the keystroke where there is one.
fn key(s: &str) -> Result<(Keystroke, Option<KeyCode>), String> {
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
    Ok((Keystroke::Key(k, mods), physical(name)))
}

/// The physical key a name stands for — letters and digits, plus the few
/// named ones a shortcut is ever bound to. `None` means "keystroke only",
/// which is all a text field needs anyway.
fn physical(name: &str) -> Option<KeyCode> {
    const LETTERS: [KeyCode; 26] = [
        KeyCode::KeyA,
        KeyCode::KeyB,
        KeyCode::KeyC,
        KeyCode::KeyD,
        KeyCode::KeyE,
        KeyCode::KeyF,
        KeyCode::KeyG,
        KeyCode::KeyH,
        KeyCode::KeyI,
        KeyCode::KeyJ,
        KeyCode::KeyK,
        KeyCode::KeyL,
        KeyCode::KeyM,
        KeyCode::KeyN,
        KeyCode::KeyO,
        KeyCode::KeyP,
        KeyCode::KeyQ,
        KeyCode::KeyR,
        KeyCode::KeyS,
        KeyCode::KeyT,
        KeyCode::KeyU,
        KeyCode::KeyV,
        KeyCode::KeyW,
        KeyCode::KeyX,
        KeyCode::KeyY,
        KeyCode::KeyZ,
    ];
    const DIGITS: [KeyCode; 10] = [
        KeyCode::Digit0,
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
        KeyCode::Digit9,
    ];
    match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => Some(KeyCode::Enter),
        "tab" => Some(KeyCode::Tab),
        "escape" | "esc" => Some(KeyCode::Escape),
        "space" => Some(KeyCode::Space),
        "delete" | "del" => Some(KeyCode::Delete),
        c => match c.chars().next().filter(|_| c.chars().count() == 1)? {
            l @ 'a'..='z' => Some(LETTERS[l as usize - 'a' as usize]),
            d @ '0'..='9' => Some(DIGITS[d as usize - '0' as usize]),
            _ => None,
        },
    }
}

/// Block until every queued step has been applied and consumed.
fn wait() -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if QUEUE.lock().is_empty() {
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

/// `rec <n> [command…]` — film the next `n` frames, optionally while running
/// `command`.
///
/// The nested command exists to remove a race: issuing the gesture and the
/// recording as two connections lets frames pass in between, and at three
/// thousand of them a second most of the gesture is over before filming
/// starts. Queued together, the two drain in lockstep.
fn rec(arg: &str) -> String {
    let (n, rest) = match arg.split_once(' ') {
        Some((n, r)) => (n, r.trim()),
        None => (arg, ""),
    };
    let Ok(n) = n.parse::<usize>() else {
        return "err rec <frames> [command…]".into();
    };
    if n == 0 || n > 600 {
        return "err rec takes 1..600 frames".into();
    }
    if !rest.is_empty() {
        let reply = dispatch(rest);
        if reply.starts_with("err") {
            return reply;
        }
    }
    let paths = crate::capture::request_many(n, None);
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if crate::capture::idle() {
            return match crate::capture::result() {
                Some(Err(e)) => format!("err {e}"),
                _ => paths.iter().map(|p| format!("{}\n", p.display())).collect(),
            };
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    "err timeout recording".into()
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

//! Global per-frame input state.
//!
//! [`Window`](crate::Window)'s event loop feeds every `WindowEvent` into the
//! accumulator via `feed_window_event` as it arrives, and clears the
//! per-frame transient state (`*_pressed` / `*_released` / deltas) via
//! `end_frame` once each frame's `Scene::update` has finished. Components
//! anywhere read it — via the free functions below or [`global`] directly —
//! with no plumbing through `Component::update` required.
//!
//! # Why no lock
//!
//! Writes (`feed_window_event`, `end_frame`) only ever happen on the
//! event-loop thread, and only ever *between* calls to `Scene::update` — never
//! while any component's `update` (and therefore any `global()` read
//! reference) is in flight. Since reads and writes are temporally
//! disjoint rather than actually concurrent, a `RwLock` here buys no safety,
//! only pointless per-component lock traffic during parallel `update` fan-out.
//! So the accumulator lives behind a raw cell instead: [`global`] hands out a
//! plain `&'static Input` for components to read lock-free, and
//! [`global_mut`] (crate-private, used only by [`Window`](crate::Window)'s
//! event loop) hands out the `&'static mut Input` used to write. Calling
//! [`global_mut`] while any `global()` reference is still alive is undefined
//! behavior — don't call it from component code, and don't hold onto a
//! `global()` reference across a frame boundary.

use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::sync::OnceLock;

use glam::Vec2;

pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode;

/// Modifier keys held when a [`Keystroke`] was produced.
///
/// Captured per keystroke rather than read back later, because a text field
/// consumes the queue *after* the events happened — by then the shift key
/// that made a selection may already be up.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Mods(u8);

impl Mods {
    pub const NONE: Self = Self(0);
    pub const SHIFT: Self = Self(1);
    pub const CTRL: Self = Self(1 << 1);
    pub const ALT: Self = Self(1 << 2);

    /// Whether *any* modifier in `m` is held.
    pub fn has(self, m: Self) -> bool {
        self.0 & m.0 != 0
    }
}

impl std::ops::BitOr for Mods {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// A key that *edits* rather than types.
///
/// Deliberately not a second copy of [`KeyCode`]: this is the editing
/// vocabulary a caret understands, and everything outside it stays with the
/// level-triggered `key_down` sets. It is also **logical**, not physical —
/// `Home` is wherever the layout puts it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Enter,
    Tab,
    Escape,
    /// A character key pressed with `Ctrl` held — `Ctrl+A`. Lowercased.
    /// Only ever emitted with a modifier, because an unmodified character is
    /// [`Keystroke::Text`] instead and nothing should have to handle both.
    Char(char),
}

/// One keyboard event, in the order it arrived.
///
/// The split is the one winit already makes and the one text editing needs:
/// *what to insert* is the OS's answer (layout, dead keys, AltGr and IME all
/// resolved before it reaches us), while *what to do* is a named key plus its
/// modifiers. A field that tried to reconstruct text from key codes would be
/// wrong on every keyboard layout but the author's.
#[derive(Clone, PartialEq, Debug)]
pub enum Keystroke {
    /// Printable text from a key press. Control characters are stripped, so
    /// `Enter` never arrives here as `"\r"`.
    Text(String),
    Key(Key, Mods),
}

/// Accumulated keyboard/mouse state for the current frame.
///
/// * `key_down` / `mouse_down` — level-triggered: true for every frame the
///   key/button is held.
/// * `key_pressed` / `key_released` (and the mouse equivalents) —
///   edge-triggered: true only for the single frame the transition happened
///   in. Cleared by [`end_frame`](Self::end_frame).
/// * `cursor_delta` / `scroll_delta` — accumulated since the last
///   `end_frame`, then reset to zero.
/// * `keystrokes` — an ordered *queue*, not a set. Typing is a sequence:
///   "ab" and "ba" differ, and two of the same character in one frame are two
///   insertions. That is exactly what the other fields cannot express, which
///   is why text input needed a channel of its own rather than a wider
///   `KeyCode` set.
pub struct Input {
    keys_down: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>,
    keys_released: HashSet<KeyCode>,
    buttons_down: HashSet<MouseButton>,
    buttons_pressed: HashSet<MouseButton>,
    buttons_released: HashSet<MouseButton>,
    cursor_position: Vec2,
    cursor_delta: Vec2,
    scroll_delta: f32,
    keystrokes: Vec<Keystroke>,
    mods: Mods,
}

impl Input {
    fn new() -> Self {
        Self {
            keys_down: HashSet::new(),
            keys_pressed: HashSet::new(),
            keys_released: HashSet::new(),
            buttons_down: HashSet::new(),
            buttons_pressed: HashSet::new(),
            buttons_released: HashSet::new(),
            cursor_position: Vec2::ZERO,
            cursor_delta: Vec2::ZERO,
            scroll_delta: 0.0,
            keystrokes: Vec::new(),
            mods: Mods::NONE,
        }
    }

    #[inline]
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }
    #[inline]
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }
    #[inline]
    pub fn key_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }

    #[inline]
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.buttons_down.contains(&button)
    }
    #[inline]
    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        self.buttons_pressed.contains(&button)
    }
    #[inline]
    pub fn mouse_released(&self, button: MouseButton) -> bool {
        self.buttons_released.contains(&button)
    }

    /// Cursor position in physical pixels, window-space (origin top-left).
    #[inline]
    pub fn cursor_position(&self) -> Vec2 {
        self.cursor_position
    }
    /// Cursor movement since the last frame, in physical pixels.
    #[inline]
    pub fn cursor_delta(&self) -> Vec2 {
        self.cursor_delta
    }
    /// Scroll wheel movement since the last frame, in "lines" (a
    /// `PixelDelta` trackpad event is normalised to the same units).
    #[inline]
    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    /// This frame's keyboard events, oldest first. Empty on the
    /// overwhelming majority of frames.
    #[inline]
    pub fn keystrokes(&self) -> &[Keystroke] {
        &self.keystrokes
    }

    /// Modifiers held right now. Level-triggered, unlike the copy each
    /// [`Keystroke`] carries.
    #[inline]
    pub fn mods(&self) -> Mods {
        self.mods
    }

    /// Feed a `winit` window event into the accumulator.
    pub(crate) fn feed_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, WindowEvent};

        match event {
            WindowEvent::ModifiersChanged(m) => {
                let s = m.state();
                self.mods = Mods(
                    (s.shift_key() as u8)
                        | ((s.control_key() as u8) << 1)
                        | ((s.alt_key() as u8) << 2),
                );
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Before the repeat guard below, deliberately: holding
                // backspace *should* keep deleting, and holding a letter
                // *should* keep typing. Repeat is noise for a level-triggered
                // key set and signal for a caret.
                if event.state == ElementState::Pressed {
                    self.push_keystroke(event);
                }
                // Only track physical keys — layout-independent, and OS key
                // repeat resends `Pressed` for a held key, which would
                // otherwise keep re-triggering `key_pressed`.
                if event.repeat {
                    return;
                }
                if let winit::keyboard::PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if self.keys_down.insert(code) {
                                self.keys_pressed.insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&code);
                            self.keys_released.insert(code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    if self.buttons_down.insert(*button) {
                        self.buttons_pressed.insert(*button);
                    }
                }
                ElementState::Released => {
                    self.buttons_down.remove(button);
                    self.buttons_released.insert(*button);
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                let cur = Vec2::new(position.x as f32, position.y as f32);
                self.cursor_delta += cur - self.cursor_position;
                self.cursor_position = cur;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 50.0,
                };
                self.scroll_delta += lines;
            }
            WindowEvent::Focused(false) => {
                // Losing focus (alt-tab, etc.) can swallow the matching
                // key/button-up event — clear held state so nothing reads
                // as stuck down for the rest of the session.
                self.keys_down.clear();
                self.buttons_down.clear();
                // Modifiers especially: alt-tab leaves `alt` held forever
                // otherwise, and every subsequent keystroke would arrive
                // wearing a modifier nobody is pressing.
                self.mods = Mods::NONE;
            }
            _ => {}
        }
    }

    /// Translate one `Pressed` key event into the editing queue.
    ///
    /// At most one [`Keystroke`] comes out. `Ctrl` is what splits the two
    /// branches: with it held a character key is a *command* (`Ctrl+A`), and
    /// the `text` the platform reports is a control code nobody wants
    /// inserted. `Alt` is deliberately not in that test — on many layouts
    /// AltGr arrives as `Alt` and produces real characters.
    fn push_keystroke(&mut self, event: &winit::event::KeyEvent) {
        use winit::keyboard::{Key as Logical, NamedKey};

        let named = match &event.logical_key {
            Logical::Named(n) => match n {
                NamedKey::Backspace => Some(Key::Backspace),
                NamedKey::Delete => Some(Key::Delete),
                NamedKey::ArrowLeft => Some(Key::Left),
                NamedKey::ArrowRight => Some(Key::Right),
                NamedKey::ArrowUp => Some(Key::Up),
                NamedKey::ArrowDown => Some(Key::Down),
                NamedKey::Home => Some(Key::Home),
                NamedKey::End => Some(Key::End),
                NamedKey::Enter => Some(Key::Enter),
                NamedKey::Tab => Some(Key::Tab),
                NamedKey::Escape => Some(Key::Escape),
                _ => None,
            },
            Logical::Character(s) if self.mods.has(Mods::CTRL) => {
                s.chars().next().map(|c| Key::Char(c.to_ascii_lowercase()))
            }
            _ => None,
        };
        if let Some(k) = named {
            self.keystrokes.push(Keystroke::Key(k, self.mods));
            return;
        }
        if self.mods.has(Mods::CTRL) {
            return;
        }
        // Control characters are filtered rather than trusted: platforms
        // disagree about whether `Enter` reports `"\r"`, `Tab` reports
        // `"\t"`, and `Escape` reports `"\u{1b}"`, and all three are already
        // named keys above.
        let text: String = event
            .text
            .iter()
            .flat_map(|s| s.chars())
            .filter(|c| !c.is_control())
            .collect();
        if !text.is_empty() {
            self.keystrokes.push(Keystroke::Text(text));
        }
    }

    /// Clear per-frame transient state (`*_pressed`, `*_released`, deltas).
    /// Called once per frame, after `Scene::update` has run, so every
    /// component's `update` for this frame observed the transition.
    pub(crate) fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.buttons_pressed.clear();
        self.buttons_released.clear();
        self.cursor_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
        self.keystrokes.clear();
    }
}

/// `UnsafeCell` isn't `Sync` on its own; wrapping it here documents (at the
/// type level) that sharing it across threads is an invariant this module
/// upholds itself — see the module-level "Why no lock" note.
struct InputCell(UnsafeCell<Input>);
unsafe impl Sync for InputCell {}

static INPUT: OnceLock<InputCell> = OnceLock::new();

fn cell() -> &'static InputCell {
    INPUT.get_or_init(|| InputCell(UnsafeCell::new(Input::new())))
}

/// Read-only access to this frame's input accumulator. Lock-free — safe to
/// call from any number of components' `update` in parallel, since it only
/// ever hands out shared references. See the module docs for why this is
/// sound without a `RwLock`.
pub fn global() -> &'static Input {
    // SAFETY: only ever aliased with `global_mut` across a frame boundary,
    // never concurrently — see the module-level "Why no lock" note.
    unsafe { &*cell().0.get() }
}

/// Mutable access to the input accumulator, for [`Window`](crate::Window)'s
/// event loop only: feeding window events and clearing per-frame transient
/// state between frames. Never call this while any `global()` reference
/// might still be alive (i.e. never during `Scene::update`'s component
/// fan-out).
pub(crate) fn global_mut() -> &'static mut Input {
    // SAFETY: see `global`'s safety comment; the event loop is the only
    // caller and never overlaps a `Scene::update` call with this one.
    unsafe { &mut *cell().0.get() }
}

// ── Convenience free functions ──────────────────────────────────────────
//
// Thin wrappers so a component's `update` can write `input::key_down(...)`
// instead of `input::global().key_down(...)`.

pub fn key_down(key: KeyCode) -> bool {
    global().key_down(key)
}
pub fn key_pressed(key: KeyCode) -> bool {
    global().key_pressed(key)
}
pub fn key_released(key: KeyCode) -> bool {
    global().key_released(key)
}
pub fn mouse_down(button: MouseButton) -> bool {
    global().mouse_down(button)
}
pub fn mouse_pressed(button: MouseButton) -> bool {
    global().mouse_pressed(button)
}
pub fn mouse_released(button: MouseButton) -> bool {
    global().mouse_released(button)
}
pub fn cursor_position() -> Vec2 {
    global().cursor_position()
}
pub fn cursor_delta() -> Vec2 {
    global().cursor_delta()
}
pub fn scroll_delta() -> f32 {
    global().scroll_delta()
}
pub fn keystrokes() -> &'static [Keystroke] {
    global().keystrokes()
}
pub fn mods() -> Mods {
    global().mods()
}

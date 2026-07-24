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

/// Accumulated keyboard/mouse state for the current frame.
///
/// * `key_down` / `mouse_down` — level-triggered: true for every frame the
///   key/button is held.
/// * `key_pressed` / `key_released` (and the mouse equivalents) —
///   edge-triggered: true only for the single frame the transition happened
///   in. Cleared by [`end_frame`](Self::end_frame).
/// * `cursor_delta` / `scroll_delta` — accumulated since the last
///   `end_frame`, then reset to zero.
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

    /// Feed a `winit` window event into the accumulator.
    pub(crate) fn feed_window_event(&mut self, event: &winit::event::WindowEvent) {
        use winit::event::{ElementState, WindowEvent};

        match event {
            WindowEvent::KeyboardInput { event, .. } => {
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
            }
            _ => {}
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

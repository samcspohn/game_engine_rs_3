//! Global per-frame input state.
//!
//! [`Window`](crate::Window)'s event loop feeds every `WindowEvent` into
//! [`global`]'s accumulator via `feed_window_event` as it arrives, and clears
//! the per-frame transient state (`*_pressed` / `*_released` / deltas) via
//! `end_frame` once each frame's `Scene::update` has had a chance to observe
//! it. Components anywhere read the global — via the free functions below or
//! [`global`] directly — with no plumbing through `Component::update`
//! required.

use std::collections::HashSet;
use std::sync::OnceLock;

use glam::Vec2;
use parking_lot::RwLock;

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

static INPUT: OnceLock<RwLock<Input>> = OnceLock::new();

/// The global input accumulator. [`Window`](crate::Window)'s event loop is
/// the sole writer; components anywhere may read it concurrently — reads
/// take a `parking_lot::RwLock` read guard, so parallel `Component::update`
/// calls never serialise on each other here.
pub fn global() -> &'static RwLock<Input> {
    INPUT.get_or_init(|| RwLock::new(Input::new()))
}

// ── Convenience free functions ──────────────────────────────────────────
//
// Thin wrappers so a component's `update` can write `input::key_down(...)`
// instead of `input::global().read().key_down(...)`.

pub fn key_down(key: KeyCode) -> bool {
    global().read().key_down(key)
}
pub fn key_pressed(key: KeyCode) -> bool {
    global().read().key_pressed(key)
}
pub fn key_released(key: KeyCode) -> bool {
    global().read().key_released(key)
}
pub fn mouse_down(button: MouseButton) -> bool {
    global().read().mouse_down(button)
}
pub fn mouse_pressed(button: MouseButton) -> bool {
    global().read().mouse_pressed(button)
}
pub fn mouse_released(button: MouseButton) -> bool {
    global().read().mouse_released(button)
}
pub fn cursor_position() -> Vec2 {
    global().read().cursor_position()
}
pub fn cursor_delta() -> Vec2 {
    global().read().cursor_delta()
}
pub fn scroll_delta() -> f32 {
    global().read().scroll_delta()
}

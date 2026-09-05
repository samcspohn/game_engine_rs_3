//! The name-keyed component-type registry a project's script dylib fills.
//!
//! `TypeId` is not stable across builds and cannot name a type in a file or
//! a menu (ADR-0010 §2), so this keys on the derive's `TYPE_NAME`.
//!
//! See [`docs/notes/scripts.md`](../../../docs/notes/scripts.md) for the
//! link mode that makes the static below one copy across the boundary.

use parking_lot::Mutex;

use crate::component::{Component, EntityMut};

/// One registered component type: its file-stable name, and how to put one
/// on an entity without naming its Rust type.
///
/// Nothing here reads or writes the component. What it holds is reflected —
/// `#[export]` says what persists, and `Export` is what walks it.
#[derive(Clone, Copy)]
pub struct ComponentType {
    pub name: &'static str,
    /// Takes an [`EntityMut`] rather than an entity id, so the one caller
    /// that has `&mut World` — the frame boundary — is the only one that can
    /// use it.
    pub add: fn(&mut EntityMut),
}

impl ComponentType {
    pub fn of<T>() -> Self
    where
        T: Component + Clone + Default + Send + Sync + 'static,
    {
        Self {
            name: T::default().type_name(),
            add: |entity| {
                entity.add_component(T::default());
            },
        }
    }
}

static TYPES: Mutex<Vec<ComponentType>> = Mutex::new(Vec::new());

/// Add `types` to the process-wide list, ignoring names already present so a
/// dylib loaded twice does not double its menu.
pub fn register(types: &[ComponentType]) {
    let mut list = TYPES.lock();
    for t in types {
        if !list.iter().any(|e| e.name == t.name) {
            list.push(*t);
        }
    }
}

pub fn types() -> Vec<ComponentType> {
    TYPES.lock().clone()
}

pub fn find(name: &str) -> Option<ComponentType> {
    TYPES.lock().iter().find(|t| t.name == name).copied()
}

/// Address of the registry this crate's code writes to.
///
/// The plugin's entry point returns its own. Equal means host and plugin
/// resolved `engine-core` to one dylib; unequal means each got a static copy,
/// and every `global()` registry in the engine is doubled the same way — a
/// mesh the plugin loads would be invisible to the renderer. It is not
/// recoverable, so the loader refuses rather than running half-connected.
pub fn registry_addr() -> usize {
    &TYPES as *const _ as usize
}

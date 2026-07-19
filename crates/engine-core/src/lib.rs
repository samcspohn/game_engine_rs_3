#![feature(sync_unsafe_cell)]
//! Core types and traits for the game engine.
//!
//! This crate is the shared vocabulary that **every** other workspace crate
//! depends on.  It has no rendering, windowing, or asset-pipeline
//! dependencies — only math, concurrency, and pure game-logic abstractions.
//!
//! # Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`transform`] | Hierarchical transform system (`TransformHierarchy`, `Transform`, `_Transform`, …) |
//! | [`component`] | ECS (`Component`, `ComponentStorage`, `ComponentRegistry`, `Entity`, `Scene`) |
//! | [`util`] | Internal containers (`Avail`, `Storage`, `SegStorage`, …) |
//! | [`mesh`] | CPU-side mesh types (`Vertex`, `Mesh`, `Aabb`) and primitive generators (`mesh::primitives`) |
//! | [`asset`] | GPU-agnostic mesh asset registry (`AssetRegistry`, `MeshId`, `MeshSlot`) with a lazy global handle |
//! | [`texture`] | GPU-agnostic texture asset registry (`TextureRegistry`, `TextureId`, `TextureSlot`) — same redirect model |
//! | [`scene_asset`] | glTF/GLB → scene-template assets (subscenes): streaming hierarchy load + queued instantiation |

pub mod transform;
pub mod component;
pub mod util;
pub mod mesh;
pub mod asset;
pub mod texture;
pub mod scene_asset;

// ---------------------------------------------------------------------------
// Re-exports — the most-commonly-used types, one `use engine_core::*;` away.
// ---------------------------------------------------------------------------

pub use component::{Component, ComponentRegistry, ComponentStorage, Entity, Scene};
pub use transform::{Transform, TransformHierarchy, _Transform};
pub use mesh::{Aabb, Mesh, Vertex};
pub use asset::{AssetRegistry, MeshId, MeshSlot};
pub use texture::{TextureData, TextureId, TextureRegistry, TextureSlot};
pub use scene_asset::{SceneId, SceneLoadState};

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/// The central application object that represents a running game instance.
///
/// Construct one with [`App::new`] and pass it to the platform layer (e.g.
/// `engine_render::Window`) to drive the game loop.
pub struct App;

impl App {
    /// Create a new, unconfigured `App`.
    pub fn new() -> Self {
        App
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

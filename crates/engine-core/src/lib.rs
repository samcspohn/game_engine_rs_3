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
//! | [`component`] | ECS (`Component`, `ComponentStorage`, `ComponentRegistry`, `Entity`, `World`) |
//! | [`util`] | Internal containers (`Avail`, `Storage`, `SegStorage`, …) |
//! | [`mesh`] | CPU-side mesh types (`Vertex`, `Mesh`, `Aabb`) and primitive generators (`mesh::primitives`) |
//! | [`asset`] | GPU-agnostic mesh asset registry (`AssetRegistry`, `MeshId`, `MeshSlot`) with a lazy global handle |
//! | [`texture`] | GPU-agnostic texture asset registry (`TextureRegistry`, `TextureId`, `TextureSlot`) — same redirect model |
//! | [`material`] | GPU-agnostic material registry (`MaterialRegistry`, `MaterialId`, `MaterialData`) — shared/deduped, immediate resolve |
//! | [`scene_asset`] | glTF/GLB → scene-template assets (subscenes): streaming hierarchy load + queued instantiation |
//! | [`reflect`] | `#[derive(Export)]` and the value model the inspector, save walk and deltas share (ADR-0010 §3) |
//! | [`worlds`] | Every world in the process (`new_world`, `WorldHandle`), ADR-0011 §3 |

/// So the paths `#[derive(Export)]` emits resolve inside this crate too.
extern crate self as engine_core;

pub mod transform;
pub mod component;
pub mod util;
pub mod mesh;
pub mod asset;
pub mod texture;
pub mod material;
pub mod scene_asset;
pub mod reflect;
pub mod worlds;

// ---------------------------------------------------------------------------
// Re-exports — the most-commonly-used types, one `use engine_core::*;` away.
// ---------------------------------------------------------------------------

pub use component::{
    Component, ComponentRegistry, ComponentStorage, Entity, EntityMut, EntityView, World,
};
pub use worlds::{new_world, WorldHandle};
pub use transform::{Transform, TransformHierarchy, WorldId, _Transform};
pub use mesh::{Aabb, Mesh, Vertex};
pub use asset::{AssetRegistry, MeshId, MeshSlot};
pub use texture::{ColorSpace, TextureData, TextureId, TextureRegistry, TextureSlot};
pub use material::{MaterialData, MaterialId, MaterialRegistry, MaterialSlot};
pub use scene_asset::{SceneId, SceneLoadState};
pub use reflect::{AssetKind, AssetRef, Export, Exportable, PropertyInfo, Value, ValueKind};

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

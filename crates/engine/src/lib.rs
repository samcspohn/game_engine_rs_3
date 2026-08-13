//! Public game-facing API for the engine.
//!
//! This umbrella crate re-exports the engine subsystems so consumers can write:
//!
//! ```no_run
//! use engine::{Window, MeshRenderer};
//! use engine::transform::{TransformHierarchy, _Transform};
//! use engine::glam::Quat;
//! use engine::mesh::primitives;
//! ```
//!
//! instead of having to name the individual implementation crates.
//!
//! # Design intent
//! * `engine` — what games depend on.  Contains no editor tooling.
//! * `engine_editor_api` — what the editor binary additionally depends on.
//!   Game code must **never** add this as a dependency.

pub use engine_core::App;

// Mesh types and primitive generators
pub use engine_core::mesh;
pub use engine_core::{Aabb, Mesh, Vertex};

// Mesh asset registry (handles + global registry).
pub use engine_core::asset;
pub use engine_core::texture;
pub use engine_core::{AssetRegistry, MeshId, MeshSlot};
pub use engine_core::{ColorSpace, TextureData, TextureId, TextureRegistry, TextureSlot};
pub use engine_core::material;
pub use engine_core::{MaterialData, MaterialId, MaterialRegistry, MaterialSlot};

// GLB scene-template assets (subscenes): request → spawn → streamed in.
pub use engine_core::scene_asset;
pub use engine_core::{SceneId, SceneLoadState};

// Transform hierarchy (CPU-side scene graph).
pub use engine_core::transform;

// ECS — Component / Entity / World live here.
pub use engine_core::component;
pub use engine_core::{
    Component, ComponentRegistry, ComponentStorage, Entity, EntityView, World, WorldId,
};
// `engine_core::worlds` — the frame's world list — is deliberately **not**
// re-exported: reaching another world is the editor's privilege (ADR-0011 §3).

// Reflection: `#[derive(Export)]` and the value model the inspector, the save
// walk and per-property deltas all read (ADR-0010 §3).
pub use engine_core::reflect;
pub use engine_core::{AssetKind, AssetRef, Export, Exportable, PropertyInfo, Value, ValueKind};
/// Where `#[derive(Export)]` aims its generated paths in a crate that depends
/// on this facade rather than on `engine-core` directly.
#[doc(hidden)]
pub use engine_core;

// Renderer + scene-frame API.
pub use engine_render::{
    active_camera, set_active_camera, CameraComponent, MeshRenderer, OrbitController, Window,
};

// Global per-frame input accumulator (keyboard + mouse), plus the winit
// key/button types its API is keyed on.
pub use engine_render::input;
pub use engine_render::{Input, KeyCode, MouseButton};

// Retained-mode UI (ADR-0006). `ui::ui()` locks the global store; build a
// tree from `main` or from a component's `init` — it owns no Vulkan, so it
// works before the window exists.
pub use engine_render::ui;

// Frame rate, frame time, and swapchain extent, published once per frame.
pub use engine_render::stats;

// Re-export glam so games don't need their own dep.
pub use glam;

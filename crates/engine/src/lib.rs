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
pub use engine_core::{TextureData, TextureId, TextureRegistry, TextureSlot};
pub use engine_core::material;
pub use engine_core::{MaterialData, MaterialId, MaterialRegistry, MaterialSlot};

// GLB scene-template assets (subscenes): request → spawn → streamed in.
pub use engine_core::scene_asset;
pub use engine_core::{SceneId, SceneLoadState};

// Transform hierarchy (CPU-side scene graph).
pub use engine_core::transform;

// ECS — Component / Entity / Scene live here.
pub use engine_core::component;
pub use engine_core::{Component, ComponentRegistry, ComponentStorage, Entity, Scene};

// Renderer + scene-frame API.
pub use engine_render::{CameraComponent, MeshRenderer, OrbitController, Window};

// Global per-frame input accumulator (keyboard + mouse), plus the winit
// key/button types its API is keyed on.
pub use engine_render::input;
pub use engine_render::{Input, KeyCode, MouseButton};

// Re-export glam so games don't need their own dep.
pub use glam;

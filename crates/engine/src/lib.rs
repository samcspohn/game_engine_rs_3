//! Public game-facing API for the engine.
//!
//! This umbrella crate re-exports the engine subsystems so consumers can write:
//!
//! ```no_run
//! use engine::{Window, RenderInstance};
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

// Transform hierarchy (CPU-side scene graph).
pub use engine_core::transform;

// ECS — Component / Entity / Scene live here.
pub use engine_core::component;
pub use engine_core::{Component, ComponentRegistry, ComponentStorage, Entity, Scene};

// Renderer + scene-frame API.
pub use engine_render::{Camera, OrbitController, RenderInstance, Window};

// Re-export glam so games don't need their own dep.
pub use glam;

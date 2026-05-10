//! Public game-facing API for the engine.
//!
//! This umbrella crate re-exports every symbol from [`engine_core`] and
//! [`engine_render`] so consumers can write:
//!
//! ```no_run
//! use engine::{App, Window};
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
pub use engine_render::Window;

// Mesh types and primitive generators
pub use engine_core::mesh;
pub use engine_core::{Aabb, Mesh, Vertex};

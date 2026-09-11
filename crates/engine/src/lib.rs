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
pub use engine_core::material;
pub use engine_core::texture;
pub use engine_core::{AssetRegistry, MeshId, MeshSlot};
pub use engine_core::{ColorSpace, TextureData, TextureId, TextureRegistry, TextureSlot};
pub use engine_core::{MaterialData, MaterialId, MaterialRegistry, MaterialSlot};

// GLB scene-template assets (subscenes): request → spawn → streamed in.
pub use engine_core::scene_asset;
pub use engine_core::{SceneId, SceneLoadState};

// Saving and loading a scene: the shape of a subtree and its components.
pub use engine_core::scene_file;

// The directory every stored asset path is relative to.
pub use engine_core::project;

// Transform hierarchy (CPU-side scene graph).
pub use engine_core::transform;

// ECS — Component / Entity / World live here.
pub use engine_core::component;
pub use engine_core::{
    Component, ComponentRegistry, ComponentStorage, Entity, EntityMut, EntityView, World, WorldId,
};
// Worlds are engine-owned and refcounted; reaching another one is an ordinary
// capability, so this is here rather than in `engine-editor-api` (ADR-0011 §3).
pub use engine_core::worlds;
pub use engine_core::WorldHandle;

/// Register a new, empty world.
///
/// Wrapped rather than re-exported so that the engine's own component types
/// are in the registry before anything can name one. A scene file names
/// components by type, and making a world is the first thing a game does —
/// which is earlier than [`Window::new`], the other place that says it.
pub fn new_world() -> WorldHandle {
    engine_render::register_builtin_components();
    engine_core::new_world()
}

// Reflection: `#[derive(Export)]` and the value model the inspector, the save
// walk and per-property deltas all read (ADR-0010 §3).
/// Where `#[derive(Export)]` aims its generated paths in a crate that depends
/// on this facade rather than on `engine-core` directly.
#[doc(hidden)]
pub use engine_core;
pub use engine_core::reflect;
pub use engine_core::script;
pub use engine_core::ComponentType;
pub use engine_core::{AssetKind, AssetRef, Export, Exportable, PropertyInfo, Value, ValueKind};

// Renderer + scene-frame API.
// No `CameraHandle`: a game's camera surface is `CameraComponent`, which
// mints and drives one from its entity's pose. Owning a camera outright is
// `engine-editor-api`'s (ADR-0011 §4).
pub use engine_render::{CameraComponent, MeshRenderer, OrbitController, Window};

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

// ---------------------------------------------------------------------------
// Scripts
// ---------------------------------------------------------------------------

/// Register each listed component type by name, both ways a project's
/// components arrive: `engine_register_scripts` is what the editor calls
/// after `dlopen`, and `register` is what a game binary — which links the
/// same crate as an rlib and never opens anything — calls itself.
///
/// Both are needed for a scene file to mean the same thing in the editor and
/// in a packaged build, because a component the registry cannot name is a
/// component the loader drops.
///
/// Defined here rather than in `engine-core` so `$crate` resolves through the
/// one crate a project actually depends on.
#[macro_export]
macro_rules! declare_scripts {
    ($($t:ty),* $(,)?) => {
        /// Put this crate's component types in the registry. Idempotent.
        pub fn register() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                $crate::script::register(&[
                    $($crate::ComponentType::of::<$t>()),*
                ]);
            });
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn engine_register_scripts() -> usize {
            register();
            $crate::script::registry_addr()
        }
    };
}

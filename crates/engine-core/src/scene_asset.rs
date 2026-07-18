//! GLB → scene-template assets ("subscenes", à la Godot's `PackedScene`).
//!
//! A `.glb` file loads as a **[`SceneTemplate`]**: the node hierarchy with
//! local transforms plus, per mesh primitive, a [`MeshRendererProxy`]
//! holding a placeholder [`MeshId`]. The template is *data* — it is never
//! rendered and owns no ECS storage. Games instantiate it into the live
//! [`Scene`] any number of times; instantiation converts each proxy into a
//! real renderer component via the `attach_renderer` callback (the render
//! crate passes `MeshRenderer::from_id`).
//!
//! # Streaming pipeline — nothing waits on vertex data
//!
//! The predecessor loader fully decoded each node (attributes, tangents,
//! blocking uploads) before visiting its children, so *nothing* appeared
//! until *everything* was decoded. Here the load decouples into phases:
//!
//! 1. **Parse** (one background task): read the file, parse the GLB
//!    container's JSON chunk. The BIN chunk is kept as a shared
//!    `Arc<[u8]>`. No image decode (the `gltf` crate's `import` feature is
//!    deliberately disabled), no accessor reads.
//! 2. **Hierarchy** (same task): walk the node tree iteratively, minting a
//!    deduped placeholder `MeshId` per primitive (virtual path
//!    `file.glb#mesh{i}/prim{j}`). The template is Ready in roughly the
//!    time it takes to read the file — milliseconds after that.
//! 3. **Mesh decodes** (one fire-and-forget background task per unique
//!    primitive): slice accessors out of the shared BIN blob, build a
//!    [`Mesh`], and [`AssetRegistry::resolve`] the redirect — a single
//!    write. Meshes pop in individually, in completion order, and because
//!    the redirect table is global, **every already-spawned instance**
//!    upgrades from placeholder simultaneously.
//! 4. **Textures**: deferred; will be the same pattern over a
//!    `TextureId` redirect table.
//!
//! The thread pool's background-occupancy cap (half the workers) keeps a
//! large decode burst from starving per-frame `parallel_for` dispatches.
//!
//! # Instantiation & GPU parent composition
//!
//! Templates store glTF **local** TRS, and instantiation preserves it:
//! each spawned entity gets its node's local TRS plus a parent link. The
//! renderer composes world transforms on the GPU — `mvp_build_cs` walks
//! the per-slot parent buffer upward each frame (fed by the hierarchy's
//! streamed parent updates), using the same composition order as
//! `TransformHierarchy::get_global_transform`. Moving the instance root
//! therefore moves the whole instance without touching any child TRS.
//!
//! Spawns are queued ([`spawn_subscene`]) and materialised by the per-frame
//! [`drain_ready_spawns`] once their template is Ready — a spawn requested
//! mid-load appears a frame or two later with placeholder meshes that then
//! stream in.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use glam::{Quat, Vec2, Vec3};

use crate::asset::{self, MeshId};
use crate::component::{Entity, Scene};
use crate::mesh::{Mesh, Vertex};
use crate::transform::_Transform;

// ─────────────────────────────────────────────────────────────────────────────
// Identifiers & template data
// ─────────────────────────────────────────────────────────────────────────────

/// Stable handle to a scene-template asset. Allocated per unique requested
/// path (deduped), valid for the process lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SceneId(pub u32);

/// A renderer *proxy* inside a template: it holds the [`MeshId`] but has no
/// ECS presence and no spawn-queue side effects — the template itself is
/// never rendered. Instantiation converts each proxy into a real renderer
/// component (which bumps the registry refcount) via the attach callback.
#[derive(Clone, Copy, Debug)]
pub struct MeshRendererProxy {
    pub mesh_id: MeshId,
}

/// One node of a parsed template. `parent` indexes `SceneTemplate::nodes`;
/// the build order guarantees parents precede their children.
struct TemplateNode {
    /// glTF-local TRS (kept local at instantiation; composed to world on
    /// the GPU via the parent chain).
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    name: String,
    parent: Option<u32>,
    renderer: Option<MeshRendererProxy>,
}

/// An immutable, instantiable scene template parsed from a GLB.
pub struct SceneTemplate {
    nodes: Vec<TemplateNode>,
}

impl SceneTemplate {
    /// Number of nodes (entities an instantiation will create, excluding
    /// the instance root).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Externally observable load state of a template.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneLoadState {
    /// Parse/hierarchy phase still running (meshes may also still stream
    /// in *after* Ready — that is tracked per-`MeshId` by the registry).
    Loading,
    /// Hierarchy available; [`spawn_subscene`] requests will materialise.
    Ready,
    /// Parse failed (reported loudly via stderr). Pending spawns for it
    /// are dropped with another loud report.
    Failed,
}

enum TemplateState {
    Loading,
    Ready(Arc<SceneTemplate>),
    Failed,
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry + spawn queue
// ─────────────────────────────────────────────────────────────────────────────

struct SceneAssets {
    /// Dedup cache: path hash → allocated `SceneId`.
    by_hash: HashMap<u64, SceneId>,
    /// Kept alongside the hash for loud diagnostics.
    paths: Vec<PathBuf>,
    states: Vec<TemplateState>,
    /// Queued `spawn_subscene` requests, drained per frame by
    /// [`drain_ready_spawns`] as their templates resolve.
    pending_spawns: Vec<(SceneId, _Transform)>,
}

static SCENE_ASSETS: OnceLock<Mutex<SceneAssets>> = OnceLock::new();

fn registry() -> &'static Mutex<SceneAssets> {
    SCENE_ASSETS.get_or_init(|| {
        Mutex::new(SceneAssets {
            by_hash: HashMap::new(),
            paths: Vec::new(),
            states: Vec::new(),
            pending_spawns: Vec::new(),
        })
    })
}

fn lock() -> std::sync::MutexGuard<'static, SceneAssets> {
    registry().lock().expect("scene asset registry mutex poisoned")
}

/// Deduped request for the scene template at `path`. On a cache miss the
/// parse + hierarchy build is handed to the pool as a background task
/// (deferred until the pool exists, like mesh loads).
pub fn request_scene(path: impl Into<PathBuf>) -> SceneId {
    let path: PathBuf = path.into();
    let hash = hash_path(&path);
    let mut reg = lock();
    if let Some(&id) = reg.by_hash.get(&hash) {
        return id;
    }
    let id = SceneId(reg.states.len() as u32);
    reg.states.push(TemplateState::Loading);
    reg.paths.push(path.clone());
    reg.by_hash.insert(hash, id);
    drop(reg);

    asset::spawn_when_pool_ready(move || match build_template(&path) {
        Ok(template) => {
            println!(
                "scene template ready: {} ({} nodes)",
                path.display(),
                template.node_count()
            );
            lock().states[id.0 as usize] = TemplateState::Ready(Arc::new(template));
        }
        Err(e) => {
            // No silent fallback: report loudly and mark Failed.
            eprintln!("scene load failed for {}: {e}", path.display());
            lock().states[id.0 as usize] = TemplateState::Failed;
        }
    });
    id
}

/// Current load state of `id`.
pub fn load_state(id: SceneId) -> SceneLoadState {
    match &lock().states[id.0 as usize] {
        TemplateState::Loading => SceneLoadState::Loading,
        TemplateState::Ready(_) => SceneLoadState::Ready,
        TemplateState::Failed => SceneLoadState::Failed,
    }
}

/// Queue an instantiation of `scene_id` at world transform `at` (its
/// `parent` field, if set, becomes the structural parent of the instance
/// root). Fire-and-forget: the instance materialises on a subsequent
/// [`drain_ready_spawns`] once the template is Ready — with placeholder
/// meshes if primitive decodes are still streaming in.
pub fn spawn_subscene(scene_id: SceneId, at: _Transform) {
    lock().pending_spawns.push((scene_id, at));
}

/// Materialise every queued spawn whose template has resolved. Called once
/// per frame by the render loop, before `Scene::update`. Spawns whose
/// template is still Loading stay queued; spawns of Failed templates are
/// dropped loudly. `attach_renderer` converts a template proxy into the
/// real renderer component for `entity` (the render crate passes
/// `MeshRenderer::from_id`). Returns the root entity of each new instance.
pub fn drain_ready_spawns(
    scene: &mut Scene,
    mut attach_renderer: impl FnMut(&mut Scene, Entity, MeshId),
) -> Vec<Entity> {
    // Decide under the lock, instantiate outside it (attach_renderer may
    // reach back into other global registries).
    let mut ready: Vec<(Arc<SceneTemplate>, _Transform)> = Vec::new();
    {
        let mut reg = lock();
        let mut kept = Vec::new();
        for (id, at) in std::mem::take(&mut reg.pending_spawns) {
            match &reg.states[id.0 as usize] {
                TemplateState::Loading => kept.push((id, at)),
                TemplateState::Ready(t) => ready.push((t.clone(), at)),
                TemplateState::Failed => {
                    eprintln!(
                        "dropping subscene spawn of failed template {}",
                        reg.paths[id.0 as usize].display()
                    );
                }
            }
        }
        reg.pending_spawns = kept;
    }
    ready
        .into_iter()
        .map(|(template, at)| {
            let root = instantiate(scene, &template, at, &mut attach_renderer);
            println!(
                "subscene instantiated: {} entities (root {})",
                template.node_count() + 1,
                root.id
            );
            root
        })
        .collect()
}

/// Create the entities for one template instance. Each entity keeps its
/// glTF **local** TRS and a parent link; the renderer composes world TRS
/// on the GPU by walking the per-slot parent buffer (`mvp_build_cs`), so
/// moving the instance root moves the whole instance for free.
fn instantiate(
    scene: &mut Scene,
    template: &SceneTemplate,
    at: _Transform,
    attach_renderer: &mut impl FnMut(&mut Scene, Entity, MeshId),
) -> Entity {
    let root = scene.new_entity(at);
    let mut entities: Vec<u32> = Vec::with_capacity(template.nodes.len());
    for node in &template.nodes {
        let parent_entity = match node.parent {
            Some(p) => entities[p as usize],
            None => root.id,
        };
        let entity = scene.new_entity(_Transform {
            position: node.position,
            rotation: node.rotation,
            scale: node.scale,
            name: node.name.clone(),
            parent: Some(parent_entity),
        });
        entities.push(entity.id);
        if let Some(proxy) = &node.renderer {
            attach_renderer(scene, entity, proxy.mesh_id);
        }
    }
    root
}

// ─────────────────────────────────────────────────────────────────────────────
// GLB parse → template build (phase 1) + per-primitive decode spawns (phase 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Read + parse a GLB and build its template. Touches only the JSON chunk;
/// per-primitive decode tasks (sharing the BIN blob via `Arc`) are spawned
/// for every *newly requested* mesh id before returning. Runs on a pool
/// background task.
fn build_template(path: &Path) -> Result<SceneTemplate, String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some(e) if e.eq_ignore_ascii_case("glb") => {}
        other => return Err(format!("unsupported scene format: {other:?} (only .glb)")),
    }
    let bytes = std::fs::read(path).map_err(|e| format!("read error: {e}"))?;
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice(&bytes).map_err(|e| format!("GLB parse error: {e}"))?;
    drop(bytes);
    let blob: Arc<[u8]> = blob
        .ok_or_else(|| "GLB has no BIN chunk".to_string())?
        .into();
    // Only the embedded BIN chunk is supported; fail loudly on external
    // buffer URIs rather than partially loading.
    for buffer in document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source() {
            return Err(format!("external buffer {uri:?} unsupported in .glb scene assets"));
        }
    }
    let document = Arc::new(document);

    let mut nodes: Vec<TemplateNode> = Vec::new();
    // Iterative DFS — the traversal never touches vertex data, so the
    // hierarchy exists as fast as the JSON can be walked.
    let mut stack: Vec<(gltf::Node, Option<u32>)> = Vec::new();
    for doc_scene in document.scenes() {
        for node in doc_scene.nodes() {
            stack.push((node, None));
        }
    }
    while let Some((node, parent)) = stack.pop() {
        let (t, r, s) = node.transform().decomposed();
        let name = node.name().unwrap_or("").to_string();
        let node_idx = nodes.len() as u32;
        nodes.push(TemplateNode {
            position: Vec3::from_array(t),
            rotation: Quat::from_array(r),
            scale: Vec3::from_array(s),
            name: name.clone(),
            parent,
            renderer: None,
        });
        if let Some(mesh) = node.mesh() {
            let prims: Vec<_> = mesh.primitives().collect();
            for prim in &prims {
                let proxy = MeshRendererProxy {
                    mesh_id: request_primitive(&document, &blob, path, mesh.index(), prim.index()),
                };
                if prims.len() == 1 {
                    // Single primitive rides on the node entity itself.
                    nodes[node_idx as usize].renderer = Some(proxy);
                } else {
                    // One child entity per primitive: a primitive is the
                    // future material boundary.
                    nodes.push(TemplateNode {
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                        name: format!("{name}:prim{}", prim.index()),
                        parent: Some(node_idx),
                        renderer: Some(proxy),
                    });
                }
            }
        }
        for child in node.children() {
            stack.push((child, Some(node_idx)));
        }
    }
    Ok(SceneTemplate { nodes })
}

/// Mint (or dedup) the `MeshId` for one primitive and, on first request,
/// spawn its background decode task.
fn request_primitive(
    document: &Arc<gltf::Document>,
    blob: &Arc<[u8]>,
    path: &Path,
    mesh_idx: usize,
    prim_idx: usize,
) -> MeshId {
    // Virtual sub-asset path: keys the registry's dedup cache, so two
    // nodes (or two templates) sharing a primitive share the MeshId.
    let virtual_path = format!("{}#mesh{mesh_idx}/prim{prim_idx}", path.display());
    let (mesh_id, needs_load) = asset::global()
        .lock()
        .expect("asset registry mutex poisoned")
        .request(Path::new(&virtual_path));
    if needs_load {
        let document = document.clone();
        let blob = blob.clone();
        asset::spawn_when_pool_ready(move || {
            match decode_primitive(&document, &blob, mesh_idx, prim_idx) {
                Ok(mesh) => {
                    asset::global()
                        .lock()
                        .expect("asset registry mutex poisoned")
                        .resolve(mesh_id, Arc::new(mesh));
                }
                Err(e) => {
                    eprintln!("asset load failed for {virtual_path}: {e}");
                    asset::global()
                        .lock()
                        .expect("asset registry mutex poisoned")
                        .fail(mesh_id);
                }
            }
        });
    }
    mesh_id
}

/// Decode one primitive's accessors out of the shared BIN blob into a CPU
/// [`Mesh`]. Runs on a pool background task, one per unique primitive.
fn decode_primitive(
    document: &gltf::Document,
    blob: &[u8],
    mesh_idx: usize,
    prim_idx: usize,
) -> Result<Mesh, String> {
    let mesh = document
        .meshes()
        .nth(mesh_idx)
        .ok_or_else(|| format!("mesh index {mesh_idx} out of range"))?;
    let prim = mesh
        .primitives()
        .nth(prim_idx)
        .ok_or_else(|| format!("primitive index {prim_idx} out of range"))?;
    if prim.mode() != gltf::mesh::Mode::Triangles {
        return Err(format!("unsupported primitive mode {:?}", prim.mode()));
    }
    let reader = prim.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => Some(blob),
        gltf::buffer::Source::Uri(_) => None,
    });
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("primitive has no POSITION attribute")?
        .collect();
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());
    if let Some(n) = &normals {
        if n.len() != positions.len() {
            return Err("NORMAL count differs from POSITION count".to_string());
        }
    }
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(0)
        .map(|it| it.into_f32().collect());
    if let Some(u) = &uvs {
        if u.len() != positions.len() {
            return Err("TEXCOORD_0 count differs from POSITION count".to_string());
        }
    }
    let vertices: Vec<Vertex> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| Vertex {
            position: Vec3::from_array(*p),
            // Missing normals zero-fill, matching the OBJ decode path.
            normal: normals
                .as_ref()
                .map(|n| Vec3::from_array(n[i]))
                .unwrap_or(Vec3::ZERO),
            uv: uvs
                .as_ref()
                .map(|u| Vec2::from_array(u[i]))
                .unwrap_or(Vec2::ZERO),
        })
        .collect();
    if vertices.is_empty() {
        return Err("primitive contained no geometry".to_string());
    }
    let indices: Vec<u32> = match reader.read_indices() {
        Some(it) => it.into_u32().collect(),
        // Non-indexed triangles: consecutive vertices per the glTF spec.
        None => (0..vertices.len() as u32).collect(),
    };
    Ok(Mesh::new(vertices, indices))
}

/// Hash a path to the dedup-cache key (same scheme as the mesh registry).
fn hash_path(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut h);
    h.finish()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::MeshSlot;

    /// Build a minimal-but-valid GLB in memory: two nodes (`root` at the
    /// origin, child `arm` translated by +1 X), both drawing the same
    /// single-triangle mesh (exercising primitive dedup).
    fn tiny_glb() -> Vec<u8> {
        // BIN chunk: 3 × vec3<f32> positions, then 3 × u32 indices.
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in v {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u32, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let json = format!(
            concat!(
                r#"{{"asset":{{"version":"2.0"}},"scene":0,"#,
                r#""scenes":[{{"nodes":[0]}}],"#,
                r#""nodes":[{{"name":"root","mesh":0,"children":[1]}},"#,
                r#"{{"name":"arm","mesh":0,"translation":[1.0,0.0,0.0]}}],"#,
                r#""meshes":[{{"name":"tri","primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],"#,
                r#""accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}},"#,
                r#"{{"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}}],"#,
                r#""bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},"#,
                r#"{{"buffer":0,"byteOffset":36,"byteLength":12}}],"#,
                r#""buffers":[{{"byteLength":{}}}]}}"#,
            ),
            bin.len()
        );
        let mut json = json.into_bytes();
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    fn init_pool() {
        let _ = crate::util::parallel::global::init(crate::util::parallel::BackendKind::MyPool, 4);
    }

    fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !cond() {
            assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// End-to-end: request → template Ready → spawn → drain instantiates
    /// the composed hierarchy → primitive decode resolves the shared mesh.
    #[test]
    fn glb_streams_hierarchy_then_meshes() {
        init_pool();
        let path = std::env::temp_dir().join(format!("engine_scene_test_{}.glb", std::process::id()));
        std::fs::write(&path, tiny_glb()).expect("write test glb");

        let id = request_scene(&path);
        assert_eq!(id, request_scene(&path), "same path must dedup to one SceneId");
        wait_until("template ready", || load_state(id) != SceneLoadState::Loading);
        assert_eq!(load_state(id), SceneLoadState::Ready);

        spawn_subscene(
            id,
            _Transform {
                position: Vec3::new(10.0, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(2.0),
                name: "instance".into(),
                parent: None,
            },
        );
        let mut scene = Scene::new();
        let mut attached: Vec<(u32, MeshId)> = Vec::new();
        let roots = drain_ready_spawns(&mut scene, |_, e, m| attached.push((e.id, m)));
        assert_eq!(roots.len(), 1);
        // instance root + "root" node + "arm" node.
        assert_eq!(scene.transform_hierarchy.len(), 3);

        // Both nodes draw the same primitive → deduped to one MeshId.
        assert_eq!(attached.len(), 2);
        assert_eq!(attached[0].1, attached[1].1);

        // Entities keep glTF-local TRS; world comes from the parent chain
        // (GPU-side per frame, `get_global_*` here): "arm"'s local +1 X
        // composes to 10 + 1 * scale(2) = 12 under the instance root.
        let arm_idx = attached
            .iter()
            .map(|(e, _)| *e)
            .max()
            .expect("two attached renderers");
        let arm = scene.transform_hierarchy.get_transform_(arm_idx);
        assert_eq!(arm.position, Vec3::new(1.0, 0.0, 0.0), "local TRS preserved");
        assert_eq!(arm.scale, Vec3::ONE, "local TRS preserved");
        assert_eq!(arm.name, "arm");
        assert_eq!(arm.parent, Some(roots[0].id + 1), "arm under the \"root\" node entity");
        {
            let arm_t = scene.transform_hierarchy.get_transform_unchecked(arm_idx);
            let g = arm_t.lock();
            assert_eq!(g.get_global_position(), Vec3::new(12.0, 0.0, 0.0));
            assert_eq!(g.get_global_scale(), Vec3::splat(2.0));
        }
        // Every instantiated entity recorded its parent link for the GPU
        // parent-scatter stream (instance root has no parent → 2 records).
        let updates = scene.transform_hierarchy.drain_parent_updates();
        assert_eq!(updates.len(), 2);
        assert!(updates.contains(&[arm_idx, roots[0].id + 1]));

        // The primitive decode resolves the redirect to a real 3-vertex mesh.
        let mesh_id = attached[0].1;
        wait_until("primitive decode", || {
            asset::global().lock().unwrap().redirect_of(mesh_id) != MeshSlot::PLACEHOLDER
        });
        let slot = asset::global().lock().unwrap().redirect_of(mesh_id);
        assert_ne!(slot, MeshSlot::ERROR, "valid GLB primitive must not fail");
        let (mesh, _) = asset::global().lock().unwrap().slot(slot);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);

        std::fs::remove_file(&path).ok();
    }

    /// A missing file marks the template Failed, and its queued spawns are
    /// dropped (loudly) instead of instantiating.
    #[test]
    fn missing_glb_fails_and_drops_spawns() {
        init_pool();
        let path = std::env::temp_dir().join(format!(
            "engine_scene_test_{}_missing.glb",
            std::process::id()
        ));
        let id = request_scene(&path);
        spawn_subscene(id, _Transform::default());
        wait_until("template failure", || load_state(id) != SceneLoadState::Loading);
        assert_eq!(load_state(id), SceneLoadState::Failed);

        let mut scene = Scene::new();
        let roots = drain_ready_spawns(&mut scene, |_, _, _| {
            panic!("failed template must not attach renderers")
        });
        assert!(roots.is_empty());
        assert_eq!(scene.transform_hierarchy.len(), 0);
    }
}

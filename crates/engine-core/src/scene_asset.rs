//! glTF/GLB → scene-template assets ("subscenes", à la Godot's
//! `PackedScene`).
//!
//! A `.glb` or `.gltf` file loads as a **[`SceneTemplate`]**: the node
//! hierarchy with local transforms plus, per mesh primitive, a
//! [`MeshRendererProxy`] holding a placeholder [`MeshId`]. The template is
//! *data* — it is never rendered and owns no ECS storage. Games instantiate
//! it into the live [`Scene`] any number of times; instantiation converts
//! each proxy into a real renderer component via the `attach_renderer`
//! callback (the render crate passes `MeshRenderer::from_id`).
//!
//! # Streaming pipeline — nothing waits on vertex data
//!
//! The predecessor loader fully decoded each node (attributes, tangents,
//! blocking uploads) before visiting its children, so *nothing* appeared
//! until *everything* was decoded. Here the load decouples into phases:
//!
//! 1. **Parse** (one background task): read **only the JSON** — for `.glb`
//!    the container framing locates the JSON chunk at the front of the
//!    file without touching the BIN chunk (potentially GBs of vertex and
//!    pixel bytes the hierarchy never needs); a `.gltf` file *is* its
//!    JSON. No buffer reads, no image decode (the `gltf` crate's `import`
//!    feature is deliberately disabled), no accessor reads.
//! 2. **Hierarchy** (same task): walk the node tree iteratively, minting a
//!    deduped placeholder `MeshId` per primitive (virtual path
//!    `file.glb#mesh{i}/prim{j}`). The template is Ready in the time it
//!    takes to read and walk the JSON — **independent of geometry and
//!    texture size**, so instantiations render placeholders while the
//!    heavy bytes stream in behind them.
//! 3. **Buffers** (second background task): **memory-map** the file-backed
//!    buffers (the BIN span, external `.bin` URIs — no bytes read, pages
//!    fault in per decode task; only `data:` URIs decode into owned
//!    bytes), then spawn the decode tasks below for every primitive/image
//!    the parse newly requested. Decodes therefore start streaming
//!    immediately after Ready, with I/O overlapping decode across the
//!    pool's capped background workers instead of gating behind one
//!    sequential whole-chunk read. A buffer failure fails every pending
//!    mesh id loudly — no placeholder may linger forever.
//! 4. **Mesh decodes** (one fire-and-forget background task per unique
//!    primitive): slice accessors out of the shared buffers, build a
//!    [`Mesh`], and [`AssetRegistry::resolve`] the redirect — a single
//!    write. Meshes pop in individually, in completion order, and because
//!    the redirect table is global, **every already-spawned instance**
//!    upgrades from placeholder simultaneously.
//! 5. **Materials + textures**: each newly-requested primitive interns its
//!    glTF material in the [`material`] registry (content-hash deduped —
//!    identical factors + textures collapse to one `MaterialId` across
//!    primitives; resolution is immediate, materials being tiny POD). The
//!    material's `baseColorTexture` is requested as a deduped
//!    [`TextureId`] (virtual path `file.glb#image{i}`) with a
//!    fire-and-forget decode — embedded bufferView images decode from
//!    zero-copy views into the shared buffers; external image URIs load
//!    from disk. The resolved mesh slot carries the MaterialId as its
//!    authored material; surfaces show the material's factors over the
//!    1×1 white placeholder until the texture decode lands.
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
use crate::material::{self, MaterialData};
use crate::texture::{self, TextureId};
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
            let t0 = std::time::Instant::now();
            let root = instantiate(scene, &template, at, &mut attach_renderer);
            println!(
                "subscene instantiated: {} entities (root {}) in {:.0}ms",
                template.node_count() + 1,
                root.id,
                t0.elapsed().as_secs_f64() * 1e3,
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

/// One primitive newly requested by the parse phase, awaiting its decode
/// spawn once the buffer bytes exist (phase 3).
struct PendingPrim {
    mesh_idx: usize,
    prim_idx: usize,
    mesh_id: MeshId,
}

/// Parse a glTF/GLB's **JSON only** and build its template. No buffer
/// bytes are read: newly requested primitives are handed to a second
/// background task ([`load_buffers_and_spawn_decodes`]) that reads the
/// buffers and spawns the per-primitive mesh + per-image texture decodes.
/// Runs on a pool background task; the caller flips the template to Ready
/// on return, so instantiations render placeholders while bytes stream.
fn build_template(path: &Path) -> Result<SceneTemplate, String> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("glb") | Some("gltf") => {}
        other => return Err(format!(
            "unsupported scene format: {other:?} (only .glb / .gltf)"
        )),
    }
    let t_read = std::time::Instant::now();
    let (json, bin_span) = read_scene_json(path)?;
    let t_parse = std::time::Instant::now();
    let gltf::Gltf { document, .. } =
        gltf::Gltf::from_slice(&json).map_err(|e| format!("glTF parse error: {e}"))?;
    drop(json);
    let document = Arc::new(document);
    let t_walk = std::time::Instant::now();

    let mut pending: Vec<PendingPrim> = Vec::new();
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
                    mesh_id: request_primitive(path, mesh.index(), prim.index(), &mut pending),
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
    eprintln!(
        "scene template phases for {}: json read {:.0}ms, parse {:.0}ms, walk {:.0}ms ({} new primitives)",
        path.display(),
        (t_parse - t_read).as_secs_f64() * 1e3,
        (t_walk - t_parse).as_secs_f64() * 1e3,
        t_walk.elapsed().as_secs_f64() * 1e3,
        pending.len(),
    );
    // Phase 3 — buffer read + decode spawns — runs as its own background
    // task. The template goes Ready without waiting on a single vertex or
    // pixel byte.
    if !pending.is_empty() {
        let document = document.clone();
        let path = path.to_path_buf();
        asset::spawn_when_pool_ready(move || {
            load_buffers_and_spawn_decodes(document, path, bin_span, pending);
        });
    }
    Ok(SceneTemplate { nodes })
}

/// Read a scene file's parseable JSON **without reading buffer bytes**:
/// for `.glb`, walk the container framing (12-byte header, then
/// `[len][type][data]` chunks) reading only the JSON chunk, and return the
/// BIN chunk's `(offset, length)` span for phase 3; a `.gltf` file is its
/// own JSON (span `None`).
fn read_scene_json(path: &Path) -> Result<(Vec<u8>, Option<(u64, u64)>), String> {
    use std::io::{Read, Seek, SeekFrom};
    let is_glb = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("glb"));
    if !is_glb {
        return Ok((std::fs::read(path).map_err(|e| format!("read error: {e}"))?, None));
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("open error: {e}"))?;
    let mut header = [0u8; 12];
    f.read_exact(&mut header)
        .map_err(|e| format!("GLB header read error: {e}"))?;
    if &header[0..4] != b"glTF" {
        return Err("not a GLB file (bad magic)".to_string());
    }
    let total = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
    let mut json: Option<Vec<u8>> = None;
    let mut bin_span: Option<(u64, u64)> = None;
    let mut offset = 12u64;
    while offset + 8 <= total {
        let mut ch = [0u8; 8];
        f.read_exact(&mut ch)
            .map_err(|e| format!("GLB chunk header read error: {e}"))?;
        let len = u32::from_le_bytes(ch[0..4].try_into().unwrap()) as u64;
        if &ch[4..8] == b"JSON" && json.is_none() {
            let mut buf = vec![0u8; len as usize];
            f.read_exact(&mut buf)
                .map_err(|e| format!("GLB JSON chunk read error: {e}"))?;
            json = Some(buf);
        } else {
            if &ch[4..8] == b"BIN\0" && bin_span.is_none() {
                bin_span = Some((offset + 8, len));
            }
            f.seek(SeekFrom::Current(len as i64))
                .map_err(|e| format!("GLB seek error: {e}"))?;
        }
        offset += 8 + len;
    }
    json.map(|j| (j, bin_span))
        .ok_or_else(|| "GLB missing JSON chunk".to_string())
}

/// One glTF buffer's bytes, shared across every decode task that reads it.
///
/// File-backed buffers (the GLB BIN chunk, external `.bin` URIs) are
/// **memory-mapped**: no bytes are read up front — each decode task faults
/// in only the pages its accessors/images touch, so I/O overlaps with
/// decode across the pool's background workers instead of gating every
/// decode behind one sequential read of the whole (potentially GBs) chunk.
/// Only `data:` URIs own their (base64-decoded) bytes.
///
/// Mapping caveat, loudly: the file must not be truncated while decodes
/// are in flight (the map would SIGBUS). Assets are treated as immutable
/// once shipped; an editor-driven hot-reload story would re-request.
#[derive(Clone)]
enum SceneBuffer {
    Mapped {
        map: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
    Owned(Arc<Vec<u8>>),
}

impl SceneBuffer {
    fn as_slice(&self) -> &[u8] {
        match self {
            SceneBuffer::Mapped { map, offset, len } => &map[*offset..*offset + *len],
            SceneBuffer::Owned(v) => v,
        }
    }
}

/// Phase 3: map every buffer, then — per pending primitive — intern its
/// authored material and spawn its mesh decode plus the texture decodes for
/// the material's images. Runs on a pool background task and is cheap —
/// mapping reads no bytes — so decodes start streaming immediately after
/// Ready. A buffer failure fails **every** pending mesh id loudly — their
/// instantiated placeholders flip to the error mesh instead of lingering
/// forever.
fn load_buffers_and_spawn_decodes(
    document: Arc<gltf::Document>,
    path: PathBuf,
    bin_span: Option<(u64, u64)>,
    pending: Vec<PendingPrim>,
) {
    let t0 = std::time::Instant::now();
    let buffers = match load_buffers(&document, &path, bin_span) {
        Ok(b) => Arc::new(b),
        Err(e) => {
            eprintln!("scene buffer load failed for {}: {e}", path.display());
            let mut reg = asset::global().lock().expect("asset registry mutex poisoned");
            for p in &pending {
                reg.fail(p.mesh_id);
            }
            return;
        }
    };
    eprintln!(
        "scene buffers mapped for {}: {} buffer(s) in {:.0}ms",
        path.display(),
        buffers.len(),
        t0.elapsed().as_secs_f64() * 1e3,
    );
    // One pass over the document's meshes matches the pending set without
    // per-primitive `nth` scans.
    let by_key: HashMap<(usize, usize), MeshId> = pending
        .iter()
        .map(|p| ((p.mesh_idx, p.prim_idx), p.mesh_id))
        .collect();
    for mesh in document.meshes() {
        for prim in mesh.primitives() {
            let Some(&mesh_id) = by_key.get(&(mesh.index(), prim.index())) else {
                continue;
            };
            // Mint the primitive's authored material (content-hash deduped —
            // identical factors + the same deduped image TextureId collapse
            // to one MaterialId across primitives). A primitive with no
            // material stays `None` → the engine default material, matching
            // the untextured look elsewhere.
            let material = prim.material().index().map(|_| {
                let m = prim.material();
                let pbr = m.pbr_metallic_roughness();
                let base_color_tex = pbr
                    .base_color_texture()
                    .map(|info| request_image(&buffers, &path, info.texture().source()));
                let data = MaterialData {
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    emissive: m.emissive_factor(),
                    base_color_tex,
                };
                material::global()
                    .lock()
                    .expect("material registry mutex poisoned")
                    .get_or_create(data)
                    .0
            });
            let virtual_path = format!("{}#mesh{}/prim{}", path.display(), mesh.index(), prim.index());
            let document = document.clone();
            let buffers = buffers.clone();
            let (mesh_idx, prim_idx) = (mesh.index(), prim.index());
            asset::spawn_when_pool_ready(move || {
                match decode_primitive(&document, &buffers, mesh_idx, prim_idx) {
                    Ok(mesh) => {
                        asset::global()
                            .lock()
                            .expect("asset registry mutex poisoned")
                            .resolve_with_material(mesh_id, Arc::new(mesh), material);
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
    }
}

/// Build the buffer table (indexed by buffer index) without reading bytes:
/// the GLB BIN span becomes a range view over the mapped scene file,
/// external URIs map their own files, and `data:` URIs (the only owned
/// case) base64-decode in place.
fn load_buffers(
    document: &gltf::Document,
    path: &Path,
    bin_span: Option<(u64, u64)>,
) -> Result<Vec<SceneBuffer>, String> {
    let dir = path.parent().unwrap_or(Path::new(""));
    // The scene file is mapped at most once, shared by every Bin buffer.
    let mut scene_map: Option<Arc<memmap2::Mmap>> = None;
    document
        .buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => {
                let (off, len) = bin_span.ok_or_else(|| {
                    "buffer references a BIN chunk but the file has none".to_string()
                })?;
                let map = match &scene_map {
                    Some(m) => m.clone(),
                    None => {
                        let m = Arc::new(map_file(path)?);
                        scene_map = Some(m.clone());
                        m
                    }
                };
                let (off, len) = (off as usize, len as usize);
                if off + len > map.len() {
                    return Err(format!(
                        "BIN span {off}+{len} exceeds file length {}",
                        map.len()
                    ));
                }
                Ok(SceneBuffer::Mapped { map, offset: off, len })
            }
            gltf::buffer::Source::Uri(uri) => {
                if uri.starts_with("data:") {
                    read_uri(uri, dir)
                        .map(|v| SceneBuffer::Owned(Arc::new(v)))
                        .map_err(|e| format!("buffer {uri:?}: {e}"))
                } else {
                    let p = dir.join(percent_decode(uri));
                    let map = map_file(&p).map_err(|e| format!("buffer {uri:?}: {e}"))?;
                    let len = map.len();
                    Ok(SceneBuffer::Mapped {
                        map: Arc::new(map),
                        offset: 0,
                        len,
                    })
                }
            }
        })
        .collect()
}

/// Memory-map a file read-only. Reads no bytes — pages fault in on access.
fn map_file(path: &Path) -> Result<memmap2::Mmap, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("open error: {e}"))?;
    // Safety: the map is read-only; per the SceneBuffer contract asset
    // files are immutable while decodes are in flight.
    unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("mmap error: {e}"))
}

/// Mint (or dedup) the `MeshId` for one primitive. On first request the
/// primitive joins `pending` — the buffer-load task (phase 3) spawns its
/// decode once the bytes exist. No buffer access here.
fn request_primitive(
    path: &Path,
    mesh_idx: usize,
    prim_idx: usize,
    pending: &mut Vec<PendingPrim>,
) -> MeshId {
    // Virtual sub-asset path: keys the registry's dedup cache, so two
    // nodes (or two templates) sharing a primitive share the MeshId.
    let virtual_path = format!("{}#mesh{mesh_idx}/prim{prim_idx}", path.display());
    let (mesh_id, needs_load) = asset::global()
        .lock()
        .expect("asset registry mutex poisoned")
        .request(Path::new(&virtual_path));
    if needs_load {
        pending.push(PendingPrim {
            mesh_idx,
            prim_idx,
            mesh_id,
        });
    }
    mesh_id
}

/// Mint (or dedup) the [`TextureId`] for one glTF image and, on first
/// request, spawn its decode: embedded bufferView images decode from a
/// zero-copy view into the mapped buffer (pages fault in inside the decode
/// task), `data:` URIs from their decoded payload; external URIs load from
/// disk relative to the scene file. A malformed source fails the id
/// immediately (→ error texture).
fn request_image(
    buffers: &Arc<Vec<SceneBuffer>>,
    path: &Path,
    image: gltf::Image<'_>,
) -> TextureId {
    let virtual_path = format!("{}#image{}", path.display(), image.index());
    let (texture_id, needs_load) = texture::global()
        .lock()
        .expect("texture registry mutex poisoned")
        .request(Path::new(&virtual_path));
    if !needs_load {
        return texture_id;
    }
    match image.source() {
        gltf::image::Source::View { view, .. } => {
            // Zero-copy: the decode task holds the buffer view and slices
            // the image range itself (out-of-range fails loudly there).
            let buf = buffers[view.buffer().index()].clone();
            let range = view.offset()..view.offset() + view.length();
            texture::request_decode_task(texture_id, virtual_path, move || {
                let bytes = buf.as_slice();
                bytes
                    .get(range.clone())
                    .ok_or_else(|| {
                        format!(
                            "bufferView {}..{} out of bounds (buffer len {})",
                            range.start,
                            range.end,
                            bytes.len()
                        )
                    })
                    .and_then(texture::decode_texture_bytes)
            });
        }
        gltf::image::Source::Uri { uri, .. } => {
            if uri.starts_with("data:") {
                match read_uri(uri, Path::new("")) {
                    Ok(bytes) => texture::request_decode_task(texture_id, virtual_path, move || {
                        texture::decode_texture_bytes(&bytes)
                    }),
                    Err(e) => {
                        eprintln!("texture load failed for {virtual_path}: {e}");
                        texture::global()
                            .lock()
                            .expect("texture registry mutex poisoned")
                            .fail(texture_id);
                    }
                }
            } else {
                let dir = path.parent().unwrap_or(Path::new(""));
                texture::request_load(texture_id, dir.join(percent_decode(uri)));
            }
        }
    }
    texture_id
}

/// Resolve a glTF URI to raw bytes: base64 `data:` URIs decode in place,
/// anything else reads as a file path relative to `dir` (with `%XX`
/// percent-escapes decoded, per the glTF spec's RFC 3986 URIs).
fn read_uri(uri: &str, dir: &Path) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (meta, payload) = rest
            .split_once(',')
            .ok_or_else(|| "malformed data URI (no comma)".to_string())?;
        if !meta.ends_with(";base64") {
            return Err(format!("unsupported data URI encoding {meta:?} (only base64)"));
        }
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|e| format!("data URI base64 decode error: {e}"))
    } else {
        let p = dir.join(percent_decode(uri));
        std::fs::read(&p).map_err(|e| format!("read error for {}: {e}", p.display()))
    }
}

/// Decode `%XX` escapes (e.g. `my%20tex.png`). Malformed escapes pass
/// through verbatim — the subsequent file read fails loudly if wrong.
fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&uri[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decode one primitive's accessors out of the shared buffer table into a
/// CPU [`Mesh`]. Runs on a pool background task, one per unique primitive.
fn decode_primitive(
    document: &gltf::Document,
    buffers: &[SceneBuffer],
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
    // Every buffer (BIN chunk, external file, data URI) has a table entry
    // from the phase-3 buffer mapping, indexed by buffer index. Mapped
    // entries fault their pages in here, on this decode task.
    let reader = prim.reader(|buffer| buffers.get(buffer.index()).map(|b| b.as_slice()));
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

    /// Encode a 2×2 RGBA PNG (distinct corner colours) via the image crate.
    fn tiny_png() -> Vec<u8> {
        let pixels: [u8; 16] = [
            255, 0, 0, 255, /**/ 0, 255, 0, 255, //
            0, 0, 255, 255, /**/ 255, 255, 255, 255,
        ];
        let mut png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &pixels,
            2,
            2,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .expect("encode test png");
        png
    }

    /// Wrap `json` + `bin` into a GLB container.
    fn wrap_glb(mut json: Vec<u8>, mut bin: Vec<u8>) -> Vec<u8> {
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

    /// Triangle geometry bytes: 3 × vec3 positions then 3 × u32 indices.
    fn tri_bin() -> Vec<u8> {
        let mut bin: Vec<u8> = Vec::new();
        for v in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in v {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u32, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin
    }

    /// A resolved mesh slot's authored material → its base-color TextureId
    /// (via the material registry), asserting both exist.
    fn slot_base_color_tex(slot: MeshSlot) -> crate::texture::TextureId {
        let mat_id = asset::global()
            .lock()
            .unwrap()
            .slot_material(slot)
            .expect("primitive must carry an authored material");
        let reg = material::global().lock().unwrap();
        reg.slot(reg.slot_of(mat_id))
            .base_color_tex
            .expect("material must carry a base-color TextureId")
    }

    /// Wait for a texture id to leave the placeholder slot, then return the
    /// decoded pixels.
    fn wait_for_texture(id: crate::texture::TextureId) -> Arc<crate::texture::TextureData> {
        wait_until("texture decode", || {
            texture::global().lock().unwrap().redirect_of(id)
                != crate::texture::TextureSlot::PLACEHOLDER
        });
        let reg = texture::global().lock().unwrap();
        let slot = reg.redirect_of(id);
        assert_ne!(slot, crate::texture::TextureSlot::ERROR, "texture decode must not fail");
        reg.slot(slot)
    }

    /// GLB with an embedded PNG bound as the primitive's baseColorTexture:
    /// the resolved mesh slot must carry a TextureId that decodes to the
    /// embedded pixels.
    #[test]
    fn glb_embedded_texture_resolves() {
        init_pool();
        let png = tiny_png();
        let mut bin = tri_bin();
        let png_offset = bin.len();
        bin.extend_from_slice(&png);
        let json = format!(
            concat!(
                r#"{{"asset":{{"version":"2.0"}},"scene":0,"#,
                r#""scenes":[{{"nodes":[0]}}],"#,
                r#""nodes":[{{"name":"tex_node","mesh":0}}],"#,
                r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],"#,
                r#""materials":[{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}}],"#,
                r#""textures":[{{"source":0}}],"#,
                r#""images":[{{"bufferView":2,"mimeType":"image/png"}}],"#,
                r#""accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}},"#,
                r#"{{"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}}],"#,
                r#""bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},"#,
                r#"{{"buffer":0,"byteOffset":36,"byteLength":12}},"#,
                r#"{{"buffer":0,"byteOffset":{png_offset},"byteLength":{png_len}}}],"#,
                r#""buffers":[{{"byteLength":{total}}}]}}"#,
            ),
            png_offset = png_offset,
            png_len = png.len(),
            total = bin.len(),
        );
        let path = std::env::temp_dir().join(format!(
            "engine_scene_test_{}_tex.glb",
            std::process::id()
        ));
        std::fs::write(&path, wrap_glb(json.into_bytes(), bin)).expect("write test glb");

        let id = request_scene(&path);
        wait_until("template ready", || load_state(id) != SceneLoadState::Loading);
        assert_eq!(load_state(id), SceneLoadState::Ready);

        // The primitive's MeshId is reachable through the dedup cache.
        let virtual_path = format!("{}#mesh0/prim0", path.display());
        let (mesh_id, fresh) = asset::global().lock().unwrap().request(Path::new(&virtual_path));
        assert!(!fresh, "template must have already requested the primitive");
        wait_until("primitive decode", || {
            asset::global().lock().unwrap().redirect_of(mesh_id) != MeshSlot::PLACEHOLDER
        });
        let slot = asset::global().lock().unwrap().redirect_of(mesh_id);
        assert_ne!(slot, MeshSlot::ERROR);
        let tex_id = slot_base_color_tex(slot);
        let data = wait_for_texture(tex_id);
        assert_eq!((data.width, data.height), (2, 2));
        assert_eq!(&data.rgba8[0..4], &[255, 0, 0, 255], "top-left pixel");
        std::fs::remove_file(&path).ok();
    }

    /// `.gltf` with an external `.bin` buffer and an external PNG image:
    /// both resolve through the same streaming pipeline.
    #[test]
    fn gltf_external_buffer_and_image_resolve() {
        init_pool();
        let dir = std::env::temp_dir().join(format!("engine_scene_test_{}_gltf", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        std::fs::write(dir.join("tri.bin"), tri_bin()).expect("write bin");
        std::fs::write(dir.join("tex.png"), tiny_png()).expect("write png");
        let json = concat!(
            r#"{"asset":{"version":"2.0"},"scene":0,"#,
            r#""scenes":[{"nodes":[0]}],"#,
            r#""nodes":[{"name":"ext_node","mesh":0}],"#,
            r#""meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]}],"#,
            r#""materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}],"#,
            r#""textures":[{"source":0}],"#,
            r#""images":[{"uri":"tex.png"}],"#,
            r#""accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]},"#,
            r#"{"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}],"#,
            r#""bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},"#,
            r#"{"buffer":0,"byteOffset":36,"byteLength":12}],"#,
            r#""buffers":[{"uri":"tri.bin","byteLength":48}]}"#,
        );
        let path = dir.join("tri.gltf");
        std::fs::write(&path, json).expect("write test gltf");

        let id = request_scene(&path);
        wait_until("template ready", || load_state(id) != SceneLoadState::Loading);
        assert_eq!(load_state(id), SceneLoadState::Ready, ".gltf must parse");

        let virtual_path = format!("{}#mesh0/prim0", path.display());
        let (mesh_id, fresh) = asset::global().lock().unwrap().request(Path::new(&virtual_path));
        assert!(!fresh);
        wait_until("primitive decode", || {
            asset::global().lock().unwrap().redirect_of(mesh_id) != MeshSlot::PLACEHOLDER
        });
        let slot = asset::global().lock().unwrap().redirect_of(mesh_id);
        assert_ne!(slot, MeshSlot::ERROR, "external-buffer primitive must decode");
        let (mesh, _) = asset::global().lock().unwrap().slot(slot);
        assert_eq!(mesh.vertices.len(), 3);
        let tex_id = slot_base_color_tex(slot);
        let data = wait_for_texture(tex_id);
        assert_eq!((data.width, data.height), (2, 2));
        std::fs::remove_dir_all(&dir).ok();
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

//! The project scene loading the editor opens on: one world per document,
//! then the editor's own rig. See
//! [editor-document-split](../../../docs/notes/editor-document-split.md).

use crate::chrome::Chrome;
use crate::{scene_path, settings};
use engine::{
    Entity, MeshRenderer, OrbitController, WorldHandle,
    glam::Vec3,
    transform::{_Transform, ROOT},
};
use engine_editor_api::CameraHandle;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// One document as the editor opens it, which is everything `Chrome` needs
/// to build a panel for it and nothing it has to look up.
pub(crate) struct Open {
    pub(crate) title: String,
    /// The file it saves to and reloads from, and what `editor.json`
    /// remembers. A scene that has never been written still has one.
    pub(crate) path: PathBuf,
    pub(crate) world: WorldHandle,
    pub(crate) camera: CameraHandle,
    /// Filled in by the spawn that mints the rig entity, at the first frame
    /// boundary — which is after the panels this opens are built.
    pub(crate) rig: Arc<OnceLock<Entity>>,
}

/// The scenes `editor.json` left open, each read into a world of its own.
///
/// A scene that will not load is reported and skipped, and a project with
/// none left opens on one empty document: the editor is a window onto a
/// document, so there is no state in which it has no document at all.
fn open_documents(open: &[PathBuf]) -> Vec<(String, PathBuf, WorldHandle)> {
    let mut scenes: Vec<_> = open.iter().filter_map(|p| read_document(p)).collect();
    if scenes.is_empty() {
        let title = "untitled".to_string();
        scenes.push((title.clone(), scene_path(&title), document_world()));
    }
    scenes
}

/// A document world: it renders, because a scene has to be seen to be
/// authored, but it does not simulate — edit mode is a registry nobody
/// sweeps (ADR-0010 §5) rather than a per-entity test.
pub(crate) fn document_world() -> WorldHandle {
    let world = engine::new_world();
    // SAFETY: no frame has started, so nothing is reading this world.
    unsafe { world.get_mut() }.set_simulating(false);
    world
}

/// Read one scene file into a world of its own, or say why it is not one.
/// The handle is all that holds the world, so a failure retires it.
fn read_document(path: &Path) -> Option<(String, PathBuf, WorldHandle)> {
    let world = document_world();
    let read = std::fs::read_to_string(path).map_err(|e| e.to_string());
    // SAFETY: nothing has been told this world exists yet.
    let loaded = read.and_then(|t| engine::scene_file::load(unsafe { world.get_mut() }, ROOT, &t));
    match loaded {
        Ok(_) => Some((stem(path), path.to_path_buf(), world)),
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            None
        }
    }
}

/// A file's name without its extension — the title its tab takes.
pub(crate) fn stem(path: &Path) -> String {
    path.file_stem().map_or_else(
        || path.display().to_string(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// The worlds the editor runs: one per document, then the editor's own rig.
///
/// Two hierarchies, not one graph with a boundary drawn through it — so
/// `parent: None`, what every spawn, subscene instantiation and
/// drop-to-top-level resolves to, cannot reach the rig from the document at
/// all. See `docs/notes/editor-document-split.md`.
pub(crate) fn load_project(
    project: &str,
    stress: usize,
    worlds: usize,
) -> (Vec<WorldHandle>, WorldHandle) {
    // What the panels edit. Under stress the hierarchy panel gets an empty
    // world rather than a million-row document, so it is not what the
    // renderer draws.
    let scenes = match stress {
        0 => open_documents(&settings::load().open),
        _ => vec![("Stress".into(), scene_path("Stress"), engine::new_world())],
    };
    let documents = match stress {
        0 => scenes.iter().map(|(.., w)| w.clone()).collect(),
        n => stress_documents(n, worlds.max(1)),
    };

    // One camera per document, owned by the editor rather than minted by a
    // `CameraComponent` — they look at worlds the rig they are driven from is
    // not part of, and they exist before any entity does. Stress mode instead
    // composites every world through one camera, so the drawn entity count is
    // the same at any `--worlds` and only the scatter varies.
    let cameras: Vec<CameraHandle> = match stress {
        0 => documents
            .iter()
            .map(|d| CameraHandle::new(d.id()))
            .collect(),
        _ => {
            let camera = CameraHandle::new(documents[0].id());
            documents[1..]
                .iter()
                .for_each(|d| camera.draw_world(d.id()));
            vec![camera]
        }
    };
    cameras.iter().for_each(|c| c.set_grid(Some(GRID)));

    let rig = engine::new_world();
    let rigs: Vec<Arc<OnceLock<Entity>>> = cameras.iter().map(|_| Arc::default()).collect();
    for (i, (camera, mine)) in cameras
        .iter()
        .cloned()
        .zip(rigs.iter().cloned())
        .enumerate()
    {
        rig.spawn(
            _Transform {
                name: format!("editor camera {i}"),
                .._Transform::default()
            },
            move |mut e| {
                mine.set(e.id()).ok();
                // `for_camera`, not `new`: it feeds that camera's matrix and
                // answers only to drags inside that camera's panel.
                e.add_component(OrbitController::for_camera(camera));
            },
        );
    }

    // An entity of the chrome's own, after the cameras so every panel it
    // builds has one to show. Not on a camera's: closing a document removes
    // the entity driving that camera, and the editor cannot be one of those.
    let open: Vec<Open> = scenes
        .iter()
        .cloned()
        .zip(cameras.iter().cloned())
        .zip(rigs)
        .map(|(((title, path, world), camera), rig)| Open {
            title,
            path,
            world,
            camera,
            rig,
        })
        .collect();
    let project = project.to_string();
    rig.spawn(
        _Transform {
            name: "editor chrome".into(),
            .._Transform::default()
        },
        move |mut e| {
            let chrome = Chrome::new(&project, open, e.world());
            e.add_component(chrome);
        },
    );
    (documents, rig)
}

/// Cell, major every ten, and the radius it fades out over — the ground plane
/// an empty document needs to read as a place rather than a void.
pub(crate) const GRID: [f32; 4] = [1.0, 10.0, 120.0, 0.0];

/// One cubic grid of `total` spinning cubes, cut into `worlds` contiguous
/// slabs — one world each. The grid is the same at any `worlds`, so the
/// scatter's *work* is fixed and only its dispatch count changes.
fn stress_documents(total: usize, worlds: usize) -> Vec<WorldHandle> {
    let side = ((total as f64).cbrt().ceil() as usize).max(1);
    let spacing = 3.0f32;
    let origin = -((side as f32) - 1.0) * 0.5 * spacing;
    let per = total.div_ceil(worlds);
    (0..worlds)
        .map(|w| {
            let world = engine::new_world();
            for k in (w * per)..((w + 1) * per).min(total) {
                let (x, y, z) = (k % side, (k / side) % side, k / (side * side));
                world.spawn(
                    _Transform {
                        position: Vec3::new(
                            origin + x as f32 * spacing,
                            origin + y as f32 * spacing,
                            origin + z as f32 * spacing,
                        ),
                        .._Transform::default()
                    },
                    |mut e| {
                        e.add_component(MeshRenderer::new("assets/cube/cube.obj"));
                    },
                );
            }
            world
        })
        .collect()
}

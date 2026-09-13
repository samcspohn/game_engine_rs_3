//! Which asset a file on disk is, keyed by extension.
//!
//! A property asks for an [`AssetKind`]; a file is an extension. This is the
//! table between them. Each loader already dispatches on its own extension,
//! but nothing could answer the question the other way round, which is what a
//! drag out of the browser needs: the row has a path and the row it is over
//! has a kind.
//!
//! A game registers its own formats at startup with [`register`], so a type
//! the engine has never heard of drags onto a property the same way a
//! built-in does. See `docs/notes/reflection.md`.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;

use crate::reflect::{AssetKind, AssetRef};
use crate::texture::ColorSpace;
use crate::{asset, scene_asset, texture};

/// One file format, as the extensions that name it and the handle it mints.
pub trait AssetType: Send + Sync {
    /// Lowercase and without the dot; matched case-insensitively.
    fn extensions(&self) -> &[&str];

    /// What a property must ask for to accept this file.
    fn kind(&self) -> AssetKind;

    /// A handle to the file, requested from the registry that owns the kind.
    /// Infallible because the bytes are not read here: the handle resolves to
    /// a placeholder, and a file that will not decode becomes the error one.
    fn reference(&self, path: &Path) -> AssetRef;
}

struct Obj;
struct Gltf;
struct Scene;
struct Image;

impl AssetType for Obj {
    fn extensions(&self) -> &[&str] {
        &["obj"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Mesh
    }

    fn reference(&self, path: &Path) -> AssetRef {
        AssetRef::Mesh(asset::request_path(path))
    }
}

impl AssetType for Gltf {
    fn extensions(&self) -> &[&str] {
        &["glb", "gltf"]
    }

    /// A scene and not a mesh: a glTF file is a node tree that instantiates,
    /// and one of its primitives is not addressable on its own.
    fn kind(&self) -> AssetKind {
        AssetKind::Scene
    }

    fn reference(&self, path: &Path) -> AssetRef {
        AssetRef::Scene(scene_asset::request_scene(path))
    }
}

impl AssetType for Scene {
    /// The extension a scene is saved with, which `project.json` and
    /// `editor.json` share — dropping one of those somewhere that wants a
    /// scene fails when it is read, not here.
    fn extensions(&self) -> &[&str] {
        &["json"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Scene
    }

    fn reference(&self, path: &Path) -> AssetRef {
        AssetRef::Scene(scene_asset::request_scene(path))
    }
}

impl AssetType for Image {
    fn extensions(&self) -> &[&str] {
        &["png", "jpg", "jpeg", "tga", "bmp", "gif", "webp"]
    }

    fn kind(&self) -> AssetKind {
        AssetKind::Texture
    }

    /// sRGB, which is what an image a person picked out of a browser is. A
    /// normal map is the exception and the material that wants one linear is
    /// where that belongs, not the file's extension.
    fn reference(&self, path: &Path) -> AssetRef {
        AssetRef::Texture(texture::request_path(path, ColorSpace::Srgb))
    }
}

fn registry() -> &'static Mutex<Vec<&'static dyn AssetType>> {
    static TYPES: OnceLock<Mutex<Vec<&'static dyn AssetType>>> = OnceLock::new();
    TYPES.get_or_init(|| Mutex::new(vec![&Obj, &Gltf, &Scene, &Image]))
}

/// Add a format. Leaked rather than owned: a registration is a startup
/// singleton, and a lookup that borrowed from the table would hold its lock
/// for as long as the caller held the type.
pub fn register(ty: Box<dyn AssetType>) {
    registry().lock().push(Box::leak(ty));
}

/// The type claiming this extension. Searched newest first, so registering a
/// format the engine already handles replaces it rather than being shadowed.
pub fn of(path: &Path) -> Option<&'static dyn AssetType> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    registry()
        .lock()
        .iter()
        .rev()
        .find(|t| t.extensions().contains(&ext.as_str()))
        .copied()
}

/// What a file is, for a drop target deciding whether to light up.
pub fn kind_of(path: &Path) -> Option<AssetKind> {
    Some(of(path)?.kind())
}

/// A handle to the file, or `None` when no type claims its extension.
pub fn reference(path: &Path) -> Option<AssetRef> {
    Some(of(path)?.reference(path))
}

/// The file a handle came from, for a panel that shows a name instead of a
/// number. `None` for one minted without a file — an empty mesh slot, or any
/// material, since no format makes one yet.
pub fn path_of(r: AssetRef) -> Option<PathBuf> {
    match r {
        AssetRef::Mesh(id) => asset::global().lock().path_of(id).map(Path::to_path_buf),
        AssetRef::Texture(id) => texture::global()
            .lock()
            .path_of(id)
            .map(|(p, _)| p.to_path_buf()),
        AssetRef::Scene(id) => scene_asset::path_of(id),
        AssetRef::Material(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_map_to_kinds() {
        assert_eq!(kind_of(Path::new("a/b.OBJ")), Some(AssetKind::Mesh));
        assert_eq!(kind_of(Path::new("a/b.gltf")), Some(AssetKind::Scene));
        assert_eq!(kind_of(Path::new("a/b.png")), Some(AssetKind::Texture));
        assert_eq!(
            kind_of(Path::new("scenes/cube.json")),
            Some(AssetKind::Scene)
        );
        assert_eq!(kind_of(Path::new("Cargo")), None);
    }

    struct Fbx;

    impl AssetType for Fbx {
        fn extensions(&self) -> &[&str] {
            &["fbx"]
        }

        fn kind(&self) -> AssetKind {
            AssetKind::Mesh
        }

        fn reference(&self, _: &Path) -> AssetRef {
            AssetRef::Mesh(crate::asset::MeshId(0))
        }
    }

    #[test]
    fn a_game_can_add_one() {
        assert_eq!(kind_of(Path::new("x.fbx")), None);
        register(Box::new(Fbx));
        assert_eq!(kind_of(Path::new("x.fbx")), Some(AssetKind::Mesh));
    }
}

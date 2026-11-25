//! Resources: an exported struct that is a file of its own, shared by every
//! handle to it — a particle system its emitters point at, a light profile.
//! A file holds the struct's `#[export]` fields the way a scene file holds a
//! component's, and edits reach every handle at once.
//! See [`docs/notes/resources.md`](../../../docs/notes/resources.md).

use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use parking_lot::RwLock;
use serde_json::Value as Json;

use crate::asset_type::{self, AssetType};
use crate::reflect::{AssetKind, AssetRef, Export, Exportable, Value, ValueKind};
use crate::scene_file;
use crate::transform::{self, Transform};

/// What `#[export(resource = "ext")]` on a `#[derive(Export)]` struct
/// implements.
pub trait Resource: Export + Default + Clone + Send + Sync + 'static {
    /// The name a property's kind and the Browser know the type by: the
    /// derive's `TYPE_NAME`.
    const NAME: &'static str;
    /// Lowercase and without the dot.
    const EXTENSION: &'static str;
    /// What a file of this type is to a drop target: its own kind, unless
    /// the type feeds a store of its own — a material drops as a `MaterialId`.
    const KIND: AssetKind = AssetKind::Resource(Self::NAME);

    /// The handle a file of this type drops as.
    fn reference(id: ResId) -> AssetRef {
        AssetRef::Resource(id)
    }

    /// Hears every edit, once it has landed — how a store derived from the
    /// resource keeps in step.
    fn edited(_id: ResId) {}
}

/// A resource of any type: what an [`AssetRef`] holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ResId(pub u32);

/// A resource of type `T`. Deduplicated by path, so two handles to one file
/// read and edit the same value.
pub struct Res<T> {
    id: ResId,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for Res<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Res<T> {}
impl<T> PartialEq for Res<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for Res<T> {}
impl<T> std::fmt::Debug for Res<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Res({:?})", self.id)
    }
}

trait Stored: Send + Sync {
    fn export(&mut self) -> &mut dyn Export;
    fn export_ref(&self) -> &dyn Export;
    fn any(&self) -> &dyn Any;
}

impl<T: Resource> Stored for T {
    fn export(&mut self) -> &mut dyn Export {
        self
    }
    fn export_ref(&self) -> &dyn Export {
        self
    }
    fn any(&self) -> &dyn Any {
        self
    }
}

struct Entry {
    path: PathBuf,
    name: &'static str,
    value: Box<dyn Stored>,
    /// Bumped by every edit, so a consumer re-uploads only what changed.
    version: u64,
    /// `false` when the file would not read: saving it would write the
    /// defaults over whatever the file says.
    read: bool,
    edited: fn(ResId),
}

#[derive(Clone, Copy)]
struct Kind {
    name: &'static str,
    extension: &'static str,
    kind: AssetKind,
    make: fn() -> Box<dyn Stored>,
    reference: fn(ResId) -> AssetRef,
    edited: fn(ResId),
}

struct State {
    entries: Vec<Entry>,
    kinds: Vec<Kind>,
}

/// Read-locked by `with`, which a sweep calls from every worker at once.
static STATE: RwLock<State> = RwLock::new(State {
    entries: Vec::new(),
    kinds: Vec::new(),
});

/// Teach the registry, the browser and scene files about `T`. Idempotent.
pub fn register<T: Resource>() {
    let kind = Kind {
        name: T::NAME,
        extension: T::EXTENSION,
        kind: T::KIND,
        make: || Box::new(T::default()),
        reference: T::reference,
        edited: T::edited,
    };
    {
        let mut state = STATE.write();
        if state.kinds.iter().any(|k| k.name == kind.name) {
            return;
        }
        state.kinds.push(kind);
    }
    asset_type::register(Box::new(FileType(kind)));
}

/// The type whose files have `path`'s extension.
pub fn name_for(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let state = STATE.read();
    Some(state.kinds.iter().find(|k| k.extension == ext)?.name)
}

/// Every registered type, as `(name, extension)`.
pub fn kinds() -> Vec<(&'static str, &'static str)> {
    STATE
        .read()
        .kinds
        .iter()
        .map(|k| (k.name, k.extension))
        .collect()
}

/// The resource at `path`, read now if this is its first request. A file
/// that is not there yet reads as the type's defaults; one that will not
/// read is reported and held at the defaults, and never saved over. `None`
/// if no type is registered as `name`.
pub fn request(name: &str, path: &Path) -> Option<ResId> {
    let kind = {
        let state = STATE.read();
        if let Some(i) = state.entries.iter().position(|e| e.path == path) {
            return Some(ResId(i as u32));
        }
        *state.kinds.iter().find(|k| k.name == name)?
    };
    // Outside the lock: a field may be a resource handle, which requests too.
    let mut value = (kind.make)();
    let read = match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => fail(path, &e.to_string()),
        Ok(text) => match serde_json::from_str::<BTreeMap<String, Json>>(&text) {
            Ok(props) => {
                scene_file::set_props_on(value.export(), &props, &transform::detached());
                true
            }
            Err(e) => fail(path, &e.to_string()),
        },
    };
    let mut state = STATE.write();
    if let Some(i) = state.entries.iter().position(|e| e.path == path) {
        return Some(ResId(i as u32));
    }
    state.entries.push(Entry {
        path: path.to_path_buf(),
        name: kind.name,
        value,
        version: 0,
        read,
        edited: kind.edited,
    });
    Some(ResId(state.entries.len() as u32 - 1))
}

fn fail(path: &Path, e: &str) -> bool {
    eprintln!("resource: cannot read {}: {e}", path.display());
    false
}

pub fn path_of(id: ResId) -> Option<PathBuf> {
    Some(STATE.read().entries.get(id.0 as usize)?.path.clone())
}

/// The type `id` was requested as.
pub fn name_of(id: ResId) -> &'static str {
    STATE.read().entries[id.0 as usize].name
}

pub fn version(id: ResId) -> u64 {
    STATE.read().entries[id.0 as usize].version
}

/// Read `id` through its exported properties. Runs under the registry's
/// lock, so `f` must not request a resource.
pub fn read<R>(id: ResId, f: impl FnOnce(&dyn Export) -> R) -> R {
    f(STATE.read().entries[id.0 as usize].value.export_ref())
}

/// Change `id` through its exported properties, and bump its version.
/// Under the lock, as [`read`] is.
pub fn edit<R>(id: ResId, f: impl FnOnce(&mut dyn Export, &Transform) -> R) -> R {
    let (out, edited) = {
        let mut state = STATE.write();
        let e = &mut state.entries[id.0 as usize];
        e.version += 1;
        (f(e.value.export(), &transform::detached()), e.edited)
    };
    edited(id);
    out
}

/// Write `id` to its file.
pub fn save(id: ResId) -> Result<(), String> {
    let (path, text) = {
        let state = STATE.read();
        let e = &state.entries[id.0 as usize];
        if !e.read {
            return Err(format!(
                "{} did not read, so saving would overwrite it",
                e.path.display()
            ));
        }
        let props = scene_file::props_of(e.value.export_ref());
        (e.path.clone(), scene_file::pretty(&props))
    };
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// A new file at `path` holding `name`'s defaults. Refuses a path that is
/// already a file.
pub fn create(name: &str, path: &Path) -> Result<ResId, String> {
    if path.exists() {
        return Err(format!("{} already exists", path.display()));
    }
    let id = request(name, path).ok_or_else(|| format!("no resource type {name}"))?;
    save(id).map(|()| id)
}

impl<T: Resource> Res<T> {
    pub fn request(path: impl AsRef<Path>) -> Self {
        register::<T>();
        let id = request(T::NAME, path.as_ref()).expect("registered on the line above");
        Self::of(id)
    }

    fn of(id: ResId) -> Self {
        Self {
            id,
            _t: PhantomData,
        }
    }

    pub fn id(self) -> ResId {
        self.id
    }

    /// `id` as a handle of type `T`, if that is what it was requested as.
    pub fn checked(id: ResId) -> Option<Self> {
        typed(id)
    }

    pub fn version(self) -> u64 {
        version(self.id)
    }

    /// Read the value. Under the registry's lock, as [`read`] is.
    pub fn with<R>(self, f: impl FnOnce(&T) -> R) -> R {
        let state = STATE.read();
        let any = state.entries[self.id.0 as usize].value.any();
        f(any.downcast_ref().expect("a Res<T> is only minted for a T"))
    }
}

/// Keeps a handle to another type out: an id names its type in the registry.
fn typed<T: Resource>(id: ResId) -> Option<Res<T>> {
    (name_of(id) == T::NAME).then(|| Res::of(id))
}

impl<T: Resource> Exportable for Res<T> {
    const KIND: ValueKind = ValueKind::Asset(AssetKind::Resource(T::NAME));
    fn to_value(&self) -> Value {
        Value::Asset(Some(AssetRef::Resource(self.id)))
    }
    fn from_value(value: Value, _t: &Transform) -> Option<Self> {
        match value {
            Value::Asset(Some(AssetRef::Resource(id))) => typed(id),
            _ => None,
        }
    }
}

impl<T: Resource> Exportable for Option<Res<T>> {
    const KIND: ValueKind = ValueKind::Asset(AssetKind::Resource(T::NAME));
    fn to_value(&self) -> Value {
        Value::Asset(self.map(|r| AssetRef::Resource(r.id)))
    }
    fn from_value(value: Value, _t: &Transform) -> Option<Self> {
        match value {
            Value::Asset(None) => Some(None),
            Value::Asset(Some(AssetRef::Resource(id))) => typed(id).map(Some),
            _ => None,
        }
    }
}

/// The browser's half: a resource's extension, so its files drag onto the
/// properties that hold one.
struct FileType(Kind);

impl AssetType for FileType {
    fn extensions(&self) -> &[&str] {
        std::slice::from_ref(&self.0.extension)
    }

    fn kind(&self) -> AssetKind {
        self.0.kind
    }

    fn reference(&self, path: &Path) -> AssetRef {
        let id = request(self.0.name, path).expect("a file type is registered with its kind");
        (self.0.reference)(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default, Export)]
    #[export(resource = "system")]
    struct System {
        #[export]
        damping: f32,
        #[export]
        name: String,
    }

    #[derive(Clone, Default, Export)]
    #[export(resource = "profile")]
    struct Profile {
        #[export]
        range: f32,
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("engine_res_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join(name)
    }

    #[test]
    fn an_edit_saves_and_reads_back_through_another_handle() {
        let path = temp("smoke.system");
        let id = create(System::NAME, &path).expect("created");
        assert_eq!(
            Res::<System>::request(&path).id(),
            id,
            "one file, one resource"
        );
        edit(id, |c, t| c.set("damping", Value::F32(0.5), t));
        assert_eq!(version(id), 1);
        save(id).expect("saved");

        let copy = temp("copy.system");
        std::fs::copy(&path, &copy).expect("copy");
        let other = Res::<System>::request(&copy);
        assert_ne!(other.id(), id);
        assert_eq!(other.with(|s| s.damping), 0.5);
        assert!(create(System::NAME, &path).is_err(), "never over a file");
    }

    #[test]
    fn a_handle_of_another_type_is_refused() {
        let system = Res::<System>::request(temp("a.system"));
        let wrong = Value::Asset(Some(AssetRef::Resource(system.id())));
        let t = transform::detached();
        assert!(<Option<Res<Profile>>>::from_value(wrong.clone(), &t).is_none());
        assert_eq!(
            <Option<Res<System>>>::from_value(wrong, &t),
            Some(Some(system))
        );
        assert_eq!(
            asset_type::kind_of(Path::new("b.system")),
            Some(AssetKind::Resource("System"))
        );
    }

    #[test]
    fn a_file_that_will_not_read_is_never_saved_over() {
        let path = temp("broken.system");
        std::fs::write(&path, "{ not json").expect("write");
        let id = Res::<System>::request(&path).id();
        assert!(save(id).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{ not json");
    }
}

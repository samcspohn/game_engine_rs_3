//! Reflection: the value model `#[derive(Export)]` speaks (ADR-0010 §3).
//!
//! One mechanism, four consumers — the inspector's rows and typed drop
//! targets, the save walk, the `TYPE_NAME` a file names a component by, and
//! per-property deltas. All of them read [`Export`], which is object-safe
//! because none of them can name the component's type.
//!
//! The set of [`Value`] variants is deliberately closed. A field type that is
//! not [`Exportable`] does not compile rather than degrading to a string.
//!
//! See [`docs/notes/reflection.md`](../../../docs/notes/reflection.md) for
//! the derive's attribute grammar and how `MeshRenderer` uses it.

use glam::{Quat, Vec3};

use crate::{
    asset::MeshId, component::Entity, material::MaterialId, scene_asset::SceneId,
    texture::TextureId, transform::Transform,
};

pub use engine_derive::Export;

/// Which registry an [`AssetRef`] points into. A drop of the wrong kind is
/// rejected on this, before any id is dereferenced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssetKind {
    Mesh,
    Material,
    Texture,
    Scene,
}

/// A typed asset handle — what an asset-browser drag carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssetRef {
    Mesh(MeshId),
    Material(MaterialId),
    Texture(TextureId),
    Scene(SceneId),
}

impl AssetRef {
    pub fn kind(self) -> AssetKind {
        match self {
            Self::Mesh(_) => AssetKind::Mesh,
            Self::Material(_) => AssetKind::Material,
            Self::Texture(_) => AssetKind::Texture,
            Self::Scene(_) => AssetKind::Scene,
        }
    }
}

/// A property's type, known without a value — which is what lets an *empty*
/// slot still type its drop target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ValueKind {
    F32,
    I32,
    Bool,
    String,
    Vec3,
    Quat,
    Color,
    Asset(AssetKind),
    Entity,
}

/// One property's value in flight between a component and a consumer.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    F32(f32),
    I32(i32),
    Bool(bool),
    String(String),
    Vec3(Vec3),
    Quat(Quat),
    /// Linear RGBA.
    Color([f32; 4]),
    /// `None` is an empty slot, not an absent property.
    Asset(Option<AssetRef>),
    Entity(Option<Entity>),
}

/// Name and type of one exported property.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PropertyInfo {
    pub name: &'static str,
    pub kind: ValueKind,
}

/// A field type that can cross the [`Value`] boundary.
pub trait Exportable: Sized {
    const KIND: ValueKind;
    fn to_value(&self) -> Value;
    /// `None` rejects — a wrong-typed drop declines instead of crashing.
    fn from_value(value: Value) -> Option<Self>;
}

/// What `#[derive(Export)]` implements: a component's properties, by name.
pub trait Export {
    /// Stable across builds, unlike `TypeId`, so it can appear in a file.
    fn type_name(&self) -> &'static str;
    fn properties(&self) -> &'static [PropertyInfo];
    fn get(&self, name: &str) -> Option<Value>;
    /// `false` rejects: unknown name, or a value of the wrong kind.
    ///
    /// Takes the transform because a setter may be a method that publishes
    /// GPU state — `MeshRenderer::set_material` is exactly that.
    fn set(&mut self, name: &str, value: Value, transform: &Transform) -> bool;
}

macro_rules! scalar {
    ($t:ty, $v:ident, $k:ident) => {
        impl Exportable for $t {
            const KIND: ValueKind = ValueKind::$k;
            fn to_value(&self) -> Value {
                Value::$v(::core::clone::Clone::clone(self))
            }
            fn from_value(value: Value) -> Option<Self> {
                match value {
                    Value::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}

scalar!(f32, F32, F32);
scalar!(i32, I32, I32);
scalar!(bool, Bool, Bool);
scalar!(String, String, String);
scalar!(Vec3, Vec3, Vec3);
scalar!(Quat, Quat, Quat);
scalar!([f32; 4], Color, Color);

/// A handle and its `Option`, which share a [`ValueKind`]: an inspector shows
/// the same slot either way, and only `Option` can be emptied.
macro_rules! asset {
    ($t:ty, $k:ident) => {
        impl Exportable for $t {
            const KIND: ValueKind = ValueKind::Asset(AssetKind::$k);
            fn to_value(&self) -> Value {
                Value::Asset(Some(AssetRef::$k(*self)))
            }
            fn from_value(value: Value) -> Option<Self> {
                match value {
                    Value::Asset(Some(AssetRef::$k(id))) => Some(id),
                    _ => None,
                }
            }
        }
        impl Exportable for Option<$t> {
            const KIND: ValueKind = ValueKind::Asset(AssetKind::$k);
            fn to_value(&self) -> Value {
                Value::Asset(self.map(AssetRef::$k))
            }
            fn from_value(value: Value) -> Option<Self> {
                match value {
                    Value::Asset(None) => Some(None),
                    Value::Asset(Some(AssetRef::$k(id))) => Some(Some(id)),
                    _ => None,
                }
            }
        }
    };
}

asset!(MeshId, Mesh);
asset!(MaterialId, Material);
asset!(TextureId, Texture);
asset!(SceneId, Scene);

impl Exportable for Entity {
    const KIND: ValueKind = ValueKind::Entity;
    fn to_value(&self) -> Value {
        Value::Entity(Some(*self))
    }
    fn from_value(value: Value) -> Option<Self> {
        match value {
            Value::Entity(Some(e)) => Some(e),
            _ => None,
        }
    }
}

impl Exportable for Option<Entity> {
    const KIND: ValueKind = ValueKind::Entity;
    fn to_value(&self) -> Value {
        Value::Entity(*self)
    }
    fn from_value(value: Value) -> Option<Self> {
        match value {
            Value::Entity(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::Component;
    use crate::transform::{TransformHierarchy, _Transform};

    #[derive(Clone, Default, Export)]
    struct Probe {
        #[export]
        speed: f32,
        #[export]
        label: String,
        #[export]
        material: Option<MaterialId>,
        #[export(get = counted, set = bump)]
        count: i32,
        /// No `#[export]`: reflection must not see it.
        hidden: bool,
    }

    impl Probe {
        fn counted(&self) -> i32 {
            self.count
        }
        fn bump(&mut self, _t: &Transform, v: i32) {
            self.count = v + 1;
        }
    }

    impl Component for Probe {}

    fn hierarchy() -> TransformHierarchy {
        let mut h = TransformHierarchy::new();
        h.create_transform(_Transform::default());
        h
    }

    #[test]
    fn properties_list_only_exported_fields() {
        let names: Vec<_> = Probe::default().properties().iter().map(|p| p.name).collect();
        assert_eq!(names, ["speed", "label", "material", "count"]);
        assert_eq!(Probe::TYPE_NAME, "Probe");
        assert_eq!(Probe::default().type_name(), "Probe");
    }

    #[test]
    fn kinds_are_known_without_a_value() {
        let p = Probe::default();
        let kind = |n: &str| p.properties().iter().find(|i| i.name == n).unwrap().kind;
        assert_eq!(kind("speed"), ValueKind::F32);
        // Empty override, and the drop target is still typed to materials.
        assert_eq!(p.get("material"), Some(Value::Asset(None)));
        assert_eq!(kind("material"), ValueKind::Asset(AssetKind::Material));
    }

    #[test]
    fn set_round_trips_through_get() {
        let h = hierarchy();
        let t = h.get_transform_unchecked(1);
        let mut p = Probe::default();
        assert!(p.set("speed", Value::F32(2.5), &t));
        assert!(p.set("label", Value::String("hull".into()), &t));
        assert!(p.set("material", Value::Asset(Some(AssetRef::Material(MaterialId(7)))), &t));
        assert_eq!(p.get("speed"), Some(Value::F32(2.5)));
        assert_eq!(p.get("label"), Some(Value::String("hull".into())));
        assert_eq!(
            p.get("material"),
            Some(Value::Asset(Some(AssetRef::Material(MaterialId(7)))))
        );
    }

    #[test]
    fn a_wrong_kind_is_declined_not_a_panic() {
        let h = hierarchy();
        let t = h.get_transform_unchecked(1);
        let mut p = Probe::default();
        // A texture dropped on a material slot — the case a drop zone must
        // survive.
        let texture = Value::Asset(Some(AssetRef::Texture(TextureId(3))));
        assert!(!p.set("material", texture, &t));
        assert!(!p.set("speed", Value::Bool(true), &t));
        assert!(!p.set("hidden", Value::Bool(true), &t), "not exported");
        assert!(!p.hidden, "and not written behind the rejection");
        assert!(!p.set("nonexistent", Value::F32(1.0), &t));
        assert_eq!(p.get("material"), Some(Value::Asset(None)), "unchanged");
    }

    #[test]
    fn accessors_route_through_the_methods() {
        let h = hierarchy();
        let t = h.get_transform_unchecked(1);
        let mut p = Probe::default();
        assert!(p.set("count", Value::I32(10), &t));
        // `bump` adds one; a plain field write would not have.
        assert_eq!(p.get("count"), Some(Value::I32(11)));
    }

    #[test]
    fn an_option_slot_can_be_emptied() {
        let h = hierarchy();
        let t = h.get_transform_unchecked(1);
        let mut p = Probe {
            material: Some(MaterialId(4)),
            ..Probe::default()
        };
        assert!(p.set("material", Value::Asset(None), &t));
        assert_eq!(p.get("material"), Some(Value::Asset(None)));
    }

    /// The consumers hold `&dyn Export`, so the trait staying object-safe is
    /// a property worth a test rather than a comment.
    #[test]
    fn export_is_object_safe() {
        let p = Probe::default();
        let dynamic: &dyn Export = &p;
        assert_eq!(dynamic.properties().len(), 4);
    }
}

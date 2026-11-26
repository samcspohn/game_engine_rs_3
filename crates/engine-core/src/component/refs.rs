//! Checked references into a world: `Copy`, non-owning, and `None` once what
//! they name is gone — including when its slot has been reused since. An
//! [`Entity`] is a raw slot id; these are not.

use std::marker::PhantomData;

use crate::component::{Component, Entity, World};
use crate::transform::{Transform, WorldId};
use crate::worlds;

/// An entity in a world, as of the slot generation it was made against.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct EntityRef {
    world: WorldId,
    entity: Entity,
    generation: u32,
}

impl EntityRef {
    /// `None` if `entity` is not alive in `world`.
    pub fn new(world: &World, entity: Entity) -> Option<Self> {
        let h = world.hierarchy();
        let generation = h.generation(entity.id).filter(|_| h.is_active(entity.id))?;
        Some(Self {
            world: world.id(),
            entity,
            generation,
        })
    }

    /// The entity a component's `init` or `update` was handed.
    pub fn of(t: &Transform) -> Self {
        Self {
            world: t.world(),
            entity: Entity::new(t.get_idx()),
            generation: t.generation(),
        }
    }

    pub fn world(self) -> WorldId {
        self.world
    }

    pub fn entity(self) -> Entity {
        self.entity
    }

    /// Run `f` on the world and entity, or `None` if either is gone.
    pub fn with<R>(self, f: impl FnOnce(&World, Entity) -> R) -> Option<R> {
        let world = worlds::world(self.world)?;
        let h = world.hierarchy();
        let live =
            h.generation(self.entity.id) == Some(self.generation) && h.is_active(self.entity.id);
        live.then(|| f(&world, self.entity))
    }
}

/// Component `T` on an entity: how a callback finds its component again
/// without a pointer into storage that moves.
pub struct ComponentRef<T> {
    entity: EntityRef,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for ComponentRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ComponentRef<T> {}

impl<T> std::fmt::Debug for ComponentRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "ComponentRef<{}>({:?})",
            std::any::type_name::<T>(),
            self.entity
        )
    }
}

impl<T: Component + Send + Sync + 'static> ComponentRef<T> {
    pub fn new(entity: EntityRef) -> Self {
        Self {
            entity,
            _t: PhantomData,
        }
    }

    /// The component on the entity a component's `init` or `update` was handed.
    pub fn of(t: &Transform) -> Self {
        Self::new(EntityRef::of(t))
    }

    pub fn entity(self) -> EntityRef {
        self.entity
    }

    /// Run `f` against the component, or `None` if it or its entity is gone.
    /// Note: takes the component's lock — see ui-threading for the order.
    pub fn with<R>(self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let found = self.entity.with(|w, e| {
            let cell = w.get_component::<T>(e)?;
            Some(f(&mut cell.lock()))
        });
        found.flatten()
    }
}

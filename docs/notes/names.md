# Names

An entity's name is a `Name`: an interned base and a numeric suffix, 8 bytes
and `Copy`. `ship12` is `ship` and 12. The default, and what `""` reads as,
is anonymous, and an anonymous entity costs nothing here: it is not indexed,
not deduplicated, and allocates nothing. Spawn projectiles that way and reach
them through the triggers that hit them, not by name.

```rust
world.get("/root/map/player")          // absolute: the root is named "root"
world.get("map/player")                // the same, from the root
world.entity(e).get("gun/muzzle")      // below an entity
let p = Path::new("/root/map/player"); // parsed once, for a hot loop
world.get(&p)
for (e, player) in world.get_all_components::<Player>() { ... }
```

## Interning

Bases are interned for the life of the process, in one table shared by every
world, so a name means the same symbol in a document and in its play copy. A
scene template interns its names when it is built, which makes an instance's
names a copy rather than a string clone. A string's leaked bytes are never
freed. That is fine for names, which are few and repeat, and wrong for
anything generated per frame, so do not build names out of counters. The
suffix is that counter, and it is not interned.

A name prints back exactly as it was spelled. A digit run becomes a suffix
only when it would print the same way: `ship012`, `12` and `ship0` keep their
digits in the base. That makes the string and the `Name` one-to-one, which
is what lets a lookup by string be a parse and a probe.

## Unique among siblings

Each hierarchy keeps one map from `(parent, name)` to slot. There is no map
per entity. Creating, renaming and re-parenting all claim a name in it, and a
taken one gets the next free suffix under that parent: `ship`, `ship1`,
`ship2`. `v2` taken goes to `v3`. A base that ends in a digit, such as `x0`,
goes to `x0_1`, since `x01` would read back as a different name. A counter
makes the next suffix a probe, not a scan. It belongs to the name that was
taken, (parent, `ship`), and is released with that name, so it cannot outlive
the parent or reach the next occupant of a freed slot. It sits in a side map
rather than in the index entry: 20-byte entries instead of 16 made every
lookup about 15 ns slower. If the original `ship` is deleted and a new one
takes the name, the next collision starts counting again at 1 and steps once
past the `ship1`…`shipK` still there.

Because siblings cannot share a name, a path names at most one entity. That
removes the case where a path names neither of two siblings. Scene templates
apply the same rule when they are built, so a glTF that repeats a sibling
name gets `Hand` and `Hand1`, and the template's paths are the instance's
paths.

Lookup takes the map's read lock once per path and probes once per segment.
A `&str` path interns nothing: a segment nobody has spelled resolves to
`None` without growing the table. Writes take the write lock, except during
creation, which holds `&mut` and needs no lock.

The editor's undo names an entity by its path, which is why the path has to
name one entity ([undo](undo.md)). Moving an entity can rename it, so the
step for a move holds the name the entity had where it is going back to, and
puts that name back as well.

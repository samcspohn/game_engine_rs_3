# UI threading

What a component on the parallel sweep may touch, and how. The store is
[`UiCore`](ui-core.md) behind one global mutex, `ui()`; this note is about
not taking it.

## Why the lock had to come off the sweep

`update_all` runs component types one after another and the instances of one
type in parallel. A component that calls `ui()` in `update` serialises every
instance of its type on that lock — a thousand enemies each setting a health
bar would run one at a time. The lock is also half of a lock-order pair: a
worker holds its own component's lock and then takes the store's, while
`dispatch` takes the store's and then a component's. The two phases never
overlap, but nothing stops a component from reaching another component through
`This::edit` while it holds the store.

So the sweep gets three things that need no lock at all, and the lock is left
for the main thread.

## Reads: the capture snapshot

`ui::pointer_captured()` and `ui::keyboard_captured()` read one atomic byte
that `publish_capture` writes after `dispatch`, before the gizmo and the
sweep. They are what a camera controller or a hotkey asks every frame, so they
are what made the lock hot. They answer as of the input fold; a focus change
made during the sweep shows next frame.

## Paint writes: `Paint`

The `style` slot array lives in an `Arc` shared between `UiCore` and
[`Paint`](../../crates/engine-render/src/ui/mod.rs), reached with
`ui::paint()`. Every `SlotArray` stores atomic words in fixed-size chunks that
are allocated on first touch and never move, so a write takes `&self`: compare
the words, store them, `fetch_or` the dirty bit. That is the transform
hierarchy's pattern (`transform/mod.rs`) without its per-entity mutex — a
paint value is one word, nothing read-modify-writes it, and the only reader is
the upload, which runs after the sweep has joined.

The rule that replaces the mutex is **one writer per widget per frame**. Two
threads writing the same slot is a race the caller has to avoid, as it is for
any field.

The upload scans chunks instead of tracking min/max watermarks, because shared
watermarks are the one cache line every writer would contend on. A per-chunk
`any` flag keeps a clean chunk unread, so an idle frame costs one load per
allocated chunk and uploads nothing.

`Progress` is the first widget built on it. Its value is a `cut` stored in the
fill's `UiStyle`: `ui.vert` shrinks the quad and crops its uv by that
fraction, so a bar moving is one word and never a relayout.
`UiCore::set_style` keeps the cut, so restyling a fill cannot refill a bar the
game emptied; only `Paint` writes it.

Glyph quads are not shared yet. A label stays on the store, because rewriting
a run needs the run's position, scale and capacity, which live in `UiCore`.

## Structural writes: `defer`

Anything that moves layout — creating, removing, restyling a node — still
needs `&mut UiCore`. From the sweep, `ui::defer(|ui| ...)` queues it on a
per-thread queue; the renderer drains every queue after `sweep_all` and before
`run_layout`, so the change lands in the frame that asked.

These are closures rather than a typed command enum, and `defer` returns no
handle: structural writes from the sweep are rare, and the common one — a
widget made for a new entity — is already on the main thread, because
`Component::init` runs at the frame boundary. A deferred closure that needs to
hand a node back reaches its component with `This::edit`; the sweep has joined
by then, so that is the same store-then-component order `dispatch` uses.
Calling `ui()` inside a deferred closure deadlocks, since the drain already
holds the store.

## What still takes the lock on the sweep

`test-game`'s `UiDemo::update` polls the store every frame. It is one
instance, so it does not contend. The editor's chrome has no `update`.

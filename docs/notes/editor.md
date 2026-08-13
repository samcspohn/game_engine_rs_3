
  1. #[export] derive + value model — new crates/engine-derive proc-macro crate, an Export trait in engine-core (TYPE_NAME + &'static
  [Property] with get/set fn pointers), and the value enum.

  The ADR is right that MeshRenderer is where a naïve model breaks, and it's worse than it says: both fields are private
  (components.rs:39-44), mesh_id has no setter at all, and set_material (:100) takes a &Transform and does retain/release refcounting. So the
  derive needs method-routed accessors (#[export(get = …, set = …)]), not field access — decide that before writing the macro.

  2. enabled bitset — enabled + active_in_hierarchy Vec<AtomicU32> next to active/has_children, grown in the same block at
  transform/mod.rs:589-594. CPU side is one AND into the word load at component/mod.rs:251 (par_iter already takes &TransformHierarchy). GPU
  side pushes [transform_id, NO_RENDERER, MATERIAL_INHERIT] through the existing queue drained at lib.rs:1912 — re-enabling needs to reach
  the live MeshRenderer to get the real ids back.

  3. Play root as a sibling — the one place the ADR understates the work. Scene::instantiate (component/mod.rs:613) deep-clones another
  Scene, walking 0..len. You need a subtree clone within one Scene, and clone_from_other borrows source and destination storage from the same
  self.components map. That's a real refactor, not a call.

  4. Node keys + delta — NodeKey on TemplateNode (scene_asset.rs:109), populated in build_template from node.index() / prim.index() (:388,
  :409) instead of the nodes.len() push order. Then SubsceneInstance, the save walk, and threading a delta through spawn_subscene since drain
  lands frames later.

  5. Per-call catch_unwind (component/mod.rs:266) — clears the entity's enabled bit, so it depends on step 2; the set_hook ring buffer feeds
  the Console pane, which is a placeholder today (main.rs:126).

  6. Inheritance — base: Option<AssetRef> and a recursive call, by which point it's small.

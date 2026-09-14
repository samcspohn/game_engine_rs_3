//! The inspector panel: the entity's transform, then every `#[export]`ed
//! property of every component on it (ADR-0010 §3).

use crate::browser::AssetDrag;
use crate::hierarchy::row_text;
use engine::{
    AssetRef, Entity, PropertyInfo, Value, ValueKind, World,
    glam::{EulerRot, Quat, Vec3},
    transform::{ROOT, Transform},
    ui::Events,
    ui::{
        Button, ButtonStyle, Label, MenuStyle, NodeId, Scrub, TextField, TextFieldStyle, UiCore,
        style::{Display, Size, Style, auto, percent, px, zero},
        theme,
    },
};
use std::path::Path;

/// The entity's own TRS. Not a component: the hierarchy owns it, so the
/// inspector reads and writes it directly rather than through `Export`.
const TRANSFORM: &str = "Transform";

/// Its three properties, in the order they are shown.
const TRS: [(&str, ValueKind); 3] = [
    ("position", ValueKind::Vec3),
    ("rotation", ValueKind::Quat),
    ("scale", ValueKind::Vec3),
];

/// How many fields a kind is edited through. Zero is a read-only line: a
/// text box you cannot type a `MeshId` into would be a worse lie than
/// showing the value.
fn arity(kind: ValueKind) -> usize {
    match kind {
        ValueKind::F32 | ValueKind::I32 | ValueKind::Bool | ValueKind::String => 1,
        ValueKind::Vec3 | ValueKind::Quat => 3,
        ValueKind::Color => 4,
        // Composite too: a nested value is saved and shown, but there is no
        // widget that edits one yet.
        ValueKind::Asset(_) | ValueKind::Entity => 0,
        ValueKind::Struct(_) | ValueKind::List(_) => 0,
    }
}

/// How much a field of this kind moves per pixel of sideways drag. Degrees
/// need a coarser step than metres; a kind with no numbers gets none, and
/// keeps drag-to-select. `I32` is excluded because these are all decimals.
fn step(kind: ValueKind) -> Option<f32> {
    match kind {
        ValueKind::Quat => Some(0.25),
        ValueKind::F32 | ValueKind::Vec3 => Some(0.01),
        ValueKind::Color => Some(0.005),
        _ => None,
    }
}

/// One number of a kind, as the panel writes it. Shared by the read-back and
/// by a drag, so a scrubbed value is not shown to a different precision than
/// the one it will be re-read at.
fn fmt(kind: ValueKind, x: f32) -> String {
    match kind {
        ValueKind::F32 => format!("{x:.4}"),
        ValueKind::Color => format!("{x:.2}"),
        // `-0.000` reads as a value rather than as zero, and a rotation
        // produces one constantly.
        _ => format!("{:.3}", if x == 0.0 { 0.0 } else { x }),
    }
}

/// A [`Value`] as one string per field — or, for a kind with none, the one
/// line its row shows instead.
///
/// A rotation is Euler degrees and not the quaternion's four numbers, which
/// nobody can author. That round trip is not the identity, so a field the
/// user is in keeps what they typed until they leave it — which is what the
/// read-back does with every field anyway.
fn parts(v: &Value) -> Vec<String> {
    let f = |x: f32| fmt(ValueKind::Vec3, x);
    match v {
        Value::F32(x) => vec![fmt(ValueKind::F32, *x)],
        Value::I32(x) => vec![x.to_string()],
        Value::Bool(x) => vec![x.to_string()],
        Value::String(s) => vec![s.clone()],
        Value::Vec3(v) => vec![f(v.x), f(v.y), f(v.z)],
        Value::Quat(q) => {
            let (x, y, z) = q.to_euler(EulerRot::XYZ);
            vec![f(x.to_degrees()), f(y.to_degrees()), f(z.to_degrees())]
        }
        Value::Color(c) => c.iter().map(|x| fmt(ValueKind::Color, *x)).collect(),
        // "none" and not an em-dash: the font atlas is ASCII, and a glyph it
        // does not have renders as `?`, which reads as an error.
        Value::Entity(e) => vec![e.map_or("none".into(), |e| format!("entity {}", e.id))],
        Value::Asset(a) => vec![a.map_or("none".into(), asset_name)],
        Value::Struct(f) => vec![format!("{} fields", f.len())],
        Value::List(items) => vec![format!("{} items", items.len())],
    }
}

/// The file a handle was minted from, which is the only part of it anyone
/// recognises. A handle with no file behind it names its kind instead — that
/// is what a renderer with nothing dropped on it yet shows.
fn asset_name(a: AssetRef) -> String {
    let path = engine::asset_type::path_of(a);
    let name = path.as_deref().and_then(Path::file_name);
    name.map_or_else(
        || format!("no {}", a.kind().name()),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// A row's fields back into a [`Value`]. `None` on anything unparseable,
/// which is a field the user is still mid-way through and not an error to
/// report.
fn assemble(kind: ValueKind, fields: &[String]) -> Option<Value> {
    let n = |i: usize| -> Option<f32> { fields.get(i)?.trim().parse().ok() };
    let first = || fields.first().map(|s| s.trim());
    match kind {
        ValueKind::F32 => n(0).map(Value::F32),
        ValueKind::I32 => first()?.parse().ok().map(Value::I32),
        ValueKind::Bool => first()?.parse().ok().map(Value::Bool),
        ValueKind::String => Some(Value::String(first()?.to_string())),
        ValueKind::Vec3 => Some(Value::Vec3(Vec3::new(n(0)?, n(1)?, n(2)?))),
        ValueKind::Quat => Some(Value::Quat(Quat::from_euler(
            EulerRot::XYZ,
            n(0)?.to_radians(),
            n(1)?.to_radians(),
            n(2)?.to_radians(),
        ))),
        ValueKind::Color => Some(Value::Color([n(0)?, n(1)?, n(2)?, n(3)?])),
        ValueKind::Asset(_) | ValueKind::Entity => None,
        ValueKind::Struct(_) | ValueKind::List(_) => None,
    }
}

/// The entity's TRS as inspector values. Local and not global: it is what
/// the entity stores, and what a parent moves.
fn trs_values(t: &Transform) -> Vec<(&'static str, &'static str, Value)> {
    let g = t.lock();
    vec![
        (TRANSFORM, "position", Value::Vec3(g.get_position())),
        (TRANSFORM, "rotation", Value::Quat(g.get_rotation())),
        (TRANSFORM, "scale", Value::Vec3(g.get_scale())),
    ]
}

/// The write half. A property whose value did not survive `assemble` never
/// gets here, so there is nothing to report — only nothing to do.
fn set_trs(t: &Transform, prop: &str, v: &Value) {
    let g = t.lock();
    match (prop, v) {
        ("position", Value::Vec3(p)) => g.set_position(*p),
        ("rotation", Value::Quat(q)) => g.set_rotation(*q),
        ("scale", Value::Vec3(s)) => g.set_scale(*s),
        _ => {}
    }
}

/// One property's row.
///
/// Addressed by `(ty, prop)` and not by position: `ComponentRegistry::inspect`
/// walks a `HashMap`, and the order it hands components back in is not
/// something to bind a row to.
#[derive(Clone)]
struct PropRow {
    /// The row itself, which is the drop target for an asset property.
    node: NodeId,
    ty: &'static str,
    prop: &'static str,
    kind: ValueKind,
    /// One per field of the kind — three for a vector, none for a value the
    /// row shows rather than edits.
    fields: Vec<TextField>,
    /// The drag anchor beside each field. The engine reports the gesture;
    /// what a pixel is worth, and how the number is written, are the panel's.
    scrubs: Vec<Scrub>,
    text: Option<Label>,
}

/// The entity's transform, then every `#[export]`ed property of every
/// component on it (ADR-0010 §3), read through `&dyn Export` — the panel
/// names no component type, so a game's own components appear here with no
/// editor change.
///
/// Not a `Component`: [`Chrome`] owns it and calls it after the hierarchy,
/// which is also where the selection it needs lives.
#[derive(Clone)]
pub(crate) struct InspectorPanel {
    pane: NodeId,
    title: Label,
    /// What this panel put in the tree, so a rebuild can take it back out.
    owned: Vec<NodeId>,
    rows: Vec<PropRow>,
    /// Rebuilt on a selection change and not per frame — the set of
    /// components on an entity does not move while you look at it.
    shown: Option<u64>,
    /// Present only with a selection, so it is rebuilt with the rows.
    add: Option<Button>,
    /// What the open menu offers, in the order it offers it — the index the
    /// choice comes back as means nothing without the list that made it.
    offered: Vec<&'static str>,
    /// The component types on the shown entity, so the menu offers only what
    /// it does not already have.
    present: Vec<&'static str>,
    /// An add is queued and lands at the next frame boundary. Until the
    /// components differ from `present`, there is nothing new to draw.
    pub(crate) awaiting: bool,
}

/// The Add Component menu's payload. A type of its own so `menu_choice` tells
/// it from the hierarchy row menu's [`EntityRef`], which is the same two
/// numbers and would otherwise be read against the wrong list of items.
#[derive(Clone, Copy)]
struct AddTo {
    world: engine::transform::WorldId,
    id: u64,
}

impl InspectorPanel {
    pub(crate) fn new(ui: &mut UiCore, pane: NodeId) -> Self {
        let title = ui.label(pane, theme().text_px, theme().text_dim, "nothing selected");
        Self {
            pane,
            title,
            owned: Vec::new(),
            rows: Vec::new(),
            shown: None,
            add: None,
            offered: Vec::new(),
            present: Vec::new(),
            awaiting: false,
        }
    }

    fn rebuild(&mut self, ui: &mut UiCore, world: &World, id: Option<u64>) {
        for n in self.owned.drain(..) {
            ui.remove_node(n);
        }
        self.rows.clear();
        self.add = None;
        self.shown = id;
        let Some(id) = id else { return };

        // Collected before any widget is built: `inspect` holds each
        // component's lock for the callback, and building UI under it would
        // hold a component lock across the whole UI store's.
        let mut specs: Vec<(&'static str, &'static [PropertyInfo])> = Vec::new();
        world
            .entity(Entity::new(id as u32))
            .inspect(|e| specs.push((e.type_name(), e.properties())));

        // First, and not out of `specs`: every entity has one. The root is
        // the identity the hierarchy composes from rather than a pose, so
        // editing its transform would move nothing — it gets no section.
        if id != ROOT as u64 {
            self.section(ui, TRANSFORM, TRS.iter().copied());
        }
        for (ty, props) in &specs {
            self.section(ui, ty, props.iter().map(|p| (p.name, p.kind)));
        }

        // Last, so it sits under the sections taffy stacks above it. The root
        // gets one too: it is a pose the hierarchy composes from, not an
        // entity that cannot carry behaviour.
        let button = ui.button(
            self.pane,
            "Add Component",
            ButtonStyle {
                text_px: theme().text_px,
                padding: 3.0,
                ..ButtonStyle::default()
            },
        );
        self.owned.push(button.node());
        self.add = Some(button);
        self.present = specs.iter().map(|(ty, _)| *ty).collect();
    }

    /// One titled block of rows.
    fn section(
        &mut self,
        ui: &mut UiCore,
        ty: &'static str,
        props: impl Iterator<Item = (&'static str, ValueKind)>,
    ) {
        let t = theme();
        self.owned
            .push(ui.label(self.pane, t.text_px, t.accent, ty).node());
        for (prop, kind) in props {
            let row = ui.node(
                self.pane,
                Style {
                    display: Display::Flex,
                    size: Size {
                        width: percent(1.0_f32),
                        height: auto(),
                    },
                    gap: Size {
                        width: px(4.0),
                        height: zero(),
                    },
                    ..Default::default()
                },
            );
            ui.label(row, t.text_px, t.text_dim, prop);
            let fields: Vec<TextField> = (0..arity(kind))
                .map(|_| {
                    ui.text_field(
                        row,
                        "",
                        TextFieldStyle {
                            width: 90.0,
                            text_px: t.text_px,
                            padding: 1.0,
                            radius: 2.0,
                            fill: t.control_held,
                            ..TextFieldStyle::default()
                        },
                    )
                })
                .collect();
            // Share the row rather than each taking 90 px: three coupled
            // fields have to fit the panel one wide one did, and the panel
            // is whatever width the user has dragged it to.
            for f in &fields {
                let mut style = ui.node_style(f.node());
                style.flex_grow = 1.0;
                style.flex_basis = px(0.0);
                style.min_size.width = px(0.0);
                ui.set_node_style(f.node(), style);
            }
            let text = fields
                .is_empty()
                .then(|| ui.label(row, t.text_px, t.text, ""));
            // A property that names an asset takes a drop, and lights up
            // for one it can hold. Innermost-wins puts the fields of an
            // editable kind ahead of this, so only a label row ever claims.
            if let ValueKind::Asset(_) = kind {
                ui.set_events(row, Events::DROP | Events::HOVER);
            }
            self.owned.push(row);
            self.rows.push(PropRow {
                node: row,
                ty,
                prop,
                kind,
                scrubs: vec![Scrub::default(); fields.len()],
                fields,
                text,
            });
        }
    }

    pub(crate) fn update(&mut self, ui: &mut UiCore, world: &World, selected: Option<u64>) {
        let h = world.hierarchy();
        if selected != self.shown {
            self.rebuild(ui, world, selected);
            let title = match selected {
                Some(id) => row_text(h, id),
                None => "nothing selected".to_string(),
            };
            self.title.set_text(ui, &title);
        }
        let Some(id) = selected else { return };

        // A queued add lands a frame later, so the panel cannot rebuild on
        // the click. It rebuilds when the entity's components stop matching
        // what is drawn, which is true whichever frame the boundary ran on.
        if self.awaiting {
            let mut now: Vec<&'static str> = Vec::new();
            world
                .entity(Entity::new(id as u32))
                .inspect(|e| now.push(e.type_name()));
            if now != self.present {
                self.awaiting = false;
                self.rebuild(ui, world, selected);
            }
        }

        if self.add.is_some_and(|b| ui.clicked(b)) {
            self.offered = engine::script::types()
                .iter()
                .map(|t| t.name)
                .filter(|n| !self.present.contains(n))
                .collect();
            if !self.offered.is_empty() {
                let at: [f32; 2] = engine::input::cursor_position().into();
                let about = AddTo {
                    world: world.id(),
                    id,
                };
                ui.context_menu(at, &self.offered, about, MenuStyle::default());
            }
        }
        // The payload says which entity the menu was opened over, so a second
        // inspector's menu is not read as this one's.
        let picked = ui
            .menu_choice::<AddTo>()
            .filter(|(_, a)| a.world == world.id() && a.id == id)
            .and_then(|(i, _)| self.offered.get(i).copied());
        if let Some(name) = picked {
            if let Some(ty) = engine::script::find(name) {
                world.edit(Entity::new(id as u32), move |mut e| (ty.add)(&mut e));
                self.awaiting = true;
            }
        }

        let t = h.get_transform_unchecked(id as u32);

        // Commit first, read back second: otherwise a submit is overwritten
        // by the value it was replacing, in the same frame. A row commits
        // when any one of its fields does, and reads the rest as they
        // stand — which is the value they were showing. A drag writes its
        // field here and commits every frame it moves: the drag *is* the
        // edit, and waiting for the release would leave the viewport a frame
        // behind the number.
        let mut edits: Vec<(&'static str, &'static str, Value)> = Vec::new();
        for r in self.rows.iter_mut() {
            let mut commit = r.fields.iter().any(|f| f.submitted(ui));
            if let Some(step) = step(r.kind) {
                for (f, s) in r.fields.iter().zip(&mut r.scrubs) {
                    let at = f.text(ui).trim().parse().ok();
                    if let Some(v) = s.update(ui, *f, step, at) {
                        f.set_text(ui, &fmt(r.kind, v));
                        commit = true;
                    }
                }
            }
            if commit {
                let fields: Vec<String> = r.fields.iter().map(|f| f.text(ui).to_string()).collect();
                if let Some(v) = assemble(r.kind, &fields) {
                    edits.push((r.ty, r.prop, v));
                }
            }
        }
        // A typed drop, and its preview: the row lights up only while a file
        // the property could hold is over it, so a refusal shows before the
        // release rather than being reported after it.
        let th = theme();
        for r in &self.rows {
            let ValueKind::Asset(want) = r.kind else {
                continue;
            };
            let fits = |d: &&AssetDrag| d.kind == Some(want);
            let taken = ui
                .dropped_on::<AssetDrag>(r.node)
                .filter(fits)
                .map(|d| d.path.clone());
            let lit = ui.dragging::<AssetDrag>().filter(fits).is_some() && ui.hovered(r.node);
            if let Some(l) = r.text {
                l.set_color(ui, if lit { th.accent } else { th.text });
            }
            if let Some(a) = taken.as_deref().and_then(engine::asset_type::reference) {
                edits.push((r.ty, r.prop, Value::Asset(Some(a))));
            }
        }

        for (_, prop, v) in edits.iter().filter(|e| e.0 == TRANSFORM) {
            set_trs(&t, prop, v);
        }
        if edits.iter().any(|e| e.0 != TRANSFORM) {
            world.entity(Entity::new(id as u32)).inspect(|e| {
                for (ty, prop, v) in &edits {
                    if e.type_name() == *ty {
                        e.set(prop, v.clone(), &t);
                    }
                }
            });
        }

        // Read back every frame rather than echoing what was typed: a setter
        // is free to refuse or to clamp, and this is what shows that. The
        // gizmo is the transform's other writer, so a drag moves these
        // fields too.
        let mut values = trs_values(&t);
        world.entity(Entity::new(id as u32)).inspect(|e| {
            let ty = e.type_name();
            for p in e.properties() {
                if let Some(v) = e.get(p.name) {
                    values.push((ty, p.name, v));
                }
            }
        });
        for r in &self.rows {
            let Some((_, _, v)) = values.iter().find(|(ty, p, _)| *ty == r.ty && *p == r.prop)
            else {
                continue;
            };
            let shown = parts(v);
            match &r.text {
                Some(l) => l.set_text(ui, &shown.join(", ")),
                // Never into a focused field: that deletes what the user is
                // halfway through typing. `Enter` is where that ends — it
                // says the value is finished, so it is written back in the
                // panel's own format, and left selected to type over.
                None => {
                    for (f, s) in r.fields.iter().zip(&shown) {
                        let done = f.submitted(ui);
                        if done || !ui.focused(*f) {
                            f.set_text(ui, s);
                        }
                        if done {
                            f.focus(ui);
                        }
                    }
                }
            }
        }
    }
}

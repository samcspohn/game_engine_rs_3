//! Handlers registered per node, run once when something happens to them.
//!
//! The retained half of [ADR-0006](../../../../docs/ADR-0006-retained-mode-ui.md):
//! a widget says what to do when it is clicked instead of being asked every
//! frame whether it was. See [ui-events](../../../../docs/notes/ui-events.md)
//! for the identity argument and the ordering rule.

use std::any::Any;
use std::marker::PhantomData;

#[cfg(test)]
use parking_lot::Mutex;

use engine_core::component::{Component, ComponentRef};

use super::UiCore;
use super::tree::{Drag, Events, NodeId};
use crate::input::Keystroke;

/// A handler that re-enqueues for itself forever is a bug, not a frame rate.
const MAX_ROUNDS: u32 = 16;

/// What happened to a node.
///
/// A drop carries no payload: it stays in the pointer for the frame, so the
/// handler reads it with [`UiCore::dropped_on`] and a target that does not
/// want this type simply declines. Anything else would make the first
/// handler to look the only one allowed to.
pub enum Event {
    Click,
    RightClick,
    DoubleClick,
    HoverEnter,
    HoverLeave,
    /// Every frame the press is held, including after the pointer leaves.
    Drag(Drag),
    /// The release that ended a drag on this node, wherever it got to.
    Dropped(Drag),
    /// A grab was released on this node. Read it with `dropped_on`.
    DroppedOn,
    /// Wheel or thumb, in pixels of content.
    Scroll(f32),
    Changed(Value),
    Submit(String),
    FocusLost,
    /// The node's absolute box moved. Queued by the placement walk, so it
    /// arrives on the frame *after* the layout that moved it.
    Resized([f32; 4]),
    /// Seconds since this subscription last fired.
    Tick(f32),
    Notify,
    Key(Keystroke),
}

/// A control's new value, for [`Event::Changed`].
pub enum Value {
    Bool(bool),
    F32(f32),
    Index(usize),
    Text(String),
}

/// Whether a handler wants the next one.
#[derive(PartialEq, Eq, Debug)]
pub enum Keep {
    Yes,
    No,
}

impl Event {
    /// Which registration bit routes this. Filtering per handler is what lets
    /// one node hold a scroll handler and a click handler without either
    /// seeing the other's events.
    fn kind(&self) -> Events {
        match self {
            Event::Click => Events::CLICK,
            Event::RightClick => Events::RIGHT_CLICK,
            Event::DoubleClick => Events::DOUBLE_CLICK,
            Event::Drag(_) | Event::Dropped(_) => Events::DRAG,
            Event::HoverEnter | Event::HoverLeave => Events::HOVER,
            Event::DroppedOn => Events::DROP,
            Event::Scroll(_) => Events::SCROLL,
            Event::Changed(_) => Events::CHANGE,
            Event::Submit(_) | Event::FocusLost => Events::FOCUS,
            Event::Resized(_) => Events::RESIZE,
            Event::Tick(_) => Events::TICK,
            Event::Notify => Events::NOTIFY,
            Event::Key(_) => Events::KEY,
        }
    }
}

type Callback = Box<dyn FnMut(&mut UiCore, &Event) -> Keep + Send>;

struct Handler {
    kinds: Events,
    f: Callback,
}

/// An [`on_any`](UiCore::on_any) subscription: no node, and no event, because
/// what it is told is that there *was* one.
struct Wake {
    kinds: Events,
    f: Box<dyn FnMut(&mut UiCore) -> Keep + Send>,
}

/// Run `f` against `r`'s component, or report that it is gone. Note: the UI
/// lock is held here, so the order is UI then component — `dispatch` runs
/// before `sweep_all`, where components take them the other way round.
fn deliver_to<T: Component + Send + Sync + 'static>(
    r: ComponentRef<T>,
    ui: &mut UiCore,
    f: impl FnOnce(&mut T, &mut UiCore),
) -> Keep {
    r.with(|c| f(c, ui)).map_or(Keep::No, |_| Keep::Yes)
}

/// State kept on a node by [`UiCore::own`].
#[derive(Default)]
pub(crate) enum Attached {
    #[default]
    Empty,
    Held(Box<dyn Any + Send>),
    /// Out on loan to an [`UiRef::with`] further up the stack.
    Lent,
}

/// A typed handle to state kept on a node with [`UiCore::own`]. `Copy`, so a
/// listener captures it; a removed node makes it answer `None`.
pub struct UiRef<T> {
    node: NodeId,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for UiRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for UiRef<T> {}

impl<T> std::fmt::Debug for UiRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "UiRef<{}>({:?})", std::any::type_name::<T>(), self.node)
    }
}

impl<T: Any + Send> UiRef<T> {
    /// The node the state lives and dies with.
    pub fn node(self) -> NodeId {
        self.node
    }

    /// Run `f` on the state, or `None` if its node is gone. Note: the state is
    /// lent to `f`, so a `with` on it from inside `f` answers `None`.
    pub fn with<R>(self, ui: &mut UiCore, f: impl FnOnce(&mut T, &mut UiCore) -> R) -> Option<R> {
        let idx = self.node.idx as usize;
        if !ui.is_live(self.node) {
            return None;
        }
        let slot = ui.attached.get_mut(idx)?;
        let mut state = match std::mem::replace(slot, Attached::Lent) {
            Attached::Held(s) => s,
            other => {
                #[cfg(debug_assertions)]
                if matches!(other, Attached::Lent) {
                    eprintln!(
                        "ui: {self:?} is already borrowed by an outer `with`; this one does nothing"
                    );
                }
                *slot = other;
                return None;
            }
        };
        let r = state.downcast_mut::<T>().map(|t| f(t, ui));
        // Freed by `f`: the slot may already be somebody else's.
        if ui.is_live(self.node) {
            ui.attached[idx] = Attached::Held(state);
        }
        r
    }
}

/// One [`UiCore::every`] subscription.
struct Ticker {
    id: TickId,
    node: NodeId,
    period: f32,
    due: f64,
    f: Box<dyn FnMut(&mut UiCore, f32) -> Keep + Send>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TickId(u64);

/// Everything dispatch owns, so `UiCore` gains one field rather than five.
#[derive(Default)]
pub(crate) struct Dispatch {
    /// Indexed by node slot, like `controls`. Cleared by `free_subtree`.
    handlers: Vec<Option<Vec<Handler>>>,
    queue: Vec<(NodeId, Event)>,
    tickers: Vec<Ticker>,
    /// Subscriptions to *something happening*, whatever it happened to.
    any: Vec<Wake>,
    /// The union of their masks, so `emit` can answer "is anyone listening?"
    /// without walking them.
    any_mask: Events,
    next_tick: u64,
    /// Handler calls made. Bookkeeping, like `Pointer::walks`: it makes "an
    /// idle frame runs nothing" a test rather than a claim.
    pub(crate) calls: u64,
    /// Guards against a handler driving a frame from inside a handler, which
    /// would take the lock pair in an order nothing else can see.
    running: bool,
}

impl UiCore {
    /// Run `f` when `kinds` happens to `n`.
    ///
    /// Registering is also what makes the node listen: the bits are OR'd into
    /// its [`Events`] mask, so a click handler is a hit target by saying so
    /// once.
    pub fn on(
        &mut self,
        n: impl Into<NodeId>,
        kinds: Events,
        f: impl FnMut(&mut UiCore, &Event) -> Keep + Send + 'static,
    ) {
        let n = n.into();
        let idx = self.live(n);
        self.add_events(n, kinds.hittable());
        if self.dispatch.handlers.len() <= idx {
            self.dispatch.handlers.resize_with(idx + 1, || None);
        }
        self.dispatch.handlers[idx]
            .get_or_insert_with(Vec::new)
            .push(Handler {
                kinds,
                f: Box::new(f),
            });
    }

    /// [`on`](Self::on), with the component the handler edits looked up for
    /// it. A handler whose entity has gone unregisters itself.
    pub fn on_this<T: Component + Send + Sync + 'static>(
        &mut self,
        n: impl Into<NodeId>,
        kinds: Events,
        this: ComponentRef<T>,
        mut f: impl FnMut(&mut T, &mut UiCore, &Event) + Send + 'static,
    ) {
        self.on(n, kinds, move |ui, ev| {
            deliver_to(this, ui, |c, ui| f(c, ui, ev))
        });
    }

    /// Run `f` once per dispatch in which anything of `kinds` happened,
    /// wherever it happened.
    ///
    /// The coarse end of the same mechanism, for a caller whose reaction is
    /// "re-read everything" rather than "this node was clicked" — the editor
    /// chrome, which owns panels across four widgets whose internals it does
    /// not name. Once per dispatch and not once per event: a click that is
    /// also a drop is one thing happening, and a caller that re-reads twice
    /// for it is paying for the difference between them without using it.
    pub fn on_any<T: Component + Send + Sync + 'static>(
        &mut self,
        kinds: Events,
        this: ComponentRef<T>,
        mut f: impl FnMut(&mut T, &mut UiCore) + Send + 'static,
    ) {
        self.dispatch.any_mask = self.dispatch.any_mask | kinds;
        self.dispatch.any.push(Wake {
            kinds,
            f: Box::new(move |ui| deliver_to(this, ui, |c, ui| f(c, ui))),
        });
    }

    /// Call `f` at `hz` while `n` lives — for a value only the simulation can
    /// move, which no event names. Stop it with [`stop`](Self::stop).
    pub fn every<T: Component + Send + Sync + 'static>(
        &mut self,
        n: impl Into<NodeId>,
        hz: f32,
        this: ComponentRef<T>,
        mut f: impl FnMut(&mut T, &mut UiCore, f32) + Send + 'static,
    ) -> TickId {
        let n = n.into();
        self.live(n);
        self.add_events(n, Events::TICK);
        let id = TickId(self.dispatch.next_tick);
        self.dispatch.next_tick += 1;
        self.dispatch.tickers.push(Ticker {
            id,
            node: n,
            period: 1.0 / hz,
            due: self.pointer.now,
            f: Box::new(move |ui, dt| deliver_to(this, ui, |c, ui| f(c, ui, dt))),
        });
        id
    }

    pub fn stop(&mut self, t: TickId) {
        self.dispatch.tickers.retain(|k| k.id != t);
    }

    /// Tell `n`'s listeners that something outside the UI changed, at the next
    /// dispatch. From another thread or the sweep, call it inside [`defer`](super::defer).
    pub fn notify(&mut self, n: impl Into<NodeId>) {
        let n = n.into();
        self.live(n);
        self.emit(n, Event::Notify);
    }

    /// Keep `value` on `n`, dropped with the node — how a widget built from
    /// several nodes, like a panel, is found again by a listener.
    pub fn own<T: Any + Send>(&mut self, n: impl Into<NodeId>, value: T) -> UiRef<T> {
        let n = n.into();
        let idx = self.live(n);
        if self.attached.len() <= idx {
            self.attached.resize_with(idx + 1, Attached::default);
        }
        self.attached[idx] = Attached::Held(Box::new(value));
        UiRef {
            node: n,
            _t: PhantomData,
        }
    }

    /// Handler calls made since startup.
    pub fn dispatched(&self) -> u64 {
        self.dispatch.calls
    }

    /// OR bits into a node's mask without disturbing what it already takes.
    fn add_events(&mut self, n: NodeId, kinds: Events) {
        let idx = self.live(n);
        let p = &mut self.pointer.listens;
        if p.len() <= idx {
            p.resize(idx + 1, Events::NONE);
        }
        p[idx] = p[idx] | kinds;
    }

    /// Queue `ev` for `n`, if anything is listening. The guard is what keeps
    /// a polled-only UI paying nothing for a layer it does not use.
    pub(crate) fn emit(&mut self, n: NodeId, ev: Event) {
        let listening = self.dispatch.any_mask.has(ev.kind())
            || (self.dispatch.handlers)
                .get(n.idx as usize)
                .is_some_and(Option::is_some);
        if listening {
            self.dispatch.queue.push((n, ev));
        }
    }

    /// Drop everything registered against a recycled slot. Called by
    /// `free_subtree`, with the rest of the index-keyed side tables.
    pub(crate) fn forget_handlers(&mut self, idx: usize) {
        if let Some(h) = self.dispatch.handlers.get_mut(idx) {
            *h = None;
        }
        self.dispatch.tickers.retain(|k| k.node.idx as usize != idx);
        self.dispatch.queue.retain(|(n, _)| n.idx as usize != idx);
    }

    /// Run this frame's handlers, and whatever they enqueue in turn.
    ///
    /// Between `update_keyboard` and `worlds::sweep_all`: input is settled, so
    /// a handler sees the same frame a polling component would, and no
    /// component is running, so taking one's lock here cannot meet the same
    /// pair being taken the other way round.
    pub(crate) fn dispatch(&mut self, now: f64) {
        assert!(!self.dispatch.running, "dispatch re-entered from a handler");
        self.dispatch.running = true;
        self.run_tickers(now);
        for _ in 0..MAX_ROUNDS {
            if self.dispatch.queue.is_empty() {
                self.dispatch.running = false;
                return;
            }
            let mut woke = Events::NONE;
            for (n, ev) in std::mem::take(&mut self.dispatch.queue) {
                woke = woke | ev.kind();
                self.deliver(n, ev);
            }
            self.deliver_any(woke);
        }
        panic!("UI event storm: {MAX_ROUNDS} dispatch rounds and the queue is still filling");
    }

    /// One event to one node's handlers, in registration order.
    fn deliver(&mut self, n: NodeId, ev: Event) {
        let idx = n.idx as usize;
        if !self.is_live(n) {
            return; // freed, and its slot already belongs to someone else
        }
        let Some(mut hs) = self.dispatch.handlers.get_mut(idx).and_then(Option::take) else {
            return; // queued for an `on_any` subscriber; this node has none
        };
        // A placeholder, not `None`: a handler that emits to its own node has
        // to find a listener there, and the table is out on loan.
        self.dispatch.handlers[idx] = Some(Vec::new());
        let kind = ev.kind();
        let mut i = 0;
        while i < hs.len() {
            if !hs[i].kinds.has(kind) {
                i += 1;
                continue;
            }
            self.dispatch.calls += 1;
            match (hs[i].f)(self, &ev) {
                Keep::Yes => i += 1,
                Keep::No => drop(hs.remove(i)),
            }
        }
        // A handler may have freed this node, or registered more against it.
        // The generation check is what stops the first case restoring a table
        // to a slot somebody else now owns.
        if !self.is_live(n) {
            return;
        }
        if let Some(added) = self.dispatch.handlers[idx].take() {
            hs.extend(added);
        }
        self.dispatch.handlers[idx] = Some(hs);
    }

    /// The round's `on_any` subscribers, once between them.
    fn deliver_any(&mut self, woke: Events) {
        if !self.dispatch.any_mask.has(woke) {
            return;
        }
        let mut hs = std::mem::take(&mut self.dispatch.any);
        let mut i = 0;
        while i < hs.len() {
            if !hs[i].kinds.has(woke) {
                i += 1;
                continue;
            }
            self.dispatch.calls += 1;
            match (hs[i].f)(self) {
                Keep::Yes => i += 1,
                Keep::No => drop(hs.remove(i)),
            }
        }
        hs.extend(std::mem::take(&mut self.dispatch.any));
        self.dispatch.any_mask = hs.iter().fold(Events::NONE, |m, h| m | h.kinds);
        self.dispatch.any = hs;
    }

    /// Fire whatever subscriptions are due. Nothing at all when there are
    /// none, which is the editor's idle case.
    fn run_tickers(&mut self, now: f64) {
        let mut due: Vec<usize> = Vec::new();
        for (i, k) in self.dispatch.tickers.iter().enumerate() {
            if now >= k.due {
                due.push(i);
            }
        }
        for i in due {
            let Some(k) = self.dispatch.tickers.get_mut(i) else {
                continue;
            };
            let dt = (now - k.due) as f32 + k.period;
            k.due = now + k.period as f64;
            let (id, mut f) = (
                k.id,
                std::mem::replace(&mut k.f, Box::new(|_, _| Keep::Yes)),
            );
            self.dispatch.calls += 1;
            let keep = f(self, dt);
            match self.dispatch.tickers.iter_mut().find(|k| k.id == id) {
                Some(k) if keep == Keep::Yes => k.f = f,
                _ => self.dispatch.tickers.retain(|k| k.id != id),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::style::{Size, Style, px};
    use super::*;
    use engine_core::component::EntityRef;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A 100x100 box at the origin, opted into nothing until a handler says.
    fn box_at(core: &mut UiCore, x: f32) -> NodeId {
        let root = core.root();
        let n = core.node(
            root,
            Style {
                position: super::super::style::Position::Absolute,
                inset: super::super::style::Rect {
                    left: px(x),
                    top: px(0.0),
                    ..super::super::style::Rect::auto()
                },
                size: Size {
                    width: px(100.0),
                    height: px(100.0),
                },
                ..Default::default()
            },
        );
        core.run_layout([400.0, 400.0]);
        n
    }

    /// A counter a handler can bump from inside its box.
    fn counter() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let c = Arc::new(AtomicUsize::new(0));
        (c.clone(), c)
    }

    fn click(core: &mut UiCore, at: [f32; 2], t: f64) {
        core.update_pointer(at, true, false, 0.0, t);
        core.update_pointer(at, false, true, 0.0, t);
        core.dispatch(t);
    }

    #[test]
    fn a_click_reaches_its_handler_once() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let quiet = box_at(&mut core, 200.0);
        let (seen, bump) = counter();
        core.on(n, Events::CLICK, move |_, _| {
            bump.fetch_add(1, Ordering::Relaxed);
            Keep::Yes
        });

        click(&mut core, [50.0, 50.0], 0.0);
        assert_eq!(seen.load(Ordering::Relaxed), 1);

        // A node nobody registered against enqueues nothing at all.
        let before = core.dispatched();
        click(&mut core, [250.0, 50.0], 1.0);
        assert_eq!(core.dispatched(), before, "{quiet:?} has no handlers");
    }

    #[test]
    fn crossing_a_node_gives_enter_then_leave() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let log = Arc::new(Mutex::new(Vec::new()));
        let out = log.clone();
        core.on(n, Events::HOVER, move |_, ev| {
            out.lock().push(match ev {
                Event::HoverEnter => "enter",
                Event::HoverLeave => "leave",
                _ => "other",
            });
            Keep::Yes
        });

        core.update_pointer([50.0, 50.0], false, false, 0.0, 0.0);
        core.dispatch(0.0);
        core.update_pointer([250.0, 50.0], false, false, 0.0, 0.1);
        core.dispatch(0.1);
        assert_eq!(*log.lock(), ["enter", "leave"]);
    }

    /// Authoring step 6: a side table that outlives its node hands a recycled slot
    /// behaviour it never asked for.
    #[test]
    fn a_removed_node_takes_its_handlers_with_it() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let (seen, bump) = counter();
        core.on(n, Events::CLICK, move |_, _| {
            bump.fetch_add(1, Ordering::Relaxed);
            Keep::Yes
        });
        core.remove_node(n);

        let again = box_at(&mut core, 0.0);
        assert_eq!(again.index(), n.index(), "the slot was recycled");
        core.set_events(again, Events::CLICK);
        click(&mut core, [50.0, 50.0], 0.0);
        assert_eq!(seen.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_handler_registered_during_dispatch_runs_on_the_next_event() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let (seen, bump) = counter();
        core.on(n, Events::CLICK, move |ui, _| {
            let bump = bump.clone();
            ui.on(n, Events::CLICK, move |_, _| {
                bump.fetch_add(1, Ordering::Relaxed);
                Keep::Yes
            });
            Keep::No
        });

        click(&mut core, [50.0, 50.0], 0.0);
        assert_eq!(seen.load(Ordering::Relaxed), 0, "not within its own round");
        click(&mut core, [50.0, 50.0], 1.0);
        assert_eq!(seen.load(Ordering::Relaxed), 1);
    }

    #[test]
    #[should_panic(expected = "event storm")]
    fn a_handler_that_re_enqueues_itself_forever_is_a_bug() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        core.on(n, Events::CLICK, move |ui, _| {
            ui.emit(n, Event::Click);
            Keep::Yes
        });
        click(&mut core, [50.0, 50.0], 0.0);
    }

    /// The identity argument, tested at its weak point: the entity is gone,
    /// and the handler that named it has to notice rather than reach.
    #[test]
    fn a_handler_whose_entity_died_unregisters_itself() {
        use engine_core::component::World;
        use engine_core::reflect::Export;
        use engine_core::transform::_Transform;

        #[derive(Clone, Default)]
        struct Counted(Arc<AtomicUsize>);
        impl Export for Counted {}
        impl Component for Counted {
            const HAS_UPDATE: bool = false;
        }

        let w = engine_core::worlds::new_world();
        let edit = || unsafe { w.get_mut() } as &mut World;
        let e = edit().new_entity(_Transform::default());
        let (seen, bump) = counter();
        edit().add_component(e, Counted(bump));

        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let this = ComponentRef::<Counted>::new(EntityRef::new(edit(), e).unwrap());
        core.on_this(n, Events::CLICK, this, |c, _, _| {
            c.0.fetch_add(1, Ordering::Relaxed);
        });

        click(&mut core, [50.0, 50.0], 0.0);
        assert_eq!(seen.load(Ordering::Relaxed), 1);

        edit().remove_entity(e);
        let calls = core.dispatched();
        click(&mut core, [50.0, 50.0], 1.0);
        assert_eq!(core.dispatched(), calls + 1, "it fires once and declines");
        click(&mut core, [50.0, 50.0], 2.0);
        assert_eq!(core.dispatched(), calls + 1, "and is gone");
    }

    /// State lives and dies with its node: a recycled slot must not hand
    /// the next owner a panel it never attached.
    #[test]
    fn owned_state_is_found_again_and_freed_with_the_node() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        let count = core.own(n, 41u32);
        let inner = count.with(&mut core, |v, ui| {
            *v += 1;
            count.with(ui, |_, _| ())
        });
        assert_eq!(inner, Some(None), "lent during the call");
        assert_eq!(count.with(&mut core, |v, _| *v), Some(42), "and returned");

        core.remove_node(n);
        assert_eq!(count.with(&mut core, |v, _| *v), None, "stale handle");
        let again = box_at(&mut core, 0.0);
        assert_eq!(again.index(), n.index());
        let fresh = core.own(again, String::from("new owner"));
        assert_eq!(count.with(&mut core, |v, _| *v), None, "recycled slot");
        assert_eq!(
            fresh.with(&mut core, |s, _| s.clone()).as_deref(),
            Some("new owner")
        );
    }

    /// The editor's shape: one subscription, and a click that is also a drag
    /// and a drop is one thing having happened.
    #[test]
    fn on_any_wakes_once_per_dispatch_however_many_events() {
        use engine_core::component::World;
        use engine_core::reflect::Export;
        use engine_core::transform::_Transform;

        #[derive(Clone, Default)]
        struct Panel(Arc<AtomicUsize>);
        impl Export for Panel {}
        impl Component for Panel {
            const HAS_UPDATE: bool = false;
        }

        let w = engine_core::worlds::new_world();
        let edit = || unsafe { w.get_mut() } as &mut World;
        let e = edit().new_entity(_Transform::default());
        let (seen, bump) = counter();
        edit().add_component(e, Panel(bump));

        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        core.set_events(n, Events::CLICK);
        let this = ComponentRef::<Panel>::new(EntityRef::new(edit(), e).unwrap());
        core.on_any(Events::CLICK | Events::DRAG, this, |c, _| {
            c.0.fetch_add(1, Ordering::Relaxed);
        });

        // A press, a move and a release: two dispatches that saw an event,
        // and five events between them.
        core.update_pointer([50.0, 50.0], true, false, 0.0, 0.0);
        core.dispatch(0.0);
        core.update_pointer([50.0, 60.0], false, true, 0.0, 0.1);
        core.dispatch(0.1);
        assert_eq!(seen.load(Ordering::Relaxed), 2);

        core.update_pointer([50.0, 60.0], false, false, 0.0, 0.2);
        core.dispatch(0.2);
        assert_eq!(seen.load(Ordering::Relaxed), 2, "an idle frame is not one");
    }

    /// A coarse subscriber stamps against this to run once a frame however
    /// many things woke it. Counted ahead of `update_pointer`'s early-out,
    /// because an idle frame is still a frame — a counter that froze would
    /// make the stamp suppress every refresh after it.
    #[test]
    fn the_frame_counter_advances_on_idle_frames_too() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        core.set_events(n, Events::CLICK);

        core.update_pointer([50.0, 50.0], false, false, 0.0, 0.0);
        let settled = core.frame();
        for f in 1..10 {
            core.update_pointer([50.0, 50.0], false, false, 0.0, f as f64);
        }
        assert_eq!(core.frame(), settled + 9);
        assert_eq!(core.pointer.walks, 1, "and none of them walked");
    }

    /// The whole point. A UI nobody is touching runs nothing and writes
    /// nothing, however many handlers are registered.
    #[test]
    fn a_hundred_idle_frames_dispatch_nothing() {
        let mut core = UiCore::new();
        let n = box_at(&mut core, 0.0);
        core.on(n, Events::CLICK | Events::HOVER, |_, _| Keep::Yes);
        core.update_pointer([50.0, 50.0], false, false, 0.0, 0.0);
        core.dispatch(0.0);
        core.run_layout([400.0, 400.0]);

        let (mut stage, mut dirty) = (vec![0u32; 1 << 16], vec![0u32; 1 << 10]);
        core.quad.upload(&mut stage, &mut dirty);
        core.style.upload(&mut stage, &mut dirty);
        let (calls, walks) = (core.dispatched(), core.pointer.walks);

        for f in 0..100 {
            core.update_pointer([50.0, 50.0], false, false, 0.0, f as f64);
            core.dispatch(f as f64);
            core.run_layout([400.0, 400.0]);
        }
        assert_eq!(core.dispatched(), calls, "no handler ran");
        assert_eq!(core.pointer.walks, walks, "no hit walk");
        assert_eq!(core.quad.upload(&mut stage, &mut dirty), (i64::MAX, -1));
    }
}

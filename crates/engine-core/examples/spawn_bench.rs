//! Spawn, instantiate and name-lookup costs, per entity.
//!
//!     cargo run -p engine-core --example spawn_bench --release [filter]
//!
//! The baseline for named entities: what a projectile, a named scene and a
//! lookup by name cost, so a change to how names are kept is measured.

use engine_core::component::{Component, Entity, World};
use engine_core::script::{self, ComponentType};
use engine_core::transform::{_Transform, Path, ROOT};
use engine_core::{Export, inherit, new_world, scene_asset};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Default, Export)]
struct Motion {
    #[export]
    speed: f32,
    #[export]
    drag: f32,
}
impl Component for Motion {}

#[derive(Clone, Default, Export)]
struct Emitter {
    #[export]
    rate: f32,
}
impl Component for Emitter {}

macro_rules! dummies {
    ($($n:ident),*) => {
        $(
            #[derive(Clone, Default, Export)]
            struct $n {
                #[export]
                x: f32,
            }
            impl Component for $n {}
        )*
        /// Thirty more component types in `w`'s registry, as a game has.
        fn register_dummies(w: &mut World, e: Entity) {
            $(w.add_component(e, $n::default());)*
        }
    };
}
dummies!(
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20,
    D21, D22, D23, D24, D25, D26, D27, D28, D29
);

const ROUNDS: usize = 7;

/// Best and median ns per despawn of what `setup` spawned, in its order.
fn despawn(label: &str, setup: impl Fn(&mut World) -> Vec<Entity>) {
    if !wanted(label) {
        return;
    }
    let mut ns: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let handle = new_world();
            let w = unsafe { handle.get_mut() };
            let es = setup(w);
            let start = Instant::now();
            for &e in &es {
                w.remove_entity(e);
            }
            start.elapsed().as_nanos() as f64 / es.len() as f64
        })
        .collect();
    ns.sort_by(f64::total_cmp);
    println!(
        "{label:<44} {:>9.1} ns  (median {:.1})",
        ns[0],
        ns[ROUNDS / 2]
    );
}

/// Best and median ns per item over `ROUNDS`, each on a fresh world.
fn time(label: &str, items: usize, mut f: impl FnMut(&mut World)) {
    if !wanted(label) {
        return;
    }
    let mut ns: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let handle = new_world();
            let w = unsafe { handle.get_mut() };
            let start = Instant::now();
            f(w);
            start.elapsed().as_nanos() as f64 / items as f64
        })
        .collect();
    ns.sort_by(f64::total_cmp);
    println!(
        "{label:<44} {:>9.1} ns  (median {:.1})",
        ns[0],
        ns[ROUNDS / 2]
    );
}

/// Whether the case `label` is among those the first argument picks.
fn wanted(label: &str) -> bool {
    std::env::args().nth(1).is_none_or(|f| label.contains(&f))
}

fn entity(w: &mut World, name: &str, parent: u32) -> Entity {
    w.new_entity(_Transform {
        name: name.into(),
        parent: Some(parent),
        .._Transform::default()
    })
}

fn main() {
    let _ = engine_core::util::parallel::global::init(
        engine_core::util::parallel::BackendKind::MyPool,
        4,
    );
    script::register(&[
        ComponentType::of::<Motion>(),
        ComponentType::of::<Emitter>(),
    ]);
    let dir = std::env::temp_dir().join(format!("spawn_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // A projectile: one entity, three properties set.
    let projectile = dir.join("projectile.scene");
    std::fs::write(
        &projectile,
        "version = 4\n[\"/projectile\"]\n\
         Motion = { speed = 40.0, drag = 0.1 }\nEmitter = { rate = 30.0 }\n",
    )
    .expect("write projectile");
    // The same, its components on the scene's root: one entity, not two.
    let bare = dir.join("bare.scene");
    std::fs::write(
        &bare,
        "version = 4\n[\"/\"]\nMotion = { speed = 40.0, drag = 0.1 }\nEmitter = { rate = 30.0 }\n",
    )
    .expect("write bare");
    let bare = scene_asset::request_scene(&bare);
    // A named scene: 100 groups of 100 leaves, no components.
    let big = dir.join("big.scene");
    let mut text = String::from("version = 4\n");
    for g in 0..100 {
        text += &format!("[\"/g{g}\"]\n");
        (0..100).for_each(|l| text += &format!("[\"/g{g}/n{l}\"]\n"));
    }
    std::fs::write(&big, text).expect("write big");
    let (projectile, big) = (
        scene_asset::request_scene(&projectile),
        scene_asset::request_scene(&big),
    );

    const N: usize = 10_000;
    println!("per entity, best of {ROUNDS} fresh worlds\n");
    time("new_entity, unnamed", N, |w| {
        for _ in 0..N {
            black_box(entity(w, "", ROOT));
        }
    });
    time("new_entity, named \"ship\", suffixed", N, |w| {
        for _ in 0..N {
            black_box(entity(w, "ship", ROOT));
        }
    });
    time("new_entity + 2 components", N, |w| {
        for _ in 0..N {
            let e = entity(w, "", ROOT);
            w.add_component(
                e,
                Motion {
                    speed: 40.0,
                    drag: 0.1,
                },
            );
            w.add_component(e, Emitter { rate: 30.0 });
        }
    });
    time("projectile scene, instantiate_into", N, |w| {
        for _ in 0..N {
            let root = entity(w, "", ROOT);
            assert!(scene_asset::instantiate_into(w, projectile, root));
        }
    });
    time("projectile as scene root components", N, |w| {
        for _ in 0..N {
            let root = entity(w, "", ROOT);
            assert!(scene_asset::instantiate_into(w, bare, root));
        }
    });
    let held = scene_asset::template(bare).expect("built");
    time("projectile, template held", N, |w| {
        for _ in 0..N {
            let root = entity(w, "", ROOT);
            scene_asset::instantiate_template(w, &held, root);
        }
    });
    time("projectile: template lookup alone", N, |_| {
        for _ in 0..N {
            black_box(scene_asset::template(black_box(bare)));
        }
    });
    time("10k-node named scene, per node", 10_100, |w| {
        let root = entity(w, "big", ROOT);
        assert!(scene_asset::instantiate_into(w, big, root));
    });

    // Lookups against one built scene, and against a 10k-sibling parent.
    let handle = new_world();
    let w = unsafe { handle.get_mut() };
    let root = entity(w, "big", ROOT);
    scene_asset::instantiate_into(w, big, root);
    let wide = entity(w, "wide", ROOT);
    for i in 0..N {
        entity(w, &format!("s{i}"), wide.id);
    }
    let lookups = |label: &str, root: u32, path: &str| {
        if !wanted(label) {
            return;
        }
        const L: usize = 100_000;
        let start = Instant::now();
        for _ in 0..L {
            black_box(inherit::resolve(w, root, black_box(path)).expect(path));
        }
        let ns = start.elapsed().as_nanos() as f64 / L as f64;
        println!("{label:<44} {ns:>9.1} ns");
    };
    println!();
    lookups("resolve \"g50/n50\" (100 x 100)", root.id, "g50/n50");
    lookups("resolve \"g99/n99\" (last of each)", root.id, "g99/n99");
    lookups("resolve \"s5000\" (10k siblings)", wide.id, "s5000");
    let found = |label: &str, f: &dyn Fn() -> bool| {
        if !wanted(label) {
            return;
        }
        const L: usize = 100_000;
        let start = Instant::now();
        for _ in 0..L {
            assert!(black_box(f()));
        }
        let ns = start.elapsed().as_nanos() as f64 / L as f64;
        println!("{label:<44} {ns:>9.1} ns");
    };
    found("world.get \"/root/big/g50/n50\"", &|| {
        w.get(black_box("/root/big/g50/n50")).is_some()
    });
    let path = Path::new("/root/big/g50/n50");
    found("world.get, Path parsed once", &|| {
        w.get(black_box(&path)).is_some()
    });

    // Despawning only: the spawning that sets it up is not timed.
    println!();
    let spawned = |w: &mut World, parent: u32| {
        let root = entity(w, "", parent);
        scene_asset::instantiate_template(w, &held, root);
        root
    };
    despawn("remove_entity, 10k siblings, oldest first", |w| {
        (0..N).map(|_| spawned(w, ROOT)).collect()
    });
    despawn("remove_entity, 10k siblings, newest first", |w| {
        (0..N)
            .rev()
            .map(|_| spawned(w, ROOT))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    });
    despawn("remove_entity, 100 siblings each", |w| {
        let groups: Vec<u32> = (0..100).map(|_| entity(w, "", ROOT).id).collect();
        (0..N).map(|i| spawned(w, groups[i % 100])).collect()
    });
    despawn("remove_entity, 100 siblings, 32 types", |w| {
        let groups: Vec<u32> = (0..100).map(|_| entity(w, "", ROOT).id).collect();
        let other = entity(w, "", ROOT);
        register_dummies(w, other);
        (0..N).map(|i| spawned(w, groups[i % 100])).collect()
    });
    std::fs::remove_dir_all(&dir).ok();
}

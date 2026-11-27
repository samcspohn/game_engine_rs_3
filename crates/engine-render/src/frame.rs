//! The device side of a frame, one method per phase. They run in the order
//! `RenderApp::about_to_wait` calls them, explained in [frame-loop](../../../docs/notes/frame-loop.md).

use std::sync::Arc;

use vulkano::{command_buffer::PrimaryAutoCommandBuffer, format::Format, image::view::ImageView};

use engine_core::util::parallel;
use engine_core::worlds::{self, WorldHandle};

use crate::FrameStats;
use crate::assets::{GpuMaterialStore, GpuMeshStore, GpuTextureStore};
use crate::camera::{self, CameraHandle, CameraResolution, DrawPlan, RenderCamera, WorldSource};
use crate::frame_slot::{FrameSlot, build_all_frame_slots, frame_slot_index};
use crate::gpu::{Gpu, Stores};
use crate::gpu_renderers::GpuRenderers;
use crate::step::{GpuStep, Rebuild, StepCx};
use crate::swapchain::{AcquiredFrame, SwapchainRenderer};
use crate::transform_gpu::{StageTimes, StagingMemory, TransformGpuShared, WorldTransformGpu};
use crate::ui::{self, UiGpu};

/// One world's GPU state. Kept per world because each world numbers its
/// transform slots from 0 (ADR-0011 §1).
pub(crate) struct WorldRender {
    pub world: WorldHandle,
    pub transforms: WorldTransformGpu,
    pub renderers: GpuRenderers,
}

impl WorldRender {
    fn new(
        world: &WorldHandle,
        gpu: &Gpu,
        shared: Arc<TransformGpuShared>,
        staging: StagingMemory,
    ) -> Self {
        let cap = world.hierarchy().len().max(1);
        Self {
            world: world.clone(),
            transforms: WorldTransformGpu::new(shared, &gpu.memory, cap, staging),
            renderers: GpuRenderers::new(gpu, cap as u32),
        }
    }

    /// The steps that run after this world's transforms, in frame order. A
    /// new feature adds a field here and an entry to `steps` and `steps_mut`.
    pub fn steps(&self) -> [&dyn GpuStep; 1] {
        [&self.renderers]
    }

    fn steps_mut(&mut self) -> (StepCx<'_>, [&mut dyn GpuStep; 1]) {
        let cx = StepCx {
            world: &self.world,
            transforms: &self.transforms,
        };
        (cx, [&mut self.renderers])
    }

    fn prepare(&mut self, gpu: &Gpu) -> Rebuild {
        let base = self.transforms.prepare(&self.world, &gpu.memory);
        let (cx, steps) = self.steps_mut();
        steps.into_iter().fold(base, |r, s| r.max(s.prepare(&cx)))
    }

    fn stage(&mut self, slot: usize, pool: Option<&parallel::Pool>) -> (StageTimes, u64) {
        let times = self.transforms.stage(&self.world, pool);
        let start = std::time::Instant::now();
        let (cx, steps) = self.steps_mut();
        for s in steps {
            s.stage(&cx, slot);
        }
        (times, start.elapsed().as_nanos() as u64)
    }
}

pub(crate) struct RenderContext {
    pub swapchain_image_views: Vec<Arc<ImageView>>,
    /// Worlds share the pipelines, staging slot and frame signal.
    pub transform_shared: Arc<TransformGpuShared>,
    pub worlds: Vec<WorldRender>,
    /// The device half of every [`CameraHandle`], in slot order.
    pub cameras: Vec<RenderCamera>,
    /// True per camera slot if the camera is dropped, switched off or in a
    /// hidden tab. Updated once a frame.
    cameras_idle: Vec<bool>,
    frame_slots: Vec<FrameSlot>,
    stores: Stores,
    /// Device side of the global `ui::ui()` store (ADR-0008).
    pub ui_gpu: UiGpu,
    /// Set when only draw-plan bases moved. `stage` writes them after the host
    /// wait.
    cheap_plans: Option<Vec<DrawPlan>>,
}

impl RenderContext {
    pub fn new(
        gpu: &Gpu,
        views: Vec<Arc<ImageView>>,
        swapchain_format: Format,
        worlds: &[WorldHandle],
        staging: StagingMemory,
    ) -> Self {
        // The descriptor sets built below bind each store's placeholder,
        // which only the first sync uploads.
        let mut stores = Stores {
            mesh: GpuMeshStore::new(gpu.memory.clone(), gpu.cbs.clone(), gpu.queue.clone()),
            texture: GpuTextureStore::new(gpu.memory.clone(), gpu.cbs.clone(), gpu.queue.clone()),
            material: GpuMaterialStore::new(gpu.memory.clone(), gpu.cbs.clone(), gpu.queue.clone()),
        };
        let _ = stores.texture.sync();
        let _ = stores.material.sync();
        let transform_shared = TransformGpuShared::new(
            gpu.queue.device().clone(),
            &gpu.memory,
            &gpu.sets,
            &gpu.cbs,
            gpu.queue.clone(),
        );
        let worlds: Vec<WorldRender> = worlds
            .iter()
            .map(|w| WorldRender::new(w, gpu, transform_shared.clone(), staging))
            .collect();
        assert!(!worlds.is_empty(), "Window::with_world takes at least one");
        let _ = stores.mesh.sync();

        let extent = extent_of(&views);
        let plans = world_plans(&stores.mesh, &worlds);
        // A game's camera doesn't exist yet. `sync_cameras` builds it once its
        // `CameraComponent` appears.
        let cameras: Vec<RenderCamera> = {
            let res = gpu.scene_resources(&transform_shared, &stores);
            (0..camera::camera_count())
                .filter_map(camera::camera)
                .map(|state| {
                    let sources = world_sources(&state, &worlds, &plans);
                    RenderCamera::new_match_swapchain(state, extent, &res, &sources)
                })
                .collect()
        };
        let mut ui_gpu = UiGpu::new(
            gpu.queue.device().clone(),
            gpu.memory.clone(),
            gpu.sets.clone(),
            gpu.cbs.clone(),
            gpu.queue.clone(),
            &stores.texture,
            &camera_targets(&cameras),
            swapchain_format,
            extent,
        );
        ui::ui().run_layout([extent[0] as f32, extent[1] as f32]);
        ui_gpu.ensure_capacity(&mut ui::ui());

        let mut rcx = Self {
            swapchain_image_views: views,
            transform_shared,
            worlds,
            cameras_idle: vec![false; cameras.len()],
            cameras,
            frame_slots: Vec::new(),
            stores,
            ui_gpu,
            cheap_plans: None,
        };
        rcx.rebuild_frame_slots(gpu);
        rcx
    }

    pub fn extent(&self) -> [u32; 2] {
        extent_of(&self.swapchain_image_views)
    }

    /// Frees the GPU state of worlds nobody else holds a handle to, such as
    /// the play world after stop.
    pub fn retire_worlds(&mut self) -> Rebuild {
        let before = self.worlds.len();
        self.worlds.retain(|wr| !wr.world.is_orphan());
        match before == self.worlds.len() {
            true => Rebuild::None,
            false => Rebuild::Cameras,
        }
    }

    /// Gives a world created after startup its GPU state once a camera shows
    /// it.
    pub fn adopt_worlds(&mut self, gpu: &Gpu, staging: StagingMemory) -> Rebuild {
        let mut adopted = Rebuild::None;
        for id in (0..camera::camera_count())
            .filter_map(camera::camera)
            .flat_map(|c| c.worlds())
        {
            if self.worlds.iter().any(|wr| wr.world.id() == id) {
                continue;
            }
            let Some(world) = worlds::world(id) else {
                continue;
            };
            let wr = WorldRender::new(&world, gpu, self.transform_shared.clone(), staging);
            self.worlds.push(wr);
            adopted = Rebuild::Cameras;
        }
        adopted
    }

    pub fn acquire(
        &mut self,
        renderer: &mut SwapchainRenderer,
        gpu: &Gpu,
    ) -> Option<AcquiredFrame> {
        renderer.acquire(|images| self.on_swapchain_recreated(images, gpu))
    }

    fn on_swapchain_recreated(&mut self, images: &[Arc<ImageView>], gpu: &Gpu) {
        self.swapchain_image_views = images.to_vec();
        let extent = self.extent();
        let res = gpu.scene_resources(&self.transform_shared, &self.stores);
        for cam in &mut self.cameras {
            cam.on_swapchain_resize(extent, &res);
        }
        self.ui_gpu
            .on_resize(extent, &self.stores.texture, &camera_targets(&self.cameras));
        self.rebuild_frame_slots(gpu);
    }

    pub fn timestamp_pool(
        &self,
        image_index: usize,
        slot: usize,
    ) -> &Arc<vulkano::query::QueryPool> {
        &self.frame_slots[frame_slot_index(image_index, slot)].timestamp_pool
    }

    pub fn frame_cb(&self, image_index: usize, slot: usize) -> Arc<PrimaryAutoCommandBuffer> {
        self.frame_slots[frame_slot_index(image_index, slot)]
            .command_buffer
            .clone()
    }

    /// Lays out the UI. Free when nothing changed.
    pub fn layout_ui(&self) {
        let [w, h] = self.extent();
        let mut ui = ui::ui();
        ui.run_deferred();
        ui.run_layout([w as f32, h as f32]);
    }

    /// Runs before the host wait.
    pub fn prepare(&mut self, gpu: &Gpu, restage: Option<StagingMemory>) -> Rebuild {
        let mut rebuild = self
            .worlds
            .iter_mut()
            .map(|wr| wr.prepare(gpu))
            .max()
            .unwrap_or_default();
        // Switch all worlds or none, because one primary runs all their scatters.
        if let Some(mode) = restage {
            let switched = self.worlds.iter_mut().fold(false, |any, wr| {
                wr.transforms.set_staging_memory(mode) || any
            });
            if switched {
                println!("[staging] switched to {}", mode.label());
                rebuild = rebuild.max(Rebuild::FrameSlots);
            }
        }
        let mesh = self.stores.mesh.sync();
        let texture = self.stores.texture.sync();
        let material = self.stores.material.sync();
        if mesh || texture || material {
            rebuild = Rebuild::Cameras;
        }
        // The UI binds its own copy of the texture array.
        if texture {
            self.ui_gpu.refresh_textures(&self.stores.texture);
        }
        if self.ui_gpu.ensure_capacity(&mut ui::ui()) {
            rebuild = rebuild.max(Rebuild::FrameSlots);
        }
        rebuild
    }

    /// Brings the cameras up to date with this frame's worlds and widget sizes.
    pub fn sync_cameras(&mut self, gpu: &Gpu, mut rebuild: Rebuild) -> Rebuild {
        let extent = self.extent();
        let res = gpu.scene_resources(&self.transform_shared, &self.stores);
        // A spawn that fits only moves the plan bases, which `stage` rewrites
        // in place. Anything else re-records the cameras.
        if rebuild >= Rebuild::Plans {
            let plans = world_plans(&self.stores.mesh, &self.worlds);
            let force = rebuild == Rebuild::Cameras;
            let structural = self.cameras.iter().any(|c| {
                c.needs_structural_rebuild(&world_sources(c.state(), &self.worlds, &plans), force)
            });
            if structural {
                for cam in &mut self.cameras {
                    cam.ensure_current(&res, &world_sources(cam.state(), &self.worlds, &plans));
                }
                rebuild = rebuild.max(Rebuild::FrameSlots);
            } else {
                self.cheap_plans = Some(plans);
            }
        }
        // One frame behind F9 by design — see `apply_pending_hiz_freeze`.
        if self
            .cameras
            .iter_mut()
            .fold(false, |any, c| c.apply_pending_hiz_freeze() || any)
        {
            rebuild = rebuild.max(Rebuild::FrameSlots);
        }

        // Build cameras created since the last frame. Slots get reused, so a
        // built slot may belong to a different camera now.
        let mut targets_moved = false;
        for i in 0..camera::camera_count() {
            let Some(state) = camera::camera(i) else {
                continue;
            };
            if self.cameras.get(i).is_some_and(|c| c.state().is(&state)) {
                continue;
            }
            let plans = world_plans(&self.stores.mesh, &self.worlds);
            let sources = world_sources(&state, &self.worlds, &plans);
            let cam = RenderCamera::new_match_swapchain(state, extent, &res, &sources);
            match self.cameras.get_mut(i) {
                Some(slot) => *slot = cam,
                None => self.cameras.push(cam),
            }
            targets_moved = true;
        }
        // Only the renderer can free a dropped camera's slot, because its
        // handle is what keeps the slot in use.
        for (i, c) in self.cameras.iter().enumerate() {
            if c.state().is_orphan() {
                camera::retire(i);
            }
        }
        let idle: Vec<bool> = self.cameras.iter().map(|c| c.state().is_dark()).collect();
        if idle != self.cameras_idle {
            rebuild = rebuild.max(Rebuild::FrameSlots);
        }
        self.cameras_idle = idle;

        // Size each camera to the widget showing it, one texel per pixel.
        for (cam, _) in self
            .cameras
            .iter_mut()
            .zip(&self.cameras_idle)
            .filter(|(_, idle)| !**idle)
        {
            let want = match cam.state().rect() {
                // No widget shows it, so it fills the window and is blitted.
                None => CameraResolution::MatchSwapchain,
                Some(r) if r[2] >= 1.0 && r[3] >= 1.0 => {
                    CameraResolution::Fixed([r[2] as u32, r[3] as u32])
                }
                // Its widget has no size this frame, so keep the current one.
                Some(_) => cam.resolution(),
            };
            if want != cam.resolution() && cam.set_resolution(want, extent, &res) {
                targets_moved = true;
            }
        }
        if targets_moved {
            self.ui_gpu
                .rebind_targets(&self.stores.texture, &camera_targets(&self.cameras));
            rebuild = rebuild.max(Rebuild::FrameSlots);
        }
        rebuild
    }

    /// F8 toggles occlusion culling for every camera.
    pub fn toggle_occlusion(&mut self, gpu: &Gpu) -> Rebuild {
        let Some(want) = self.cameras.first().map(|c| !c.occlusion_enabled()) else {
            return Rebuild::None;
        };
        let res = gpu.scene_resources(&self.transform_shared, &self.stores);
        let changed = self
            .cameras
            .iter_mut()
            .fold(false, |any, c| c.set_occlusion_enabled(want, &res) || any);
        match changed {
            true => Rebuild::FrameSlots,
            false => Rebuild::None,
        }
    }

    /// F9 (and `ENGINE_CULL_AWAY=1`) freezes the cull frustum while the view
    /// keeps moving. With `away` the frozen frustum is moved off the scene.
    pub fn toggle_cull_lock(&mut self, away: bool) {
        let Some(cam) = self.cameras.first_mut() else {
            return;
        };
        let view_proj = cam.state().view_proj().0;
        let frozen = match away {
            true => view_proj * glam::Mat4::from_translation(glam::Vec3::splat(1.0e7)),
            false => view_proj,
        };
        let lock = !cam.cull_lock();
        cam.set_cull_lock(lock, frozen.to_cols_array());
    }

    pub fn cull_locked(&self) -> bool {
        self.cameras.first().is_some_and(|c| c.cull_lock())
    }

    /// Runs after the resolution sync, so the aspect is final. A camera on an
    /// entity reads its pose here because mid-sweep it could race the parents.
    pub fn drive_cameras(&self) {
        for cam in &self.cameras {
            cam.state().set_aspect(cam.aspect());
        }
        camera::drive_bound_cameras();
    }

    pub fn rebuild_frame_slots(&mut self, gpu: &Gpu) {
        // Drop first, because an old primary may hold a `MultipleSubmit` secondary.
        self.frame_slots.clear();
        self.frame_slots = build_all_frame_slots(
            gpu,
            &self.transform_shared,
            &self.swapchain_image_views,
            &shown(&self.cameras, &self.cameras_idle),
            &self.worlds,
            &self.ui_gpu,
        );
    }

    /// Runs after the host wait and writes everything the primary for `slot`
    /// reads. An idle frame still writes zero counts so old data isn't replayed.
    pub fn stage(&mut self, slot: usize, pool: Option<&parallel::Pool>, fps: &mut FrameStats) {
        if let Some(plans) = self.cheap_plans.take() {
            for cam in &self.cameras {
                cam.write_template_bases(&world_sources(cam.state(), &self.worlds, &plans));
            }
        }
        for cam in &self.cameras {
            let (view_proj, eye) = cam.state().view_proj();
            cam.write_view_proj(slot, view_proj, eye);
            cam.write_cull_view_proj(slot, view_proj.to_cols_array());
            cam.write_overlay(slot, view_proj, eye);
        }
        for wr in &mut self.worlds {
            let (times, steps) = wr.stage(slot, pool);
            fps.record_stage(&times, steps);
        }
        self.ui_gpu.write_staging(slot, &mut ui::ui());
    }
}

fn extent_of(views: &[Arc<ImageView>]) -> [u32; 2] {
    let [w, h, _] = views[0].image().extent();
    [w, h]
}

/// One draw plan per world, in `worlds` order. Every plan has the same mesh
/// slots but counts only its own world's instances (ADR-0011 §4).
fn world_plans(store: &GpuMeshStore, worlds: &[WorldRender]) -> Vec<DrawPlan> {
    worlds
        .iter()
        .map(|wr| build_draw_plan(store, &store.slot_totals(wr.renderers.mesh_instances())))
        .collect()
}

/// The worlds a camera draws, in composite order. A camera that lists none
/// of this window's worlds draws the first one.
fn world_sources<'a>(
    state: &CameraHandle,
    worlds: &'a [WorldRender],
    plans: &[DrawPlan],
) -> Vec<WorldSource<'a>> {
    let source = |i: usize| WorldSource {
        id: worlds[i].world.id(),
        transforms: &worlds[i].transforms,
        renderers: &worlds[i].renderers,
        plan: plans[i].clone(),
    };
    let named: Vec<WorldSource<'a>> = state
        .worlds()
        .iter()
        .filter_map(|id| worlds.iter().position(|wr| wr.world.id() == *id))
        .map(source)
        .collect();
    match named.is_empty() && !worlds.is_empty() {
        true => vec![source(0)],
        false => named,
    }
}

/// One indirect command per mesh slot. Each slot's instances start where
/// the previous slot's end.
fn build_draw_plan(mesh_store: &GpuMeshStore, slot_totals: &[u32]) -> DrawPlan {
    let mut base = 0u32;
    let commands = slot_totals
        .iter()
        .enumerate()
        .map(|(slot, &total)| {
            let geom = mesh_store.slot_geometry(slot as u32);
            let first_instance = base;
            base += total;
            vulkano::command_buffer::DrawIndexedIndirectCommand {
                index_count: geom.map(|g| g.index_count).unwrap_or(0),
                instance_count: 0,
                first_index: geom.map(|g| g.first_index).unwrap_or(0),
                vertex_offset: geom.map(|g| g.vertex_offset as u32).unwrap_or(0),
                first_instance,
            }
        })
        .collect();
    DrawPlan {
        commands,
        total_renderers: base,
    }
}

/// Includes idle cameras, because the UI indexes this by camera slot.
fn camera_targets(cameras: &[RenderCamera]) -> Vec<Arc<ImageView>> {
    cameras.iter().map(|c| c.color_view().clone()).collect()
}

fn shown<'a>(cameras: &'a [RenderCamera], idle: &[bool]) -> Vec<&'a RenderCamera> {
    cameras
        .iter()
        .zip(idle)
        .filter(|(_, idle)| !**idle)
        .map(|(c, _)| c)
        .collect()
}

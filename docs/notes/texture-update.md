❯ looks good. lets go on a tangent. no code. it is useful for the camera to have accessible textures for the sake of screens, mirrors, water
  reflection, etc. we also need to support a dynamic, arbitrary number of textures. also I'd like to implement mega texures at some point.
  obviously we cant render to a mega texture and maybe the solution is truly a seperate array/reservered slots. with those points in mind
  what is the way forward with that

● Good tangent — and the four asks pull in the same direction once you sort images by lifetime and mutability rather than by "texture-ness".
  Three classes fall out, and they want three different bindings.

  Why render targets are genuinely their own array

  Your instinct is right, and there are four independent reasons, any one of which is sufficient:

  You can't atlas one. A colour attachment has to be its own image (or a whole array layer with separate-layout care). Sizes vary per camera,
  they change on resize, and they're written every frame. Nothing about that fits a packed page.

  The self-read hazard. One shared descriptor array means every draw has every target bound, including the one the current pass is writing.
  That's what RESERVED_SLOTS and the UI's private copy of the array are working around today — a stopgap for having exactly one target and
  nowhere to put it.

  Address mode. The store binds one Repeat/Linear sampler. A mirror sampling a hair past its edge with Repeat wraps to the opposite side — a
  bright, obvious artifact. Targets want ClampToEdge, no mips, no aniso. Separate binding, separate sampler, no per-texture sampler table.

  Colour space. Targets are R16G16B16A16_SFLOAT holding linear radiance; material textures are sRGB-encoded albedo. If the shader knows which
  array it sampled, the decode rule is implicit. If they share an array, every material has to carry a "don't decode this one" bit.

  So: binding 1 stays material textures, binding 2 becomes render targets — small, fixed cap (64 is not a lie the way 1024 is), its own
  sampler. ui::CAMERA_TARGET and the substitution both retire, and the UI binds the same array everyone else does.

  The read-write problem, and what solves it cheaply

  Double-buffer every target and let readers sample the retired one. It's the hiz_current / hiz_prev pattern already in the codebase, and it
  buys more than hazard-freedom: it makes camera ordering irrelevant. A screen showing camera B doesn't need B to run before A — which
  matters enormously here, because the frame is pre-recorded command buffers. With ordering required, N cameras mean a dependency graph baked
  into the recording; without it, they're N independent recordings you can submit in any order. One frame of latency on a mirror is the
  standard trade and nobody has ever noticed it.

  The part I'd emphasise: express "which image is readable right now" in the redirect buffer, not in the descriptor set. You already built
  that indirection. Both physical images live in the array permanently; the redirect entry for a target flips between two slot indices each
  frame. That's one u32 write per target per frame, versus a descriptor-set rebuild plus a secondary re-record plus every frame slot. The
  redirect is the thing that makes this whole design cheap, and it's already there.

  Arbitrary count

  Two levers, and they overlap:

  descriptorBindingVariableDescriptorCount + descriptorBindingPartiallyBound makes the array runtime-sized — declare u_textures[] unbounded
  in GLSL, allocate the set with the current capacity, and MAX_TEXTURES becomes a growth policy instead of a wall. Cheap, and worth doing
  regardless.

  descriptorBindingSampledImageUpdateAfterBind is the bigger prize: it lets a newly decoded texture be written into a bound set, which
  retires the arrival → force_full → rebuild-everything path. That's the actual cost of streaming today.

  But note the interaction with your third ask: if mega textures land, the material array collapses to a handful of descriptors and neither
  lever matters much for it — a new texture becomes a layer index, and no descriptor changes at all. So if you're confident mega textures are
  coming, take the variable-count change (small) and skip the update-after-bind machinery (not small), because the atlas subsumes it.

  Mega textures: array layers, not a packed page

  The repeat-addressing problem decides this. Material textures tile; a packed atlas can't express Repeat without frac() in the shader and
  manual mip selection, and its mip chain bleeds across neighbours unless you pad every entry. A sampler2DArray of fixed-size layers keeps
  hardware wrapping and per-layer mips intact and still collapses to one binding.

  The cost is that layers are one size and one format, so it's really several arrays keyed by (format, size class) — BC7 1K, BC7 2K, BC5
  normals, R8, and so on — with a residency policy that rescales or promotes between classes. Packed atlases stay the right answer for things
  that never tile and are small: glyphs, icons, UI sprites. Your glyph atlas already is one, and its comment already articulates exactly
  this reasoning.

  What changes downstream is only the redirect's value: instead of a slot index, it becomes a packed (array, layer) — a few bits and a shift.
  The shader's index math changes; materials, the store's API and every call site do not. That's the payoff for having put the indirection
  in a buffer.

  What actually blocks mirrors

  Not the texture plumbing — plural cameras. main_camera is singular and the whole frame is built around it: one cull set, one Hi-Z pair, one
  set of secondaries, one view_proj in the SoT. Mirrors and screens need a camera list with per-camera resolution (which Fixed now gives
  you), and planar reflection specifically needs an oblique near-plane projection — the one place a hand-modified matrix genuinely belongs,
  unlike the viewport skew we just deleted.

  Two globals die that same day: scene::set_viewport's static and ui::CAMERA_TARGET. Viewport::new would take a camera handle, and the
  widget's published box would resize that camera. Worth knowing now so neither calcifies.

  Order I'd take it in

  1. Split targets into their own binding + sampler, double-buffered, addressed through the redirect. Smallest change, retires the two hacks,
  and it's the prerequisite for everything else. A material can then name a target, which is mirrors-and-screens shaped even before there
  are two cameras.
  2. Plural cameras. This is the real feature. Do it once (1) means a second camera has somewhere to put its output.
  3. Variable descriptor count on the material array — a few lines, removes the wall.
  4. Mega textures as format-classed layer arrays, and decide about update-after-bind only if you conclude atlasing isn't coming.

  The thing to hold onto through all of it: the redirect buffer is the seam. Every one of these changes is either "a different value in the
  redirect" or "a different array behind it", and none of them has to touch a material, a call site, or the shape of GpuTextureStore's API.

# Scene loading should occur after engine / thread-pool initialization

Asset decoding runs as `spawn_background` tasks on the engine's global
thread pool (`parallel::global`), not on a dedicated loader thread. A
`MeshRenderer` constructed before the pool exists cannot spawn its load —
the request parks in `asset::PENDING_LOADS` until engine init calls
`asset::flush_pending_loads()` right after building the pool.

Today the games do the opposite order (`test-game` builds its whole scene
in `main` before `Window::run()` initializes the engine), which works only
because of that pending-queue escape hatch: every early load sits deferred
through scene construction and starts in a burst at init.

Preferred order for new code:

1. Initialize the engine / thread pool.
2. Build or load the scene (component constructors may then hand their
   asset loads straight to the pool, no deferral).

This lets loads start immediately and overlap with the rest of scene
construction, instead of queueing behind it. The pending queue stays as a
correctness net for the legacy order, not something to design around.

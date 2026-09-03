# engine-core::util

`engine-core::util` is ~2467 tokens of signatures — too many to load whole.
Each file below is one part of it; open only the ones you need.

| Module | Digest | ~tok | Public symbols |
|---|---|---|---|
| `engine-core::util::avail` | [avail.md](avail.md) | 113 | `Avail`, `commit()`, `len()`, `pop()`, `push()` |
| `engine-core::util::container` | [container.md](container.md) | 173 | `Container`, `commit()`, `for_each()`, `insert()`, `len()`, `remove()` |
| `engine-core::util::mod` | [mod.md](mod.md) | 83 | `get_chunk_size()` |
| `engine-core::util::my_thread_pool` | [my_thread_pool.md](my_thread_pool.md) | 339 | `BitmapTaskLayout`, `ThreadPool`, `background_cap()`, `bitmap_task_layout()`, `idle_workers()`, `init()`, `init_with_options()`, `is_initialized()`, `num_threads()`, `parallel_for()`, `pool()`, `spawn_background()`, `with_options()`, `work_stealing()` |
| `engine-core::util::numa` | [numa.md](numa.md) | 269 | `NumaNode`, `NumaTopology`, `cpus_of_node()`, `current_affinity()`, `detect()`, `gpu_numa_node()`, `nodes()`, `restrict_affinity_to()`, `single_node()` |
| `engine-core::util::numa_mem` | [numa_mem.md](numa_mem.md) | 266 | `MempolicyGuard`, `bind_to_node()`, `mbind_policy_to_node()`, `mbind_to_node()`, `page_residency()`, `page_size()`, `verify_residency_single_node()` |
| `engine-core::util::numa_pool` | [numa_pool.md](numa_pool.md) | 568 | `Config`, `Scope`, `ThreadPool`, `init()`, `init_threads()`, `is_initialized()`, `join()`, `num_threads()`, `parallel_for()`, `parallel_for_grain()`, `parallel_for_static()`, `pool()`, `run()`, `scope()`, `spawn()`, `spawn_background()`, `with_config()`, `with_threads()` |
| `engine-core::util::parallel` | [parallel.md](parallel.md) | 378 | `BackendKind`, `Pool`, `backend()`, `bitmap_task_layout()`, `from_env()`, `init()`, `init_with_options()`, `is_initialized()`, `num_threads()`, `parallel_for()`, `pool()`, `spawn_background()`, `with_options()` |
| `engine-core::util::seg_storage` | [seg_storage.md](seg_storage.md) | 218 | `SegStorage`, `drop()`, `get_from_slice()`, `get_from_slice_unchecked()`, `get_segment_chunk()`, `get_segment_chunk_unchecked()`, `get_unchecked()`, `get_unchecked_mut()`, `len()`, `set()` |
| `engine-core::util::storage` | [storage.md](storage.md) | 141 | `Storage`, `get()`, `get_mut()`, `insert()`, `len()`, `remove()` |
| `engine-core::util::thread_pool` | [thread_pool.md](thread_pool.md) | 388 | `BitmapTaskLayout`, `DispatchTiming`, `PoolConfig`, `ThreadPool`, `bitmap_task_layout()`, `for_chunks()`, `global()`, `init_global()`, `is_initialised()`, `num_threads()`, `num_workers()`, `parallel_for()`, `set_current_thread_affinity_mask()`, `verify_enabled()` |

[engine index](../../index.md)

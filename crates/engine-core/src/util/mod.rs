use std::ops::Div;

mod avail;
pub use avail::Avail;
mod storage;
pub use storage::Storage;
pub mod container;
// a segvec like structure that holds items with min chunk size or 32 corresponding to the active bits of the ComponentStorafe container
pub mod seg_storage;
pub mod thread_pool;

pub fn get_chunk_size(num_items: usize) -> usize {
    let chunk_size = ((num_items as f32).sqrt().ceil() as usize).max(1);
    let num_chunks = num_items.div_ceil(chunk_size);
    let n_threads = thread_pool::global().num_threads();
    if chunk_size != 1 && num_chunks < n_threads {
        return (num_items as f32).div(n_threads as f32).ceil() as usize;
    }
    chunk_size
}

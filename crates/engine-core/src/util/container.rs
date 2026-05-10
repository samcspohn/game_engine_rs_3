// container for parallel computing
// optimized for parallel allocation/deallocation and iteration
// uses array of arrays (1 per thread)
// each thread iterates over its own array
// each sub-array keeps track of its global offsets
// re-balance periodically to keep active elements evenly distributed
// insert returns global index that is stable across re-balancing

use std::collections::VecDeque;
use rayon::prelude::*;
use crate::util::Avail;

struct SubContainer<T> {
    data: VecDeque<Option<T>>,
    avail: Avail,
    offset: usize,
    active: u32,
}
impl<T> SubContainer<T> {
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
            avail: Avail::new(),
            offset: 0,
            active: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn insert(&mut self, item: T) -> (u32, bool) {
        if let Some(i) = self.avail.pop() {
            self.data[i as usize] = Some(item);
            (i + self.offset as u32, false)
        } else {
            let i = self.data.len() as u32;
            self.data.push_back(Some(item));
            (i + self.offset as u32, true)
        }
    }
    pub fn remove(&mut self, i: u32) -> Option<T> {
        let local_idx = i as usize - self.offset;
        if local_idx < self.data.len() {
            let v = self.data[local_idx].take();
            self.avail.push(local_idx as u32);
            v
        } else {
            None
        }
    }
}

pub struct Container<T> {
    data: Vec<SubContainer<T>>,
    ordered_by_active: Vec<usize>,
    num_items: usize,
}

impl<T> Container<T>
where
    T: Send + Sync,
{
    pub fn new(num_threads: usize) -> Self {
        Self {
            data: (0..num_threads).map(|_| SubContainer::new()).collect(),
            ordered_by_active: (0..num_threads).collect(),
            num_items: 0,
        }
    }
    pub fn commit(&mut self) {
        for sub in &mut self.data {
            sub.avail.commit();
        }
    }
    pub fn insert(&mut self, item: T) -> u32 {
        let thread_id = self.ordered_by_active.first().unwrap();
        let (idx, resized) = self.data[*thread_id].insert(item);
        if resized {
            // update offsets
            for i in (thread_id + 1)..self.data.len() {
                self.data[i].offset += 1;
            }
        }
        self.num_items += 1;
        // bubble sort first element
        let mut i = 0;
        let o = &mut self.ordered_by_active;
        let data = &mut self.data;
        while i + 1 < o.len() {
            if data[o[i]].active > data[o[i + 1]].active {
                o.swap(i, i + 1);
                i += 1;
            } else {
                break;
            }
        }
        idx
    }
    pub fn remove(&mut self, i: u32) -> Option<T> {
        // find which sub-container with binary search
        let sub_idx = self.data.binary_search_by(|probe| {
            if probe.offset + probe.data.len() <= i as usize {
                std::cmp::Ordering::Less
            } else if probe.offset > i as usize {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        }).ok()?;
        let v = self.data[sub_idx].remove(i);
        if v.is_some() {
            self.num_items -= 1;
        }
        v
    }
    pub fn for_each<F>(&mut self, f: F)
    where
        F: Fn(&mut T) + Send + Sync,
    {
        self.data.par_iter_mut().for_each(|v| {
            for item_opt in v.data.iter_mut() {
                if let Some(item) = item_opt {
                    f(item);
                }
            }
        });
    }
    
}
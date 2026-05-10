use segvec::SegVec;

use crate::util::Avail;


pub struct Storage<T> {
    pub data: SegVec<Option<T>>,
    pub avail: Avail,
}

impl<T> Storage<T> {
    pub fn new() -> Self {
        Self {
            data: SegVec::new(),
            avail: Avail::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn insert(&mut self, v: T) -> u32 {
        if let Some(i) = self.avail.pop() {
            self.data[i as usize] = Some(v);
            i
        } else {
            let i = self.data.len() as u32;
            self.data.push(Some(v));
            i
        }
    }
    pub fn remove(&mut self, i: u32) -> Option<T> {
        if (i as usize) < self.data.len() {
            let v = self.data[i as usize].take();
            if v.is_some() {
                self.avail.push(i);
            }
            v
        } else {
            None
        }
    }
    pub fn get(&self, i: u32) -> Option<&T> {
        if (i as usize) < self.data.len() {
            self.data[i as usize].as_ref()
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, i: u32) -> Option<&mut T> {
        if (i as usize) < self.data.len() {
            self.data[i as usize].as_mut()
        } else {
            None
        }
    }
}

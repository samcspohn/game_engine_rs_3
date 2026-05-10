use std::cmp::Reverse;

use dary_heap::DaryHeap;
use parking_lot::Mutex;


pub struct Avail {
    pub data: DaryHeap<Reverse<u32>, 4>,
    new_ids: Mutex<Vec<Reverse<u32>>>,
}
impl Avail {
    pub fn new() -> Self {
        Self {
            data: DaryHeap::new(),
            new_ids: Mutex::new(Vec::new()),
        }
    }
    pub fn commit(&mut self) {
        let mut a = self.new_ids.lock();
        for i in a.drain(..) {
            self.data.push(i);
        }
    }
    pub fn push(&self, i: u32) {
        self.new_ids.lock().push(Reverse(i));
        // self.data.push(Reverse(i));
    }
    pub fn pop(&mut self) -> Option<u32> {
        match self.data.pop() {
            Some(Reverse(a)) => Some(a),
            None => None,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

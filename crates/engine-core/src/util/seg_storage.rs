use std::mem::MaybeUninit;

// uses vec like growth patterns with each consecutive segment being double the size of the previous one with a base size of 32
pub struct SegStorage<T> {
    segments: Vec<Vec<MaybeUninit<T>>>,
    len: usize,
}

#[inline(always)]
fn get_seg_index(index: usize) -> (usize, usize) {
    // Optimal index calculation:
    // - Use index+32 to shift ranges to powers of 2
    // - Find highest bit position with leading_zeros
    // - Compute segment index and local index with minimal ops
    let temp = index + 32;
    let h = 63 - temp.leading_zeros();
    let power = 1usize << h;
    let seg_index = h as usize - 5;
    let local_index = index - (power - 32);
    (seg_index, local_index)
}

#[inline(always)]
fn get_seg_index_(idx: usize) -> (usize, usize) {
 // Adjust index to make math cleaner: this maps segment boundaries to powers of 2
    let adjusted = (idx + 32) >> 5;

    // Find segment: position of most significant bit
    // This is equivalent to floor(log2(adjusted))
    let segment = (usize::BITS - 1 - adjusted.leading_zeros()) as usize;

    // Calculate local index within segment
    let local = idx + 32 - (32 << segment);

    (segment, local)
}

impl<T> SegStorage<T> {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, value: T) {
        let (seg_index, local_index) = get_seg_index(index);
        self.len = self.len.max(index + 1);

        if self.segments.len() <= seg_index {
            let mut seg_len = self.segments.len();
            self.segments.resize_with(seg_index + 1, || {
                let size = 32 << seg_len;
                let mut vec = Vec::with_capacity(size);
                for _ in 0..size {
                    vec.push(MaybeUninit::uninit());
                }
                seg_len += 1;
                vec
            });
        }
        self.segments[seg_index][local_index] = MaybeUninit::new(value);
    }
    #[inline(always)]
    pub fn drop(&mut self, index: usize) {
        let (seg_index, local_index) = get_seg_index(index);
        if let Some(segment) = self.segments.get_mut(seg_index) {
            unsafe {
                std::ptr::drop_in_place(segment[local_index].as_mut_ptr());
            }
        }
    }

    #[inline(always)]
    pub fn get_unchecked(&self, index: usize) -> &T {
        let (seg_index, local_index) = get_seg_index(index);
        unsafe { &*self.segments.get_unchecked(seg_index).get_unchecked(local_index).as_ptr() }
    }

    #[inline(always)]
    pub fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        let (seg_index, local_index) = get_seg_index(index);
        unsafe { &mut *self.segments[seg_index][local_index].as_mut_ptr() }
    }

    // gets a chunk of size 32 starting from start_index
    #[inline(always)]
    pub fn get_segment_chunk(&self, start_index: usize) -> Option<&[MaybeUninit<T>]> {
        let (seg_index, local_index) = get_seg_index(start_index);
        self.segments.get(seg_index).map(|segment| {
			&segment[local_index..local_index + 32.min(segment.len() - local_index)]
		})
    }
    #[inline(always)]
    pub fn get_segment_chunk_unchecked(&self, start_index: usize) -> &[MaybeUninit<T>] {
		let (seg_index, local_index) = get_seg_index(start_index);
		// &self.segments.seg_index][local_index..local_index + 32]
		unsafe { self.segments.get_unchecked(seg_index).get_unchecked(local_index..local_index + 32) }
	}

    pub fn len(&self) -> usize {
        self.len
    }
}

#[inline(always)]
pub fn get_from_slice<T>(s: &[MaybeUninit<T>], local_index: usize) -> &T {
    unsafe { &*s[local_index].as_ptr() }
}


#[inline(always)]
pub fn get_from_slice_unchecked<T>(s: &[MaybeUninit<T>], local_index: usize) -> &T {
    unsafe { &*s.get_unchecked(local_index).as_ptr() }
}

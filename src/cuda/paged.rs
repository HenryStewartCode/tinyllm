//! What cute-flashinfer's paged attention kernels read from the host: FlashInfer's page table and
//! its decode plan, built from tinyllm's rows.
//!
//! The kernels (cute-rs's `flashinfer` crate, after FlashInfer's `fa2` decode) see the KV cache as
//! `[pages, page_size, kv_heads, head_dim]` and find a request's tokens through three arrays:
//! request `b`'s pages are `kv_indices[kv_indptr[b]..kv_indptr[b + 1]]`, and its last page holds
//! `kv_last_page_len[b]` tokens. tinyllm's rows carry the same thing as a list of slots, where
//! slot `s` is entry `s % page_size` of page `s / page_size`.
//!
//! A long request is split into chunks that separate blocks read at once, then merged. FlashInfer
//! picks the chunk size from how full the GPU is, so a sequence's rounding depends on what else is
//! in the batch. tinyllm doesn't allow that (results are bit-identical alone or batched), so
//! [`DecodePlans`] splits each request by its own length only: a fixed chunk size, and requests
//! that fit in one chunk go to the kernel that writes results directly while the others go to the
//! one that writes partial states to merge. Which kernel a request gets, and where its chunks
//! fall, never depends on its neighbours.

/// FlashInfer's page table for a batch of requests.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PageTable {
    pub kv_indptr: Vec<i32>,
    pub kv_indices: Vec<i32>,
    pub kv_last_page_len: Vec<i32>,
}

impl PageTable {
    /// The table for requests given as their slots in order (a [`crate::model::Row`]'s `slots`).
    /// Every request needs at least one slot, and each page's slots must be consecutive from the
    /// page's first, as the pool hands them out.
    pub fn from_slots<'a>(requests: impl IntoIterator<Item = &'a [u32]>, page_size: usize) -> Self {
        let mut table = PageTable { kv_indptr: vec![0], ..Default::default() };
        for slots in requests {
            assert!(!slots.is_empty(), "a request needs a KV slot");
            for (i, page) in slots.chunks(page_size).enumerate() {
                assert!(
                    page[0] as usize % page_size == 0
                        && page.iter().enumerate().all(|(j, &s)| s == page[0] + j as u32),
                    "page {i} of a request isn't one whole page of the pool"
                );
                table.kv_indices.push((page[0] as usize / page_size) as i32);
            }
            table.kv_indptr.push(table.kv_indices.len() as i32);
            table.kv_last_page_len.push(((slots.len() - 1) % page_size + 1) as i32);
        }
        table
    }

    pub fn batch(&self) -> usize {
        self.kv_last_page_len.len()
    }

    /// How many pages request `b` has.
    pub fn pages(&self, b: usize) -> usize {
        (self.kv_indptr[b + 1] - self.kv_indptr[b]) as usize
    }
}

/// One launch's plan, with the fields of cute-flashinfer's `decode::Plan`: block `i` takes chunk
/// `kv_tile_indices[i]` of request `request_indices[i]`, and request `b`'s partial states are
/// rows `o_indptr[b]..o_indptr[b + 1]`. Requests are numbered within `requests`, which gives each
/// one's place in the batch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodePlan {
    pub requests: Vec<u32>,
    pub chunk_pages: usize,
    pub request_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub o_indptr: Vec<i32>,
}

/// A batch's decode plans: the requests read whole, and the ones read in chunks and merged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecodePlans {
    pub whole: DecodePlan,
    pub split: DecodePlan,
}

impl DecodePlans {
    /// Plans for requests of `pages[b]` pages each, in chunks of `chunk_pages` pages.
    pub fn new(pages: &[usize], chunk_pages: usize) -> Self {
        assert!(chunk_pages > 0);
        let mut plans = DecodePlans::default();
        for plan in [&mut plans.whole, &mut plans.split] {
            plan.chunk_pages = chunk_pages;
            plan.o_indptr.push(0);
        }
        for (b, &n) in pages.iter().enumerate() {
            let chunks = n.max(1).div_ceil(chunk_pages);
            let plan = if chunks == 1 { &mut plans.whole } else { &mut plans.split };
            let request = plan.requests.len() as i32;
            plan.requests.push(b as u32);
            for chunk in 0..chunks {
                plan.request_indices.push(request);
                plan.kv_tile_indices.push(chunk as i32);
            }
            plan.o_indptr.push(plan.request_indices.len() as i32);
        }
        plans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_table_follows_the_slots() {
        // Pages 3 and 1 with five tokens, then page 2 exactly full.
        let a: Vec<u32> = (12..16).chain(4..5).collect();
        let b: Vec<u32> = (8..12).collect();
        let table = PageTable::from_slots([&a[..], &b[..]], 4);
        assert_eq!(table.kv_indptr, [0, 2, 3]);
        assert_eq!(table.kv_indices, [3, 1, 2]);
        assert_eq!(table.kv_last_page_len, [1, 4]);
        assert_eq!((table.batch(), table.pages(0), table.pages(1)), (2, 2, 1));
    }

    #[test]
    #[should_panic(expected = "isn't one whole page")]
    fn page_table_refuses_a_broken_page() {
        PageTable::from_slots([&[4u32, 6][..]], 4);
    }

    #[test]
    fn plans_split_by_a_request_s_own_length() {
        let plans = DecodePlans::new(&[1, 9, 4, 5], 4);
        assert_eq!(plans.whole.requests, [0, 2]);
        assert_eq!(plans.whole.request_indices, [0, 1]);
        assert_eq!(plans.whole.o_indptr, [0, 1, 2]);
        assert_eq!(plans.split.requests, [1, 3]);
        assert_eq!(plans.split.request_indices, [0, 0, 0, 1, 1]);
        assert_eq!(plans.split.kv_tile_indices, [0, 1, 2, 0, 1]);
        assert_eq!(plans.split.o_indptr, [0, 3, 5]);
    }

    /// A request's chunks are the same whatever it's batched with.
    #[test]
    fn plans_do_not_depend_on_the_batch() {
        let chunks_of = |pages: &[usize], b: u32| {
            let plans = DecodePlans::new(pages, 8);
            [(&plans.whole, false), (&plans.split, true)].into_iter().find_map(|(plan, split)| {
                let r = plan.requests.iter().position(|&x| x == b)?;
                Some((split, plan.o_indptr[r + 1] - plan.o_indptr[r]))
            })
        };
        for n in [1, 8, 9, 100] {
            let alone = chunks_of(&[n], 0);
            assert_eq!(alone, chunks_of(&[3, n, 500, 7], 1));
            assert_eq!(alone, Some((n > 8, n.div_ceil(8) as i32)));
        }
    }
}

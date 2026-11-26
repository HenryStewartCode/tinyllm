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
        let mut table = PageTable {
            kv_indptr: vec![0],
            ..Default::default()
        };
        for slots in requests {
            assert!(!slots.is_empty(), "a request needs a KV slot");
            for (i, page) in slots.chunks(page_size).enumerate() {
                assert!(
                    page[0] as usize % page_size == 0
                        && page
                            .iter()
                            .enumerate()
                            .all(|(j, &s)| s == page[0] + j as u32),
                    "page {i} of a request isn't one whole page of the pool"
                );
                table.kv_indices.push((page[0] as usize / page_size) as i32);
            }
            table.kv_indptr.push(table.kv_indices.len() as i32);
            table
                .kv_last_page_len
                .push(((slots.len() - 1) % page_size + 1) as i32);
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

/// A step's rows grouped into FlashInfer requests. tinyllm gives every token its own row, and a
/// prompt's rows each see one more slot than the row before; FlashInfer takes those as one
/// request of several queries under a causal mask. Request `b` is rows
/// `qo_indptr[b]..qo_indptr[b + 1]`, and its keys and values are its last row's slots. A row
/// generating a token is a request of one query.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Requests {
    pub qo_indptr: Vec<i32>,
}

impl Requests {
    /// Groups rows, given as their slots, into runs that each extend the row before by one slot.
    pub fn from_slots(rows: &[&[u32]]) -> Self {
        let mut qo_indptr = vec![0];
        for (i, pair) in rows.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            // Rows of one prompt are slices of the same list, which saves comparing them.
            let extends =
                b.len() == a.len() + 1 && (a.as_ptr() == b.as_ptr() || b[..a.len()] == *a);
            if !extends {
                qo_indptr.push(i as i32 + 1);
            }
        }
        if !rows.is_empty() {
            qo_indptr.push(rows.len() as i32);
        }
        Requests { qo_indptr }
    }

    pub fn batch(&self) -> usize {
        self.qo_indptr.len() - 1
    }

    /// The row whose slots are request `b`'s whole context.
    pub fn last_row(&self, b: usize) -> usize {
        self.qo_indptr[b + 1] as usize - 1
    }

    /// The requests' page table.
    pub fn page_table(&self, rows: &[&[u32]], page_size: usize) -> PageTable {
        PageTable::from_slots((0..self.batch()).map(|b| rows[self.last_row(b)]), page_size)
    }

    /// Where each row's keys and values go, as FlashInfer's `append_paged_kv_cache` takes them:
    /// the row's request and its position in that request's sequence.
    pub fn append_indices(&self, rows: &[&[u32]]) -> (Vec<i32>, Vec<i32>) {
        let mut batch_indices = Vec::with_capacity(rows.len());
        for b in 0..self.batch() {
            batch_indices.extend((self.qo_indptr[b]..self.qo_indptr[b + 1]).map(|_| b as i32));
        }
        (
            batch_indices,
            rows.iter().map(|slots| slots.len() as i32 - 1).collect(),
        )
    }
}

/// One launch of cute-flashinfer's `tinyllm` attention kernel: rows `first_row..first_row +
/// rows` of a step, as FlashInfer requests, every request split into chunks of a fixed number of
/// tokens counted from position 0, one block (CTA) per tile of packed queries per chunk. Each
/// chunk's partial state is a row of scratch; row `r` of the launch merges scratch rows
/// `merge_start[r]..merge_start[r] + merge_count[r]` in order.
///
/// A row's chunks, and the order they are merged in, depend only on its own position, so it
/// gets the same bits in any step. (A request's rows can be cut into several requests, the
/// earlier ones seeing only their own last row's keys: that changes how queries are packed into
/// tiles, which rows don't see.)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttentionLaunch {
    pub first_row: usize,
    pub rows: usize,
    pub qo_indptr: Vec<i32>,
    pub table: PageTable,
    pub request_indices: Vec<i32>,
    pub qo_tile_indices: Vec<i32>,
    pub kv_tile_indices: Vec<i32>,
    pub o_indptr: Vec<i32>,
    pub merge_start: Vec<i32>,
    pub merge_count: Vec<i32>,
}

impl AttentionLaunch {
    /// Scratch rows the launch's partial states take.
    pub fn scratch_rows(&self) -> usize {
        *self.o_indptr.last().unwrap_or(&0) as usize
    }
}

/// The launches for a step's rows, given as their slots (a [`crate::model::Row`]'s), with
/// `group` query heads per KV head, `tile_q` packed query rows per block, `chunk` tokens per
/// chunk and pages of `page_size` slots: as few launches as keep each one's partial states
/// within `scratch` rows.
pub fn attention_launches(
    rows: &[&[u32]],
    group: usize,
    tile_q: usize,
    chunk: usize,
    page_size: usize,
    scratch: usize,
) -> Vec<AttentionLaunch> {
    let chunks_of = |slots: &[u32]| slots.len().div_ceil(chunk);
    assert!(
        rows.iter().all(|slots| chunks_of(slots) <= scratch),
        "a row's chunks don't fit the scratch"
    );
    let requests = Requests::from_slots(rows);
    let mut launches: Vec<AttentionLaunch> = Vec::new();
    let mut current = AttentionLaunch {
        qo_indptr: vec![0],
        o_indptr: vec![0],
        ..Default::default()
    };
    current.table.kv_indptr.push(0);
    let finish =
        |launch: &mut AttentionLaunch, launches: &mut Vec<AttentionLaunch>, next: usize| {
            if launch.rows > 0 {
                let done = std::mem::replace(
                    launch,
                    AttentionLaunch {
                        first_row: next,
                        qo_indptr: vec![next as i32],
                        o_indptr: vec![0],
                        ..Default::default()
                    },
                );
                launch.table.kv_indptr.push(0);
                launches.push(done);
            }
        };
    for b in 0..requests.batch() {
        let (start, end) = (
            requests.qo_indptr[b] as usize,
            requests.qo_indptr[b + 1] as usize,
        );
        let mut first = start;
        while first < end {
            // As many of the request's rows as fit what's left of the scratch: each takes the
            // chunks of the last one's keys.
            let room = scratch - current.scratch_rows();
            let mut last = first;
            while last + 1 < end && (last + 2 - first) * chunks_of(rows[last + 1]) <= room {
                last += 1;
            }
            if chunks_of(rows[last]) * (last + 1 - first) > room {
                finish(&mut current, &mut launches, first);
                continue;
            }
            let (qo_len, slots) = (last + 1 - first, rows[last]);
            let chunks = chunks_of(slots);
            let index = current.qo_indptr.len() as i32 - 1;
            current.qo_indptr.push((last + 1) as i32);
            for page in slots.chunks(page_size) {
                current
                    .table
                    .kv_indices
                    .push((page[0] as usize / page_size) as i32);
            }
            current
                .table
                .kv_indptr
                .push(current.table.kv_indices.len() as i32);
            current
                .table
                .kv_last_page_len
                .push(((slots.len() - 1) % page_size + 1) as i32);
            for q_tile in 0..(qo_len * group).div_ceil(tile_q) {
                for kv_tile in 0..chunks {
                    current.request_indices.push(index);
                    current.qo_tile_indices.push(q_tile as i32);
                    current.kv_tile_indices.push(kv_tile as i32);
                }
            }
            let base = current.scratch_rows() as i32;
            for i in 0..qo_len {
                current.merge_start.push(base + (i * chunks) as i32);
                current.merge_count.push(chunks as i32);
            }
            current.o_indptr.push(base + (qo_len * chunks) as i32);
            current.rows += qo_len;
            first = last + 1;
        }
    }
    finish(&mut current, &mut launches, rows.len());
    launches
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
            let plan = if chunks == 1 {
                &mut plans.whole
            } else {
                &mut plans.split
            };
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
    fn rows_group_into_requests() {
        // A three-token prompt after a two-token prefix, a row generating, and a second prompt.
        let prompt: Vec<u32> = (8..13).collect();
        let other = [16u32, 17, 18, 19];
        let rows: Vec<&[u32]> = vec![
            &prompt[..3],
            &prompt[..4],
            &prompt[..5],
            &[4, 5],
            &other[..3],
            &other[..4],
        ];
        let requests = Requests::from_slots(&rows);
        assert_eq!(requests.qo_indptr, [0, 3, 4, 6]);
        assert_eq!(
            (requests.batch(), requests.last_row(0), requests.last_row(2)),
            (3, 2, 5)
        );
        let table = requests.page_table(&rows, 4);
        assert_eq!(table.kv_indices, [2, 3, 1, 4]);
        assert_eq!(table.kv_last_page_len, [1, 2, 4]);
        let (batch_indices, positions) = requests.append_indices(&rows);
        assert_eq!(batch_indices, [0, 0, 0, 1, 2, 2]);
        assert_eq!(positions, [2, 3, 4, 1, 2, 3]);
        assert_eq!(Requests::from_slots(&[]).batch(), 0);
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
            [(&plans.whole, false), (&plans.split, true)]
                .into_iter()
                .find_map(|(plan, split)| {
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
    #[test]
    fn attention_launches_fit_the_scratch_and_cover_every_row() {
        let pool: Vec<u32> = (0..100).collect();
        let other: Vec<u32> = (200..240).collect();
        // A prompt of 60 rows, then one row of another sequence.
        let mut rows: Vec<&[u32]> = (41..=100).map(|n| &pool[..n]).collect();
        rows.push(&other[..33]);
        for scratch in [7, 10, 40, 1000] {
            let launches = attention_launches(&rows, 2, 64, 16, 16, scratch);
            let mut next = 0;
            for launch in &launches {
                assert_eq!(launch.first_row, next);
                assert!(launch.scratch_rows() <= scratch);
                assert_eq!(launch.merge_start.len(), launch.rows);
                next += launch.rows;
                for b in 0..launch.qo_indptr.len() - 1 {
                    // Each request's keys are its last row's.
                    let last = launch.qo_indptr[b + 1] as usize - 1;
                    let pages = launch.table.kv_indptr[b + 1] - launch.table.kv_indptr[b];
                    assert_eq!(pages as usize, rows[last].len().div_ceil(16));
                }
                // A row merges its request's chunks, at least as many as its own keys fill.
                for (r, &count) in launch.merge_count.iter().enumerate() {
                    assert!(count as usize >= rows[launch.first_row + r].len().div_ceil(16));
                }
            }
            assert_eq!(next, rows.len());
        }
    }
}

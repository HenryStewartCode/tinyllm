//! The KV cache: storage for every token's attention keys and values, and a radix tree that
//! finds and shares work already done for an earlier request.
//!
//! This is the idea SGLang calls RadixAttention. Most requests to a server start the same way
//! (system prompt, few-shot examples, the conversation so far), so a new request's first tokens
//! have usually been computed before. Every computed token gets a slot in one shared pool, and
//! the radix tree maps token sequences to their slots:
//!
//! ```text
//! root ─ "You are helpful." ┬─ "What is 2+2?" ─ "4"
//!                           └─ "Name a colour" ─ "Red"
//! ```
//!
//! A new request walks the tree to find its longest cached prefix and starts from there. Those
//! slots are shared, not copied. Least-recently-used branches are evicted when the pool is full;
//! branches that a running request depends on are pinned and can't be.
//!
//! Slots are handed out in pages. On the CPU a page is one slot, but the GPU's tensor cores work
//! on 16 tokens at a time, so there a page is 16 consecutive slots holding 16 consecutive tokens,
//! with token `p` of a sequence always in slot `p % 16` of its page. Only whole pages are shared.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Fixed storage for the keys and values of up to `capacity` tokens.
///
/// The memory is requested zeroed, so the operating system only commits the parts that have
/// actually been written.
pub struct KvPool {
    pub capacity: usize,
    /// Per layer, `[capacity, kv_dim]`.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvPool {
    pub fn new(layers: usize, kv_dim: usize, capacity: usize) -> Self {
        let arrays = || {
            (0..layers)
                .map(|_| vec![0.0f32; capacity * kv_dim])
                .collect()
        };
        Self {
            capacity,
            k: arrays(),
            v: arrays(),
        }
    }
}

/// A place in the radix tree that a sequence holds on to.
pub type NodeId = u32;

const ROOT: NodeId = 0;

/// Don't bother evicting fewer tokens than this at a time.
const MIN_EVICTION: usize = 64;

struct Node {
    /// The tokens along the edge from the parent to here, and their KV slots.
    tokens: Vec<u32>,
    slots: Vec<u32>,
    parent: NodeId,
    /// Children by the first page of tokens along their edge.
    children: HashMap<Box<[u32]>, NodeId>,
    /// How many running sequences include this node in their prefix.
    pins: u32,
    last_used: u64,
}

/// Decides which KV slot each token goes in.
///
/// A page of slots is in one of three states: free, private to a running sequence (computed but
/// not yet shared), or owned by the tree.
pub struct PrefixCache {
    nodes: Vec<Node>,
    unused_nodes: Vec<NodeId>,
    free_pages: Vec<u32>,
    /// Slots per page.
    page_size: usize,
    pages: usize,
    /// Pages that running sequences have allocated and not yet committed.
    private: usize,
    clock: u64,
}

impl PrefixCache {
    /// A cache of `capacity` slots, in pages of `page_size`.
    pub fn new(capacity: usize, page_size: usize) -> Self {
        let pages = capacity / page_size;
        let root = Node {
            tokens: Vec::new(),
            slots: Vec::new(),
            parent: ROOT,
            children: HashMap::new(),
            pins: 0,
            last_used: 0,
        };
        Self {
            nodes: vec![root],
            unused_nodes: Vec::new(),
            // Reversed, so that the lowest pages are handed out first and the untouched part of
            // the pool stays uncommitted for as long as possible.
            free_pages: (0..pages as u32).rev().collect(),
            page_size,
            pages,
            private: 0,
            clock: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.pages * self.page_size
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Slots held by the tree.
    pub fn cached(&self) -> usize {
        (self.pages - self.free_pages.len() - self.private) * self.page_size
    }

    /// Slots that could be handed out right now: free ones, plus anything evictable.
    pub fn available(&self) -> usize {
        let evictable: usize = self
            .nodes
            .iter()
            .filter(|n| n.pins == 0)
            .map(|n| n.slots.len())
            .sum();
        self.free_pages.len() * self.page_size + evictable
    }

    /// `tokens`, rounded down to whole pages.
    fn whole_pages(&self, tokens: usize) -> usize {
        tokens - tokens % self.page_size
    }

    /// The child of `node` that continues with the page of `tokens` starting at `pos`, if any.
    fn child_for(&self, node: NodeId, tokens: &[u32], pos: usize) -> Option<NodeId> {
        let page = tokens.get(pos..pos + self.page_size)?;
        self.nodes[node as usize].children.get(page).copied()
    }

    /// How many leading tokens of `tokens` are cached, without changing anything.
    pub fn match_len(&self, tokens: &[u32]) -> usize {
        let (mut node, mut pos) = (ROOT, 0);
        while let Some(child) = self.child_for(node, tokens, pos) {
            let edge = &self.nodes[child as usize].tokens;
            let matched = self.whole_pages(common_prefix(edge, &tokens[pos..]));
            pos += matched;
            if matched < edge.len() {
                break;
            }
            node = child;
        }
        pos
    }

    /// Find the longest cached prefix of `tokens`, append its slots to `slots`, and pin it so
    /// that it stays until [`release`](Self::release).
    pub fn acquire(&mut self, tokens: &[u32], slots: &mut Vec<u32>) -> NodeId {
        self.clock += 1;
        let (mut node, mut pos) = (ROOT, 0);
        while let Some(child) = self.child_for(node, tokens, pos) {
            let common = common_prefix(&self.nodes[child as usize].tokens, &tokens[pos..]);
            let matched = self.whole_pages(common);
            let child = self.split_if_partial(child, matched);
            let n = &mut self.nodes[child as usize];
            n.last_used = self.clock;
            slots.extend_from_slice(&n.slots);
            pos += matched;
            node = child;
        }
        self.pin(node);
        node
    }

    /// Stop holding on to a node returned by `acquire` or `commit`.
    pub fn release(&mut self, node: NodeId) {
        let mut at = node;
        while at != ROOT {
            let n = &mut self.nodes[at as usize];
            debug_assert!(n.pins > 0);
            n.pins -= 1;
            at = n.parent;
        }
    }

    /// Take `n` pages for sequences' new tokens, evicting old branches if the pool is full.
    /// Returns false, having taken nothing, if there aren't enough.
    pub fn allocate(&mut self, n: usize, out: &mut Vec<u32>) -> bool {
        if self.free_pages.len() < n {
            let minimum = (MIN_EVICTION / self.page_size).max(1);
            self.evict((n - self.free_pages.len()).max(minimum));
        }
        if self.free_pages.len() < n {
            return false;
        }
        let start = self.free_pages.len() - n;
        out.extend(self.free_pages.drain(start..).rev());
        self.private += n;
        true
    }

    /// Give back a page that a sequence allocated and never committed.
    pub fn free_page(&mut self, page: u32) {
        self.free_pages.push(page);
        self.private -= 1;
    }

    /// Hand a sequence's computed tokens (whole pages of them) over to the tree so other requests
    /// can use them, and move its pin from `pinned` to the end of them. Where another sequence
    /// got there first, this sequence's duplicate pages are freed and replaced with the shared
    /// ones.
    pub fn commit(&mut self, pinned: NodeId, tokens: &[u32], slots: &mut [u32]) -> NodeId {
        assert_eq!(tokens.len(), slots.len());
        assert!(tokens.len().is_multiple_of(self.page_size));
        let page_size = self.page_size;
        self.clock += 1;
        let (mut node, mut pos) = (ROOT, 0);
        while let Some(child) = self.child_for(node, tokens, pos) {
            let common = common_prefix(&self.nodes[child as usize].tokens, &tokens[pos..]);
            let matched = self.whole_pages(common);
            let child = self.split_if_partial(child, matched);
            let n = &mut self.nodes[child as usize];
            n.last_used = self.clock;
            for (mine, theirs) in slots[pos..pos + matched]
                .chunks_exact_mut(page_size)
                .zip(n.slots.chunks_exact(page_size))
            {
                if mine[0] != theirs[0] {
                    self.free_pages.push(mine[0] / page_size as u32);
                    self.private -= 1;
                    mine.copy_from_slice(theirs);
                }
            }
            pos += matched;
            node = child;
        }
        if pos < tokens.len() {
            self.private -= (tokens.len() - pos) / page_size;
            node = self.add_node(node, tokens[pos..].to_vec(), slots[pos..].to_vec());
        }
        self.pin(node);
        self.release(pinned);
        node
    }

    /// Drop everything that isn't pinned.
    pub fn clear(&mut self) {
        self.evict(usize::MAX);
    }

    fn pin(&mut self, node: NodeId) {
        let mut at = node;
        while at != ROOT {
            let n = &mut self.nodes[at as usize];
            n.pins += 1;
            at = n.parent;
        }
    }

    /// Make sure `matched` tokens into `node`'s edge is a node boundary, returning the node that
    /// ends there.
    ///
    /// The new node is the upper part, so that `node` still means the same place for anyone
    /// holding it. Everyone pinning `node` passes through both halves.
    fn split_if_partial(&mut self, node: NodeId, matched: usize) -> NodeId {
        let page_size = self.page_size;
        let n = &mut self.nodes[node as usize];
        if matched == n.tokens.len() {
            return node;
        }
        let tokens: Vec<u32> = n.tokens.drain(..matched).collect();
        let slots: Vec<u32> = n.slots.drain(..matched).collect();
        let tail_key: Box<[u32]> = n.tokens[..page_size].into();
        let (parent, pins, last_used) = (n.parent, n.pins, n.last_used);
        let upper = self.add_node(parent, tokens, slots);
        let u = &mut self.nodes[upper as usize];
        u.pins = pins;
        u.last_used = last_used;
        u.children.insert(tail_key, node);
        self.nodes[node as usize].parent = upper;
        upper
    }

    /// Create a node as a child of `parent`, replacing any child with the same first page.
    fn add_node(&mut self, parent: NodeId, tokens: Vec<u32>, slots: Vec<u32>) -> NodeId {
        let key: Box<[u32]> = tokens[..self.page_size].into();
        let node = Node {
            tokens,
            slots,
            parent,
            children: HashMap::new(),
            pins: 0,
            last_used: self.clock,
        };
        let id = match self.unused_nodes.pop() {
            Some(id) => {
                self.nodes[id as usize] = node;
                id
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as NodeId
            }
        };
        self.nodes[parent as usize].children.insert(key, id);
        id
    }

    /// Free up to `wanted` pages, oldest branches first, trimming from the ends of leaves.
    fn evict(&mut self, wanted: usize) {
        let page_size = self.page_size;
        let is_candidate = |id: usize, n: &Node| {
            id != ROOT as usize && n.pins == 0 && n.children.is_empty() && !n.tokens.is_empty()
        };
        let mut heap: BinaryHeap<_> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|&(id, n)| is_candidate(id, n))
            .map(|(id, n)| Reverse((n.last_used, id)))
            .collect();
        let mut freed = 0;
        while freed < wanted {
            let Some(Reverse((_, id))) = heap.pop() else {
                break;
            };
            let n = &mut self.nodes[id];
            let pages = n.slots.len() / page_size;
            let keep = pages.saturating_sub(wanted - freed);
            freed += pages - keep;
            for page in n.slots[keep * page_size..].chunks_exact(page_size) {
                self.free_pages.push(page[0] / page_size as u32);
            }
            n.slots.truncate(keep * page_size);
            n.tokens.truncate(keep * page_size);
            if keep > 0 {
                continue; // that was enough
            }
            // The leaf is gone, which may have made its parent a leaf.
            let parent = n.parent as usize;
            let first = std::mem::take(&mut n.children);
            debug_assert!(first.is_empty());
            self.nodes[parent]
                .children
                .retain(|_, child| *child as usize != id);
            self.unused_nodes.push(id as NodeId);
            if is_candidate(parent, &self.nodes[parent]) {
                heap.push(Reverse((self.nodes[parent].last_used, parent)));
            }
        }
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate a sequence computing `tokens`: reuse what's cached, allocate the rest, commit.
    fn run(cache: &mut PrefixCache, tokens: &[u32]) -> (NodeId, Vec<u32>, usize) {
        let mut slots = Vec::new();
        let pinned = cache.acquire(tokens, &mut slots);
        let hit = slots.len();
        assert!(cache.allocate(tokens.len() - hit, &mut slots));
        let node = cache.commit(pinned, tokens, &mut slots);
        (node, slots, hit)
    }

    #[test]
    fn finds_and_shares_the_longest_prefix() {
        let mut cache = PrefixCache::new(100, 1);
        let (a, slots_a, hit) = run(&mut cache, &[1, 2, 3, 4, 5]);
        assert_eq!(hit, 0);
        cache.release(a);
        let (b, slots_b, hit) = run(&mut cache, &[1, 2, 3, 9, 9]);
        assert_eq!(hit, 3);
        assert_eq!(slots_a[..3], slots_b[..3]);
        assert_ne!(slots_a[3..], slots_b[3..]);
        cache.release(b);
        assert_eq!(cache.match_len(&[1, 2, 3, 4, 5, 6]), 5);
        assert_eq!(cache.match_len(&[1, 2, 3, 9, 7]), 4);
        assert_eq!(cache.match_len(&[2, 2]), 0);
        assert_eq!(cache.cached(), 7);
    }

    #[test]
    fn a_fully_cached_sequence_needs_no_new_slots() {
        let mut cache = PrefixCache::new(10, 1);
        let (a, _, _) = run(&mut cache, &[1, 2, 3]);
        cache.release(a);
        let (b, _, hit) = run(&mut cache, &[1, 2, 3]);
        assert_eq!(hit, 3);
        cache.release(b);
        assert_eq!(cache.cached(), 3);
    }

    #[test]
    fn duplicates_are_freed_when_two_sequences_race() {
        let mut cache = PrefixCache::new(20, 1);
        // Both start before either has committed.
        let (mut s1, mut s2) = (Vec::new(), Vec::new());
        let p1 = cache.acquire(&[1, 2, 3], &mut s1);
        let p2 = cache.acquire(&[1, 2, 3, 4], &mut s2);
        assert!(cache.allocate(3, &mut s1));
        assert!(cache.allocate(4, &mut s2));
        let n1 = cache.commit(p1, &[1, 2, 3], &mut s1);
        let n2 = cache.commit(p2, &[1, 2, 3, 4], &mut s2);
        assert_eq!(
            s1[..3],
            s2[..3],
            "the second sequence adopts the first's slots"
        );
        assert_eq!(cache.cached(), 4);
        assert_eq!(cache.private, 0);
        cache.release(n1);
        cache.release(n2);
        assert_eq!(cache.available(), 20);
    }

    #[test]
    fn evicts_least_recently_used_first_and_never_pinned() {
        let mut cache = PrefixCache::new(MIN_EVICTION + 10, 1);
        let long_a: Vec<u32> = (0..MIN_EVICTION as u32 / 2).map(|t| t + 1000).collect();
        let long_b: Vec<u32> = (0..MIN_EVICTION as u32 / 2).map(|t| t + 2000).collect();
        let (a, _, _) = run(&mut cache, &long_a);
        cache.release(a);
        let (b, _, _) = run(&mut cache, &long_b); // stays pinned
        // Refresh nothing; ask for almost everything. Only `a` can go.
        let mut slots = Vec::new();
        assert!(cache.allocate(10 + MIN_EVICTION / 2, &mut slots));
        assert_eq!(cache.match_len(&long_a), 0);
        assert_eq!(cache.match_len(&long_b), long_b.len());
        // More than what's left can't be satisfied, and nothing is taken.
        let free_before = cache.free_pages.len();
        assert!(!cache.allocate(1, &mut Vec::new()));
        assert_eq!(cache.free_pages.len(), free_before);
        cache.release(b);
    }

    #[test]
    fn eviction_trims_a_branch_from_its_end() {
        let mut cache = PrefixCache::new(MIN_EVICTION + 10, 1);
        let tokens: Vec<u32> = (1..=(MIN_EVICTION + 10) as u32).collect();
        let (a, _, _) = run(&mut cache, &tokens);
        cache.release(a);
        assert!(cache.allocate(MIN_EVICTION, &mut Vec::new()));
        assert_eq!(cache.match_len(&tokens), 10);
    }

    #[test]
    fn splitting_keeps_pins_correct() {
        let mut cache = PrefixCache::new(20, 1);
        let (a, _, _) = run(&mut cache, &[1, 2, 3, 4]); // holds [1, 2, 3, 4]
        let (b, _, _) = run(&mut cache, &[1, 2, 9]); // splits it at 2
        cache.release(b);
        // [9] is evictable, but the split-off upper half of a's edge isn't.
        assert_eq!(cache.available(), 20 - 4);
        cache.release(a);
        assert_eq!(cache.available(), 20);
    }

    /// Simulate a sequence on a paged cache: the slots of its pages, committed whole.
    fn run_paged(cache: &mut PrefixCache, tokens: &[u32]) -> (NodeId, Vec<u32>, usize) {
        let page = cache.page_size();
        let mut slots = Vec::new();
        let pinned = cache.acquire(tokens, &mut slots);
        let hit = slots.len();
        let new_tokens = tokens.len() - hit;
        let mut pages = Vec::new();
        assert!(cache.allocate(new_tokens.div_ceil(page), &mut pages));
        for (i, position) in (hit..tokens.len()).enumerate() {
            slots.push(pages[i / page] * page as u32 + (position % page) as u32);
        }
        let whole = tokens.len() - tokens.len() % page;
        let node = cache.commit(pinned, &tokens[..whole], &mut slots[..whole]);
        if !tokens.len().is_multiple_of(page) {
            cache.free_page(pages[pages.len() - 1]);
        }
        (node, slots, hit)
    }

    #[test]
    fn pages_share_only_whole_pages() {
        let mut cache = PrefixCache::new(16 * 8, 16);
        let first: Vec<u32> = (0..40).collect();
        let (a, slots_a, hit) = run_paged(&mut cache, &first);
        assert_eq!(hit, 0);
        cache.release(a);
        // 32 of the 40 tokens are whole pages; the last 8 were never committed.
        assert_eq!(cache.match_len(&first), 32);
        assert_eq!(cache.cached(), 32);
        assert_eq!(slots_a[..16], (0..16).collect::<Vec<u32>>()[..]);
        assert_eq!(
            slots_a[20] % 16,
            4,
            "token p lives in slot p % 16 of its page"
        );

        // Shares 23 tokens with `first`, but only the 16 of a whole page can be reused.
        let mut second = first[..23].to_vec();
        second.extend([900, 901, 902, 903]);
        assert_eq!(cache.match_len(&second), 16);
        let (b, slots_b, hit) = run_paged(&mut cache, &second);
        assert_eq!(hit, 16);
        assert_eq!(slots_a[..16], slots_b[..16]);
        assert_ne!(
            slots_a[16], slots_b[16],
            "the partly shared page is computed afresh"
        );
        cache.release(b);
    }

    #[test]
    fn page_eviction_frees_whole_pages() {
        let mut cache = PrefixCache::new(16 * 8, 16);
        let tokens: Vec<u32> = (1..=112).collect();
        let (a, _, _) = run_paged(&mut cache, &tokens);
        cache.release(a);
        assert_eq!(cache.cached(), 112);
        assert_eq!(cache.available(), 16 * 8);
        // One page is free. Asking for two evicts at least MIN_EVICTION tokens' worth (4 pages)
        // from the end of the branch.
        let mut pages = Vec::new();
        assert!(cache.allocate(2, &mut pages));
        assert_eq!(cache.match_len(&tokens), 112 - 4 * 16);
        assert_eq!(cache.cached(), 112 - 4 * 16);
        // Everything is pinned or held now except the rest of the branch.
        assert!(!cache.allocate(7, &mut pages), "only 6 pages are left");
        assert!(cache.allocate(6, &mut pages));
        assert_eq!(cache.match_len(&tokens), 0);
    }

    #[test]
    fn pages_that_differ_after_the_first_token_are_separate_branches() {
        let mut cache = PrefixCache::new(16 * 16, 16);
        let first: Vec<u32> = (0..16).chain(100..116).collect();
        let (a, _, _) = run_paged(&mut cache, &first);
        cache.release(a);
        // The same first token, but a different first page.
        let mut second = first.clone();
        second[5] = 999;
        let (b, slots_b, hit) = run_paged(&mut cache, &second);
        assert_eq!(hit, 0, "no whole page in common");
        cache.release(b);
        // Both branches are still there, on different pages, and evicting frees each page once.
        assert_eq!(cache.match_len(&first), 32);
        assert_eq!(cache.match_len(&second), 32);
        assert_eq!(cache.available(), 16 * 16);
        cache.clear();
        assert_eq!(cache.cached(), 0);
        let mut pages = Vec::new();
        assert!(cache.allocate(16, &mut pages));
        pages.sort_unstable();
        pages.dedup();
        assert_eq!(pages.len(), 16, "every page comes back exactly once");
        assert!(slots_b.len() == 32);
    }
}

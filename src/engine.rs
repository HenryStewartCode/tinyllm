//! The serving loop: many requests at once on one model, one batch at a time.
//!
//! This is *continuous batching*. Each step runs one batch containing the next token of every
//! sequence that's generating, plus as many prompt tokens as fit. Requests join and leave
//! between steps, so a short request is never stuck behind a long one, and every step does as
//! much useful work as it can.
//!
//! A sequence is just its tokens and the KV slots of the ones computed so far. Whatever is
//! `tokens[slots.len()..]` still has to go through the model, whether that's a long prompt or
//! the single token that was just generated, so both go through the same code path.

use crate::backend::{Backend, Logits};
use crate::cache::{NodeId, PrefixCache};
use crate::grammar::{Constraint, Grammar, TokenSet};
use crate::model::{LogitUse, Row};
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::Tokenizer;
use std::collections::{HashSet, VecDeque};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc;
use std::time::Instant;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

/// Every step gets at least this many prompt tokens, however many sequences are generating, so
/// waiting prompts always make progress.
const MIN_PREFILL_ROWS: usize = 8;

/// Don't hold up a new request for an in-flight prompt unless that saves at least this many
/// tokens of work.
const MIN_SHARED_PREFIX: usize = 32;

/// Stack for the threads that run grammars, whose automaton is recursive.
pub const STACK_BYTES: usize = 8 << 20;

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub threads: usize,
    /// How many sequences can be in progress at once.
    pub max_parallel: usize,
    /// How many tokens go through the model in each step.
    pub max_batch: usize,
    /// KV slots, i.e. how many tokens of context can be held in memory across all requests.
    pub cache_tokens: usize,
    /// The longest a single sequence can get.
    pub ctx_size: usize,
}

#[derive(Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_tokens: HashSet<u32>,
    pub stop_strings: Vec<String>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    /// Tokenizers that mark a word boundary with a leading space add one to the start of a
    /// reply. Set this to remove it, when the text isn't a continuation of earlier text.
    pub strip_leading_space: bool,
    /// Report the log-probability of each token, along with this many alternatives.
    pub logprobs: Option<usize>,
    pub grammar: Option<Arc<Grammar>>,
    /// If not empty, the reply is free until one of these turns up in it. The grammar applies
    /// from there on, and has to be able to start with the text that set it off.
    pub grammar_triggers: Vec<GrammarTrigger>,
    /// Special tokens to write out even with `skip_special_tokens` on.
    pub keep_tokens: Vec<u32>,
    /// If the reply may think out loud before it answers, how to tell when it is.
    pub thinking: Option<Thinking>,
    /// Instead of generating, send the final hidden state of the prompt's last token (see
    /// [`Event::Embedding`]), as embedding models with last-token pooling use it.
    pub embed: bool,
}

/// How a reply thinks out loud before it answers (see [`crate::reasoning`]). While it thinks,
/// its grammar and the triggers for it wait, so that only the answer is held to them.
#[derive(Clone, Debug)]
pub struct Thinking {
    /// The prompt left a block of thinking open, so the reply starts inside it.
    pub open: bool,
    /// The token that opens a block, so that a reply can open one itself.
    pub start_token: Option<u32>,
    /// What closes the block.
    pub end: String,
    /// The token that closes it, for a budget to force.
    pub end_token: Option<u32>,
    /// A newline token, to put a forced end on a line of its own.
    pub newline_token: Option<u32>,
    /// The most tokens to spend thinking, after which the block is closed for the model.
    pub budget: Option<usize>,
}

/// Where a reply is, as far as thinking goes.
enum Phase {
    /// Answering, or never going to think: the grammar applies.
    Answering,
    /// Nothing generated yet, and the first token may open a block of thinking.
    MayThink,
    /// In a block of thinking. `recent` is the end of it, as much as the end marker could be
    /// spread across.
    Thinking { recent: Vec<u8>, spent: usize },
}

/// Text in a reply that sets off its grammar.
#[derive(Clone, Debug)]
pub struct GrammarTrigger {
    pub text: String,
    /// It only counts as the first thing in the reply, apart from whitespace.
    pub at_start: bool,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            sampling: SamplingParams::default(),
            stop_tokens: HashSet::new(),
            stop_strings: Vec::new(),
            ignore_eos: false,
            skip_special_tokens: true,
            strip_leading_space: false,
            logprobs: None,
            grammar: None,
            grammar_triggers: Vec::new(),
            keep_tokens: Vec::new(),
            thinking: None,
            embed: false,
        }
    }
}

impl GenerateParams {
    /// How the grammar starts: either right away, or watching for what sets it off.
    fn start_grammar(
        &self,
        strip_leading_space: bool,
    ) -> (Option<Constraint>, Option<TriggerWatch>) {
        match &self.grammar {
            Some(_) if !self.grammar_triggers.is_empty() => (None, Some(TriggerWatch::new())),
            Some(grammar) => {
                let stop_tokens = self.stop_tokens.iter().copied().collect();
                (Some(grammar.start(strip_leading_space, stop_tokens)), None)
            }
            None => (None, None),
        }
    }
}

pub struct Request {
    pub tokens: Vec<u32>,
    pub params: GenerateParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token or stop string.
    Stop,
    /// Hit `max_tokens` or the end of the context.
    Length,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// How much of the prompt was already in the cache.
    pub cached_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Debug)]
pub enum Event {
    /// More of the reply. The logprobs are for tokens that finished since the last one.
    Text {
        text: String,
        logprobs: Vec<TokenLogprob>,
    },
    Done {
        reason: FinishReason,
        usage: Usage,
        /// The stop string that ended the reply, if one did.
        stop_string: Option<String>,
    },
    /// For a request with [`GenerateParams::embed`]: the final hidden state of the prompt's last
    /// token, after the model's last norm. `Done` follows.
    Embedding(Vec<f32>),
    Error(String),
}

/// Counters that can be read while the engine runs.
#[derive(Default)]
pub struct Stats {
    pub requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub cached_prompt_tokens: AtomicU64,
    pub generated_tokens: AtomicU64,
    pub steps: AtomicU64,
    pub step_micros: AtomicU64,
    pub rows: AtomicU64,
    pub running: AtomicU64,
    pub waiting: AtomicU64,
    pub cache_used: AtomicU64,
    pub cache_capacity: AtomicU64,
    pub retractions: AtomicU64,
}

enum Command {
    Submit {
        request: Box<Request>,
        events: UnboundedSender<Event>,
    },
    FlushCache,
}

/// A connection to a running engine. Dropping every handle shuts the engine down.
#[derive(Clone)]
pub struct EngineHandle {
    commands: mpsc::Sender<Command>,
    stats: Arc<Stats>,
    config: Arc<EngineConfig>,
}

impl EngineHandle {
    /// Queue a request. Results arrive on the returned channel; dropping it cancels the request.
    pub fn submit(&self, request: Request) -> UnboundedReceiver<Event> {
        let (events, receiver) = unbounded_channel();
        let command = Command::Submit {
            request: Box::new(request),
            events,
        };
        if let Err(mpsc::SendError(Command::Submit { events, .. })) = self.commands.send(command) {
            let _ = events.send(Event::Error("the engine has stopped".to_string()));
        }
        receiver
    }

    /// Throw away everything in the prefix cache that isn't in use.
    pub fn flush_cache(&self) {
        let _ = self.commands.send(Command::FlushCache);
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

/// Start an engine on its own thread.
pub fn start(
    backend: Box<dyn Backend>,
    tokenizer: Arc<Tokenizer>,
    config: EngineConfig,
    vocab_size: usize,
) -> EngineHandle {
    let (commands, receiver) = mpsc::channel();
    let stats = Arc::new(Stats::default());
    stats
        .cache_capacity
        .store(config.cache_tokens as u64, Relaxed);
    let handle = EngineHandle {
        commands,
        stats: stats.clone(),
        config: Arc::new(config.clone()),
    };
    std::thread::Builder::new()
        .name("tinyllm-engine".to_string())
        .stack_size(STACK_BYTES)
        .spawn(move || Engine::new(backend, tokenizer, config, stats, vocab_size).run(receiver))
        .expect("couldn't start the engine thread");
    handle
}

/// One request's progress.
struct Sequence {
    /// The prompt plus everything generated so far.
    tokens: Vec<u32>,
    /// KV slots for `tokens[..slots.len()]`. The rest haven't been through the model yet.
    slots: Vec<u32>,
    /// The sequence's place in the prefix cache, which keeps its shared slots from being evicted.
    pinned: Option<NodeId>,
    /// How many of `tokens` the prefix cache has been told about.
    prompt_len: usize,
    prompt_committed: bool,
    cached_tokens: Option<usize>,
    generated: usize,
    /// Rows this sequence has in the current step.
    take: usize,
    /// Which logits row belongs to this sequence in the current step, if any.
    logits_row: Option<usize>,
    /// Result of sampling in the current step.
    sampled: u32,
    token_logprob: Option<TokenLogprob>,
    params: GenerateParams,
    sampler: Sampler,
    constraint: Option<Constraint>,
    /// Looks out for what sets off the grammar, until it starts.
    watch: Option<TriggerWatch>,
    phase: Phase,
    allowed: Option<Arc<TokenSet>>,
    text: TextStream,
    pending_logprobs: Vec<TokenLogprob>,
    events: UnboundedSender<Event>,
}

impl Sequence {
    fn pending(&self) -> usize {
        self.tokens.len() - self.slots.len()
    }

    fn is_prefilling(&self) -> bool {
        !self.prompt_committed
    }

    fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_len,
            completion_tokens: self.generated,
            cached_tokens: self.cached_tokens.unwrap_or(0),
        }
    }

    /// The token the thinking budget forces next, if it's been used up.
    fn forced(&self) -> Option<u32> {
        let Phase::Thinking { recent, spent } = &self.phase else {
            return None;
        };
        let thinking = self.params.thinking.as_ref()?;
        let end = thinking.end_token?;
        if *spent < thinking.budget? {
            return None;
        }
        match thinking.newline_token {
            Some(newline) if recent.last() != Some(&b'\n') => Some(newline),
            _ => Some(end),
        }
    }

    /// Whether choosing the next token needs every token's score, rather than a shortlist.
    fn needs_all_logits(&self) -> bool {
        self.constraint.is_some() || self.params.logprobs.is_some() || self.forced().is_some()
    }

    /// The tokens that may come next, if they're limited.
    fn allowed_tokens(&mut self, vocab_size: usize) -> Option<Arc<TokenSet>> {
        if let Some(token) = self.forced() {
            let mut only = TokenSet::new(vocab_size);
            only.insert(token);
            return Some(Arc::new(only));
        }
        let allowed = self.constraint.as_mut()?.allowed_tokens();
        match (&self.phase, &self.params.thinking) {
            // The answer has to follow the grammar, but it can think first.
            (
                Phase::MayThink,
                Some(Thinking {
                    start_token: Some(start),
                    ..
                }),
            ) => {
                let mut allowed = (*allowed).clone();
                allowed.insert(*start);
                Some(Arc::new(allowed))
            }
            _ => Some(allowed),
        }
    }
}

struct Engine {
    backend: Box<dyn Backend>,
    tokenizer: Arc<Tokenizer>,
    config: EngineConfig,
    stats: Arc<Stats>,
    vocab_size: usize,
    cache: PrefixCache,
    waiting: VecDeque<Sequence>,
    /// Oldest first.
    active: Vec<Sequence>,
    new_pages: Vec<u32>,
}

impl Engine {
    fn new(
        backend: Box<dyn Backend>,
        tokenizer: Arc<Tokenizer>,
        config: EngineConfig,
        stats: Arc<Stats>,
        vocab_size: usize,
    ) -> Self {
        Self {
            cache: PrefixCache::new(config.cache_tokens, backend.page_size()),
            backend,
            tokenizer,
            config,
            stats,
            vocab_size,
            waiting: VecDeque::new(),
            active: Vec::new(),
            new_pages: Vec::new(),
        }
    }

    fn run(mut self, commands: mpsc::Receiver<Command>) {
        self.backend.attach();
        loop {
            if self.active.is_empty() && self.waiting.is_empty() {
                // Nothing to do: wait for something to arrive.
                match commands.recv() {
                    Ok(command) => self.handle(command),
                    Err(_) => return,
                }
            }
            while let Ok(command) = commands.try_recv() {
                self.handle(command);
            }
            // A bug shouldn't take the whole server down with it.
            let step = panic::catch_unwind(AssertUnwindSafe(|| self.step()));
            if step.is_err() {
                self.recover();
            }
        }
    }

    /// After a panic, nothing about the requests in progress can be trusted. Fail them and
    /// start again with an empty cache.
    fn recover(&mut self) {
        for seq in self.active.drain(..).chain(self.waiting.drain(..)) {
            let _ = seq
                .events
                .send(Event::Error("internal error while generating".to_string()));
        }
        self.cache = PrefixCache::new(self.config.cache_tokens, self.backend.page_size());
        self.update_gauges();
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Submit { request, events } => self.enqueue(*request, events),
            Command::FlushCache => self.cache.clear(),
        }
    }

    fn enqueue(&mut self, request: Request, events: UnboundedSender<Event>) {
        let Request { tokens, params } = request;
        let prompt_len = tokens.len();
        let limit = self.config.ctx_size.min(self.config.cache_tokens);
        if tokens.is_empty() {
            let _ = events.send(Event::Error("the prompt is empty".to_string()));
            return;
        }
        if prompt_len >= limit {
            let _ = events.send(Event::Error(format!(
                "the prompt is {prompt_len} tokens but the context limit is {limit}"
            )));
            return;
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= self.vocab_size) {
            let _ = events.send(Event::Error(format!(
                "token id {bad} is outside the vocabulary"
            )));
            return;
        }
        self.stats.requests.fetch_add(1, Relaxed);
        self.stats
            .prompt_tokens
            .fetch_add(prompt_len as u64, Relaxed);
        let strip = if params.strip_leading_space {
            self.tokenizer.leading_spaces_to_strip()
        } else {
            0
        };
        let phase = match &params.thinking {
            Some(thinking) if thinking.open => Phase::Thinking {
                recent: Vec::new(),
                spent: 0,
            },
            Some(thinking) if thinking.start_token.is_some() => Phase::MayThink,
            _ => Phase::Answering,
        };
        let (constraint, watch) = match phase {
            Phase::Thinking { .. } => (None, None),
            _ => params.start_grammar(strip > 0),
        };
        self.waiting.push_back(Sequence {
            tokens,
            slots: Vec::new(),
            pinned: None,
            prompt_len,
            prompt_committed: false,
            cached_tokens: None,
            generated: 0,
            take: 0,
            logits_row: None,
            sampled: 0,
            token_logprob: None,
            sampler: Sampler::new(params.sampling.clone(), self.vocab_size),
            constraint,
            watch,
            phase,
            allowed: None,
            text: TextStream::new(params.stop_strings.clone(), strip),
            pending_logprobs: Vec::new(),
            params,
            events,
        });
    }

    /// One trip through the model.
    fn step(&mut self) {
        let start = Instant::now();
        self.drop_cancelled();
        self.admit();
        self.update_gauges();
        if self.active.is_empty() {
            return;
        }
        if !self.plan_rows() {
            return;
        }

        // Run the batch.
        let mut rows = Vec::new();
        let mut logits_rows = 0;
        for seq in &mut self.active {
            seq.logits_row = None;
            let end = seq.slots.len();
            for pos in end - seq.take..end {
                let want_logits = pos + 1 == seq.tokens.len();
                if want_logits {
                    seq.logits_row = Some(logits_rows);
                    logits_rows += 1;
                }
                let all_logits = seq.needs_all_logits();
                rows.push(Row {
                    token: seq.tokens[pos],
                    slots: &seq.slots[..=pos],
                    want_logits,
                    sampling: match (seq.params.embed, all_logits) {
                        (true, _) => LogitUse::Hidden,
                        (false, true) => LogitUse::All,
                        (false, false) => seq.sampler.logit_use(seq.generated),
                    },
                });
            }
        }
        self.backend.forward(&rows);
        let row_count = rows.len();
        drop(rows);

        self.sample();
        self.advance();

        self.stats.steps.fetch_add(1, Relaxed);
        self.stats.rows.fetch_add(row_count as u64, Relaxed);
        self.stats
            .step_micros
            .fetch_add(start.elapsed().as_micros() as u64, Relaxed);
        self.update_gauges();
    }

    fn update_gauges(&self) {
        self.stats.running.store(self.active.len() as u64, Relaxed);
        self.stats.waiting.store(self.waiting.len() as u64, Relaxed);
        self.stats
            .cache_used
            .store(self.cache.cached() as u64, Relaxed);
    }

    /// Requests whose client has gone away don't need finishing.
    fn drop_cancelled(&mut self) {
        self.waiting.retain(|seq| !seq.events.is_closed());
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].events.is_closed() {
                let mut seq = self.active.remove(i);
                self.retire(&mut seq);
            } else {
                i += 1;
            }
        }
    }

    /// Start as many waiting requests as there's room for.
    fn admit(&mut self) {
        let mut i = 0;
        while i < self.waiting.len() && self.active.len() < self.config.max_parallel {
            let seq = &self.waiting[i];
            // The last token is always run, so that there are logits to sample from.
            let cacheable = seq.tokens.len() - 1;
            let cached = self.cache.match_len(&seq.tokens[..cacheable]);

            // If a prompt in progress shares a long beginning with this one, wait for it to
            // finish and reuse its work, instead of computing the same thing twice.
            let wait = self.active.iter().any(|other| {
                other.is_prefilling()
                    && common_prefix_len(&other.tokens, &seq.tokens) >= cached + MIN_SHARED_PREFIX
            });
            if wait {
                i += 1;
                continue;
            }

            // Leave some room for the requests already running to grow.
            let needed = seq.tokens.len() - cached;
            let headroom = 16 * (self.active.len() + 1);
            if !self.active.is_empty() && needed + headroom > self.cache.available() {
                break; // out of memory for now, and it's first come, first served
            }
            let mut seq = self.waiting.remove(i).expect("index is in range");
            let pinned = self.cache.acquire(&seq.tokens[..cacheable], &mut seq.slots);
            seq.pinned = Some(pinned);
            if seq.cached_tokens.is_none() {
                seq.cached_tokens = Some(seq.slots.len());
                self.stats
                    .cached_prompt_tokens
                    .fetch_add(seq.slots.len() as u64, Relaxed);
            }
            self.active.push(seq);
        }
    }

    /// Decide how many tokens each sequence runs this step and allocate their KV slots.
    /// Returns false if there's nothing to run.
    fn plan_rows(&mut self) -> bool {
        loop {
            let generating = self.active.iter().filter(|s| s.pending() == 1).count();
            let budget = self
                .config
                .max_batch
                .saturating_sub(generating)
                .max(MIN_PREFILL_ROWS);
            let mut prefill_left = budget;
            let mut needed = 0;
            let mut new_pages = 0;
            let page_size = self.cache.page_size();
            for seq in &mut self.active {
                let pending = seq.pending();
                seq.take = if pending <= 1 {
                    pending
                } else {
                    let take = pending.min(prefill_left);
                    prefill_left -= take;
                    take
                };
                needed += seq.take;
                let first = seq.slots.len();
                new_pages += (first + seq.take).div_ceil(page_size) - first.div_ceil(page_size);
            }
            if needed == 0 {
                return false;
            }

            self.new_pages.clear();
            if self.cache.allocate(new_pages, &mut self.new_pages) {
                let mut new_pages = self.new_pages.drain(..);
                for seq in &mut self.active {
                    let first = seq.slots.len();
                    for position in first..first + seq.take {
                        let page = match position % page_size {
                            0 => new_pages.next().expect("counted above"),
                            _ => seq.slots[position - 1] / page_size as u32,
                        };
                        seq.slots
                            .push(page * page_size as u32 + (position % page_size) as u32);
                    }
                }
                return true;
            }

            // The pool is full of slots that running sequences need. Put the newest one back in
            // the queue. Its work stays in the prefix cache, so unless that gets evicted, it
            // picks up where it left off.
            if self.active.len() <= 1 {
                let Some(mut seq) = self.active.pop() else {
                    return false;
                };
                let _ = seq.events.send(Event::Error(
                    "the KV cache is too small for this request".to_string(),
                ));
                self.retire(&mut seq);
                return false;
            }
            let mut seq = self.active.pop().expect("more than one is active");
            self.retire(&mut seq);
            seq.slots.clear();
            seq.prompt_committed = false;
            self.stats.retractions.fetch_add(1, Relaxed);
            self.waiting.push_front(seq);
        }
    }

    /// Choose the next token for every sequence that has logits from this step.
    fn sample(&mut self) {
        // Working out which tokens a grammar allows can touch shared state, so it happens here.
        for seq in &mut self.active {
            if seq.logits_row.is_some() {
                seq.allowed = seq.allowed_tokens(self.vocab_size);
            }
        }
        let views: Vec<Option<Logits>> = self
            .active
            .iter()
            .map(|seq| match seq.params.embed {
                true => None,
                false => seq.logits_row.map(|row| self.backend.logits(row)),
            })
            .collect();
        let pool = self.backend.pool();
        pool.for_chunks(&mut self.active, 1, 2, |i, seq| {
            let seq = &mut seq[0];
            let Some(logits) = views[i] else {
                return;
            };
            let allowed = seq.allowed.as_deref();
            seq.sampled = seq
                .sampler
                .sample(logits, &seq.tokens, seq.generated, allowed);
            seq.token_logprob = match (seq.params.logprobs, logits) {
                (Some(top), Logits::All(scores)) => Some(logprobs(scores, seq.sampled, top)),
                _ => None,
            };
        });
    }

    /// Act on the sampled tokens: stream them out, and finish sequences that are done.
    fn advance(&mut self) {
        let mut i = 0;
        while i < self.active.len() {
            let seq = &mut self.active[i];
            let mut finished = None;
            if seq.logits_row.is_some() {
                if !seq.prompt_committed {
                    // The prompt is done, so other requests can start using it.
                    seq.prompt_committed = true;
                    Self::commit_into(&mut self.cache, seq);
                }
                finished = match seq.params.embed {
                    true => {
                        let row = seq.logits_row.expect("checked above");
                        let hidden = self.backend.hidden(row).to_vec();
                        let _ = seq.events.send(Event::Embedding(hidden));
                        Some(FinishReason::Stop)
                    }
                    false => self.accept_token(i),
                };
            }
            match finished {
                Some(reason) => {
                    let mut seq = self.active.remove(i);
                    self.finish(&mut seq, reason);
                }
                None => i += 1,
            }
        }
    }

    /// Add the sampled token to its sequence. Returns why the sequence ended, if it has.
    fn accept_token(&mut self, index: usize) -> Option<FinishReason> {
        let seq = &mut self.active[index];
        let token = seq.sampled;
        seq.generated += 1;
        self.stats.generated_tokens.fetch_add(1, Relaxed);
        if let Some(logprob) = seq.token_logprob.take() {
            seq.pending_logprobs.push(logprob);
        }
        if !seq.params.ignore_eos && seq.params.stop_tokens.contains(&token) {
            return Some(FinishReason::Stop);
        }
        seq.tokens.push(token);
        let mut answering = true;
        let mut finished_thinking = false;
        match &mut seq.phase {
            Phase::MayThink => {
                let start = seq.params.thinking.as_ref().and_then(|t| t.start_token);
                if start == Some(token) {
                    // The grammar waits until the thinking is done, and then starts afresh.
                    seq.phase = Phase::Thinking {
                        recent: Vec::new(),
                        spent: 0,
                    };
                    seq.constraint = None;
                    seq.watch = None;
                    answering = false;
                } else {
                    seq.phase = Phase::Answering;
                }
            }
            Phase::Thinking { recent, spent } => {
                answering = false;
                *spent += 1;
                recent.extend_from_slice(&self.tokenizer.token_bytes(token));
                let end = seq.params.thinking.as_ref().map_or("", |t| t.end.as_str());
                let end = end.as_bytes();
                if !end.is_empty() && recent.windows(end.len()).any(|w| w == end) {
                    finished_thinking = true;
                } else {
                    let excess = recent.len().saturating_sub(end.len().max(2) - 1);
                    recent.drain(..excess);
                }
            }
            Phase::Answering => {}
        }
        if finished_thinking {
            seq.phase = Phase::Answering;
            (seq.constraint, seq.watch) = seq.params.start_grammar(false);
        }
        if !answering {
            // Thinking isn't held to the grammar.
        } else if let Some(constraint) = &mut seq.constraint {
            constraint.advance(token);
        } else if let Some(watch) = &mut seq.watch
            && let Some(written) = watch.push(
                &self.tokenizer.token_bytes(token),
                &seq.params.grammar_triggers,
            )
        {
            seq.watch = None;
            let grammar = seq.params.grammar.as_ref().expect("it has a trigger");
            let stop_tokens = seq.params.stop_tokens.iter().copied().collect();
            let mut constraint = grammar.start(false, stop_tokens);
            // If it can't fit, this wasn't what the trigger looked for after all.
            if constraint.feed(&written) {
                seq.constraint = Some(constraint);
            }
        }

        let hidden = seq.params.skip_special_tokens
            && self.tokenizer.is_special(token)
            && !seq.params.keep_tokens.contains(&token);
        let piece = match hidden {
            true => Vec::new(),
            false => self.tokenizer.token_bytes(token),
        };
        let (text, stopped) = seq.text.push(&piece);
        if !text.is_empty() {
            let logprobs = std::mem::take(&mut seq.pending_logprobs);
            let _ = seq.events.send(Event::Text { text, logprobs });
        }
        let limit = self.config.ctx_size.min(self.config.cache_tokens);
        if stopped {
            Some(FinishReason::Stop)
        } else if seq.generated >= seq.params.max_tokens || seq.tokens.len() >= limit {
            Some(FinishReason::Length)
        } else {
            None
        }
    }

    fn finish(&mut self, seq: &mut Sequence, reason: FinishReason) {
        let text = seq.text.finish();
        let logprobs = std::mem::take(&mut seq.pending_logprobs);
        if !text.is_empty() || !logprobs.is_empty() {
            let _ = seq.events.send(Event::Text { text, logprobs });
        }
        let _ = seq.events.send(Event::Done {
            reason,
            usage: seq.usage(),
            stop_string: seq.text.matched.take(),
        });
        self.retire(seq);
    }

    /// Hand what a sequence has computed to the prefix cache, and let go of what it held.
    fn retire(&mut self, seq: &mut Sequence) {
        Self::commit_into(&mut self.cache, seq);
        if let Some(pinned) = seq.pinned.take() {
            self.cache.release(pinned);
        }
        // Only whole pages go in the cache, so a page still being filled is simply given back.
        let page_size = self.cache.page_size();
        if !seq.slots.len().is_multiple_of(page_size) {
            let last = seq.slots[seq.slots.len() - 1];
            self.cache.free_page(last / page_size as u32);
        }
    }

    /// Give the sequence's computed tokens to the prefix cache, as far as whole pages go.
    fn commit_into(cache: &mut PrefixCache, seq: &mut Sequence) {
        if let Some(pinned) = seq.pinned {
            let whole = seq.slots.len() - seq.slots.len() % cache.page_size();
            seq.pinned = Some(cache.commit(pinned, &seq.tokens[..whole], &mut seq.slots[..whole]));
        }
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The log-probability of `token`, plus the `top` most likely tokens and theirs.
fn logprobs(logits: &[f32], token: u32, top: usize) -> TokenLogprob {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln() + max;
    let mut best: Vec<(u32, f32)> = Vec::new();
    if top > 0 {
        let mut all: Vec<(u32, f32)> = (0..logits.len() as u32)
            .zip(logits.iter().copied())
            .collect();
        let by_logit_desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if top < all.len() {
            all.select_nth_unstable_by(top - 1, by_logit_desc);
            all.truncate(top);
        }
        all.sort_unstable_by(by_logit_desc);
        best = all.into_iter().map(|(t, l)| (t, l - log_sum)).collect();
    }
    TokenLogprob {
        token,
        logprob: logits[token as usize] - log_sum,
        top: best,
    }
}

/// Watches a reply for the text that sets off its grammar.
struct TriggerWatch {
    /// The end of the reply so far, as much as a trigger could still be spread across.
    recent: Vec<u8>,
    /// Nothing but whitespace has been written so far.
    at_start: bool,
}

impl TriggerWatch {
    fn new() -> Self {
        Self {
            recent: Vec::new(),
            at_start: true,
        }
    }

    /// Add the next token. If a trigger has appeared, returns everything from there on.
    fn push(&mut self, bytes: &[u8], triggers: &[GrammarTrigger]) -> Option<Vec<u8>> {
        self.recent.extend_from_slice(bytes);
        let first_text = self.recent.iter().position(|b| !b.is_ascii_whitespace());
        let found = triggers
            .iter()
            .filter_map(|trigger| {
                let text = trigger.text.as_bytes();
                if trigger.at_start {
                    let start = first_text.filter(|_| self.at_start)?;
                    self.recent[start..].starts_with(text).then_some(start)
                } else {
                    self.recent.windows(text.len()).position(|w| w == text)
                }
            })
            .min();
        if let Some(start) = found {
            return Some(self.recent[start..].to_vec());
        }
        self.at_start &= first_text.is_none();
        let longest = triggers.iter().map(|t| t.text.len()).max().unwrap_or(1);
        let excess = self.recent.len().saturating_sub(longest - 1);
        self.recent.drain(..excess);
        None
    }
}

/// Turns token bytes into text for the client: holds back incomplete UTF-8 characters, and any
/// text that might turn out to be the start of a stop string.
struct TextStream {
    utf8: Vec<u8>,
    held: String,
    stops: Vec<String>,
    /// Leading spaces still to remove from the start of the reply.
    strip: usize,
    /// Everything has been decided; ignore further input.
    stopped: bool,
    /// The stop string that was found.
    matched: Option<String>,
}

impl TextStream {
    fn new(stops: Vec<String>, strip: usize) -> Self {
        Self {
            utf8: Vec::new(),
            held: String::new(),
            stops,
            strip,
            stopped: false,
            matched: None,
        }
    }

    /// Add bytes. Returns the text that's safe to send, and whether a stop string was found.
    fn push(&mut self, bytes: &[u8]) -> (String, bool) {
        let decoded = self.decode(bytes);
        self.hold_back(decoded)
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            return String::new();
        }
        let rest = String::from_utf8_lossy(&std::mem::take(&mut self.utf8)).into_owned();
        let (mut text, _) = self.hold_back(rest);
        text.push_str(&std::mem::take(&mut self.held));
        text
    }

    fn decode(&mut self, bytes: &[u8]) -> String {
        self.utf8.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.utf8) {
                Ok(text) => {
                    out.push_str(text);
                    self.utf8.clear();
                    return out;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.utf8[..valid]).expect("just checked"));
                    match err.error_len() {
                        None => {
                            // An incomplete character at the end: wait for the rest.
                            self.utf8.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.utf8.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    fn hold_back(&mut self, mut decoded: String) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        while self.strip > 0 && !decoded.is_empty() {
            if decoded.starts_with(' ') {
                decoded.remove(0);
                self.strip -= 1;
            } else {
                self.strip = 0;
            }
        }
        self.held.push_str(&decoded);
        let found = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()).map(|at| (at, stop)))
            .min_by_key(|&(at, _)| at);
        if let Some((at, stop)) = found {
            self.stopped = true;
            self.matched = Some(stop.clone());
            self.held.truncate(at);
            return (std::mem::take(&mut self.held), true);
        }
        let keep = self
            .stops
            .iter()
            .map(|stop| partial_match_len(&self.held, stop))
            .max()
            .unwrap_or(0);
        let tail = self.held.split_off(self.held.len() - keep);
        (std::mem::replace(&mut self.held, tail), false)
    }
}

/// The length of the longest end of `text` that's the beginning of `stop`.
pub(crate) fn partial_match_len(text: &str, stop: &str) -> usize {
    (1..stop.len().min(text.len() + 1))
        .rev()
        .find(|&n| stop.is_char_boundary(n) && text.ends_with(&stop[..n]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triggers_are_found_across_tokens() {
        let triggers = [GrammarTrigger {
            text: "<call>".to_string(),
            at_start: false,
        }];
        let mut watch = TriggerWatch::new();
        assert_eq!(watch.push(b"some <ca", &triggers), None);
        assert_eq!(watch.push(b"ll>rest", &triggers).unwrap(), b"<call>rest");
    }

    #[test]
    fn triggers_at_the_start_only_count_there() {
        let triggers = [GrammarTrigger {
            text: "{".to_string(),
            at_start: true,
        }];
        let mut watch = TriggerWatch::new();
        assert_eq!(watch.push(b" \n", &triggers), None);
        assert_eq!(watch.push(b"{\"", &triggers).unwrap(), b"{\"");
        let mut watch = TriggerWatch::new();
        assert_eq!(watch.push(b"ok", &triggers), None);
        assert_eq!(watch.push(b"{", &triggers), None);
    }

    fn stream(stops: &[&str]) -> TextStream {
        TextStream::new(stops.iter().map(|s| s.to_string()).collect(), 0)
    }

    #[test]
    fn text_waits_for_whole_characters() {
        let mut s = stream(&[]);
        let euro = "€".as_bytes();
        assert_eq!(s.push(&euro[..1]), (String::new(), false));
        assert_eq!(s.push(&euro[1..]), ("€".to_string(), false));
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn stop_strings_end_the_text() {
        let mut s = stream(&["END"]);
        assert_eq!(s.push(b"hello E"), ("hello ".to_string(), false));
        assert_eq!(s.push(b"N"), (String::new(), false));
        assert_eq!(s.push(b"Dxyz"), (String::new(), true));
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn false_alarms_are_released() {
        let mut s = stream(&["END"]);
        assert_eq!(s.push(b"E"), (String::new(), false));
        assert_eq!(s.push(b"Nope"), ("ENope".to_string(), false));
        let mut s = stream(&["END"]);
        assert_eq!(s.push(b"EN"), (String::new(), false));
        assert_eq!(s.finish(), "EN");
    }

    #[test]
    fn earliest_stop_string_wins() {
        let mut s = stream(&["bb", "a"]);
        assert_eq!(s.push(b"xxbbyya"), ("xx".to_string(), true));
        assert_eq!(s.matched.as_deref(), Some("bb"));
    }

    #[test]
    fn leading_spaces_can_be_stripped() {
        let mut s = TextStream::new(Vec::new(), 1);
        assert_eq!(s.push(b" Hello"), ("Hello".to_string(), false));
        assert_eq!(s.push(b" world"), (" world".to_string(), false));
    }
}

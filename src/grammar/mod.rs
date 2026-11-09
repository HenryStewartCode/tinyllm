//! Structured output: making a reply follow a grammar, such as a JSON schema or a regex.
//!
//! Before each token is sampled, every token that can't continue the grammar is ruled out, so
//! the reply is valid by construction. There are three parts:
//!
//! 1. Every kind of constraint is compiled to one form: a context-free grammar over *bytes*
//!    (`Builder`). Bytes, because tokens can split a character.
//! 2. An `Automaton` tracks which parses are still possible after each byte. Grammars can
//!    nest, so this is a pushdown automaton, and ambiguity means several parses at once.
//!    Each distinct set of parses becomes a numbered state the first time it's seen, with its
//!    transitions remembered, which makes it behave like a very fast DFA after warming up.
//! 3. The vocabulary is kept as a trie of byte strings. Walking it alongside the automaton
//!    finds all allowed tokens in one sweep, because tokens that share a prefix share the work,
//!    and whole branches are discarded at once. The result is remembered for each state.

mod ebnf;
mod regex;
mod schema;

use crate::tokenizer::Tokenizer;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// A set of token ids.
#[derive(Clone)]
pub struct TokenSet {
    bits: Vec<u64>,
}

impl TokenSet {
    pub fn new(vocab: usize) -> Self {
        Self {
            bits: vec![0; vocab.div_ceil(64)],
        }
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.bits
            .get(id as usize / 64)
            .is_some_and(|word| word >> (id % 64) & 1 == 1)
    }

    pub fn insert(&mut self, id: u32) {
        if let Some(word) = self.bits.get_mut(id as usize / 64) {
            *word |= 1 << (id % 64);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&word| word == 0)
    }

    fn fill(&mut self) {
        self.bits.fill(u64::MAX);
    }
}

// ---------------------------------------------------------------------------------------------
// Grammars
// ---------------------------------------------------------------------------------------------

/// A set of byte values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) struct ByteSet([u64; 4]);

impl ByteSet {
    pub(crate) fn range(start: u8, end: u8) -> Self {
        let mut set = Self::default();
        set.insert_range(start, end);
        set
    }

    pub(crate) fn insert_range(&mut self, start: u8, end: u8) {
        for byte in start..=end {
            self.0[byte as usize / 64] |= 1 << (byte % 64);
        }
    }

    pub(crate) fn remove(&mut self, byte: u8) {
        self.0[byte as usize / 64] &= !(1 << (byte % 64));
    }

    #[inline]
    fn contains(&self, byte: u8) -> bool {
        self.0[byte as usize / 64] >> (byte % 64) & 1 == 1
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0 == [0; 4]
    }
}

pub(crate) type RuleId = u32;

/// One position in a rule: match one byte, or match a whole other rule.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Item {
    Set(ByteSet),
    Call(RuleId),
}

/// A sequence of items. A rule is a list of alternative sequences.
pub(crate) type Seq = Vec<Item>;

/// Guards against a pattern like `(a{1000}){1000}` using unlimited memory.
const MAX_GRAMMAR_ITEMS: usize = 1 << 18;

/// Assembles a grammar.
#[derive(Default)]
pub(crate) struct Builder {
    rules: Vec<Vec<Seq>>,
    /// Rules already made, so that identical ones are shared.
    interned: HashMap<Vec<Seq>, RuleId>,
    items: usize,
}

impl Builder {
    /// A rule to be filled in later with [`define`](Self::define), for recursion.
    pub(crate) fn declare(&mut self) -> RuleId {
        self.rules.push(Vec::new());
        (self.rules.len() - 1) as RuleId
    }

    pub(crate) fn define(&mut self, rule: RuleId, alternatives: Vec<Seq>) -> Result<()> {
        self.items += alternatives.iter().map(Vec::len).sum::<usize>();
        if self.items > MAX_GRAMMAR_ITEMS {
            bail!("the grammar is too large");
        }
        self.rules[rule as usize] = alternatives;
        Ok(())
    }

    pub(crate) fn rule(&mut self, alternatives: Vec<Seq>) -> Result<RuleId> {
        if let Some(&rule) = self.interned.get(&alternatives) {
            return Ok(rule);
        }
        let rule = self.declare();
        self.define(rule, alternatives.clone())?;
        self.interned.insert(alternatives, rule);
        Ok(rule)
    }

    pub(crate) fn call(&mut self, alternatives: Vec<Seq>) -> Result<Item> {
        Ok(Item::Call(self.rule(alternatives)?))
    }

    /// Match this sequence of items, or nothing.
    pub(crate) fn optional(&mut self, seq: Seq) -> Result<Item> {
        self.call(vec![seq, Vec::new()])
    }

    /// Zero or more repeats. Recursing at the very end of a rule uses no stack.
    pub(crate) fn star(&mut self, seq: Seq) -> Result<Item> {
        let rule = self.declare();
        let mut body = seq;
        body.push(Item::Call(rule));
        self.define(rule, vec![body, Vec::new()])?;
        Ok(Item::Call(rule))
    }

    /// Between `min` and `max` repeats (`None` = no upper limit).
    pub(crate) fn repeat(&mut self, seq: Seq, min: usize, max: Option<usize>) -> Result<Seq> {
        if max.is_some_and(|max| max < min) {
            bail!("a repetition's maximum is less than its minimum");
        }
        // Sharing the body as one rule keeps big repeat counts small.
        let body = if seq.len() == 1 {
            seq[0]
        } else {
            self.call(vec![seq])?
        };
        let count = max.unwrap_or(min);
        if count > MAX_GRAMMAR_ITEMS {
            bail!("the grammar is too large");
        }
        let mut result = vec![body; min];
        match max {
            None => result.push(self.star(vec![body])?),
            Some(max) => {
                // x{0,3} becomes (x (x (x)?)?)?
                let mut tail: Seq = Vec::new();
                for _ in min..max {
                    let mut inner = vec![body];
                    inner.extend(tail);
                    tail = vec![self.optional(inner)?];
                }
                result.extend(tail);
            }
        }
        self.items += result.len();
        if self.items > MAX_GRAMMAR_ITEMS {
            bail!("the grammar is too large");
        }
        Ok(result)
    }

    /// Refuse rules that begin by calling themselves, which would never finish expanding.
    fn check_left_recursion(&self) -> Result<()> {
        let mut nullable = vec![false; self.rules.len()];
        loop {
            let mut changed = false;
            for (r, alternatives) in self.rules.iter().enumerate() {
                if !nullable[r]
                    && alternatives.iter().any(|seq| {
                        seq.iter().all(|item| match item {
                            Item::Call(c) => nullable[*c as usize],
                            Item::Set(_) => false,
                        })
                    })
                {
                    nullable[r] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // Which rules can each rule start with?
        let starts: Vec<HashSet<RuleId>> = self
            .rules
            .iter()
            .map(|alternatives| {
                let mut calls = HashSet::new();
                for seq in alternatives {
                    for item in seq {
                        match *item {
                            Item::Call(c) => {
                                calls.insert(c);
                                if !nullable[c as usize] {
                                    break;
                                }
                            }
                            Item::Set(_) => break,
                        }
                    }
                }
                calls
            })
            .collect();
        // Depth-first search for a cycle, without recursing, since rule chains can be long.
        #[derive(Clone, Copy, PartialEq)]
        enum Mark {
            New,
            InProgress,
            Done,
        }
        let starts: Vec<Vec<RuleId>> = starts
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect();
        let mut marks = vec![Mark::New; self.rules.len()];
        for root in 0..self.rules.len() {
            if marks[root] != Mark::New {
                continue;
            }
            marks[root] = Mark::InProgress;
            let mut path = vec![(root, 0)];
            while let Some(&mut (rule, ref mut next)) = path.last_mut() {
                let Some(&callee) = starts[rule].get(*next) else {
                    marks[rule] = Mark::Done;
                    path.pop();
                    continue;
                };
                *next += 1;
                match marks[callee as usize] {
                    Mark::InProgress => {
                        bail!("the grammar is left-recursive, which isn't supported")
                    }
                    Mark::Done => {}
                    Mark::New => {
                        marks[callee as usize] = Mark::InProgress;
                        path.push((callee as usize, 0));
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn literal(bytes: &[u8]) -> Seq {
    bytes
        .iter()
        .map(|&b| Item::Set(ByteSet::range(b, b)))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The vocabulary
// ---------------------------------------------------------------------------------------------

struct TrieNode {
    children: Vec<(u8, u32)>,
    tokens: Vec<u32>,
}

/// Every ordinary token's bytes, as a trie.
pub struct Vocabulary {
    nodes: Vec<TrieNode>,
    size: usize,
    /// Each token's bytes, for following a token through the automaton.
    bytes: Vec<Box<[u8]>>,
}

impl Vocabulary {
    pub fn new(tokenizer: &Tokenizer, size: usize) -> Self {
        Self::with_markers(tokenizer, size, &[])
    }

    /// A vocabulary in which these special tokens can be written like ordinary ones, for
    /// grammars that have to include them.
    pub fn with_markers(tokenizer: &Tokenizer, size: usize, markers: &[u32]) -> Self {
        let tokens = (0..size as u32).map(|id| {
            if tokenizer.is_special(id) && !markers.contains(&id) {
                Vec::new()
            } else {
                tokenizer.token_bytes(id)
            }
        });
        Self::from_tokens(tokens.collect())
    }

    /// Tokens with no bytes (special ones) can't appear in the middle of structured text.
    fn from_tokens(tokens: Vec<Vec<u8>>) -> Self {
        let mut nodes = vec![TrieNode {
            children: Vec::new(),
            tokens: Vec::new(),
        }];
        for (id, bytes) in tokens.iter().enumerate() {
            if bytes.is_empty() {
                continue;
            }
            let mut at = 0;
            for &byte in bytes {
                at = match nodes[at].children.iter().find(|(b, _)| *b == byte) {
                    Some(&(_, child)) => child as usize,
                    None => {
                        let child = nodes.len();
                        nodes.push(TrieNode {
                            children: Vec::new(),
                            tokens: Vec::new(),
                        });
                        nodes[at].children.push((byte, child as u32));
                        child
                    }
                };
            }
            nodes[at].tokens.push(id as u32);
        }
        Self {
            nodes,
            size: tokens.len(),
            bytes: tokens.into_iter().map(Vec::into_boxed_slice).collect(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The automaton
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Element {
    Set(u32),
    Call(RuleId),
    End,
}

/// A grammar flattened into one array of elements, which is what the automaton walks through.
struct Program {
    elements: Vec<Element>,
    sets: Vec<ByteSet>,
    /// For each rule, where each alternative starts.
    starts: Vec<Vec<u32>>,
    root: RuleId,
}

impl Program {
    fn new(builder: &Builder, root: RuleId) -> Self {
        let mut program = Program {
            elements: Vec::new(),
            sets: Vec::new(),
            starts: Vec::with_capacity(builder.rules.len()),
            root,
        };
        let mut set_ids: HashMap<ByteSet, u32> = HashMap::new();
        for alternatives in &builder.rules {
            let mut starts = Vec::with_capacity(alternatives.len());
            for seq in alternatives {
                starts.push(program.elements.len() as u32);
                for item in seq {
                    program.elements.push(match *item {
                        Item::Call(rule) => Element::Call(rule),
                        Item::Set(set) => Element::Set(*set_ids.entry(set).or_insert_with(|| {
                            program.sets.push(set);
                            (program.sets.len() - 1) as u32
                        })),
                    });
                }
                program.elements.push(Element::End);
            }
            program.starts.push(starts);
        }
        program
    }
}

/// The state with no possible parses left.
const DEAD: u32 = 0;
const UNKNOWN: u32 = u32::MAX;
/// The bottom of every stack, and what marks a finished parse in a state's list.
const EMPTY_STACK: u32 = 0;

struct State {
    /// Stacks whose top is a byte set (or `EMPTY_STACK` for a parse that could end here).
    stacks: Box<[u32]>,
    next: Box<[u32; 256]>,
}

/// Start again with an empty automaton past this many states, to bound memory use.
const MAX_STATES: usize = 1 << 17;
const MAX_MASKS: usize = 4096;
const MAX_EXPANSION_DEPTH: usize = 4096;

struct Automaton {
    program: Arc<Program>,
    vocabulary: Arc<Vocabulary>,
    /// Stacks as linked lists, each element being (position, rest of stack), so that they share
    /// their tails. Numbering them makes comparing stacks free.
    stack_nodes: Vec<(u32, u32)>,
    stack_ids: HashMap<(u32, u32), u32>,
    states: Vec<State>,
    state_ids: HashMap<Box<[u32]>, u32>,
    masks: HashMap<(u32, bool), Arc<TokenSet>>,
    start: u32,
    scratch: Vec<u32>,
    visited: HashSet<(u32, u32)>,
}

impl Automaton {
    fn new(program: Arc<Program>, vocabulary: Arc<Vocabulary>) -> Self {
        let mut automaton = Self {
            program,
            vocabulary,
            stack_nodes: vec![(u32::MAX, u32::MAX)],
            stack_ids: HashMap::new(),
            states: Vec::new(),
            state_ids: HashMap::new(),
            masks: HashMap::new(),
            start: DEAD,
            scratch: Vec::new(),
            visited: HashSet::new(),
        };
        automaton.intern_state(Vec::new()); // DEAD
        let root = automaton.program.root as usize;
        let mut out = Vec::new();
        for i in 0..automaton.program.starts[root].len() {
            let start = automaton.program.starts[root][i];
            automaton.expand(start, EMPTY_STACK, &mut out, 0);
        }
        automaton.visited.clear();
        automaton.start = automaton.intern_state(out);
        automaton
    }

    fn intern_stack(&mut self, position: u32, rest: u32) -> u32 {
        *self.stack_ids.entry((position, rest)).or_insert_with(|| {
            self.stack_nodes.push((position, rest));
            (self.stack_nodes.len() - 1) as u32
        })
    }

    fn intern_state(&mut self, mut stacks: Vec<u32>) -> u32 {
        stacks.sort_unstable();
        stacks.dedup();
        if let Some(&id) = self.state_ids.get(stacks.as_slice()) {
            return id;
        }
        let id = self.states.len() as u32;
        // Nothing leaves the dead state.
        let filler = if id == DEAD { DEAD } else { UNKNOWN };
        let stacks = stacks.into_boxed_slice();
        self.state_ids.insert(stacks.clone(), id);
        self.states.push(State {
            stacks,
            next: Box::new([filler; 256]),
        });
        id
    }

    /// Find every way to get from `position` to a byte set, following calls and returns.
    fn expand(&mut self, position: u32, rest: u32, out: &mut Vec<u32>, depth: usize) {
        // Only a pathological grammar (thousands of nested optional parts) gets this deep.
        // Dropping that parse is better than overflowing the stack.
        if depth > MAX_EXPANSION_DEPTH || !self.visited.insert((position, rest)) {
            return;
        }
        match self.program.elements[position as usize] {
            Element::Set(_) => {
                let stack = self.intern_stack(position, rest);
                out.push(stack);
            }
            Element::Call(rule) => {
                // A call in last position has nothing to come back to, so repetition (which
                // calls itself at the end) doesn't make the stack grow.
                let next = position + 1;
                let rest = match self.program.elements[next as usize] {
                    Element::End => rest,
                    _ => self.intern_stack(next, rest),
                };
                for i in 0..self.program.starts[rule as usize].len() {
                    let start = self.program.starts[rule as usize][i];
                    self.expand(start, rest, out, depth + 1);
                }
            }
            Element::End if rest == EMPTY_STACK => out.push(EMPTY_STACK),
            Element::End => {
                let (position, rest) = self.stack_nodes[rest as usize];
                self.expand(position, rest, out, depth + 1);
            }
        }
    }

    fn step(&mut self, state: u32, byte: u8) -> u32 {
        let next = self.states[state as usize].next[byte as usize];
        if next != UNKNOWN {
            return next;
        }
        let mut out = std::mem::take(&mut self.scratch);
        out.clear();
        for i in 0..self.states[state as usize].stacks.len() {
            let stack = self.states[state as usize].stacks[i];
            if stack == EMPTY_STACK {
                continue;
            }
            let (position, rest) = self.stack_nodes[stack as usize];
            let Element::Set(set) = self.program.elements[position as usize] else {
                unreachable!("a state's stacks always have a byte set on top");
            };
            if self.program.sets[set as usize].contains(byte) {
                self.expand(position + 1, rest, &mut out, 0);
            }
        }
        self.visited.clear();
        let id = if out.is_empty() {
            DEAD
        } else {
            self.intern_state(out.clone())
        };
        self.scratch = out;
        self.states[state as usize].next[byte as usize] = id;
        id
    }

    fn is_accepting(&self, state: u32) -> bool {
        self.states[state as usize].stacks.first() == Some(&EMPTY_STACK)
    }

    /// Every token that can come next. With `ignore_space`, a token's leading space doesn't
    /// count, for the first token of a reply from a tokenizer where it gets removed.
    fn allowed(&mut self, state: u32, ignore_space: bool) -> Arc<TokenSet> {
        if let Some(mask) = self.masks.get(&(state, ignore_space)) {
            return mask.clone();
        }
        let vocabulary = self.vocabulary.clone();
        let mut mask = TokenSet::new(vocabulary.size);
        for &(byte, child) in &vocabulary.nodes[0].children {
            let child = child as usize;
            let next = if ignore_space && byte == b' ' {
                // This space won't reach the client, so it doesn't move the parse along.
                state
            } else {
                self.step(state, byte)
            };
            if next != DEAD {
                for &token in &vocabulary.nodes[child].tokens {
                    mask.insert(token);
                }
                self.walk(&vocabulary, child, next, &mut mask);
            }
        }
        let mask = Arc::new(mask);
        if self.masks.len() >= MAX_MASKS {
            self.masks.clear();
        }
        self.masks.insert((state, ignore_space), mask.clone());
        mask
    }

    fn walk(&mut self, vocabulary: &Vocabulary, node: usize, state: u32, mask: &mut TokenSet) {
        for &(byte, child) in &vocabulary.nodes[node].children {
            let next = self.step(state, byte);
            if next == DEAD {
                continue;
            }
            for &token in &vocabulary.nodes[child as usize].tokens {
                mask.insert(token);
            }
            self.walk(vocabulary, child as usize, next, mask);
        }
    }
}

/// Text made of tool calls, each one a fixed frame around JSON arguments.
#[derive(Serialize, Deserialize)]
pub struct CallsGrammar {
    /// One of these comes first.
    pub opening: Vec<String>,
    /// The tools there are to call.
    pub calls: Vec<CallFrame>,
    /// What goes between calls.
    pub separator: String,
    /// What comes after the last call.
    pub closing: String,
    /// How many calls there can be. There's always at least one.
    pub max_calls: Option<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct CallFrame {
    pub before: String,
    /// What the arguments have to match.
    pub schema: serde_json::Value,
    /// Whether a schema that can't be followed is an error, rather than allowing any object.
    pub strict: bool,
    pub after: String,
}

/// A compiled grammar, which can be shared by any number of requests.
pub struct Grammar {
    program: Arc<Program>,
    vocabulary: Arc<Vocabulary>,
    automaton: Mutex<Arc<Mutex<Automaton>>>,
}

impl Grammar {
    /// Compile a constraint: `json` (any JSON object), `schema:<JSON schema>`, `regex:<pattern>`,
    /// `ebnf:<grammar>` or `calls:<CallsGrammar as JSON>`.
    pub fn compile(spec: &str, vocabulary: Arc<Vocabulary>) -> Result<Self> {
        let mut builder = Builder::default();
        let root = if spec == "json" {
            schema::compile_any_object(&mut builder)?
        } else if let Some(text) = spec.strip_prefix("schema:") {
            schema::compile(&serde_json::from_str(text)?, &mut builder)?
        } else if let Some(pattern) = spec.strip_prefix("regex:") {
            let seq = regex::compile(pattern, &mut builder, false)?;
            builder.rule(vec![seq])?
        } else if let Some(text) = spec.strip_prefix("ebnf:") {
            ebnf::compile(text, &mut builder)?
        } else if let Some(text) = spec.strip_prefix("calls:") {
            schema::compile_calls(&serde_json::from_str(text)?, &mut builder)?
        } else {
            bail!("unknown kind of constraint");
        };
        builder.check_left_recursion()?;
        let program = Arc::new(Program::new(&builder, root));
        let automaton = Automaton::new(program.clone(), vocabulary.clone());
        Ok(Self {
            program,
            vocabulary,
            automaton: Mutex::new(Arc::new(Mutex::new(automaton))),
        })
    }

    /// Begin at the start of the grammar. `strip_leading_space` says the reply's first space
    /// will be removed; `stop_tokens` are allowed wherever the grammar could end.
    pub fn start(&self, strip_leading_space: bool, stop_tokens: Vec<u32>) -> Constraint {
        let mut current = self.automaton.lock().unwrap();
        if current.lock().unwrap().states.len() >= MAX_STATES {
            // Requests already running keep the old one until they finish.
            *current = Arc::new(Mutex::new(Automaton::new(
                self.program.clone(),
                self.vocabulary.clone(),
            )));
        }
        let automaton = current.clone();
        let state = automaton.lock().unwrap().start;
        Constraint {
            automaton,
            state,
            ignore_space: strip_leading_space,
            stop_tokens,
        }
    }
}

/// One request's position in a grammar.
pub struct Constraint {
    automaton: Arc<Mutex<Automaton>>,
    state: u32,
    ignore_space: bool,
    stop_tokens: Vec<u32>,
}

impl Constraint {
    /// The tokens that may come next.
    pub fn allowed_tokens(&mut self) -> Arc<TokenSet> {
        let mut automaton = self.automaton.lock().unwrap();
        let mask = automaton.allowed(self.state, self.ignore_space);
        let at_end = automaton.is_accepting(self.state);
        // The mask has no way to represent stop tokens (they have no bytes), and it's shared
        // by requests that may each have different ones.
        if !at_end && !mask.is_empty() {
            return mask;
        }
        let mut mask = (*mask).clone();
        for &token in &self.stop_tokens {
            mask.insert(token);
        }
        if mask.is_empty() {
            // Nothing can continue, which only happens if the vocabulary can't spell what the
            // grammar needs and the model has no stop token. Better to emit junk than crash.
            mask.fill();
        }
        Arc::new(mask)
    }

    /// Move past text that's already been written. Returns false if it can't be part of the
    /// grammar.
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        let mut automaton = self.automaton.lock().unwrap();
        for &byte in bytes {
            self.state = automaton.step(self.state, byte);
        }
        self.state != DEAD
    }

    /// Move past a token that was generated.
    pub fn advance(&mut self, token: u32) {
        let mut automaton = self.automaton.lock().unwrap();
        let vocabulary = automaton.vocabulary.clone();
        let Some(bytes) = vocabulary.bytes.get(token as usize) else {
            return;
        };
        let mut bytes: &[u8] = bytes;
        if self.ignore_space && !bytes.is_empty() {
            self.ignore_space = false;
            // The client won't see this space.
            bytes = bytes.strip_prefix(b" ").unwrap_or(bytes);
        }
        for &byte in bytes {
            self.state = automaton.step(self.state, byte);
        }
    }

    #[cfg(test)]
    fn is_accepting(&self) -> bool {
        self.automaton.lock().unwrap().is_accepting(self.state)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Does the grammar accept exactly this text?
    pub(crate) fn matches(builder: &Builder, root: RuleId, text: &str) -> bool {
        builder.check_left_recursion().unwrap();
        let program = Arc::new(Program::new(builder, root));
        let vocabulary = Arc::new(Vocabulary::from_tokens(Vec::new()));
        let mut automaton = Automaton::new(program, vocabulary);
        let mut state = automaton.start;
        for &byte in text.as_bytes() {
            state = automaton.step(state, byte);
        }
        automaton.is_accepting(state)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::matches;
    use super::*;

    fn abc_grammar() -> (Builder, RuleId) {
        let mut b = Builder::default();
        let seq = regex::compile("a(bc)*d?", &mut b, false).unwrap();
        let root = b.rule(vec![seq]).unwrap();
        (b, root)
    }

    #[test]
    fn automaton_follows_a_grammar() {
        let (b, root) = abc_grammar();
        assert!(matches(&b, root, "a"));
        assert!(matches(&b, root, "abcbcd"));
        assert!(!matches(&b, root, "ab"));
        assert!(!matches(&b, root, "ad d"));
    }

    fn vocabulary() -> Arc<Vocabulary> {
        let tokens = ["a", "b", "c", "bc", "d", "ab", "x", "ax", ""];
        Arc::new(Vocabulary::from_tokens(
            tokens.iter().map(|t| t.as_bytes().to_vec()).collect(),
        ))
    }

    fn allowed_list(constraint: &mut Constraint, vocab: u32) -> Vec<u32> {
        let mask = constraint.allowed_tokens();
        (0..vocab).filter(|&t| mask.contains(t)).collect()
    }

    #[test]
    fn masks_contain_exactly_the_tokens_that_fit() {
        let grammar = Grammar::compile("regex:a(bc)*d?", vocabulary()).unwrap();
        let mut c = grammar.start(false, vec![8]);
        // At the start only tokens beginning with "a" that keep going: "a", "ab".
        assert_eq!(allowed_list(&mut c, 9), [0, 5]);
        c.advance(0); // a
        assert!(c.is_accepting());
        // "b" alone is a valid prefix of "bc". The stop token (8) is allowed here.
        assert_eq!(allowed_list(&mut c, 9), [1, 3, 4, 8]);
        c.advance(1); // b: now only c fits
        assert!(!c.is_accepting());
        assert_eq!(allowed_list(&mut c, 9), [2]);
        c.advance(2); // c
        c.advance(4); // d: nothing can follow, so only the stop token is left
        assert_eq!(allowed_list(&mut c, 9), [8]);
    }

    #[test]
    fn a_leading_space_can_be_ignored_once() {
        let vocabulary = Arc::new(Vocabulary::from_tokens(
            [" a", "a", " b", "b", " ", "  a"]
                .iter()
                .map(|t| t.as_bytes().to_vec())
                .collect(),
        ));
        let grammar = Grammar::compile("regex:ab", vocabulary).unwrap();
        let mut c = grammar.start(true, vec![]);
        assert_eq!(
            allowed_list(&mut c, 6),
            [0, 1, 4],
            "' ' alone is harmless, '  a' isn't"
        );
        c.advance(0); // " a", whose space will be removed
        assert_eq!(allowed_list(&mut c, 6), [3]);
    }

    fn call_grammar() -> String {
        let frame = |name: &str, schema: serde_json::Value| CallFrame {
            before: format!("<c>{name}:"),
            schema,
            strict: false,
            after: "</c>".to_string(),
        };
        let spec = CallsGrammar {
            opening: vec!["".to_string(), "<|t|>".to_string()],
            calls: vec![
                frame(
                    "a",
                    serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
                ),
                frame("b", serde_json::json!({})),
            ],
            separator: ";".to_string(),
            closing: "!".to_string(),
            max_calls: None,
        };
        format!("calls:{}", serde_json::to_string(&spec).unwrap())
    }

    fn accepts(grammar: &Grammar, text: &str) -> bool {
        let mut c = grammar.start(false, vec![]);
        c.feed(text.as_bytes()) && c.is_accepting()
    }

    #[test]
    fn tool_calls_follow_their_frames_and_schemas() {
        let grammar = Grammar::compile(&call_grammar(), vocabulary()).unwrap();
        for good in [
            "<c>a:{\"x\": 1}</c>!",
            "<|t|><c>a:{\"x\": 1}</c>!",
            "<c>b:{}</c>!",
            "<c>b:{\"anything\": [1, {}]}</c>;<c>a:{\"x\": 2}</c>!",
        ] {
            assert!(accepts(&grammar, good), "{good}");
        }
        for bad in [
            "",
            "!",
            "<c>a:{}</c>!",
            "<c>a:{\"x\": \"y\"}</c>!",
            "<c>c:{}</c>!",
            "<c>b:[]</c>!",
            "<c>b:{}</c>",
            "<c>b:{}</c>;!",
        ] {
            assert!(!accepts(&grammar, bad), "{bad}");
        }
    }

    #[test]
    fn the_number_of_calls_can_be_limited() {
        let spec = CallsGrammar {
            opening: vec!["".to_string()],
            calls: vec![CallFrame {
                before: "c".to_string(),
                schema: serde_json::json!({"type": "object", "properties": {}, "additionalProperties": false}),
                strict: false,
                after: String::new(),
            }],
            separator: ",".to_string(),
            closing: String::new(),
            max_calls: Some(1),
        };
        let spec = format!("calls:{}", serde_json::to_string(&spec).unwrap());
        let grammar = Grammar::compile(&spec, vocabulary()).unwrap();
        assert!(accepts(&grammar, "c{}"));
        assert!(!accepts(&grammar, "c{},c{}"));
    }

    #[test]
    fn a_schema_that_cant_be_followed_only_matters_if_strict() {
        let spec = |strict: bool| {
            let spec = CallsGrammar {
                opening: vec!["".to_string()],
                calls: vec![CallFrame {
                    before: "c".to_string(),
                    schema: serde_json::json!({"type": "varchar"}),
                    strict,
                    after: String::new(),
                }],
                separator: String::new(),
                closing: String::new(),
                max_calls: Some(1),
            };
            format!("calls:{}", serde_json::to_string(&spec).unwrap())
        };
        let loose = Grammar::compile(&spec(false), vocabulary()).unwrap();
        assert!(accepts(&loose, "c{\"any\": 1}"));
        assert!(!accepts(&loose, "c1"), "still has to be an object");
        assert!(Grammar::compile(&spec(true), vocabulary()).is_err());
    }

    #[test]
    fn feeding_reports_dead_ends() {
        let grammar = Grammar::compile("regex:ab", vocabulary()).unwrap();
        let mut c = grammar.start(false, vec![]);
        assert!(c.feed(b"a"));
        assert!(!c.feed(b"x"));
    }

    #[test]
    fn absurd_grammars_are_refused_or_survived() {
        // Too big to build.
        assert!(Grammar::compile("regex:a{300000}", vocabulary()).is_err());
        // A long chain of optional parts that can all be empty mustn't overflow the stack.
        let grammar = Grammar::compile("regex:(a?){0,50000}b", vocabulary());
        if let Ok(grammar) = grammar {
            let mut c = grammar.start(false, vec![]);
            assert!(!allowed_list(&mut c, 9).is_empty());
        }
    }

    #[test]
    fn left_recursion_through_empty_rules_is_refused() {
        let spec = "ebnf:root ::= a root \"x\" | \"y\"\na ::= \"z\"?";
        let err = Grammar::compile(spec, vocabulary())
            .err()
            .expect("should be refused");
        assert!(err.to_string().contains("left-recursive"));
    }

    #[test]
    fn left_recursion_is_refused() {
        let err = Grammar::compile("ebnf:root ::= root \"a\" | \"b\"", vocabulary())
            .err()
            .expect("should be refused");
        assert!(err.to_string().contains("left-recursive"));
    }
}

//! Choosing the next token from the model's logits.
//!
//! Penalties and biases adjust the logits first, and a grammar (for structured output) can rule
//! tokens out entirely. Greedy decoding (temperature 0) then takes the most likely token.
//! Otherwise the logits are divided by the temperature and turned into probabilities, the
//! candidates are cut down by `top_k`, `top_p` and `min_p`, and one is drawn at random.

use crate::backend::Logits;
use crate::grammar::TokenSet;
use crate::model::LogitUse;

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Keep only the k most likely tokens (0 = no limit).
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities add up to p.
    pub top_p: f32,
    /// Drop tokens less than `min_p` times as likely as the most likely one.
    pub min_p: f32,
    /// Hugging Face style: divide or multiply the logits of recently seen tokens. 1 = off.
    pub repeat_penalty: f32,
    /// OpenAI style: subtract this much per previous use in the reply. 0 = off.
    pub frequency_penalty: f32,
    /// OpenAI style: subtract this much if used at all in the reply. 0 = off.
    pub presence_penalty: f32,
    /// Added to the logits of specific tokens.
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

const REPEAT_LAST_N: usize = 64;

/// How far behind the most likely token (in logits at temperature 1) a token can be and still
/// count. e^-32 is about 1 part in 10^14, which a 24-bit random number could never pick.
const NEGLIGIBLE_LOGIT_GAP: f32 = 32.0;

pub struct Sampler {
    params: SamplingParams,
    vocab: usize,
    candidates: Vec<(f32, u32)>,
    seen: Vec<u32>,
    /// A copy of the logits with penalties and biases applied, when there are any.
    adjusted: Vec<f32>,
}

impl Sampler {
    pub fn new(params: SamplingParams, vocab: usize) -> Self {
        Self {
            params,
            vocab,
            candidates: Vec::new(),
            seen: Vec::new(),
            adjusted: Vec::new(),
        }
    }

    /// How far behind the best score a token can be and still have a chance of being chosen.
    pub fn logit_gap(&self) -> f32 {
        let params = &self.params;
        let mut gap = NEGLIGIBLE_LOGIT_GAP;
        if params.top_p < 1.0 {
            // If token j is at least ln(vocab / (1 - p)) behind the best, it and everything after
            // it add up to at most 1 - p of the probability. So the tokens ahead of j already
            // make up at least p, and top-p would never get as far as j.
            gap = gap.min((self.vocab as f32 / (1.0 - params.top_p)).ln());
        }
        if params.min_p > 0.0 {
            gap = gap.min(-params.min_p.ln());
        }
        gap * params.temperature.max(0.0)
    }

    /// How many of the top tokens could possibly be chosen (0 = no limit).
    pub fn top_limit(&self) -> usize {
        match self.params.top_k {
            0 => 0,
            // Penalties can push some of the best tokens down, letting others in.
            k => {
                k + if self.needs_adjusting() {
                    REPEAT_LAST_N
                } else {
                    0
                }
            }
        }
    }

    /// Whether choosing a token needs every token's score, as opposed to just the best few.
    pub fn needs_all_logits(&self) -> bool {
        // A positive bias could lift a token that's currently far behind.
        self.params.logit_bias.iter().any(|&(_, bias)| bias > 0.0)
    }

    /// What choosing token number `generated` of the reply needs from the model's scores.
    pub fn logit_use(&self, generated: usize) -> LogitUse {
        let params = &self.params;
        if self.needs_all_logits() {
            LogitUse::All
        } else if params.temperature > 0.0
            && params.top_k == 0
            && params.top_p >= 1.0
            && params.min_p <= 0.0
            && !self.needs_adjusting()
        {
            LogitUse::Draw {
                temperature: params.temperature,
                key: draw_key(params.seed, generated),
            }
        } else {
            LogitUse::Candidates {
                gap: self.logit_gap(),
                limit: self.top_limit(),
            }
        }
    }

    /// Pick a token. `history` is every token so far, of which the last `generated` are the
    /// reply. `allowed` restricts the choice, unless it rules out every token there is.
    /// `allowed` needs [`Logits::All`], to know how the allowed tokens scored.
    pub fn sample(
        &mut self,
        logits: Logits,
        history: &[u32],
        generated: usize,
        allowed: Option<&TokenSet>,
    ) -> u32 {
        let mut candidates = std::mem::take(&mut self.candidates);
        candidates.clear();
        match logits {
            Logits::All(scores) => {
                let allowed =
                    allowed.filter(|set| (0..scores.len() as u32).any(|id| set.contains(id)));
                self.gather(scores, history, generated, allowed, &mut candidates);
            }
            Logits::Candidates(list) => {
                candidates.extend_from_slice(list);
                self.adjust_candidates(&mut candidates, history, generated);
            }
        }
        // Scores can change the order of ties between runs, so ties are settled by token id.
        let token = if self.params.temperature <= 0.0 {
            candidates
                .iter()
                .min_by(|a, b| by_score_then_id(a, b))
                .expect("the model always produces scores")
                .1
        } else {
            self.draw(&mut candidates, generated)
        };
        self.candidates = candidates;
        token
    }

    /// Reduce the full list of scores to the tokens that matter, as `(score, token)`.
    fn gather(
        &mut self,
        scores: &[f32],
        history: &[u32],
        generated: usize,
        allowed: Option<&TokenSet>,
        candidates: &mut Vec<(f32, u32)>,
    ) {
        let mut adjusted = std::mem::take(&mut self.adjusted);
        let adjusting = self.needs_adjusting();
        if adjusting {
            adjusted.clear();
            adjusted.extend_from_slice(scores);
            self.adjust(&mut adjusted, history, generated);
        }
        let scores: &[f32] = if adjusting { &adjusted } else { scores };
        let best = match allowed {
            None => maximum(scores),
            Some(set) => scores
                .iter()
                .enumerate()
                .filter(|&(id, _)| set.contains(id as u32))
                .fold(f32::NEG_INFINITY, |max, (_, &score)| max.max(score)),
        };
        // Tokens this far behind the best are so unlikely that they could never be drawn, and
        // leaving them out saves sorting the whole vocabulary for every token.
        collect_above(scores, best - self.logit_gap(), candidates);
        if let Some(set) = allowed {
            candidates.retain(|&(_, id)| set.contains(id));
        }
        self.adjusted = adjusted;
    }

    fn needs_adjusting(&self) -> bool {
        let p = &self.params;
        p.repeat_penalty != 1.0
            || p.frequency_penalty != 0.0
            || p.presence_penalty != 0.0
            || !p.logit_bias.is_empty()
    }

    /// Apply penalties and biases to `logits`, which is indexed by token id.
    fn adjust(&mut self, logits: &mut [f32], history: &[u32], generated: usize) {
        let params = &self.params;
        if params.repeat_penalty != 1.0 {
            self.seen.clear();
            self.seen
                .extend_from_slice(&history[history.len().saturating_sub(REPEAT_LAST_N)..]);
            self.seen.sort_unstable();
            self.seen.dedup();
            for &id in &self.seen {
                if let Some(logit) = logits.get_mut(id as usize) {
                    *logit = if *logit > 0.0 {
                        *logit / params.repeat_penalty
                    } else {
                        *logit * params.repeat_penalty
                    };
                }
            }
        }
        if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
            self.seen.clear();
            self.seen
                .extend_from_slice(&history[history.len() - generated..]);
            self.seen.sort_unstable();
            for run in self.seen.chunk_by(|a, b| a == b) {
                if let Some(logit) = logits.get_mut(run[0] as usize) {
                    *logit -= params.presence_penalty + params.frequency_penalty * run.len() as f32;
                }
            }
        }
        for &(id, bias) in &params.logit_bias {
            if let Some(logit) = logits.get_mut(id as usize) {
                *logit += bias;
            }
        }
    }

    /// Apply penalties and biases to candidates, then drop any that fell out of contention.
    fn adjust_candidates(
        &mut self,
        candidates: &mut Vec<(f32, u32)>,
        history: &[u32],
        generated: usize,
    ) {
        if !self.needs_adjusting() {
            return;
        }
        let params = &self.params;
        let recent = &history[history.len().saturating_sub(REPEAT_LAST_N)..];
        let replied = &history[history.len() - generated..];
        for (score, id) in candidates.iter_mut() {
            if params.repeat_penalty != 1.0 && recent.contains(id) {
                *score = if *score > 0.0 {
                    *score / params.repeat_penalty
                } else {
                    *score * params.repeat_penalty
                };
            }
            if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
                let uses = replied.iter().filter(|t| *t == id).count();
                if uses > 0 {
                    *score -= params.presence_penalty + params.frequency_penalty * uses as f32;
                }
            }
        }
        for &(token, bias) in &params.logit_bias {
            if let Some((score, _)) = candidates.iter_mut().find(|(_, id)| *id == token) {
                *score += bias;
            }
        }
        let best = candidates
            .iter()
            .fold(f32::NEG_INFINITY, |max, c| max.max(c.0));
        let gap = self.logit_gap();
        candidates.retain(|&(score, _)| score >= best - gap);
    }

    /// Draw token number `generated` of the reply from the candidates. The random number depends
    /// only on the seed and `generated`, so checking a guessed token (speculative decoding) picks
    /// what drawing it in the ordinary way would have.
    fn draw(&mut self, candidates: &mut Vec<(f32, u32)>, generated: usize) -> u32 {
        let (temperature, top_k, top_p, min_p) = (
            self.params.temperature,
            self.params.top_k,
            self.params.top_p,
            self.params.min_p,
        );
        if top_k > 0 && top_k < candidates.len() {
            candidates.select_nth_unstable_by(top_k - 1, by_score_then_id);
            candidates.truncate(top_k);
        }
        candidates.sort_unstable_by(by_score_then_id);

        // Softmax with temperature. Candidates are sorted, so the first has the largest logit.
        let max = candidates[0].0;
        let mut sum = 0.0;
        for (p, _) in candidates.iter_mut() {
            *p = ((*p - max) / temperature).exp();
            sum += *p;
        }
        for (p, _) in candidates.iter_mut() {
            *p /= sum;
        }

        if top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut keep = candidates.len();
            for (i, (p, _)) in candidates.iter().enumerate() {
                cumulative += p;
                if cumulative >= top_p {
                    keep = i + 1;
                    break;
                }
            }
            candidates.truncate(keep);
        }
        if min_p > 0.0 {
            let threshold = candidates[0].0 * min_p;
            let keep = candidates.partition_point(|&(p, _)| p >= threshold);
            candidates.truncate(keep.max(1));
        }

        let total: f32 = candidates.iter().map(|(p, _)| p).sum();
        let r = self.random_f32(generated) * total;
        let mut cumulative = 0.0;
        for &(p, id) in candidates.iter() {
            cumulative += p;
            if r < cumulative {
                return id;
            }
        }
        candidates.last().unwrap().1
    }

    /// Uniform in [0, 1), for drawing token number `generated` of the reply.
    fn random_f32(&self, generated: usize) -> f32 {
        // A different stream from the GPU's draws, which use the key for Gumbel noise.
        let bits = draw_key(self.params.seed ^ 0x5DEE_CE66_D1CE_4E5B, generated) >> 40;
        bits as f32 / (1u32 << 24) as f32
    }
}

/// The key for the random numbers of one draw (one step of SplitMix64): a function of the seed
/// and how far through the reply this is, so it comes out the same whatever else is running.
pub(crate) fn draw_key(seed: u64, position: usize) -> u64 {
    let mut z = seed.wrapping_add((position as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Highest score first, and the lowest token id among equals.
fn by_score_then_id(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// How many values to compare at once in the loops below. Several accumulators side by side
/// let the compiler use SIMD, and these loops run over the whole vocabulary for every token.
const SCAN_LANES: usize = 16;

/// The largest value in a slice (of finite numbers).
fn maximum(values: &[f32]) -> f32 {
    let (chunks, tail) = values.as_chunks::<SCAN_LANES>();
    let mut best = [f32::NEG_INFINITY; SCAN_LANES];
    for chunk in chunks {
        for i in 0..SCAN_LANES {
            if chunk[i] > best[i] {
                best[i] = chunk[i];
            }
        }
    }
    let mut max = f32::NEG_INFINITY;
    for &value in best.iter().chain(tail) {
        if value > max {
            max = value;
        }
    }
    max
}

/// Append `(value, index)` for every value that's at least `cutoff`.
fn collect_above(values: &[f32], cutoff: f32, out: &mut Vec<(f32, u32)>) {
    let (chunks, tail) = values.as_chunks::<SCAN_LANES>();
    let mut push_matches = |offset: usize, slice: &[f32]| {
        for (i, &value) in slice.iter().enumerate() {
            if value >= cutoff {
                out.push((value, (offset + i) as u32));
            }
        }
    };
    for (c, chunk) in chunks.iter().enumerate() {
        let mut any = false;
        for &value in chunk {
            any |= value >= cutoff;
        }
        if any {
            push_matches(c * SCAN_LANES, chunk);
        }
    }
    push_matches(chunks.len() * SCAN_LANES, tail);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(scores: &[f32]) -> Logits<'_> {
        Logits::All(scores)
    }

    fn sampler(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler::new(
            SamplingParams {
                temperature,
                top_k,
                top_p,
                seed,
                ..Default::default()
            },
            1000,
        )
    }

    #[test]
    fn the_gap_never_excludes_a_token_top_p_would_keep() {
        // A long-tailed distribution: scores fall off slowly enough for top-p to reach far down.
        let scores: Vec<f32> = (0..1000).map(|i| -(i as f32) * 0.02).collect();
        let s = sampler(1.0, 0, 0.9, 1);
        let gap = s.logit_gap();
        let total: f32 = scores.iter().map(|x| x.exp()).sum();
        let mut sum = 0.0;
        let kept = scores
            .iter()
            .take_while(|x| {
                sum += x.exp();
                sum - x.exp() < 0.9 * total
            })
            .count();
        assert!(scores[kept - 1] >= -gap, "gap {gap} cut token {}", kept - 1);
    }

    #[test]
    fn a_plain_draw_is_left_to_the_backend_only_when_nothing_else_applies() {
        let draw = |params: SamplingParams| match Sampler::new(params, 100).logit_use(3) {
            LogitUse::Draw { temperature, key } => Some((temperature, key)),
            _ => None,
        };
        let plain = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            seed: 7,
            ..Default::default()
        };
        let (temperature, key) = draw(plain.clone()).expect("this one can be drawn on the GPU");
        assert_eq!(temperature, 0.8);
        assert_eq!(key, draw_key(7, 3));
        assert_ne!(key, draw_key(7, 4));
        assert_ne!(key, draw_key(8, 3));

        let with = |change: fn(&mut SamplingParams)| {
            let mut params = plain.clone();
            change(&mut params);
            draw(params)
        };
        assert!(with(|p| p.temperature = 0.0).is_none());
        assert!(with(|p| p.top_k = 40).is_none());
        assert!(with(|p| p.top_p = 0.9).is_none());
        assert!(with(|p| p.min_p = 0.05).is_none());
        assert!(with(|p| p.repeat_penalty = 1.1).is_none());
        assert!(with(|p| p.frequency_penalty = 0.5).is_none());
        assert!(with(|p| p.presence_penalty = 0.5).is_none());
        assert!(with(|p| p.logit_bias = vec![(5, -1.0)]).is_none());
        let mut biased = plain.clone();
        biased.logit_bias = vec![(5, 1.0)];
        assert!(matches!(
            Sampler::new(biased, 100).logit_use(0),
            LogitUse::All
        ));
    }

    #[test]
    fn scanning_finds_everything() {
        let values: Vec<f32> = (0..1000).map(|i| ((i * 37) % 1000) as f32).collect();
        assert_eq!(maximum(&values), 999.0);
        let mut found = Vec::new();
        collect_above(&values, 990.0, &mut found);
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(found.len(), 10);
        for (i, &(value, id)) in found.iter().enumerate() {
            assert_eq!(value, 990.0 + i as f32);
            assert_eq!(values[id as usize], value);
        }
    }

    #[test]
    fn greedy_picks_the_largest_logit() {
        let mut s = sampler(0.0, 40, 0.9, 1);
        assert_eq!(s.sample(all(&[0.1, 3.0, -1.0, 2.9]), &[], 0, None), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = sampler(1.0, 1, 1.0, 7);
        for _ in 0..20 {
            assert_eq!(s.sample(all(&[0.1, 3.0, -1.0, 2.9]), &[], 0, None), 1);
        }
    }

    #[test]
    fn repeat_penalty_discourages_seen_tokens() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 0.0,
                repeat_penalty: 2.0,
                ..Default::default()
            },
            4,
        );
        assert_eq!(s.sample(all(&[0.1, 3.0, -1.0, 2.0]), &[1], 1, None), 3);
    }

    #[test]
    fn frequency_penalty_grows_with_use() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 0.0,
                frequency_penalty: 1.0,
                ..Default::default()
            },
            3,
        );
        let logits = [2.5, 0.0, 2.0];
        assert_eq!(s.sample(all(&logits), &[0], 1, None), 2);
        assert_eq!(
            s.sample(all(&logits), &[0], 0, None),
            0,
            "only the reply counts"
        );
    }

    #[test]
    fn logit_bias_applies() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 0.0,
                logit_bias: vec![(2, 10.0)],
                ..Default::default()
            },
            3,
        );
        assert_eq!(s.sample(all(&[1.0, 2.0, 0.0]), &[], 0, None), 2);
    }

    #[test]
    fn allowed_set_restricts_the_choice() {
        let mut allowed = TokenSet::new(4);
        allowed.insert(0);
        allowed.insert(3);
        let mut s = sampler(0.0, 0, 1.0, 1);
        assert_eq!(
            s.sample(all(&[0.1, 3.0, -1.0, 2.0]), &[], 0, Some(&allowed)),
            3
        );
        let mut s = sampler(1.0, 0, 1.0, 1);
        for _ in 0..50 {
            let t = s.sample(all(&[0.1, 3.0, -1.0, 2.0]), &[], 0, Some(&allowed));
            assert!(t == 0 || t == 3);
        }
    }

    #[test]
    fn min_p_drops_unlikely_tokens() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.5,
                seed: 3,
                ..Default::default()
            },
            4,
        );
        let logits = [0.0, -0.1, -5.0, -6.0];
        for _ in 0..100 {
            assert!(s.sample(all(&logits), &[], 0, None) < 2);
        }
    }

    #[test]
    fn samples_follow_the_distribution() {
        let mut s = sampler(1.0, 0, 1.0, 42);
        let logits = [0.0, (3.0f32).ln()]; // probabilities 0.25 and 0.75
        let ones = (0..10_000)
            .filter(|&i| s.sample(all(&logits), &[], i, None) == 1)
            .count();
        assert!((7_200..7_800).contains(&ones), "{ones}");
    }

    #[test]
    fn a_draw_depends_only_on_the_seed_and_position() {
        let logits: Vec<f32> = (0..50).map(|i| -(i as f32) * 0.1).collect();
        let mut a = sampler(1.0, 0, 1.0, 7);
        let mut b = sampler(1.0, 0, 1.0, 7);
        let forward: Vec<u32> = (0..20)
            .map(|i| a.sample(all(&logits), &[], i, None))
            .collect();
        let backward: Vec<u32> = (0..20)
            .rev()
            .map(|i| b.sample(all(&logits), &[], i, None))
            .collect();
        assert_eq!(forward, backward.into_iter().rev().collect::<Vec<_>>());
    }
}

//! Speculative decoding: guess several tokens ahead cheaply, then check all the guesses in one
//! pass of the model, which costs little more than generating one token when the batch is small
//! (reading the weights is most of the work either way).
//!
//! The guesses here come from the text itself, as SGLang's NGRAM algorithm (and "prompt lookup
//! decoding") does: find the latest earlier place where the last few tokens appeared, and guess
//! that what followed them then follows them now. Replies that quote or edit their prompt, or
//! repeat themselves (code, JSON, retrieval answers), are full of these.
//!
//! Checking a guess means sampling each position as ordinary decoding would and seeing whether
//! the guess is what came out. Since every draw depends only on the request's seed and position
//! (see [`crate::sampler`]), and the model's results don't depend on how rows are batched, the
//! reply is exactly what it would have been without guessing, only sooner.

use std::collections::HashMap;

/// The shortest run of tokens worth matching. Single tokens match almost anywhere, and guesses
/// from them are rarely right.
pub const MIN_NGRAM: usize = 2;
/// The longest. Longer matches are more likely to continue the same way, so they're tried first.
pub const MAX_NGRAM: usize = 5;

/// An index of a sequence's n-grams, kept up to date as the sequence grows.
#[derive(Default)]
pub struct NgramIndex {
    /// For each length `MIN_NGRAM + i`, an n-gram's hash to the position just after its latest
    /// occurrence (the token that followed it).
    maps: [HashMap<u64, u32>; MAX_NGRAM - MIN_NGRAM + 1],
    /// Positions before this have been indexed.
    indexed: usize,
}

fn hash(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &t| {
        (h ^ t as u64)
            .wrapping_mul(0x0000_0100_0000_01B3)
            .rotate_left(17)
    })
}

impl NgramIndex {
    /// Index every n-gram of `tokens` that has a token after it.
    pub fn update(&mut self, tokens: &[u32]) {
        for end in self.indexed.max(MIN_NGRAM)..tokens.len() {
            for (i, map) in self.maps.iter_mut().enumerate() {
                let n = MIN_NGRAM + i;
                if end >= n {
                    map.insert(hash(&tokens[end - n..end]), end as u32);
                }
            }
        }
        self.indexed = self.indexed.max(tokens.len());
    }

    /// Up to `max` tokens that might come next, judging by what followed the longest earlier
    /// match of the tokens at the end. Call [`Self::update`] first.
    pub fn propose<'a>(&self, tokens: &'a [u32], max: usize) -> &'a [u32] {
        let len = tokens.len();
        for n in (MIN_NGRAM..=MAX_NGRAM).rev() {
            if len < n + 1 || max == 0 {
                continue;
            }
            let suffix = &tokens[len - n..];
            let Some(&end) = self.maps[n - MIN_NGRAM].get(&hash(suffix)) else {
                continue;
            };
            let end = end as usize;
            // A different n-gram with the same hash is possible, if unlikely.
            if end < len && tokens[end - n..end] == *suffix {
                return &tokens[end..(end + max).min(len)];
            }
        }
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(tokens: &[u32]) -> NgramIndex {
        let mut index = NgramIndex::default();
        index.update(tokens);
        index
    }

    #[test]
    fn guesses_what_followed_the_last_match() {
        let tokens = [1, 2, 3, 4, 5, 9, 9, 2, 3];
        assert_eq!(index(&tokens).propose(&tokens, 3), &[4, 5, 9]);
        // Never past the end of what's known.
        assert_eq!(index(&tokens).propose(&tokens, 10), &[4, 5, 9, 9, 2, 3]);
    }

    #[test]
    fn prefers_longer_and_later_matches() {
        // "2 3" is followed by 7 the first time and 8 the second; "1 2 3" only by 7.
        let tokens = [1, 2, 3, 7, 0, 2, 3, 8, 0, 1, 2, 3];
        assert_eq!(index(&tokens).propose(&tokens, 1), &[7]);
        let tokens = [5, 2, 3, 7, 0, 2, 3, 8, 0, 1, 2, 3];
        assert_eq!(index(&tokens).propose(&tokens, 1), &[8]);
    }

    #[test]
    fn nothing_without_a_match() {
        let tokens = [1, 2, 3, 4, 5, 6];
        assert!(index(&tokens).propose(&tokens, 4).is_empty());
    }

    #[test]
    fn indexing_as_it_grows_is_the_same_as_all_at_once() {
        let tokens: Vec<u32> = (0..200).map(|i| (i * 7 % 13) as u32).collect();
        let mut grown = NgramIndex::default();
        for len in 1..=tokens.len() {
            grown.update(&tokens[..len]);
            assert_eq!(
                grown.propose(&tokens[..len], 4),
                index(&tokens[..len]).propose(&tokens[..len], 4)
            );
        }
    }
}

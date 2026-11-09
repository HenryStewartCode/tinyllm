//! Regular expressions, parsed by the `regex-syntax` crate and turned into grammar rules.
//!
//! The pattern has to match the whole reply, so `^` and `$` at the ends mean nothing extra.

use super::{Builder, ByteSet, Item, Seq, literal};
use anyhow::{Result, bail};
use regex_syntax::ParserBuilder;
use regex_syntax::hir::{Class, ClassUnicode, Hir, HirKind, Look};
use regex_syntax::utf8::Utf8Sequences;

/// Compile a pattern. With `json_safe` the text will appear inside a JSON string, so characters
/// that would end or corrupt the string can't be chosen by a character class.
pub(super) fn compile(pattern: &str, builder: &mut Builder, json_safe: bool) -> Result<Seq> {
    let hir = ParserBuilder::new().build().parse(pattern)?;
    convert(&hir, builder, json_safe)
}

fn convert(hir: &Hir, builder: &mut Builder, json_safe: bool) -> Result<Seq> {
    Ok(match hir.kind() {
        HirKind::Empty => Vec::new(),
        HirKind::Literal(lit) => literal(&lit.0),
        HirKind::Class(Class::Unicode(class)) => class_items(class, builder, json_safe)?,
        HirKind::Class(Class::Bytes(class)) => {
            let mut set = ByteSet::default();
            for range in class.ranges() {
                set.insert_range(range.start(), range.end());
            }
            vec![Item::Set(set)]
        }
        HirKind::Look(look) => match look {
            Look::Start
            | Look::End
            | Look::StartLF
            | Look::EndLF
            | Look::StartCRLF
            | Look::EndCRLF => Vec::new(),
            other => bail!("{other:?} assertions aren't supported in constraints"),
        },
        HirKind::Repetition(repetition) => {
            let body = convert(&repetition.sub, builder, json_safe)?;
            builder.repeat(
                body,
                repetition.min as usize,
                repetition.max.map(|max| max as usize),
            )?
        }
        HirKind::Capture(capture) => convert(&capture.sub, builder, json_safe)?,
        HirKind::Concat(parts) => {
            let mut seq = Vec::new();
            for part in parts {
                seq.extend(convert(part, builder, json_safe)?);
            }
            seq
        }
        HirKind::Alternation(parts) => {
            let mut alternatives = Vec::with_capacity(parts.len());
            for part in parts {
                alternatives.push(convert(part, builder, json_safe)?);
            }
            vec![builder.call(alternatives)?]
        }
    })
}

/// The bytes of one character from a set of characters. Characters of different encoded
/// lengths or leading bytes become separate alternatives.
pub(super) fn class_items(
    class: &ClassUnicode,
    builder: &mut Builder,
    json_safe: bool,
) -> Result<Seq> {
    let mut ascii = ByteSet::default();
    let mut alternatives: Vec<Seq> = Vec::new();
    for range in class.ranges() {
        for sequence in Utf8Sequences::new(range.start(), range.end()) {
            let ranges = sequence.as_slice();
            if ranges.len() == 1 {
                ascii.insert_range(ranges[0].start, ranges[0].end);
            } else {
                alternatives.push(
                    ranges
                        .iter()
                        .map(|r| Item::Set(ByteSet::range(r.start, r.end)))
                        .collect(),
                );
            }
        }
    }
    if json_safe {
        for byte in (0..0x20).chain(*b"\"\\") {
            ascii.remove(byte);
        }
    }
    if !ascii.is_empty() {
        if alternatives.is_empty() {
            return Ok(vec![Item::Set(ascii)]);
        }
        alternatives.insert(0, vec![Item::Set(ascii)]);
    }
    if alternatives.is_empty() {
        bail!("a character class in the pattern can't match anything");
    }
    Ok(vec![builder.call(alternatives)?])
}

#[cfg(test)]
mod tests {
    use super::super::test_support::matches;
    use super::*;

    fn check(pattern: &str, good: &[&str], bad: &[&str]) {
        let mut builder = Builder::default();
        let seq = compile(pattern, &mut builder, false).unwrap();
        let root = builder.rule(vec![seq]).unwrap();
        for text in good {
            assert!(
                matches(&builder, root, text),
                "{pattern} should match {text:?}"
            );
        }
        for text in bad {
            assert!(
                !matches(&builder, root, text),
                "{pattern} shouldn't match {text:?}"
            );
        }
    }

    #[test]
    fn literals_classes_and_alternation() {
        check("cat|dog", &["cat", "dog"], &["ca", "catdog", ""]);
        check("[a-c]x", &["ax", "cx"], &["dx", "a"]);
        check("^ab$", &["ab"], &["abc"]);
    }

    #[test]
    fn repetition() {
        check("a{2,3}", &["aa", "aaa"], &["a", "aaaa"]);
        check("(ab)+", &["ab", "abab"], &["", "aba"]);
        check("x*", &["", "xxxx"], &["y"]);
        check("a{3}", &["aaa"], &["aa", "aaaa"]);
    }

    #[test]
    fn unicode() {
        check(".", &["a", "é", "€", "🙂"], &["", "ab", "\n"]);
        check(r"\d{3}-\d{4}", &["555-1234"], &["5551234", "55-1234"]);
        check("[é-ë]+", &["é", "êë"], &["e", "éa"]);
        check(r"[^a]", &["b", "é"], &["a"]);
    }

    #[test]
    fn unsupported_assertions_are_reported() {
        let mut builder = Builder::default();
        assert!(compile(r"\bfoo", &mut builder, false).is_err());
    }

    #[test]
    fn json_safe_classes_stay_inside_the_string() {
        let mut builder = Builder::default();
        let seq = compile(r".*", &mut builder, true).unwrap();
        let root = builder.rule(vec![seq]).unwrap();
        assert!(matches(&builder, root, "hello é"));
        assert!(!matches(&builder, root, "a\"b"));
        assert!(!matches(&builder, root, "a\\b"));
    }
}

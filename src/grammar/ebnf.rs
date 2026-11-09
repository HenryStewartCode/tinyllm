//! Grammars written in EBNF, in the same dialect as GBNF and XGrammar:
//!
//! ```text
//! root   ::= answer ("," answer)*
//! answer ::= "yes" | "no" | [0-9]+   # comments run to the end of the line
//! ```

use super::{Builder, ByteSet, Item, RuleId, Seq, literal, regex};
use anyhow::{Context, Result, bail};
use regex_syntax::hir::{ClassUnicode, ClassUnicodeRange};
use std::collections::HashMap;

pub(super) fn compile(source: &str, builder: &mut Builder) -> Result<RuleId> {
    let mut parser = Parser {
        text: source.as_bytes(),
        at: 0,
        builder,
        rules: HashMap::new(),
        defined: HashMap::new(),
        depth: 0,
    };
    parser.skip_space();
    while parser.at < parser.text.len() {
        parser.parse_rule()?;
        parser.skip_space();
    }
    for name in parser.rules.keys() {
        if !parser.defined.contains_key(name) {
            bail!("rule '{name}' is used but never defined");
        }
    }
    parser
        .defined
        .get("root")
        .copied()
        .context("the grammar needs a rule called 'root'")
}

struct Parser<'a> {
    text: &'a [u8],
    at: usize,
    builder: &'a mut Builder,
    /// Rule ids by name, including rules that have only been referred to so far.
    rules: HashMap<String, RuleId>,
    defined: HashMap<String, RuleId>,
    /// How many parentheses are open.
    depth: usize,
}

/// Parentheses nested deeper than this are certainly a mistake or an attack.
const MAX_NESTING: usize = 64;

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    fn skip_space(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.at += 1;
            } else if c == b'#' {
                while self.peek().is_some_and(|c| c != b'\n') {
                    self.at += 1;
                }
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.text[self.at..].starts_with(token.as_bytes()) {
            self.at += token.len();
            true
        } else {
            false
        }
    }

    fn name(&mut self) -> Option<String> {
        self.skip_space();
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        {
            self.at += 1;
        }
        (self.at > start).then(|| String::from_utf8_lossy(&self.text[start..self.at]).into_owned())
    }

    /// Is the next thing the start of a new rule (a name followed by `::=`)?
    fn at_rule_start(&mut self) -> bool {
        let saved = self.at;
        let found = self.name().is_some() && self.eat("::=");
        self.at = saved;
        found
    }

    fn rule_id(&mut self, name: &str) -> RuleId {
        if let Some(&id) = self.rules.get(name) {
            return id;
        }
        let id = self.builder.declare();
        self.rules.insert(name.to_string(), id);
        id
    }

    fn parse_rule(&mut self) -> Result<()> {
        let name = self
            .name()
            .with_context(|| format!("expected a rule name at byte {}", self.at))?;
        if !self.eat("::=") {
            bail!("expected '::=' after '{name}'");
        }
        let alternatives = self.parse_alternatives()?;
        let id = self.rule_id(&name);
        if self.defined.insert(name.clone(), id).is_some() {
            bail!("rule '{name}' is defined twice");
        }
        self.builder.define(id, alternatives)
    }

    fn parse_alternatives(&mut self) -> Result<Vec<Seq>> {
        let mut alternatives = vec![self.parse_sequence()?];
        while self.eat("|") {
            alternatives.push(self.parse_sequence()?);
        }
        Ok(alternatives)
    }

    fn parse_sequence(&mut self) -> Result<Seq> {
        let mut seq = Vec::new();
        loop {
            self.skip_space();
            match self.peek() {
                None | Some(b'|' | b')') => return Ok(seq),
                _ if self.at_rule_start() => return Ok(seq),
                _ => {}
            }
            let atom = self.parse_atom()?;
            seq.extend(self.parse_postfix(atom)?);
        }
    }

    fn parse_atom(&mut self) -> Result<Seq> {
        match self.peek() {
            Some(quote @ (b'"' | b'\'')) => {
                self.at += 1;
                let mut bytes = Vec::new();
                loop {
                    match self.next_char()? {
                        (c, false) if c as u32 == quote as u32 => break,
                        (c, _) => {
                            let mut buffer = [0; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                        }
                    }
                }
                Ok(literal(&bytes))
            }
            Some(b'[') => {
                self.at += 1;
                self.parse_class()
            }
            Some(b'(') => {
                self.at += 1;
                self.depth += 1;
                if self.depth > MAX_NESTING {
                    bail!("parentheses are nested too deeply");
                }
                let alternatives = self.parse_alternatives()?;
                self.depth -= 1;
                if !self.eat(")") {
                    bail!("missing ')' at byte {}", self.at);
                }
                Ok(vec![self.builder.call(alternatives)?])
            }
            _ => {
                let name = self
                    .name()
                    .with_context(|| format!("unexpected character at byte {}", self.at))?;
                Ok(vec![Item::Call(self.rule_id(&name))])
            }
        }
    }

    fn parse_postfix(&mut self, mut atom: Seq) -> Result<Seq> {
        loop {
            self.skip_space();
            atom = match self.peek() {
                Some(b'*') => {
                    self.at += 1;
                    vec![self.builder.star(atom)?]
                }
                Some(b'+') => {
                    self.at += 1;
                    self.builder.repeat(atom, 1, None)?
                }
                Some(b'?') => {
                    self.at += 1;
                    vec![self.builder.optional(atom)?]
                }
                Some(b'{') => {
                    self.at += 1;
                    let (min, max) = self.parse_bounds()?;
                    self.builder.repeat(atom, min, max)?
                }
                _ => return Ok(atom),
            };
        }
    }

    fn parse_bounds(&mut self) -> Result<(usize, Option<usize>)> {
        let number = |parser: &mut Self| -> Option<usize> {
            parser.skip_space();
            let start = parser.at;
            while parser.peek().is_some_and(|c| c.is_ascii_digit()) {
                parser.at += 1;
            }
            std::str::from_utf8(&parser.text[start..parser.at])
                .ok()?
                .parse()
                .ok()
        };
        let min = number(self).unwrap_or(0);
        let bounds = if self.eat(",") {
            (min, number(self))
        } else {
            (min, Some(min))
        };
        if !self.eat("}") {
            bail!("missing '}}' at byte {}", self.at);
        }
        Ok(bounds)
    }

    /// One character from inside a string or class. Returns it and whether it was escaped.
    fn next_char(&mut self) -> Result<(char, bool)> {
        let rest = std::str::from_utf8(&self.text[self.at..]).context("the grammar isn't UTF-8")?;
        let c = rest
            .chars()
            .next()
            .context("unexpected end of the grammar")?;
        self.at += c.len_utf8();
        if c != '\\' {
            return Ok((c, false));
        }
        let rest = std::str::from_utf8(&self.text[self.at..]).context("the grammar isn't UTF-8")?;
        let escaped = rest
            .chars()
            .next()
            .context("unexpected end of the grammar")?;
        self.at += escaped.len_utf8();
        let mut hex = |digits: usize| -> Result<char> {
            let text = std::str::from_utf8(
                self.text
                    .get(self.at..self.at + digits)
                    .context("bad escape")?,
            )?;
            self.at += digits;
            char::from_u32(u32::from_str_radix(text, 16)?).context("bad escape")
        };
        let c = match escaped {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'x' => hex(2)?,
            'u' => hex(4)?,
            'U' => hex(8)?,
            other => other,
        };
        Ok((c, true))
    }

    fn parse_class(&mut self) -> Result<Seq> {
        let negated = self.peek() == Some(b'^');
        if negated {
            self.at += 1;
        }
        let mut ranges = Vec::new();
        loop {
            let (start, escaped) = self.next_char()?;
            if start == ']' && !escaped {
                break;
            }
            let mut end = start;
            if self.peek() == Some(b'-') && self.text.get(self.at + 1) != Some(&b']') {
                self.at += 1;
                end = self.next_char()?.0;
            }
            ranges.push(ClassUnicodeRange::new(start, end));
        }
        let mut class = ClassUnicode::new(ranges);
        if negated {
            class.negate();
        }
        if class.ranges().is_empty() {
            return Ok(vec![Item::Set(ByteSet::default())]);
        }
        regex::class_items(&class, self.builder, false)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::matches;
    use super::*;

    fn check(grammar: &str, good: &[&str], bad: &[&str]) {
        let mut builder = Builder::default();
        let root = compile(grammar, &mut builder).unwrap();
        for text in good {
            assert!(matches(&builder, root, text), "should match {text:?}");
        }
        for text in bad {
            assert!(!matches(&builder, root, text), "shouldn't match {text:?}");
        }
    }

    #[test]
    fn rules_and_references() {
        let grammar = r#"
            root ::= answer ("," answer)*   # a list
            answer ::= "yes" | "no" | [0-9]+
        "#;
        check(grammar, &["yes", "no,12,yes"], &["maybe", "yes,", ""]);
    }

    #[test]
    fn recursion_nests() {
        let grammar = r#"root ::= "(" root ")" | "x""#;
        check(grammar, &["x", "((x))"], &["(x", "()"]);
    }

    #[test]
    fn repetition_escapes_and_classes() {
        check(
            r#"root ::= "a"{2,3} "\n""#,
            &["aa\n", "aaa\n"],
            &["a\n", "aaaa\n"],
        );
        check(r#"root ::= [^a-c]+"#, &["xyz", "é"], &["a", "xa", ""]);
        check(r#"root ::= [\x41-\x43]"#, &["A", "C"], &["D"]);
        check(r#"root ::= 'é'? "b""#, &["éb", "b"], &["é"]);
    }

    #[test]
    fn mistakes_are_reported() {
        for (grammar, expected) in [
            ("root ::= other", "never defined"),
            ("a ::= \"x\"", "'root'"),
            ("root ::= \"a\"\nroot ::= \"b\"", "twice"),
            (
                &format!("root ::= {}\"a\"{}", "(".repeat(5000), ")".repeat(5000)),
                "too deeply",
            ),
        ] {
            let err = compile(grammar, &mut Builder::default()).unwrap_err();
            assert!(err.to_string().contains(expected), "{err}");
        }
    }
}

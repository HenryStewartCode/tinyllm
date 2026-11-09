//! JSON schemas, turned into grammar rules.
//!
//! The usual keywords are supported: `type`, `properties`, `required`, `additionalProperties`,
//! `items`, `prefixItems`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`
//! (of objects), `$ref`, `pattern`, `format`, `minLength`, `maxLength`, and the numeric bounds
//! for integers. As with OpenAI's strict mode and XGrammar, an object may have only the
//! properties listed, in the listed order, unless `additionalProperties` says otherwise.

use super::{Builder, ByteSet, CallsGrammar, Item, RuleId, Seq, literal, regex};
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

/// A number with more digits than this is almost certainly a model stuck repeating digits.
const MAX_DIGITS: usize = 15;
const MAX_INTEGER: i64 = 999_999_999_999_999; // 15 digits

pub(super) fn compile(schema: &Value, builder: &mut Builder) -> Result<RuleId> {
    let mut converter = Converter::new(schema, builder)?;
    let body = converter.convert(schema)?;
    converter.wrap(body)
}

/// Any JSON object, which is what OpenAI's `json_object` response format asks for.
pub(super) fn compile_any_object(builder: &mut Builder) -> Result<RuleId> {
    let null = Value::Null;
    let mut converter = Converter::new(&null, builder)?;
    let (object, _) = converter.any_value()?;
    converter.wrap(vec![Item::Call(object)])
}

/// Tool calls: fixed text around each tool's arguments.
pub(super) fn compile_calls(spec: &CallsGrammar, builder: &mut Builder) -> Result<RuleId> {
    let mut alternatives = Vec::with_capacity(spec.calls.len());
    for frame in &spec.calls {
        let mut converter = Converter::new(&frame.schema, builder)?;
        // An empty schema allows anything, but arguments are always an object.
        let is_empty = frame.schema.as_object().is_some_and(Map::is_empty);
        let converted = match is_empty {
            true => Ok(None),
            false => converter.convert(&frame.schema).map(Some),
        };
        let arguments = match converted {
            Ok(Some(seq)) => seq,
            Err(err) if frame.strict => {
                return Err(err.context(format!("the parameters of '{}'", frame.before)));
            }
            // Only strict tools promise to follow their schema, so it isn't worth failing over.
            _ => vec![Item::Call(converter.any_value()?.0)],
        };
        let mut call = literal(frame.before.as_bytes());
        call.extend(arguments);
        call.extend(literal(frame.after.as_bytes()));
        alternatives.push(call);
    }
    let call = builder.call(alternatives)?;

    let mut more = literal(spec.separator.as_bytes());
    more.push(call);
    let extra_calls = spec.max_calls.map(|max| max.saturating_sub(1));
    let rest = builder.repeat(more, 0, extra_calls)?;

    let openings = spec.opening.iter().map(|o| literal(o.as_bytes())).collect();
    let mut root = vec![builder.call(openings)?, call];
    root.extend(rest);
    root.extend(literal(spec.closing.as_bytes()));
    builder.rule(vec![root])
}

struct Converter<'a> {
    root: &'a Value,
    builder: &'a mut Builder,
    refs: HashMap<String, RuleId>,
    /// Optional whitespace between JSON tokens.
    ws: Item,
    string: Item,
    string_char: Item,
    /// Rules for (any object, any value), once they have been needed.
    any: Option<(RuleId, RuleId)>,
}

impl<'a> Converter<'a> {
    fn new(root: &'a Value, builder: &'a mut Builder) -> Result<Self> {
        // At most one line break plus indentation. Allowing unlimited whitespace lets a weak
        // model fall into printing blank lines until it runs out of tokens.
        let indent = builder.repeat(literal(b" "), 0, Some(32))?;
        let tabs = builder.repeat(literal(b"\t"), 0, Some(8))?;
        let mut line_break = literal(b"\n");
        line_break.push(builder.call(vec![indent, tabs])?);
        let ws = builder.call(vec![Vec::new(), literal(b" "), line_break])?;

        let mut plain = ByteSet::range(0x20, 0x7f);
        plain.remove(b'"');
        plain.remove(b'\\');
        let cont = || Item::Set(ByteSet::range(0x80, 0xbf));
        let set = |a: u8, b: u8| Item::Set(ByteSet::range(a, b));
        let hex = {
            let mut digits = ByteSet::range(b'0', b'9');
            digits.insert_range(b'a', b'f');
            digits.insert_range(b'A', b'F');
            Item::Set(digits)
        };
        let mut escape_chars = ByteSet::default();
        for c in b"\"\\/bfnrt" {
            escape_chars.insert_range(*c, *c);
        }
        let string_char = builder.call(vec![
            vec![Item::Set(plain)],
            vec![set(0xc2, 0xdf), cont()],
            vec![set(0xe0, 0xe0), set(0xa0, 0xbf), cont()],
            vec![set(0xe1, 0xec), cont(), cont()],
            vec![set(0xed, 0xed), set(0x80, 0x9f), cont()],
            vec![set(0xee, 0xef), cont(), cont()],
            vec![set(0xf0, 0xf0), set(0x90, 0xbf), cont(), cont()],
            vec![set(0xf1, 0xf3), cont(), cont(), cont()],
            vec![set(0xf4, 0xf4), set(0x80, 0x8f), cont(), cont()],
            vec![set(b'\\', b'\\'), Item::Set(escape_chars)],
            vec![set(b'\\', b'\\'), set(b'u', b'u'), hex, hex, hex, hex],
        ])?;
        let mut string = literal(b"\"");
        string.push(builder.star(vec![string_char])?);
        string.extend(literal(b"\""));
        let string = builder.call(vec![string])?;

        Ok(Self {
            root,
            builder,
            refs: HashMap::new(),
            ws,
            string,
            string_char,
            any: None,
        })
    }

    /// The top level: the value, with room for whitespace around it.
    fn wrap(&mut self, body: Seq) -> Result<RuleId> {
        let mut seq = vec![self.ws];
        seq.extend(body);
        seq.push(self.ws);
        self.builder.rule(vec![seq])
    }

    fn convert(&mut self, schema: &Value) -> Result<Seq> {
        let map = match schema {
            Value::Bool(true) => return self.any_value_seq(),
            Value::Bool(false) => bail!("a schema of 'false' can't match anything"),
            Value::Object(map) => map,
            other => bail!("expected a schema, found {other}"),
        };

        if let Some(reference) = map.get("$ref") {
            let reference = reference.as_str().context("$ref must be a string")?;
            return Ok(vec![Item::Call(self.reference(reference)?)]);
        }
        let mut alternatives = self.alternatives(map)?;
        if map.get("nullable") == Some(&Value::Bool(true)) {
            alternatives.push(literal(b"null"));
        }
        if alternatives.len() == 1 {
            return Ok(alternatives.pop().expect("there is one"));
        }
        Ok(vec![self.builder.call(alternatives)?])
    }

    fn alternatives(&mut self, map: &Map<String, Value>) -> Result<Vec<Seq>> {
        if let Some(value) = map.get("const") {
            return Ok(vec![literal(serde_json::to_string(value)?.as_bytes())]);
        }
        if let Some(values) = map.get("enum") {
            let values = values.as_array().context("enum must be a list")?;
            if values.is_empty() {
                bail!("enum must not be empty");
            }
            return values
                .iter()
                .map(|v| Ok(literal(serde_json::to_string(v)?.as_bytes())))
                .collect();
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(options) = map.get(keyword) {
                let options = options.as_array().context("anyOf must be a list")?;
                if options.is_empty() {
                    bail!("{keyword} must not be empty");
                }
                return options.iter().map(|o| self.convert(o)).collect();
            }
        }
        if let Some(all) = map.get("allOf") {
            let merged = merge_all_of(all.as_array().context("allOf must be a list")?, self.root)?;
            return Ok(vec![self.convert(&merged)?]);
        }

        match map.get("type") {
            Some(Value::Array(types)) => {
                if types.is_empty() {
                    bail!("type must not be an empty list");
                }
                types
                    .iter()
                    .map(|t| self.of_type(t.as_str().context("type must be a string")?, map))
                    .collect()
            }
            Some(Value::String(kind)) => Ok(vec![self.of_type(kind, map)?]),
            Some(other) => bail!("invalid type: {other}"),
            None if map.contains_key("properties")
                || map.contains_key("required")
                || map.contains_key("additionalProperties") =>
            {
                Ok(vec![self.of_type("object", map)?])
            }
            None if map.contains_key("items") || map.contains_key("prefixItems") => {
                Ok(vec![self.of_type("array", map)?])
            }
            None if map.contains_key("pattern") => Ok(vec![self.of_type("string", map)?]),
            None => Ok(vec![self.any_value_seq()?]),
        }
    }

    fn of_type(&mut self, kind: &str, map: &Map<String, Value>) -> Result<Seq> {
        match kind {
            "string" => self.string_schema(map),
            "integer" => self.integer(map),
            "number" => self.number(map),
            "boolean" => Ok(vec![
                self.builder
                    .call(vec![literal(b"true"), literal(b"false")])?,
            ]),
            "null" => Ok(literal(b"null")),
            "object" => self.object(map),
            "array" => self.array(map),
            other => bail!("unknown type '{other}'"),
        }
    }

    /// Resolve a `$ref`. Each target gets one rule, reserved before it's filled in, so
    /// schemas can refer to themselves.
    fn reference(&mut self, reference: &str) -> Result<RuleId> {
        if let Some(&rule) = self.refs.get(reference) {
            return Ok(rule);
        }
        let pointer = reference.strip_prefix('#').with_context(|| {
            format!("only references within the schema work, not '{reference}'")
        })?;
        let target = self
            .root
            .pointer(pointer)
            .with_context(|| format!("the reference '{reference}' doesn't lead anywhere"))?;
        let rule = self.builder.declare();
        self.refs.insert(reference.to_string(), rule);
        let body = self.convert(target)?;
        self.builder.define(rule, vec![body])?;
        Ok(rule)
    }

    fn string_schema(&mut self, map: &Map<String, Value>) -> Result<Seq> {
        if let Some(pattern) = map.get("pattern") {
            let pattern = pattern.as_str().context("pattern must be a string")?;
            return self.quoted_pattern(pattern);
        }
        if let Some(pattern) = map
            .get("format")
            .and_then(Value::as_str)
            .and_then(format_pattern)
        {
            return self.quoted_pattern(pattern);
        }
        let min = uint(map, "minLength")?.unwrap_or(0);
        let max = uint(map, "maxLength")?;
        if min == 0 && max.is_none() {
            return Ok(vec![self.string]);
        }
        let mut seq = literal(b"\"");
        seq.extend(self.builder.repeat(vec![self.string_char], min, max)?);
        seq.extend(literal(b"\""));
        Ok(seq)
    }

    fn quoted_pattern(&mut self, pattern: &str) -> Result<Seq> {
        let mut seq = literal(b"\"");
        seq.extend(regex::compile(pattern, self.builder, true)?);
        seq.extend(literal(b"\""));
        Ok(seq)
    }

    fn integer(&mut self, map: &Map<String, Value>) -> Result<Seq> {
        let bound = |name: &str| map.get(name).and_then(Value::as_f64);
        let mut low = bound("minimum").map(|v| v.ceil() as i64);
        if let Some(v) = bound("exclusiveMinimum") {
            low = low.into_iter().chain([v.floor() as i64 + 1]).max();
        }
        let mut high = bound("maximum").map(|v| v.floor() as i64);
        if let Some(v) = bound("exclusiveMaximum") {
            high = high.into_iter().chain([v.ceil() as i64 - 1]).min();
        }
        if low.is_none() && high.is_none() {
            return self.plain_integer();
        }
        let low = low.unwrap_or(-MAX_INTEGER).max(-MAX_INTEGER);
        let high = high.unwrap_or(MAX_INTEGER).min(MAX_INTEGER);
        if low > high {
            bail!("the integer range {low}..{high} is empty");
        }
        let mut alternatives = Vec::new();
        if high >= 0 {
            digit_ranges(low.max(0) as u64, high as u64, &mut alternatives);
        }
        if low < 0 {
            let mut negative = Vec::new();
            digit_ranges((-high).max(1) as u64, (-low) as u64, &mut negative);
            alternatives.extend(negative.into_iter().map(|digits| {
                let mut seq = literal(b"-");
                seq.extend(digits);
                seq
            }));
        }
        Ok(vec![self.builder.call(alternatives)?])
    }

    /// An integer with a sign, no leading zeros, and a sensible number of digits.
    fn plain_integer(&mut self) -> Result<Seq> {
        let mut seq = vec![self.builder.optional(literal(b"-"))?];
        seq.extend(self.natural()?);
        Ok(seq)
    }

    fn natural(&mut self) -> Result<Seq> {
        let digit = Item::Set(ByteSet::range(b'0', b'9'));
        let mut nonzero = vec![Item::Set(ByteSet::range(b'1', b'9'))];
        nonzero.extend(self.builder.repeat(vec![digit], 0, Some(MAX_DIGITS - 1))?);
        Ok(vec![self.builder.call(vec![literal(b"0"), nonzero])?])
    }

    fn number(&mut self, map: &Map<String, Value>) -> Result<Seq> {
        let digit = Item::Set(ByteSet::range(b'0', b'9'));
        // Only the sign can be enforced for fractional numbers.
        let non_negative = ["minimum", "exclusiveMinimum"]
            .iter()
            .filter_map(|name| map.get(*name).and_then(Value::as_f64))
            .any(|v| v >= 0.0);
        let mut seq = Vec::new();
        if !non_negative {
            seq.push(self.builder.optional(literal(b"-"))?);
        }
        seq.extend(self.natural()?);
        let mut fraction = literal(b".");
        fraction.extend(self.builder.repeat(vec![digit], 1, Some(MAX_DIGITS))?);
        seq.push(self.builder.optional(fraction)?);
        let mut exponent = vec![Item::Set({
            let mut e = ByteSet::range(b'e', b'e');
            e.insert_range(b'E', b'E');
            e
        })];
        exponent.push(self.builder.optional(vec![Item::Set({
            let mut sign = ByteSet::range(b'+', b'+');
            sign.insert_range(b'-', b'-');
            sign
        })])?);
        exponent.extend(self.builder.repeat(vec![digit], 1, Some(3))?);
        seq.push(self.builder.optional(exponent)?);
        Ok(seq)
    }

    fn separator(&self) -> Seq {
        let mut seq = vec![self.ws];
        seq.extend(literal(b","));
        seq.push(self.ws);
        seq
    }

    fn member(&self, key: Seq, value: Seq) -> Seq {
        let mut seq = key;
        seq.push(self.ws);
        seq.extend(literal(b":"));
        seq.push(self.ws);
        seq.extend(value);
        seq
    }

    fn object(&mut self, map: &Map<String, Value>) -> Result<Seq> {
        let empty = Map::new();
        let properties = match map.get("properties") {
            Some(properties) => properties
                .as_object()
                .context("properties must be an object")?,
            None => &empty,
        };
        let required: HashSet<&str> = map
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        for name in &required {
            if !properties.contains_key(*name)
                && map.get("additionalProperties") == Some(&Value::Bool(false))
            {
                bail!("'{name}' is required but not allowed");
            }
        }

        // Properties not listed are allowed only if the schema says so, or if it lists none.
        let extra: Option<Seq> = match map.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(self.any_value_seq()?),
            Some(schema) => Some(self.convert(schema)?),
            None if properties.is_empty() => Some(self.any_value_seq()?),
            None => None,
        };

        // Working backwards through the properties, `rest` is what can follow once at least
        // one property has been written, and `first` is what can follow if none has yet.
        let (mut rest, mut first): (Seq, Seq) = match extra {
            None => (Vec::new(), Vec::new()),
            Some(value) => {
                let pair = self.member(vec![self.string], value);
                let mut more = self.separator();
                more.extend(pair.clone());
                let tail = self.builder.star(more)?;
                let mut with_one = pair;
                with_one.push(tail);
                (vec![tail], vec![self.builder.optional(with_one)?])
            }
        };
        for (name, schema) in properties.iter().rev() {
            let key = literal(serde_json::to_string(name)?.as_bytes());
            let value = self.convert(schema)?;
            let mut member = self.member(key, value);
            member.extend(rest.iter().copied());
            let member = vec![self.builder.call(vec![member])?];
            let mut continued = self.separator();
            continued.extend(member.iter().copied());
            if required.contains(name.as_str()) {
                rest = continued;
                first = member;
            } else {
                rest = vec![self.builder.call(vec![continued, rest])?];
                first = vec![self.builder.call(vec![member, first])?];
            }
        }

        let mut seq = literal(b"{");
        seq.push(self.ws);
        seq.extend(first);
        seq.push(self.ws);
        seq.extend(literal(b"}"));
        Ok(seq)
    }

    fn array(&mut self, map: &Map<String, Value>) -> Result<Seq> {
        let min = uint(map, "minItems")?.unwrap_or(0);
        let max = uint(map, "maxItems")?;
        let open = |s: &Self| {
            let mut seq = literal(b"[");
            seq.push(s.ws);
            seq
        };
        let close = |s: &Self, seq: &mut Seq| {
            seq.push(s.ws);
            seq.extend(literal(b"]"));
        };

        if let Some(prefix) = map.get("prefixItems") {
            let prefix = prefix.as_array().context("prefixItems must be a list")?;
            let mut seq = open(self);
            for (i, schema) in prefix.iter().enumerate() {
                if i > 0 {
                    seq.extend(self.separator());
                }
                seq.extend(self.convert(schema)?);
            }
            if let Some(schema) = map.get("items").filter(|s| **s != Value::Bool(false)) {
                let mut more = self.separator();
                more.extend(self.convert(schema)?);
                seq.push(self.builder.star(more)?);
            }
            close(self, &mut seq);
            return Ok(seq);
        }

        let item = match map.get("items") {
            Some(schema) => self.convert(schema)?,
            None => self.any_value_seq()?,
        };
        if max == Some(0) {
            let mut seq = open(self);
            close(self, &mut seq);
            return Ok(seq);
        }
        if max.is_some_and(|max| max < min) {
            bail!("maxItems is less than minItems");
        }
        let mut more = self.separator();
        more.extend(item.iter().copied());
        let rest = self
            .builder
            .repeat(more, min.saturating_sub(1), max.map(|max| max - 1))?;
        let mut items = item;
        items.extend(rest);
        let mut seq = open(self);
        if min == 0 {
            seq.push(self.builder.optional(items)?);
        } else {
            seq.extend(items);
        }
        close(self, &mut seq);
        Ok(seq)
    }

    /// Rules for any JSON at all: (any object, any value).
    fn any_value(&mut self) -> Result<(RuleId, RuleId)> {
        if let Some(any) = self.any {
            return Ok(any);
        }
        let value = self.builder.declare();
        let object = self.builder.declare();
        let array = self.builder.declare();
        self.any = Some((object, value));

        let pair = self.member(vec![self.string], vec![Item::Call(value)]);
        let mut more = self.separator();
        more.extend(pair.iter().copied());
        let tail = self.builder.star(more)?;
        let mut members = pair;
        members.push(tail);
        let mut body = literal(b"{");
        body.push(self.ws);
        body.push(self.builder.optional(members)?);
        body.push(self.ws);
        body.extend(literal(b"}"));
        self.builder.define(object, vec![body])?;

        let mut more = self.separator();
        more.push(Item::Call(value));
        let tail = self.builder.star(more)?;
        let mut body = literal(b"[");
        body.push(self.ws);
        body.push(self.builder.optional(vec![Item::Call(value), tail])?);
        body.push(self.ws);
        body.extend(literal(b"]"));
        self.builder.define(array, vec![body])?;

        let number = self.number(&Map::new())?;
        self.builder.define(
            value,
            vec![
                vec![self.string],
                number,
                vec![Item::Call(object)],
                vec![Item::Call(array)],
                literal(b"true"),
                literal(b"false"),
                literal(b"null"),
            ],
        )?;
        Ok((object, value))
    }

    fn any_value_seq(&mut self) -> Result<Seq> {
        Ok(vec![Item::Call(self.any_value()?.1)])
    }
}

fn uint(map: &Map<String, Value>, name: &str) -> Result<Option<usize>> {
    match map.get(name) {
        None => Ok(None),
        Some(value) => match value.as_u64() {
            Some(n) => Ok(Some(n as usize)),
            None => bail!("{name} must be a non-negative integer"),
        },
    }
}

/// Patterns for the common values of the `format` keyword. Any other format means no restriction.
fn format_pattern(format: &str) -> Option<&'static str> {
    Some(match format {
        "date" => r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])",
        "time" => r"([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d{1,9})?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)",
        "date-time" => {
            r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])T([01]\d|2[0-3]):[0-5]\d:[0-5]\d(\.\d{1,9})?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)"
        }
        "uuid" => r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "email" => r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}",
        "ipv4" => r"((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)",
        "uri" => r"[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]+",
        _ => return None,
    })
}

/// Sequences of digit ranges that together match the integers in `low..=high`, with no leading
/// zeros: `17..=132` becomes `1[7-9]`, `[2-9][0-9]`, `1[0-2][0-9]`, `13[0-2]`.
fn digit_ranges(low: u64, high: u64, out: &mut Vec<Seq>) {
    let digits = |n: u64| n.to_string().into_bytes();
    let (low_len, high_len) = (digits(low).len(), digits(high).len());
    for len in low_len..=high_len {
        // Clip the range to the numbers with exactly `len` digits.
        let floor = if len == 1 {
            0
        } else {
            10u64.pow(len as u32 - 1)
        };
        let ceiling = 10u64.pow(len as u32) - 1;
        same_length_ranges(
            &digits(low.max(floor)),
            &digits(high.min(ceiling)),
            &mut Vec::new(),
            out,
        );
    }
}

fn same_length_ranges(low: &[u8], high: &[u8], prefix: &mut Seq, out: &mut Vec<Seq>) {
    let digit = |a: u8, b: u8| Item::Set(ByteSet::range(a, b));
    if low.is_empty() {
        out.push(prefix.clone());
        return;
    }
    if low[0] == high[0] {
        prefix.push(digit(low[0], low[0]));
        same_length_ranges(&low[1..], &high[1..], prefix, out);
        prefix.pop();
        return;
    }
    // Numbers from `low` up to the end of its first digit.
    let mut first = low[0];
    if low[1..].iter().any(|&d| d != b'0') {
        prefix.push(digit(low[0], low[0]));
        same_length_ranges(&low[1..], &vec![b'9'; low.len() - 1], prefix, out);
        prefix.pop();
        first += 1;
    }
    // Numbers from the start of `high`'s first digit up to `high`.
    let mut last = high[0];
    if high[1..].iter().any(|&d| d != b'9') {
        prefix.push(digit(high[0], high[0]));
        same_length_ranges(&vec![b'0'; high.len() - 1], &high[1..], prefix, out);
        prefix.pop();
        last -= 1;
    }
    // Whole first digits in between: any digits at all after them.
    if first <= last {
        let mut seq = prefix.clone();
        seq.push(digit(first, last));
        seq.extend(std::iter::repeat_n(digit(b'0', b'9'), low.len() - 1));
        out.push(seq);
    }
}

/// `allOf` is only supported for combining objects, which is how it's nearly always used.
fn merge_all_of(schemas: &[Value], root: &Value) -> Result<Value> {
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();
    let mut additional = None;
    for schema in schemas {
        let mut schema = schema.clone();
        // Follow references, so that the merge sees the object underneath.
        let mut hops = 0;
        while let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            hops += 1;
            if hops > 64 {
                bail!("the reference '{reference}' goes round in circles");
            }
            let pointer = reference
                .strip_prefix('#')
                .context("only local references work")?;
            schema = root
                .pointer(pointer)
                .with_context(|| format!("the reference '{reference}' doesn't lead anywhere"))?
                .clone();
        }
        let map = schema
            .as_object()
            .context("allOf must be a list of schemas")?;
        if map.get("type").is_some_and(|t| t != "object") || !map.contains_key("properties") {
            if schemas.len() == 1 {
                return Ok(schema);
            }
            bail!("allOf is only supported for combining objects");
        }
        if let Some(listed) = map["properties"].as_object() {
            properties.extend(listed.clone());
        }
        required.extend(
            map.get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .cloned(),
        );
        if let Some(value) = map.get("additionalProperties") {
            additional = Some(value.clone());
        }
    }
    let mut merged = Map::new();
    merged.insert("type".into(), "object".into());
    merged.insert("properties".into(), properties.into());
    merged.insert("required".into(), required.into());
    if let Some(value) = additional {
        merged.insert("additionalProperties".into(), value);
    }
    Ok(merged.into())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::matches;
    use super::*;
    use serde_json::json;

    fn check(schema: Value, good: &[&str], bad: &[&str]) {
        let mut builder = Builder::default();
        let root = compile(&schema, &mut builder).unwrap();
        for text in good {
            assert!(
                matches(&builder, root, text),
                "{schema} should match {text:?}"
            );
        }
        for text in bad {
            assert!(
                !matches(&builder, root, text),
                "{schema} shouldn't match {text:?}"
            );
        }
    }

    #[test]
    fn scalars() {
        check(
            json!({"type": "string"}),
            &[r#""hi""#, r#""a\nb""#, "\"é🙂\"", r#""é""#],
            &["hi", r#""a"b""#, "\"\n\""],
        );
        check(json!({"type": "boolean"}), &["true", "false"], &["yes"]);
        check(json!({"type": "null"}), &["null"], &["0"]);
        check(
            json!({"type": "integer"}),
            &["0", "-12", "1000"],
            &["01", "1.5", "-"],
        );
        check(
            json!({"type": "number"}),
            &["0", "-1.5", "2e10", "1.5E-3"],
            &["1.", ".5", "01"],
        );
    }

    #[test]
    fn integer_ranges() {
        let schema = json!({"type": "integer", "minimum": 17, "maximum": 132});
        let mut builder = Builder::default();
        let root = compile(&schema, &mut builder).unwrap();
        for n in -50..300 {
            assert_eq!(
                matches(&builder, root, &n.to_string()),
                (17..=132).contains(&n),
                "{n}"
            );
        }
        let schema = json!({"type": "integer", "minimum": -25, "exclusiveMaximum": 8});
        let mut builder = Builder::default();
        let root = compile(&schema, &mut builder).unwrap();
        for n in -100..100 {
            assert_eq!(
                matches(&builder, root, &n.to_string()),
                (-25..8).contains(&n),
                "{n}"
            );
        }
        check(
            json!({"type": "integer", "minimum": 0}),
            &["0", "7", "123456"],
            &["-1"],
        );
    }

    #[test]
    fn enums_and_constants() {
        check(
            json!({"enum": ["red", 3, null]}),
            &[r#""red""#, "3", "null"],
            &["red", "4"],
        );
        check(
            json!({"const": {"a": [1, 2]}}),
            &[r#"{"a":[1,2]}"#],
            &[r#"{"a":[1]}"#],
        );
    }

    #[test]
    fn objects() {
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "ok": {"type": "boolean"}},
            "required": ["age"],
        });
        check(
            schema,
            &[
                r#"{"age":1}"#,
                r#"{"name":"x","age":1}"#,
                r#"{"age":1,"ok":true}"#,
                r#"{"name":"x","age":1,"ok":false}"#,
                "{\n  \"name\": \"x\",\n  \"age\": 1\n}",
            ],
            &[
                r#"{}"#,
                r#"{"name":"x"}"#,
                r#"{"age":1,"name":"x"}"#,
                r#"{"age":1,}"#,
                r#"{"age":1,"z":2}"#,
            ],
        );
    }

    #[test]
    fn additional_properties() {
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}, "additionalProperties": {"type": "string"}});
        check(
            schema,
            &[r#"{}"#, r#"{"a":1,"b":"x","c":"y"}"#, r#"{"b":"x"}"#],
            &[r#"{"b":1}"#],
        );
        check(
            json!({"type": "object"}),
            &[r#"{}"#, r#"{"any":[1,{"b":null}]}"#],
            &["[]"],
        );
    }

    #[test]
    fn arrays() {
        check(
            json!({"type": "array", "items": {"type": "integer"}}),
            &["[]", "[1]", "[1, 2,3]"],
            &["[1,]", "[,]", "[\"a\"]"],
        );
        check(
            json!({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 3}),
            &["[1,2]", "[1,2,3]"],
            &["[1]", "[1,2,3,4]"],
        );
        check(
            json!({"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]}),
            &[r#"["a",1]"#],
            &[r#"["a"]"#, r#"[1,"a"]"#],
        );
    }

    #[test]
    fn strings_with_limits_and_patterns() {
        check(
            json!({"type": "string", "minLength": 2, "maxLength": 3}),
            &[r#""ab""#, r#""é€🙂""#],
            &[r#""a""#, r#""abcd""#],
        );
        check(
            json!({"type": "string", "pattern": "^[a-z]+$"}),
            &[r#""abc""#],
            &[r#""ABC""#, r#""""#],
        );
        check(
            json!({"type": "string", "format": "date"}),
            &[r#""2024-02-29""#],
            &[r#""2024-13-01""#, r#""soon""#],
        );
        check(
            json!({"type": "string", "format": "uuid"}),
            &[r#""123e4567-e89b-12d3-a456-426614174000""#],
            &[r#""123""#],
        );
    }

    #[test]
    fn combinations() {
        check(
            json!({"anyOf": [{"type": "integer"}, {"type": "string"}]}),
            &["1", r#""a""#],
            &["true"],
        );
        check(
            json!({"type": ["integer", "null"]}),
            &["1", "null"],
            &["\"a\""],
        );
        check(
            json!({"type": "integer", "nullable": true}),
            &["1", "null"],
            &["\"a\""],
        );
        check(
            json!({"allOf": [
                {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
            ]}),
            &[r#"{"a":1,"b":"x"}"#],
            &[r#"{"a":1}"#],
        );
    }

    #[test]
    fn references_and_recursion() {
        let schema = json!({
            "$defs": {"node": {"type": "object", "properties": {
                "value": {"type": "integer"},
                "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]},
            }, "required": ["value", "next"]}},
            "$ref": "#/$defs/node",
        });
        check(
            schema,
            &[
                r#"{"value":1,"next":null}"#,
                r#"{"value":1,"next":{"value":2,"next":null}}"#,
            ],
            &[r#"{"value":1}"#, r#"{"value":1,"next":{"value":2}}"#],
        );
    }

    #[test]
    fn any_value() {
        check(
            json!({}),
            &["1", "[]", r#"{"a":{"b":[true,null,"x",1.5]}}"#],
            &["{", "tru"],
        );
    }

    #[test]
    fn whitespace_is_limited() {
        check(
            json!({"type": "array", "items": {"type": "integer"}}),
            &["[1, 2]", "[\n  1,\n  2\n]"],
            &["[1,\n\n\n2]"],
        );
    }

    #[test]
    fn surrounding_whitespace_is_allowed() {
        check(json!({"type": "integer"}), &[" 1", "1\n", "\n1 "], &["  1"]);
    }

    #[test]
    fn mistakes_are_reported() {
        for schema in [
            json!({"type": "wibble"}),
            json!({"$ref": "#/nowhere"}),
            json!({"$ref": "http://example.com/x"}),
            json!({"enum": []}),
            json!({"type": "integer", "minimum": 5, "maximum": 1}),
            json!(false),
            json!({"$ref": "#"}),
            json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "allOf": [{"$ref": "#/$defs/a"}]}),
        ] {
            let mut builder = Builder::default();
            let result =
                compile(&schema, &mut builder).and_then(|_| builder.check_left_recursion());
            assert!(result.is_err(), "{schema}");
        }
    }

    #[test]
    fn any_object_rejects_other_values() {
        let mut builder = Builder::default();
        let root = compile_any_object(&mut builder).unwrap();
        assert!(matches(&builder, root, r#"{"a":[1,2]}"#));
        assert!(!matches(&builder, root, "[1]"));
    }
}

//! Chat prompt formatting.
//!
//! Chat models are fine-tuned on conversations laid out with particular marker tokens, and only
//! answer well when prompted in that same layout. Hugging Face models describe their layout with
//! a Jinja template, which this module runs the same way `transformers` does.

use crate::tokenizer::Tokenizer;
use crate::tools::ToolFormat;
use anyhow::{Context, Result, anyhow};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::{Kwargs, Serde, ValueKind};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Map, Value, json};
use std::collections::HashSet;
use std::path::Path;

/// A `{"role": ..., "content": ...}` message.
pub fn message(role: &str, content: &str) -> Value {
    json!({ "role": role, "content": content })
}

pub struct ChatTemplate {
    env: Environment<'static>,
    /// The source of the template conversations normally use.
    source: String,
    has_tool_template: bool,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Find the model's chat template. `None` if it doesn't have one.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let read_json = |name: &str| -> Value {
            let text = std::fs::read_to_string(dir.join(name)).unwrap_or_default();
            serde_json::from_str(&text).unwrap_or(Value::Null)
        };
        let config = read_json("tokenizer_config.json");
        let mut templates: Vec<(String, String)> = Vec::new();
        match &config["chat_template"] {
            Value::String(template) => templates.push(("default".into(), template.clone())),
            Value::Array(named) => {
                for entry in named {
                    if let (Some(name), Some(template)) =
                        (entry["name"].as_str(), entry["template"].as_str())
                    {
                        templates.push((name.to_string(), template.to_string()));
                    }
                }
            }
            _ => {
                if let Ok(template) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
                    templates.push(("default".into(), template));
                } else if let Some(template) =
                    read_json("chat_template.json")["chat_template"].as_str()
                {
                    templates.push(("default".into(), template.to_string()));
                }
            }
        }
        if templates.is_empty() {
            return Ok(None);
        }
        let token = |name: &str| match &config[name] {
            Value::String(token) => token.clone(),
            other => other["content"].as_str().unwrap_or_default().to_string(),
        };
        Self::from_templates(templates, token("bos_token"), token("eos_token")).map(Some)
    }

    fn from_templates(
        templates: Vec<(String, String)>,
        bos_token: String,
        eos_token: String,
    ) -> Result<Self> {
        let mut env = Environment::new();
        env.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .expect("the default delimiters are valid"),
        );
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        let has_tool_template = templates.iter().any(|(name, _)| name == "tool_use");
        let source = templates
            .iter()
            .find(|(name, _)| name == "default")
            .map(|(_, source)| source.clone())
            .unwrap_or_default();
        for (name, source) in templates {
            env.add_template_owned(name.clone(), source)
                .with_context(|| format!("couldn't parse the chat template '{name}'"))?;
        }
        env.get_template("default")
            .context("the model has no default chat template")?;
        Ok(Self {
            env,
            source,
            has_tool_template,
            bos_token,
            eos_token,
        })
    }

    /// The source of the template conversations normally use.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The notation this model writes tool calls in, if its template handles them.
    pub fn tool_format(&self) -> Option<ToolFormat> {
        let name = if self.has_tool_template {
            "tool_use"
        } else {
            "default"
        };
        let template = self.env.get_template(name).expect("checked in load");
        ToolFormat::detect(template.source())
    }

    /// Lay out a conversation, ending with the header that cues the assistant's reply.
    /// `messages` and `tools` are in OpenAI's format. `extra` becomes extra template variables.
    pub fn render(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        extra: Option<&Map<String, Value>>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let mut variables = extra.cloned().unwrap_or_default();
        variables.insert("messages".into(), messages.clone());
        variables.insert("add_generation_prompt".into(), add_generation_prompt.into());
        variables.insert("bos_token".into(), self.bos_token.clone().into());
        variables.insert("eos_token".into(), self.eos_token.clone().into());
        if let Some(tools) = tools {
            variables.insert("tools".into(), tools.clone());
        }
        let name = if tools.is_some() && self.has_tool_template {
            "tool_use"
        } else {
            "default"
        };
        let template = self.env.get_template(name).expect("checked in load");
        // Errors raised by the template are its way of saying the conversation isn't valid (say,
        // roles that don't alternate), so that message is the useful part.
        template.render(Serde(&variables)).map_err(|err| {
            anyhow!(
                "{}",
                err.detail().map_or_else(|| err.to_string(), String::from)
            )
        })
    }

    /// Render a conversation and tokenize it, ready to feed to the model.
    pub fn encode(&self, tokenizer: &Tokenizer, messages: &Value) -> Result<Vec<u32>> {
        let text = self.render(messages, None, None, true)?;
        Ok(encode_rendered(tokenizer, &text))
    }
}

/// Tokenize a rendered conversation. Some models (Llama 2 style) expect a beginning-of-sequence
/// token that the template leaves out, so it's added if it's missing.
pub fn encode_rendered(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    let mut tokens = tokenizer.encode(text, false);
    let bos = tokenizer.prefix_tokens();
    if !tokens.starts_with(bos) {
        tokens.splice(0..0, bos.iter().copied());
    }
    tokens
}

/// Token ids that end a reply: the model's end-of-sequence tokens plus the usual end-of-turn
/// markers.
pub fn stop_tokens(dir: &Path, config_eos: Vec<u32>, tokenizer: &Tokenizer) -> HashSet<u32> {
    let mut stop: HashSet<u32> = config_eos.into_iter().collect();
    let read_json = |name: &str| -> Value {
        let text = std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        serde_json::from_str(&text).unwrap_or(Value::Null)
    };
    match &read_json("generation_config.json")["eos_token_id"] {
        Value::Number(id) => stop.extend(id.as_u64().map(|id| id as u32)),
        Value::Array(ids) => stop.extend(ids.iter().filter_map(Value::as_u64).map(|id| id as u32)),
        _ => {}
    }
    let tokenizer_config = read_json("tokenizer_config.json");
    let eos = &tokenizer_config["eos_token"];
    let eos_name = eos.as_str().or_else(|| eos["content"].as_str());
    let markers = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|endoftext|>",
        "</s>",
        "<|eom_id|>",
        "<end_of_turn>",
        "<|end|>",
    ];
    for name in eos_name.into_iter().chain(markers) {
        if let Some(id) = tokenizer
            .token_id(name)
            .filter(|&id| tokenizer.is_special(id))
        {
            stop.insert(id);
        }
    }
    stop
}

fn strftime_now(format: String) -> Result<String, Error> {
    use std::fmt::Write;
    let mut out = String::new();
    write!(out, "{}", chrono::Local::now().format(&format))
        .map_err(|_| Error::new(ErrorKind::InvalidOperation, "invalid strftime format"))?;
    Ok(out)
}

/// The `tojson` filter as `transformers` defines it, which is Python's `json.dumps` with
/// non-ASCII characters left alone. Models were trained on this exact formatting of tools.
fn tojson(value: minijinja::Value, kwargs: Kwargs) -> Result<minijinja::Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let _sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    let separators: Option<minijinja::Value> = kwargs.get("separators")?;
    kwargs.assert_all_used()?;
    let (item, key) = match separators {
        Some(s) => (
            s.get_item_by_index(0)?.to_string(),
            s.get_item_by_index(1)?.to_string(),
        ),
        None => (
            (if indent.is_some() { "," } else { ", " }).to_string(),
            ": ".to_string(),
        ),
    };
    let style = JsonStyle {
        indent,
        ensure_ascii: ensure_ascii.unwrap_or(false),
        item,
        key,
    };
    let mut out = String::new();
    write_json(&value, &style, 0, &mut out)?;
    // Marked safe so nothing downstream escapes it.
    Ok(minijinja::Value::from_safe_string(out))
}

struct JsonStyle {
    indent: Option<usize>,
    ensure_ascii: bool,
    item: String,
    key: String,
}

fn write_json(
    value: &minijinja::Value,
    style: &JsonStyle,
    depth: usize,
    out: &mut String,
) -> Result<(), Error> {
    let newline = |out: &mut String, depth: usize| {
        if let Some(indent) = style.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', indent * depth));
        }
    };
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => match f64::try_from(value.clone()) {
            // Python writes floats with a decimal point and integers without.
            Ok(number) if !value.is_integer() => {
                let text = number.to_string();
                out.push_str(&text);
                if !text.contains(['.', 'e', 'E']) {
                    out.push_str(".0");
                }
            }
            _ => out.push_str(&value.to_string()),
        },
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<_> = value.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item);
                }
                newline(out, depth + 1);
                write_json(item, style, depth + 1, out)?;
            }
            newline(out, depth);
            out.push(']');
        }
        ValueKind::Map => {
            let keys: Vec<_> = value.try_iter()?.collect();
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&style.item);
                }
                newline(out, depth + 1);
                write_json_string(&key.to_string(), style, out);
                out.push_str(&style.key);
                write_json(&value.get_item(key)?, style, depth + 1, out)?;
            }
            newline(out, depth);
            out.push('}');
        }
        _ => write_json_string(&value.to_string(), style, out),
    }
    Ok(())
}

fn write_json_string(text: &str, style: &JsonStyle, out: &mut String) {
    use std::fmt::Write;
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (style.ensure_ascii && !c.is_ascii()) => {
                let mut units = [0; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(source: &str) -> ChatTemplate {
        ChatTemplate::from_templates(
            vec![("default".to_string(), source.to_string())],
            "<s>".to_string(),
            "</s>".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn renders_like_transformers() {
        let t = template(
            "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'].strip() }}{{ eos_token }}\n{% endfor %}\
             {% if add_generation_prompt %}<|assistant|>{% endif %}",
        );
        let messages = json!([message("user", " Hi ")]);
        assert_eq!(
            t.render(&messages, None, None, true).unwrap(),
            "<|user|>Hi</s>\n<|assistant|>"
        );
        assert_eq!(
            t.render(&messages, None, None, false).unwrap(),
            "<|user|>Hi</s>\n"
        );
    }

    #[test]
    fn tojson_is_formatted_like_python() {
        let t = template("{{ tools | tojson }}|{{ tools[0] | tojson(indent=2) }}");
        let tools = json!([{"name": "é", "n": 1, "x": 1.5, "o": {"b": [true, null]}, "e": []}]);
        assert_eq!(
            t.render(&json!([]), Some(&tools), None, false).unwrap(),
            "[{\"name\": \"é\", \"n\": 1, \"x\": 1.5, \"o\": {\"b\": [true, null]}, \"e\": []}]|\
             {\n  \"name\": \"é\",\n  \"n\": 1,\n  \"x\": 1.5,\n  \"o\": {\n    \"b\": [\n      true,\n      null\n    ]\n  },\n  \"e\": []\n}"
        );
    }

    #[test]
    fn raise_exception_reports_its_message() {
        let t = template("{{ raise_exception('roles must alternate') }}");
        let err = t.render(&json!([]), None, None, false).unwrap_err();
        assert_eq!(err.to_string(), "roles must alternate");
    }
}

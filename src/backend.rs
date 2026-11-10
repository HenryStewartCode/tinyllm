//! Where the model runs: the engine only needs something that can take a batch of tokens and
//! produce logits, which keeps scheduling, caching and sampling independent of the hardware.

use crate::cache::KvPool;
use crate::config::Config;
use crate::lora::LoraAdapter;
use crate::model::{Model, Row, Workspace};
use crate::pool::Pool;
use anyhow::{Result, bail};
use std::path::Path;

pub trait Backend: Send {
    /// Called on the thread that's going to use the backend, before anything else.
    fn attach(&mut self) {}

    /// What the model is running on.
    fn describe(&self) -> String;

    /// Run a batch of tokens through the model, storing their keys and values in their KV slots.
    /// Rows from one sequence must be next to each other, in order.
    fn forward(&mut self, rows: &[Row]);

    /// The scores from the last `forward` for the `row`-th row that wanted logits.
    fn logits(&self, row: usize) -> Logits<'_>;

    /// The final hidden state from the last `forward` for the `row`-th row that wanted logits,
    /// if it asked for it with [`crate::model::LogitUse::Hidden`].
    fn hidden(&self, row: usize) -> &[f32];

    /// CPU threads for parallel work between steps.
    fn pool(&self) -> &Pool;

    /// How many tokens' keys and values fit in the KV cache.
    fn capacity(&self) -> usize;

    /// KV slots come in pages of this many consecutive slots, holding consecutive tokens, with
    /// token `p` of a sequence in slot `p % page_size` of its page.
    fn page_size(&self) -> usize {
        1
    }

    /// Add a LoRA adapter, which rows then ask for by number: the first one added is 1, the
    /// next 2, and so on.
    fn add_adapter(&mut self, _adapter: &LoraAdapter) -> Result<()> {
        bail!("{} can't run LoRA adapters yet", self.describe())
    }

    /// From now on, keep each row's residual stream after each of `layers` in every `forward`,
    /// for [`Backend::features`]: the hidden states an EAGLE head guesses from.
    fn capture_layers(&mut self, _layers: &[usize]) -> Result<()> {
        bail!("{} can't keep hidden states yet", self.describe())
    }

    /// What was kept for row `row` in the last `forward` (counting every row, not only those
    /// that wanted logits): the residual stream after each layer asked for, one after another.
    fn features(&self, _row: usize) -> &[f32] {
        &[]
    }

    /// For an EAGLE-3 head: the hidden state each row of the next `forward` starts from,
    /// `width` floats a row. That's its model's kept features (`3 * dim`), which the head folds
    /// into one, or the head's own features (`dim`) from the row for the position before.
    fn set_hidden_inputs(&mut self, _width: usize, _values: &[f32]) -> Result<()> {
        bail!("{} isn't an EAGLE head", self.describe())
    }
}

/// Scores for the next token.
#[derive(Clone, Copy, Debug)]
pub enum Logits<'a> {
    /// Every token's score, by token id.
    All(&'a [f32]),
    /// `(score, token)` for just the tokens in contention, in no particular order.
    Candidates(&'a [(f32, u32)]),
}

/// A model that's been loaded onto some backend, and what was learned doing it.
pub struct LoadedModel<B> {
    pub config: Config,
    pub num_params: usize,
    pub backend: B,
}

/// Runs the model on the CPU.
pub struct CpuBackend {
    model: Model,
    kv: KvPool,
    workspace: Workspace,
    threads: usize,
}

impl CpuBackend {
    pub fn load(dir: &Path, threads: usize, cache_bytes: usize) -> Result<LoadedModel<Self>> {
        Self::load_with(dir, None, threads, cache_bytes)
    }

    /// `load`, taking weights the checkpoint doesn't have (an EAGLE head's embeddings) from the
    /// one in `fallback`.
    pub fn load_with(
        dir: &Path,
        fallback: Option<&Path>,
        threads: usize,
        cache_bytes: usize,
    ) -> Result<LoadedModel<Self>> {
        let model = Model::load_with(dir, fallback)?;
        let capacity = (cache_bytes / model.kv_bytes_per_token()).max(16);
        Ok(LoadedModel {
            config: model.config.clone(),
            num_params: model.num_params(),
            backend: Self {
                kv: model.new_kv_pool(capacity),
                model,
                workspace: Workspace::new(threads),
                threads,
            },
        })
    }
}

impl Backend for CpuBackend {
    fn describe(&self) -> String {
        format!("CPU ({} threads)", self.threads)
    }

    fn forward(&mut self, rows: &[Row]) {
        self.model.forward(rows, &mut self.kv, &mut self.workspace);
    }

    fn logits(&self, row: usize) -> Logits<'_> {
        Logits::All(self.workspace.logits(row))
    }

    fn hidden(&self, row: usize) -> &[f32] {
        self.workspace.hidden(row)
    }

    fn pool(&self) -> &Pool {
        self.workspace.pool()
    }

    fn capacity(&self) -> usize {
        self.kv.capacity
    }

    fn add_adapter(&mut self, adapter: &LoraAdapter) -> Result<()> {
        self.model.add_adapter(adapter)
    }

    fn capture_layers(&mut self, layers: &[usize]) -> Result<()> {
        self.workspace.capture_layers(layers);
        Ok(())
    }

    fn features(&self, row: usize) -> &[f32] {
        self.workspace.features(row)
    }

    fn set_hidden_inputs(&mut self, width: usize, values: &[f32]) -> Result<()> {
        self.workspace.set_hidden_inputs(width, values);
        Ok(())
    }
}

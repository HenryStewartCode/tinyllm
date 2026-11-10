//! Stands in for the Metal backend where there's no Metal: anywhere but macOS.

use crate::backend::{Backend, LoadedModel, Logits};
use crate::model::Row;
use crate::pool::Pool;
use anyhow::{Result, bail};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Options {
    /// Memory for the KV cache, in bytes.
    pub cache_bytes: usize,
    /// CPU threads for work between GPU steps (choosing the next tokens).
    pub host_threads: usize,
}

/// Never made, since there's no Metal here: [`MetalBackend::load`] always fails.
pub struct MetalBackend(());

const NO_METAL: &str = "Metal is only on Macs";

impl MetalBackend {
    pub fn available() -> bool {
        false
    }

    pub fn load(_dir: &Path, _options: &Options) -> Result<LoadedModel<Self>> {
        bail!(NO_METAL)
    }

    #[doc(hidden)]
    pub fn profile(&mut self, _rows: &[Row]) -> Result<Vec<(String, f64)>> {
        bail!(NO_METAL)
    }
}

impl Backend for MetalBackend {
    fn describe(&self) -> String {
        unreachable!("{NO_METAL}")
    }

    fn forward(&mut self, _rows: &[Row]) {
        unreachable!("{NO_METAL}")
    }

    fn logits(&self, _row: usize) -> Logits<'_> {
        unreachable!("{NO_METAL}")
    }

    fn hidden(&self, _row: usize) -> &[f32] {
        unreachable!("{NO_METAL}")
    }

    fn pool(&self) -> &Pool {
        unreachable!("{NO_METAL}")
    }

    fn capacity(&self) -> usize {
        unreachable!("{NO_METAL}")
    }
}

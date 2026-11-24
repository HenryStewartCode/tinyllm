//! One model split across several GPUs (tensor parallelism).

use super::api::{Event, Gpu};
use super::{CudaBackend, Options, PADDING_ROWS, TpLink};
use crate::backend::{Backend, LoadedModel, Logits};
use crate::model::Row;
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use anyhow::{Result, ensure};
use std::path::Path;
use std::sync::{Arc, Barrier};

/// A model split across several GPUs (tensor parallelism): each holds its share of every layer's
/// attention heads and MLP, and they add up their outputs after each block. All of them run every
/// step, each from its own thread; the first's results are the ones read.
pub struct TensorParallel {
    ranks: Vec<CudaBackend>,
}

impl TensorParallel {
    /// Load a model split across the first `size` GPUs.
    pub fn load(dir: &Path, options: &Options, size: usize) -> Result<LoadedModel<Self>> {
        let available = Gpu::count()?;
        ensure!(
            size <= available,
            "{size} GPUs asked for, but there are {available}"
        );
        // Each GPU loads its share from its own thread, at the same time.
        let loaded = std::thread::scope(|scope| {
            let loads: Vec<_> = (0..size)
                .map(|rank| {
                    scope.spawn(move || CudaBackend::load_shard(dir, options, (rank, size)))
                })
                .collect();
            (loads.into_iter())
                .map(|load| load.join().expect("loading panicked"))
                .collect::<Result<Vec<_>>>()
        })?;
        let config = loaded[0].config.clone();
        // Embeddings, the output projection and norms are on every GPU: count them once.
        let shared = config.vocab_size
            * config.hidden_size
            * match SafeTensors::open_with(dir, None)?.contains("lm_head.weight") {
                true => 2,
                false => 1,
            };
        let num_params = loaded.iter().map(|l| l.num_params).sum::<usize>() - (size - 1) * shared;
        let mut ranks: Vec<CudaBackend> = loaded.into_iter().map(|l| l.backend).collect();
        for a in 0..size {
            for b in (0..size).filter(|&b| b != a) {
                ranks[a].gpu.enable_peer_access(&ranks[b].gpu)?;
            }
        }
        let points = 2 * ranks[0].layers.len();
        let rows = ranks[0].max_batch + PADDING_ROWS;
        let dim = ranks[0].dim;
        let (mut partials, mut events) = (Vec::new(), vec![None; points * size]);
        for (r, rank) in ranks.iter().enumerate() {
            rank.gpu.attach()?;
            let mut own = Vec::with_capacity(points);
            for p in 0..points {
                own.push(rank.gpu.alloc(rows * dim)?);
                events[p * size + r] = Some(rank.gpu.event()?);
            }
            partials.push(own);
        }
        let events: Arc<Vec<Event>> = Arc::new(events.into_iter().map(Option::unwrap).collect());
        let barrier = Arc::new(Barrier::new(size));
        for (r, rank) in ranks.iter_mut().enumerate() {
            rank.gpu.attach()?;
            let mut tables = Vec::with_capacity(points);
            for p in 0..points {
                let addresses: Vec<u64> = partials.iter().map(|own| own[p].0).collect();
                let table = rank.gpu.alloc(2 * size)?;
                rank.gpu.write_now(table, &addresses)?;
                tables.push(table);
            }
            rank.tp = Some(TpLink {
                rank: r,
                size,
                barrier: barrier.clone(),
                events: events.clone(),
                partials: partials[r].clone(),
                tables,
            });
        }
        ranks[0].gpu.attach()?;
        Ok(LoadedModel {
            config,
            num_params,
            backend: Self { ranks },
        })
    }
}

impl Backend for TensorParallel {
    fn attach(&mut self) {
        self.ranks[0].attach();
    }

    fn describe(&self) -> String {
        format!(
            "{}, split across {} GPUs",
            self.ranks[0].describe(),
            self.ranks.len()
        )
    }

    fn forward(&mut self, rows: &[Row]) {
        let (first, rest) = self.ranks.split_first_mut().expect("at least one GPU");
        std::thread::scope(|scope| {
            for rank in rest {
                scope.spawn(move || {
                    rank.attach();
                    rank.forward(rows);
                });
            }
            first.attach();
            first.forward(rows);
        });
    }

    fn logits(&self, row: usize) -> Logits<'_> {
        self.ranks[0].logits(row)
    }

    fn hidden(&self, row: usize) -> &[f32] {
        self.ranks[0].hidden(row)
    }

    fn pool(&self) -> &Pool {
        self.ranks[0].pool()
    }

    fn capacity(&self) -> usize {
        self.ranks.iter().map(|r| r.capacity()).min().unwrap_or(0)
    }

    fn page_size(&self) -> usize {
        self.ranks[0].page_size()
    }
}

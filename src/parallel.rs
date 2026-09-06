//! Coarse, disjoint output partitions with a serial path for small images.
use rayon::prelude::*;

pub(crate) const MIN_PARALLEL_BYTES: usize = 256 * 1024;
pub(crate) const CHUNK_PIXELS: usize = 16 * 1024;

pub(crate) fn chunks_mut(
    output: &mut [u8],
    chunk_size: usize,
    operation: impl Fn(usize, &mut [u8]) + Sync + Send,
) {
    chunks_mut_above(output, chunk_size, MIN_PARALLEL_BYTES, operation);
}

/// Memory-only work needs more bytes to amortize scheduling than arithmetic.
pub(crate) fn chunks_mut_above(
    output: &mut [u8],
    chunk_size: usize,
    minimum_bytes: usize,
    operation: impl Fn(usize, &mut [u8]) + Sync + Send,
) {
    if output.is_empty() {
        return;
    }
    if output.len() >= minimum_bytes {
        output
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| operation(i, chunk));
    } else {
        output
            .chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| operation(i, chunk));
    }
}

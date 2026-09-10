//! Coarse, disjoint output partitions with a serial path for small images.
use rayon::prelude::*;

pub(crate) const MIN_PARALLEL_BYTES: usize = 256 * 1024;
pub(crate) const CHUNK_PIXELS: usize = 16 * 1024;

/// Use consistent units for `work`, `chunk_size`, and `minimum_work` (bytes
/// or items). Preserve operation-specific cost cutoffs, but only schedule work
/// when there are multiple partitions and workers to execute them.
pub(crate) fn should_parallel(work: usize, chunk_size: usize, minimum_work: usize) -> bool {
    work >= minimum_work && work > chunk_size && rayon::current_num_threads() > 1
}

pub(crate) fn chunks_mut(output: &mut [u8], chunk_size: usize, operation: impl Fn(usize, &mut [u8]) + Sync + Send) {
    chunks_mut_above(output, chunk_size, MIN_PARALLEL_BYTES, operation);
}

/// Memory-only work needs more bytes to amortize scheduling than arithmetic.
pub(crate) fn chunks_mut_above(output: &mut [u8], chunk_size: usize, minimum_bytes: usize, operation: impl Fn(usize, &mut [u8]) + Sync + Send) {
    if output.is_empty() {
        return;
    }
    if should_parallel(output.len(), chunk_size, minimum_bytes) {
        output.par_chunks_mut(chunk_size).enumerate().for_each(|(i, chunk)| operation(i, chunk));
    } else {
        output.chunks_mut(chunk_size).enumerate().for_each(|(i, chunk)| operation(i, chunk));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduling_requires_work_partitions_and_workers() {
        for threads in [1, 2] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                assert!(!should_parallel(0, 16, 32));
                assert!(!should_parallel(31, 16, 32));
                assert!(!should_parallel(32, 32, 32));
                assert_eq!(should_parallel(32, 16, 32), threads > 1);
            });
        }
    }

    #[test]
    fn chunk_paths_preserve_indices_and_partial_tail() {
        for threads in [1, 2] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                for minimum in [0, usize::MAX] {
                    let mut output = [0; 35];
                    chunks_mut_above(&mut output, 16, minimum, |i, chunk| chunk.fill(i as u8 + 1));
                    assert_eq!(&output[..16], &[1; 16]);
                    assert_eq!(&output[16..32], &[2; 16]);
                    assert_eq!(&output[32..], &[3; 3]);
                }
            });
        }
    }
}

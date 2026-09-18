//! Coarse, disjoint output partitions with a serial path for small images.
use rayon::prelude::*;

pub(crate) const MIN_PARALLEL_BYTES: usize = 256 * 1024;
pub(crate) const CHUNK_PIXELS: usize = 16 * 1024;

/// Independent single-threaded codec candidates. Limit concurrent encoders
/// and retain only a batch of results, rather than every candidate buffer.
/// Ordered collection preserves tie breaking and error order across pools.
pub(crate) fn try_candidates<T: Sync, E: Send>(
    output: &mut Vec<u8>, jobs: &[T], work_bytes: usize, encode: impl Fn(&T) -> Result<Vec<u8>, E> + Sync,
) -> Result<(), E> {
    let workers = if work_bytes >= 128 * 1024 && jobs.len() > 1 {
        // Allow roughly four input-sized working buffers per encoder within
        // a 256 MiB budget. Very large images stay serial to bound memory.
        rayon::current_num_threads().min(4).min((64 * 1024 * 1024 / work_bytes.max(1)).max(1))
    } else {
        1
    };
    if workers == 1 {
        for job in jobs {
            let candidate = encode(job)?;
            if candidate.len() < output.len() {
                *output = candidate;
            }
        }
    } else {
        for batch in jobs.chunks(workers) {
            let candidates: Vec<_> = batch.par_iter().map(&encode).collect();
            for candidate in candidates {
                let candidate = candidate?;
                if candidate.len() < output.len() {
                    *output = candidate;
                }
            }
        }
    }
    Ok(())
}

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
    fn codec_search_preserves_ties_and_error_order() {
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let mut output = vec![0; 10];
                try_candidates::<_, ()>(&mut output, &[3, 2, 1], 256 * 1024, |&value| Ok(vec![value; 5])).unwrap();
                assert_eq!(output, vec![3; 5]);
                let error = try_candidates(&mut output, &[3, 2, 1], 256 * 1024, |&value| Err(value));
                assert_eq!(error, Err(3));
            });
        }
    }

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

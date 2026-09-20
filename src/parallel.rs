//! Coarse, disjoint output partitions with a serial path for small images.
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const MIN_PARALLEL_BYTES: usize = 256 * 1024;
pub(crate) const CHUNK_PIXELS: usize = 16 * 1024;

/// Independent single-threaded codec candidates, with deterministic ties.
pub(crate) fn try_candidates<T: Sync, E: Send>(
    output: &mut Vec<u8>, jobs: &[T], work_bytes: usize, encode: impl Fn(&T) -> Result<Vec<u8>, E> + Sync,
) -> Result<(), E> {
    if let Some(candidate) = best_candidate(jobs, work_bytes, output.len(), |job| encode(job).map(Some))? {
        *output = candidate;
    }
    Ok(())
}

/// A bounded number of workers pull jobs as soon as they become available.
/// Each retains only its smallest result; slow filters cannot hold up the next
/// batch. Indices, rather than completion order, break ties and order errors.
/// `None` lets native codecs discard losing buffers without copying to a Vec.
pub(crate) fn best_candidate<T: Sync, E: Send>(
    jobs: &[T], work_bytes: usize, limit: usize, encode: impl Fn(&T) -> Result<Option<Vec<u8>>, E> + Sync,
) -> Result<Option<Vec<u8>>, E> {
    let workers = candidate_workers(work_bytes, jobs.len());
    let next = AtomicUsize::new(0);
    let run = |_| {
        let mut result = CandidateResult::default();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = jobs.get(index) else { break };
            match encode(job) {
                Ok(Some(candidate)) if candidate.len() < limit => result.keep(index, candidate),
                Ok(_) => (),
                Err(error) => {
                    result.error = Some((index, error));
                    // All earlier jobs are already assigned. They still finish
                    // so the earliest error is independent of worker timing.
                    break;
                }
            }
        }
        result
    };
    let result = if workers == 1 {
        run(0)
    } else {
        (0..workers).into_par_iter().map(run).reduce(CandidateResult::default, |mut a, b| {
            if let Some((index, candidate)) = b.best {
                a.keep(index, candidate);
            }
            if let Some((index, error)) = b.error
                && a.error.as_ref().is_none_or(|(old, _)| index < *old)
            {
                a.error = Some((index, error));
            }
            a
        })
    };
    match result.error {
        Some((_, error)) => Err(error),
        None => Ok(result.best.map(|(_, bytes)| bytes)),
    }
}

pub(crate) fn candidate_workers(work_bytes: usize, jobs: usize) -> usize {
    if work_bytes >= 128 * 1024 && jobs > 1 {
        // Allow roughly four input-sized working buffers per encoder within
        // a 256 MiB budget. Very large images stay serial to bound memory.
        rayon::current_num_threads()
            .min(4)
            .min(jobs)
            .min((64 * 1024 * 1024 / work_bytes.max(1)).max(1))
    } else {
        1
    }
}

struct CandidateResult<E> {
    best: Option<(usize, Vec<u8>)>,
    error: Option<(usize, E)>,
}

impl<E> Default for CandidateResult<E> {
    fn default() -> Self {
        Self { best: None, error: None }
    }
}

impl<E> CandidateResult<E> {
    fn keep(&mut self, index: usize, candidate: Vec<u8>) {
        if self
            .best
            .as_ref()
            .is_none_or(|(old, bytes)| (candidate.len(), index) < (bytes.len(), *old))
        {
            self.best = Some((index, candidate));
        }
    }
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
    fn codec_queue_handles_more_jobs_than_workers_and_discards_losers() {
        for threads in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let jobs: Vec<_> = (0..31).collect();
                let mut output = vec![0; 100];
                try_candidates::<_, ()>(&mut output, &jobs, 256 * 1024, |&index| {
                    // Winners occur beyond the first worker batch, with ties.
                    Ok(vec![index as u8; if index >= 11 { 5 } else { 10 }])
                })
                .unwrap();
                assert_eq!(output, vec![11; 5]);
                let empty = best_candidate::<_, ()>(&jobs, 256 * 1024, 5, |_| Ok(None)).unwrap();
                assert!(empty.is_none());
                let error = best_candidate(&jobs, 256 * 1024, 5, |&index| {
                    if index == 7 || index == 19 { Err(index) } else { Ok(Some(vec![0; 4])) }
                });
                assert_eq!(error, Err(7));
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

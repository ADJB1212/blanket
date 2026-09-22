//! Coarse, disjoint output partitions with a serial path for small images.
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const MIN_PARALLEL_BYTES: usize = 256 * 1024;
pub(crate) const CHUNK_PIXELS: usize = 16 * 1024;

/// Live exclusive bound. Encoders may recheck it after expensive work, before
/// copying or decoding a candidate that a concurrent job has already beaten.
pub(crate) struct CandidateLimit(AtomicUsize);

impl CandidateLimit {
    pub(crate) fn new(limit: usize) -> Self {
        Self(AtomicUsize::new(limit))
    }

    pub(crate) fn get(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn set(&self, limit: usize) {
        self.0.store(limit, Ordering::Relaxed);
    }
}

/// Independent single-threaded codec candidates, with deterministic ties.
#[cfg(test)]
pub(crate) fn try_candidates<T: Sync, E: Send>(
    output: &mut Vec<u8>, jobs: &[T], work_bytes: usize, encode: impl Fn(&T) -> Result<Vec<u8>, E> + Sync,
) -> Result<(), E> {
    if let Some(candidate) = best_candidate(jobs, work_bytes, output.len(), |job, _| encode(job).map(Some))? {
        *output = candidate;
    }
    Ok(())
}

/// A bounded number of workers pull jobs as soon as they become available.
/// Each retains only its smallest result; slow filters cannot hold up the next
/// batch. Indices, rather than completion order, break ties and order errors.
/// `None` lets native codecs discard losing buffers without copying to a Vec.
/// The encoder receives an exclusive size bound, tightened after each winner.
/// Allow ties with that winner because an earlier job may still be running.
pub(crate) fn best_candidate<T: Sync, E: Send>(
    jobs: &[T], work_bytes: usize, limit: usize, encode: impl Fn(&T, &CandidateLimit) -> Result<Option<Vec<u8>>, E> + Sync,
) -> Result<Option<Vec<u8>>, E> {
    let workers = candidate_workers(work_bytes, jobs.len());
    let next = AtomicUsize::new(0);
    let bound = CandidateLimit::new(limit);
    let run = |_| {
        let mut result = CandidateResult::default();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(job) = jobs.get(index) else { break };
            match encode(job, &bound) {
                Ok(Some(candidate)) if candidate.len() < limit => {
                    bound.0.fetch_min(candidate.len() + 1, Ordering::Relaxed);
                    result.keep(index, candidate);
                }
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
        // Candidates are single-threaded, so every pool thread can run one.
        // Allow roughly four input-sized working buffers per encoder within
        // a 256 MiB budget. Very large images stay serial to bound memory.
        rayon::current_num_threads().min(jobs).min((64 * 1024 * 1024 / work_bytes.max(1)).max(1))
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
pub(crate) fn chunks_mut_above<T: Send>(
    output: &mut [T], chunk_size: usize, minimum_bytes: usize, operation: impl Fn(usize, &mut [T]) + Sync + Send,
) {
    if output.is_empty() {
        return;
    }
    if should_parallel(output.len(), chunk_size, minimum_bytes.div_ceil(std::mem::size_of::<T>().max(1))) {
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
                let empty = best_candidate::<_, ()>(&jobs, 256 * 1024, 5, |_, _| Ok(None)).unwrap();
                assert!(empty.is_none());
                let error = best_candidate(&jobs, 256 * 1024, 5, |&index, _| {
                    if index == 7 || index == 19 { Err(index) } else { Ok(Some(vec![0; 4])) }
                });
                assert_eq!(error, Err(7));
            });
        }
    }

    #[test]
    fn live_candidate_bound_skips_verification_after_a_concurrent_winner() {
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let (send, receive) = std::sync::mpsc::channel();
            let receive = std::sync::Mutex::new(receive);
            let verified = AtomicUsize::new(0);
            let winner = best_candidate::<_, ()>(&[0, 1, 2], 256 * 1024, 100, |&index, limit| {
                if index == 0 {
                    // Simulate an encode finishing after job 1 publishes its
                    // winner and job 2 observes the tightened bound.
                    receive.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10)).unwrap();
                    if 50 >= limit.get() {
                        return Ok(None);
                    }
                    verified.fetch_add(1, Ordering::Relaxed);
                    Ok(Some(vec![0; 50]))
                } else if index == 1 {
                    Ok(Some(vec![1; 5]))
                } else {
                    assert_eq!(limit.get(), 6);
                    send.send(()).unwrap();
                    Ok(None)
                }
            })
            .unwrap()
            .unwrap();
            assert_eq!(winner, [1; 5]);
            assert_eq!(verified.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn candidate_bounds_tighten_without_discarding_earlier_ties() {
        let mut observed = Vec::new();
        let bounds = std::sync::Mutex::new(&mut observed);
        let winner = best_candidate::<_, ()>(&[9, 7, 7, 5], 0, 20, |&size, limit| {
            bounds.lock().unwrap().push(limit.get());
            Ok((size < limit.get()).then(|| vec![0; size]))
        })
        .unwrap()
        .unwrap();
        assert_eq!(winner.len(), 5);
        assert_eq!(observed, [20, 10, 8, 8]);

        // Job 0 finishes after job 1 has tightened the bound. Job 2 sees
        // that bound and releases job 0; equal sizes must still favor job 0.
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let (send, receive) = std::sync::mpsc::channel();
            let receive = std::sync::Mutex::new(receive);
            let winner = best_candidate::<_, ()>(&[0, 1, 2], 256 * 1024, 20, |&index, limit| {
                if index == 0 {
                    receive.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10)).unwrap();
                } else if index == 2 {
                    assert_eq!(limit.get(), 6);
                    send.send(()).unwrap();
                }
                Ok((5 < limit.get()).then(|| vec![index; 5]))
            })
            .unwrap()
            .unwrap();
            assert_eq!(winner, [0; 5]);
        });
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

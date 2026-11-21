use std::slice::ChunksExact;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

fn hash_chunks(packed: &[u8], width: usize) -> PyResult<ChunksExact<'_, u8>> {
    if width == 0 || !packed.len().is_multiple_of(width) {
        return Err(PyValueError::new_err("invalid packed hash width or buffer length"));
    }
    Ok(packed.chunks_exact(width))
}

fn hamming_distance(left: &[u8], right: &[u8]) -> usize {
    let (left_words, left_tail) = left.as_chunks::<8>();
    let (right_words, right_tail) = right.as_chunks::<8>();
    let words = left_words.iter().zip(right_words).map(|(&a, &b)| (u64::from_ne_bytes(a) ^ u64::from_ne_bytes(b)).count_ones() as usize);
    let tail = left_tail.iter().zip(right_tail).map(|(&a, &b)| (a ^ b).count_ones() as usize);
    words.chain(tail).sum()
}

fn use_early_exit(width: usize, limit: usize) -> bool {
    width > 8 && limit <= 64 && limit < width.saturating_mul(2)
}

#[inline]
fn distance_within<const EARLY_EXIT: bool>(left: &[u8], right: &[u8], limit: usize) -> Option<usize> {
    if !EARLY_EXIT {
        let distance = hamming_distance(left, right);
        return (distance <= limit).then_some(distance);
    }
    let (left_words, left_tail) = left.as_chunks::<8>();
    let (right_words, right_tail) = right.as_chunks::<8>();
    let mut distance = 0;
    for (&a, &b) in left_words.iter().zip(right_words) {
        distance += (u64::from_ne_bytes(a) ^ u64::from_ne_bytes(b)).count_ones() as usize;
        if distance > limit {
            return None;
        }
    }
    distance += left_tail.iter().zip(right_tail).map(|(&a, &b)| (a ^ b).count_ones() as usize).sum::<usize>();
    (distance <= limit).then_some(distance)
}

#[pyfunction]
pub(super) fn hash_nearest_neighbors(py: Python<'_>, query: &[u8], packed: &[u8], max_distance: usize) -> PyResult<Vec<(usize, usize)>> {
    let hashes = hash_chunks(packed, query.len())?;
    Ok(py.detach(|| {
        if use_early_exit(query.len(), max_distance) {
            nearest_neighbors::<true>(query, hashes, max_distance)
        } else {
            nearest_neighbors::<false>(query, hashes, max_distance)
        }
    }))
}

fn nearest_neighbors<const EARLY_EXIT: bool>(query: &[u8], hashes: ChunksExact<'_, u8>, max_distance: usize) -> Vec<(usize, usize)> {
    let mut matches: Vec<_> = hashes
        .enumerate()
        .filter_map(|(position, hash)| distance_within::<EARLY_EXIT>(query, hash, max_distance).map(|distance| (position, distance)))
        .collect();
    matches.sort_unstable_by_key(|&(position, distance)| (distance, position));
    matches
}

#[pyfunction]
pub(super) fn hash_group(py: Python<'_>, packed: &[u8], width: usize, max_distance: usize) -> PyResult<Vec<Vec<usize>>> {
    let hashes = hash_chunks(packed, width)?;
    Ok(py.detach(|| group_hashes(hashes, max_distance)))
}

fn group_hashes<'a>(hashes: impl Iterator<Item = &'a [u8]>, max_distance: usize) -> Vec<Vec<usize>> {
    let mut unique = Vec::new();
    let mut indices = FxHashMap::default();
    let input_indices: Vec<_> = hashes
        .map(|hash| {
            *indices.entry(hash).or_insert_with(|| {
                let index = unique.len();
                unique.push(hash);
                index
            })
        })
        .collect();

    let mut group_by_hash: Vec<_> = (0..unique.len()).collect();
    if max_distance == 1 && unique.len() >= 256 && unique[0].len() == 8 {
        // On 64-bit hashes, enumerating the single-bit neighbors bounds each search to 64 lookups.
        let mut frontier = Vec::new();
        for seed in 0..unique.len() {
            if group_by_hash[seed] != seed {
                continue;
            }
            frontier.push(seed);
            while let Some(current) = frontier.pop() {
                let value = u64::from_ne_bytes(unique[current].try_into().unwrap());
                for bit in 0..u64::BITS {
                    let neighbor = (value ^ (1_u64 << bit)).to_ne_bytes();
                    if let Some(&candidate) = indices.get(neighbor.as_slice())
                        && candidate > seed
                        && group_by_hash[candidate] == candidate
                    {
                        group_by_hash[candidate] = seed;
                        frontier.push(candidate);
                    }
                }
            }
        }
    } else if max_distance > 0 && !unique.is_empty() {
        if use_early_exit(unique[0].len(), max_distance) {
            connect_hashes::<true>(&unique, &mut group_by_hash, max_distance);
        } else {
            connect_hashes::<false>(&unique, &mut group_by_hash, max_distance);
        }
    }

    let mut groups = vec![Vec::new(); unique.len()];
    for (position, index) in input_indices.into_iter().enumerate() {
        groups[group_by_hash[index]].push(position);
    }
    groups.retain(|group| !group.is_empty());
    groups
}

fn connect_hashes<const EARLY_EXIT: bool>(unique: &[&[u8]], group_by_hash: &mut [usize], max_distance: usize) {
    // Keep the earliest input hash as the representative of each group.
    let mut remaining: Vec<_> = (0..unique.len()).rev().collect();
    let mut frontier = Vec::new();
    while let Some(seed) = remaining.pop() {
        frontier.push(seed);
        while let Some(current) = frontier.pop() {
            remaining.retain(|&candidate| {
                if distance_within::<EARLY_EXIT>(unique[current], unique[candidate], max_distance).is_none() {
                    return true;
                }
                group_by_hash[candidate] = seed;
                frontier.push(candidate);
                false
            });
        }
    }
}

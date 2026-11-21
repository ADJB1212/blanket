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

#[pyfunction]
pub(super) fn hash_nearest_neighbors(py: Python<'_>, query: &[u8], packed: &[u8], max_distance: usize) -> PyResult<Vec<(usize, usize)>> {
    let hashes = hash_chunks(packed, query.len())?;
    Ok(py.detach(|| {
        let mut matches: Vec<_> = hashes
            .enumerate()
            .filter_map(|(position, hash)| {
                let distance = hamming_distance(query, hash);
                (distance <= max_distance).then_some((position, distance))
            })
            .collect();
        matches.sort_unstable_by_key(|&(position, distance)| (distance, position));
        matches
    }))
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
    } else if max_distance > 0 {
        // Keep the earliest input hash as the representative of each group.
        let mut remaining: Vec<_> = (0..unique.len()).rev().collect();
        let mut frontier = Vec::new();
        while let Some(seed) = remaining.pop() {
            frontier.push(seed);
            while let Some(current) = frontier.pop() {
                remaining.retain(|&candidate| {
                    if hamming_distance(unique[current], unique[candidate]) > max_distance {
                        return true;
                    }
                    group_by_hash[candidate] = seed;
                    frontier.push(candidate);
                    false
                });
            }
        }
    }

    let mut groups = vec![Vec::new(); unique.len()];
    for (position, index) in input_indices.into_iter().enumerate() {
        groups[group_by_hash[index]].push(position);
    }
    groups.retain(|group| !group.is_empty());
    groups
}

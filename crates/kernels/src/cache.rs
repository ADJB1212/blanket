use std::hash::{BuildHasher, Hash};
use std::num::NonZeroUsize;

use lru::LruCache;
use rustc_hash::FxBuildHasher;

pub struct ByteLru<K, V, S = FxBuildHasher> {
    entries: LruCache<K, (V, usize), S>,
    bytes: usize,
    limit: usize,
}

impl<K: Hash + Eq, V> ByteLru<K, V, FxBuildHasher> {
    pub fn new(entries: NonZeroUsize, limit: usize) -> Self {
        Self { entries: LruCache::with_hasher(entries, FxBuildHasher), bytes: 0, limit }
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> ByteLru<K, V, S> {
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.entries.get(key).map(|(value, _)| value)
    }

    pub fn insert(&mut self, key: K, value: V, bytes: usize) {
        if bytes > self.limit {
            return;
        }
        if let Some((_, old_bytes)) = self.entries.pop(&key) {
            self.bytes -= old_bytes;
        }
        while self.bytes > self.limit - bytes || self.entries.len() == self.entries.cap().get() {
            let (_, (_, old_bytes)) = self.entries.pop_lru().unwrap();
            self.bytes -= old_bytes;
        }
        self.entries.put(key, (value, bytes));
        self.bytes += bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_bytes_and_entries_and_refreshes_hits() {
        let mut cache = ByteLru::new(NonZeroUsize::new(2).unwrap(), 10);
        cache.insert(1, 11, 4);
        cache.insert(2, 22, 4);
        assert_eq!(cache.get(&1), Some(&11));
        cache.insert(3, 33, 4);
        assert_eq!(cache.get(&2), None);
        cache.insert(4, 44, 7);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&3), None);
        cache.insert(5, 55, 11);
        assert_eq!(cache.get(&4), Some(&44));
        assert_eq!(cache.get(&5), None);
        cache.insert(4, 45, 2);
        assert_eq!(cache.bytes, 2);
        assert_eq!(cache.get(&4), Some(&45));
    }
}

//! Exact squared RGBA distance searches, with first-entry tie breaking.
use std::simd::{Simd, num::SimdUint};
type Color = [u8; 4];

pub(crate) struct PaletteSearch<'a> {
    entries: &'a [Color],
}

impl<'a> PaletteSearch<'a> {
    pub(crate) fn new(entries: &'a [Color]) -> Self {
        Self { entries }
    }

    pub(crate) fn nearest(&self, color: Color) -> usize {
        let mut best = (u32::MAX, 0);
        let end = self.entries.len() / 4 * 4;
        for i in (0..end).step_by(4) {
            let mut distances = Simd::<i32, 4>::splat(0);
            for (c, &channel) in color.iter().enumerate() {
                let values = Simd::<u8, 4>::gather_or_default(self.entries[i..i + 4].as_flattened(), Simd::from_array([c, c + 4, c + 8, c + 12]));
                let difference = values.cast::<i32>() - Simd::splat(i32::from(channel));
                distances += difference * difference;
            }
            for (lane, d) in distances.to_array().into_iter().enumerate() {
                if (d as u32) < best.0 {
                    best = (d as u32, i + lane);
                }
            }
        }
        for (i, &entry) in self.entries.iter().enumerate().skip(end) {
            let d = distance(color, entry);
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }
}

fn distance(a: Color, b: Color) -> u32 {
    a.into_iter().zip(b).map(|(a, b)| (i32::from(a) - i32::from(b)).pow(2) as u32).sum()
}

#[cfg(test)]
fn scalar(color: Color, entries: &[Color]) -> usize {
    entries
        .iter()
        .enumerate()
        .min_by_key(|(_, entry)| distance(color, **entry))
        .map_or(0, |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_matches_scalar_with_tails_ties_and_alpha() {
        let colors: Vec<Color> = (0..257).map(|i| [i as u8, (i * 79) as u8, (i * 137) as u8, (i * 213) as u8]).collect();
        for length in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 255, 256] {
            // Offset by one entry to exercise unaligned loads.
            let palette = &colors[1..length + 1];
            let search = PaletteSearch::new(palette);
            for &color in &colors {
                assert_eq!(search.nearest(color), scalar(color, palette));
            }
        }
        for color in [[0; 4], [255; 4], [255, 0, 255, 0]] {
            let entries = [color; 9];
            assert_eq!(PaletteSearch::new(&entries).nearest(color), 0);
        }
        let entries = [[255; 4], [254; 4], [253; 4], [0; 4]];
        assert_eq!(PaletteSearch::new(&entries).nearest([0; 4]), 3);
    }
}

//! Exact squared RGBA distance searches, with first-entry tie breaking.
type Color = [u8; 4];
type Search = fn(Color, &[Color]) -> usize;

pub(crate) struct PaletteSearch<'a> {
    entries: &'a [Color],
    search: Search,
}

impl<'a> PaletteSearch<'a> {
    pub(crate) fn new(entries: &'a [Color]) -> Self {
        #[cfg(target_arch = "aarch64")]
        let search: Search = neon;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let search: Search = if std::arch::is_x86_feature_detected!("sse2") { sse2 } else { scalar };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        let search: Search = scalar;
        Self { entries, search }
    }

    pub(crate) fn nearest(&self, color: Color) -> usize {
        (self.search)(color, self.entries)
    }
}

fn distance(a: Color, b: Color) -> u32 {
    a.into_iter().zip(b).map(|(a, b)| (i32::from(a) - i32::from(b)).pow(2) as u32).sum()
}

#[cfg(any(test, not(target_arch = "aarch64")))]
fn scalar(color: Color, entries: &[Color]) -> usize {
    entries
        .iter()
        .enumerate()
        .min_by_key(|(_, entry)| distance(color, **entry))
        .map_or(0, |(i, _)| i)
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
fn finish(color: Color, entries: &[Color], start: usize, mut best: (u32, usize)) -> usize {
    for (i, &entry) in entries.iter().enumerate().skip(start) {
        let d = distance(color, entry);
        if d < best.0 {
            best = (d, i);
        }
    }
    best.1
}

#[cfg(target_arch = "aarch64")]
fn neon(color: Color, entries: &[Color]) -> usize {
    use std::arch::aarch64::*;
    let mut done = 0;
    let mut best = (u32::MAX, 0);
    // SAFETY: NEON is mandatory on AArch64; each load contains four full
    // palette entries, and stores target a four-element local array.
    unsafe {
        let query = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(u32::from_ne_bytes(color)))));
        while done + 4 <= entries.len() {
            let values = vld1q_u8(entries.as_ptr().add(done).cast());
            let lo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(values))), query);
            let hi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(values))), query);
            // Squares fit unsigned 16 bits; widen before adding channels.
            let lo = vpaddlq_u16(vreinterpretq_u16_s16(vmulq_s16(lo, lo)));
            let hi = vpaddlq_u16(vreinterpretq_u16_s16(vmulq_s16(hi, hi)));
            let mut distances = [0; 4];
            vst1q_u32(distances.as_mut_ptr(), vpaddq_u32(lo, hi));
            for (lane, d) in distances.into_iter().enumerate() {
                if d < best.0 {
                    best = (d, done + lane);
                }
            }
            done += 4;
        }
    }
    finish(color, entries, done, best)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn sse2(color: Color, entries: &[Color]) -> usize {
    // SAFETY: selected only after the SSE2 feature check in PaletteSearch.
    unsafe { sse2_inner(color, entries) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sse2_inner(color: Color, entries: &[Color]) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut done = 0;
    let mut best = (u32::MAX, 0);
    // SAFETY: each unaligned load covers four validated entries; no padding
    // or alignment assumptions are made about the input or output.
    unsafe {
        let zero = _mm_setzero_si128();
        let query = _mm_unpacklo_epi8(_mm_set1_epi32(i32::from_le_bytes(color)), zero);
        while done + 4 <= entries.len() {
            let values = _mm_loadu_si128(entries.as_ptr().add(done).cast());
            let lo = _mm_sub_epi16(_mm_unpacklo_epi8(values, zero), query);
            let hi = _mm_sub_epi16(_mm_unpackhi_epi8(values, zero), query);
            let lo = _mm_madd_epi16(lo, lo);
            let hi = _mm_madd_epi16(hi, hi);
            let lo = _mm_add_epi32(lo, _mm_shuffle_epi32::<0xb1>(lo));
            let hi = _mm_add_epi32(hi, _mm_shuffle_epi32::<0xb1>(hi));
            let sums = _mm_unpacklo_epi64(_mm_shuffle_epi32::<0x88>(lo), _mm_shuffle_epi32::<0x88>(hi));
            let mut distances = [0u32; 4];
            _mm_storeu_si128(distances.as_mut_ptr().cast(), sums);
            for (lane, d) in distances.into_iter().enumerate() {
                if d < best.0 {
                    best = (d, done + lane);
                }
            }
            done += 4;
        }
    }
    finish(color, entries, done, best)
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

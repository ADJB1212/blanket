#[derive(Debug)]
pub(super) enum Error {
    NeedInput,
    Invalid,
}

pub(super) type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Default)]
pub(super) struct Bits {
    buffer: u64,
    remaining: u8,
}

pub(super) struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub bits: Bits,
}

impl Reader<'_> {
    /// Return whole unread entropy bytes to the input before committing its position.
    pub(super) fn release_lookahead(&mut self) {
        while self.bits.remaining >= 8 {
            self.pos -= 1;
            if self.data[self.pos] == 0 {
                while self.pos > 0 && self.data[self.pos - 1] == 255 {
                    self.pos -= 1;
                }
            }
            self.bits.buffer >>= 8;
            self.bits.remaining -= 8;
        }
    }

    /// Buffer entropy bytes up to the next marker or the end of the input.
    fn refill(&mut self) {
        while self.bits.remaining <= 56 {
            if let Some(word) = self.data.get(self.pos..self.pos + 8) {
                let word = u64::from_be_bytes(word.try_into().unwrap());
                let inverted = !word;
                if (inverted.wrapping_sub(0x0101_0101_0101_0101) & !inverted & 0x8080_8080_8080_8080) == 0 {
                    let count = ((64 - self.bits.remaining) / 8).min(7);
                    let shift = u32::from(count) * 8;
                    self.bits.buffer = (self.bits.buffer << shift) | (word >> (64 - shift));
                    self.bits.remaining += count * 8;
                    self.pos += usize::from(count);
                    continue;
                }
            }
            let Some(&byte) = self.data.get(self.pos) else {
                return;
            };
            let mut next = self.pos + 1;
            if byte == 255 {
                while self.data.get(next) == Some(&255) {
                    next += 1;
                }
                if self.data.get(next) != Some(&0) {
                    return;
                }
                next += 1;
            }
            self.pos = next;
            self.bits.buffer = (self.bits.buffer << 8) | u64::from(byte);
            self.bits.remaining += 8;
        }
    }

    #[inline]
    fn fill(&mut self, count: u8) -> Result<()> {
        if self.bits.remaining < count {
            self.refill();
            if self.bits.remaining < count {
                // Short at a marker is corrupt; short at the end of input waits for more.
                let mut next = self.pos;
                while self.data.get(next) == Some(&255) {
                    next += 1;
                }
                return Err(if next > self.pos && next < self.data.len() { Error::Invalid } else { Error::NeedInput });
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn read(&mut self, count: u8) -> Result<u16> {
        self.fill(count)?;
        self.bits.remaining -= count;
        Ok(((self.bits.buffer >> self.bits.remaining) & ((1 << count) - 1)) as u16)
    }

    #[inline]
    fn peek(&mut self, count: u8) -> Option<usize> {
        if self.bits.remaining < count {
            self.refill();
            if self.bits.remaining < count {
                return None;
            }
        }
        Some(((self.bits.buffer >> (self.bits.remaining - count)) & ((1 << count) - 1)) as usize)
    }

    pub(super) fn signed(&mut self, count: u8) -> Result<i16> {
        if count == 0 {
            return Ok(0);
        }
        if count > 11 {
            return Err(Error::Invalid);
        }
        let value = i32::from(self.read(count)?);
        Ok(if value < 1 << (count - 1) { value + 1 - (1 << count) } else { value } as i16)
    }

    pub(super) fn restart(&mut self, expected: u8) -> Result<()> {
        self.release_lookahead();
        self.bits = Bits::default();
        if self.data.get(self.pos) != Some(&255) {
            return Err(if self.pos == self.data.len() { Error::NeedInput } else { Error::Invalid });
        }
        while self.data.get(self.pos) == Some(&255) {
            self.pos += 1;
        }
        let marker = *self.data.get(self.pos).ok_or(Error::NeedInput)?;
        if marker != 0xd0 + expected {
            return Err(Error::Invalid);
        }
        self.pos += 1;
        Ok(())
    }
}

pub(super) struct Huffman {
    counts: [u8; 16],
    values: Vec<u8>,
    fast: [u16; 1024],
    fast_value: [u32; 1024],
    first_long: usize,
    offset_long: usize,
}

impl Huffman {
    pub(super) fn new(counts: [u8; 16], values: &[u8]) -> Result<Self> {
        let mut slots = 1i32;
        for count in counts {
            slots = slots * 2 - i32::from(count);
            if slots <= 0 {
                return Err(Error::Invalid);
            }
        }
        if values.is_empty() || values.len() > 256 || counts.iter().map(|&n| usize::from(n)).sum::<usize>() != values.len() {
            return Err(Error::Invalid);
        }
        let mut fast = [0; 1024];
        let mut code = 0;
        let mut offset = 0;
        for (index, &count) in counts.iter().enumerate().take(10) {
            let length = index + 1;
            for _ in 0..count {
                let entry = ((length as u16) << 8) | u16::from(values[offset]);
                fast[code << (10 - length)..(code + 1) << (10 - length)].fill(entry);
                code += 1;
                offset += 1;
            }
            code <<= 1;
        }
        let mut fast_value = [0; 1024];
        for (prefix, &entry) in fast.iter().enumerate() {
            let length = u32::from(entry >> 8);
            let size = u32::from(entry & 15);
            if length == 0 || length + size > 10 {
                continue;
            }
            let amplitude = (prefix as u32 >> (10 - length - size)) & ((1 << size) - 1);
            let value = if size != 0 && amplitude < 1 << (size - 1) { amplitude as i32 + 1 - (1 << size) } else { amplitude as i32 };
            fast_value[prefix] = ((length + size) << 24) | (u32::from(entry & 255) << 16) | u32::from(value as i16 as u16);
        }
        Ok(Self { counts, values: values.to_vec(), fast, fast_value, first_long: code, offset_long: offset })
    }

    #[inline]
    fn coefficient(&self, reader: &mut Reader<'_>) -> Result<(u8, i16)> {
        if let Some(prefix) = reader.peek(10) {
            let entry = self.fast_value[prefix];
            if entry != 0 {
                reader.bits.remaining -= (entry >> 24) as u8;
                return Ok(((entry >> 16) as u8, entry as i16));
            }
            let entry = self.fast[prefix];
            if entry != 0 {
                reader.bits.remaining -= (entry >> 8) as u8;
                let symbol = entry as u8;
                return Ok((symbol, reader.signed(symbol & 15)?));
            }
        }
        let symbol = self.symbol(reader)?;
        Ok((symbol, reader.signed(symbol & 15)?))
    }

    #[inline]
    fn symbol(&self, reader: &mut Reader<'_>) -> Result<u8> {
        let (mut code, mut first, mut offset, start) = if let Some(prefix) = reader.peek(10) {
            let entry = self.fast[prefix];
            if entry != 0 {
                reader.bits.remaining -= (entry >> 8) as u8;
                return Ok(entry as u8);
            }
            reader.bits.remaining -= 10;
            (prefix, self.first_long, self.offset_long, 10)
        } else {
            (0, 0, 0, 0)
        };
        for &count in &self.counts[start..] {
            code = (code << 1) | usize::from(reader.read(1)?);
            if code < first + usize::from(count) {
                return self.values.get(offset + code - first).copied().ok_or(Error::Invalid);
            }
            offset += usize::from(count);
            first = (first + usize::from(count)) << 1;
        }
        Err(Error::Invalid)
    }
}

pub(super) const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
    36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

pub(super) fn dc(reader: &mut Reader<'_>, table: &Huffman, predictor: &mut i16, block: &mut [i16; 64], al: u8) -> Result<()> {
    let (category, value) = table.coefficient(reader)?;
    if category > 11 {
        return Err(Error::Invalid);
    }
    *predictor = predictor.checked_add(value).ok_or(Error::Invalid)?;
    block[0] = predictor.checked_mul(1 << al).ok_or(Error::Invalid)?;
    Ok(())
}

fn correction(reader: &mut Reader<'_>, coefficient: &mut i16, bit: i16) -> Result<()> {
    if reader.read(1)? != 0 && (*coefficient & bit) == 0 {
        *coefficient = coefficient.checked_add(if *coefficient > 0 { bit } else { -bit }).ok_or(Error::Invalid)?;
        if !(-1023..=1023).contains(coefficient) {
            return Err(Error::Invalid);
        }
    }
    Ok(())
}

pub(super) fn ac(
    reader: &mut Reader<'_>, table: &Huffman, block: &mut [i16; 64], range: (u8, u8), al: u8, refine: bool, eob: &mut u16,
) -> Result<()> {
    let (start, end) = (usize::from(range.0.max(1)), usize::from(range.1));
    let bit = 1i16 << al;
    let mut k = start;
    if !refine && *eob == 0 {
        while k <= end {
            let (symbol, amplitude) = table.coefficient(reader)?;
            let (run, size) = (usize::from(symbol >> 4), symbol & 15);
            if size == 0 {
                if run != 15 {
                    if range.0 == 0 && run != 0 {
                        return Err(Error::Invalid);
                    }
                    *eob = (1u16 << run) + reader.read(run as u8)?;
                    break;
                }
                k += 16;
                if k > end + 1 {
                    return Err(Error::Invalid);
                }
                continue;
            }
            let value = amplitude.checked_mul(bit).ok_or(Error::Invalid)?;
            k += run;
            if size > 10 || !(-1023..=1023).contains(&value) || k > end {
                return Err(Error::Invalid);
            }
            block[k] = value;
            k += 1;
        }
    } else if *eob == 0 {
        while k <= end {
            let (symbol, amplitude) = if refine { (table.symbol(reader)?, 0) } else { table.coefficient(reader)? };
            let mut run = usize::from(symbol >> 4);
            let size = symbol & 15;
            if !refine && size > 10 {
                return Err(Error::Invalid);
            }
            let value = if refine {
                if size > 1 {
                    return Err(Error::Invalid);
                }
                if size == 1 { if reader.read(1)? == 1 { bit } else { -bit } } else { 0 }
            } else {
                amplitude.checked_mul(bit).ok_or(Error::Invalid)?
            };
            if !(-1023..=1023).contains(&value) {
                return Err(Error::Invalid);
            }
            if size == 0 && run != 15 {
                if range.0 == 0 && run != 0 {
                    return Err(Error::Invalid);
                }
                *eob = (1u16 << run) + reader.read(run as u8)?;
                break;
            }
            if size == 0 {
                run = 16;
            }
            while k <= end {
                let coefficient = &mut block[k];
                if refine && *coefficient != 0 {
                    correction(reader, coefficient, bit)?;
                } else if run == 0 {
                    break;
                } else {
                    run -= 1;
                }
                k += 1;
                if size == 0 && run == 0 {
                    break;
                }
            }
            if run != 0 {
                return Err(Error::Invalid);
            }
            if size != 0 {
                if k > end {
                    return Err(Error::Invalid);
                }
                block[k] = value;
                k += 1;
            }
        }
    }
    if *eob != 0 {
        if refine {
            for coefficient in &mut block[k..=end] {
                if *coefficient != 0 {
                    correction(reader, coefficient, bit)?;
                }
            }
        }
        *eob -= 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_lookup_matches_signed_amplitudes_and_byte_stuffing() {
        let table = Huffman::new([1; 16], &(0..16).collect::<Vec<_>>()).unwrap();
        let mut bits = Vec::new();
        let mut expected = Vec::new();
        for size in 0..=10 {
            let length = size + 1;
            let code = (1u32 << length) - 2;
            for amplitude in 0..1u32 << size {
                bits.extend((0..length).rev().map(|shift| ((code >> shift) & 1) as u8));
                bits.extend((0..size).rev().map(|shift| ((amplitude >> shift) & 1) as u8));
                let value = if size > 0 && amplitude < 1 << (size - 1) { amplitude as i32 + 1 - (1 << size) } else { amplitude as i32 };
                expected.push((size as u8, value as i16));
            }
        }
        bits.resize(bits.len().next_multiple_of(8), 1);
        let mut data = Vec::new();
        for byte in bits.as_chunks::<8>().0 {
            let byte = byte.iter().fold(0, |value, bit| (value << 1) | bit);
            data.push(byte);
            if byte == 255 {
                data.push(0);
            }
        }
        let end = data.len();
        data.extend_from_slice(&[255, 217]);
        let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
        for coefficient in expected {
            assert_eq!(table.coefficient(&mut reader).unwrap(), coefficient);
        }
        assert_eq!(reader.pos, end);
    }

    #[test]
    fn huffman_lookup_handles_all_code_lengths_and_marker_boundaries() {
        let values: Vec<u8> = (0..16).collect();
        let table = Huffman::new([1; 16], &values).unwrap();
        for lengths in [(1..=16).collect::<Vec<_>>(), vec![1], vec![8], vec![16]] {
            let mut bits = Vec::new();
            for &length in &lengths {
                let code = (1u32 << length) - 2;
                bits.extend((0..length).rev().map(|shift| ((code >> shift) & 1) as u8));
            }
            bits.resize(bits.len().next_multiple_of(8), 1);
            let mut data = Vec::new();
            for chunk in bits.as_chunks::<8>().0 {
                let byte = chunk.iter().fold(0, |value, &bit| (value << 1) | bit);
                data.push(byte);
                if byte == 255 {
                    data.push(0);
                }
            }
            let end = data.len();
            data.extend_from_slice(&[255, 217]);
            let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
            for length in lengths {
                assert_eq!(table.symbol(&mut reader).unwrap(), length as u8 - 1);
            }
            assert_eq!(reader.pos, end);
        }
    }

    #[test]
    fn short_codes_decode_without_a_full_lookahead_byte() {
        let table = Huffman::new([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[73]).unwrap();
        for suffix in [vec![], vec![255], vec![255, 217]] {
            let mut data = vec![0];
            data.extend_from_slice(&suffix);
            let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
            assert_eq!(reader.read(7).unwrap(), 0);
            assert_eq!(table.symbol(&mut reader).unwrap(), 73);
            assert_eq!(reader.pos, 1);
        }
    }

    #[test]
    fn released_lookahead_preserves_partial_bits_and_stuffed_bytes() {
        let data = [0xa5, 255, 255, 0, 0x3c, 255, 0xd0];
        let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
        assert_eq!(reader.read(3).unwrap(), 5);
        reader.release_lookahead();
        assert_eq!(reader.pos, 1);
        assert_eq!(reader.bits.remaining, 5);
        assert_eq!(reader.read(5).unwrap(), 5);
        assert_eq!(reader.read(8).unwrap(), 255);
        reader.release_lookahead();
        assert_eq!(reader.pos, 4);
        assert!(matches!(reader.restart(0), Err(Error::Invalid)));
        assert_eq!(reader.read(8).unwrap(), 0x3c);
        reader.restart(0).unwrap();
    }

    #[test]
    fn bit_reservoir_preserves_stuffing_and_restart_boundaries() {
        let data = [0xa5, 255, 0, 0x3c, 255, 0xd0, 0x8f];
        let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
        assert_eq!(reader.read(4).unwrap(), 10);
        assert_eq!(reader.read(12).unwrap(), 0x5ff);
        assert_eq!(reader.read(3).unwrap(), 1);
        reader.restart(0).unwrap();
        assert_eq!(reader.read(4).unwrap(), 8);
        let mut reader = Reader { data: &data, pos: 0, bits: Bits::default() };
        assert_eq!(reader.read(16).unwrap(), 0xa5ff);
        assert_eq!(reader.read(8).unwrap(), 0x3c);
        reader.restart(1).unwrap_err();
    }
}

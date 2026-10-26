#[derive(Debug)]
pub(super) enum Error {
    NeedInput,
    Invalid,
}

pub(super) type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Default)]
pub(super) struct Bits {
    buffer: u32,
    remaining: u8,
}

pub(super) struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
    pub bits: Bits,
}

impl Reader<'_> {
    #[inline]
    fn fill(&mut self, count: u8) -> Result<()> {
        while self.bits.remaining < count {
            let mut byte = *self.data.get(self.pos).ok_or(Error::NeedInput)?;
            self.pos += 1;
            if byte == 255 {
                loop {
                    byte = *self.data.get(self.pos).ok_or(Error::NeedInput)?;
                    self.pos += 1;
                    if byte != 255 {
                        break;
                    }
                }
                if byte != 0 {
                    return Err(Error::Invalid);
                }
                byte = 255;
            }
            self.bits.buffer = (self.bits.buffer << 8) | u32::from(byte);
            self.bits.remaining += 8;
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
    fn peek_byte(&mut self) -> Option<u8> {
        if self.bits.remaining < 8 {
            let checkpoint = (self.pos, self.bits);
            if self.fill(8).is_err() {
                (self.pos, self.bits) = checkpoint;
                return None;
            }
        }
        Some((self.bits.buffer >> (self.bits.remaining - 8)) as u8)
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
    fast: [u16; 256],
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
        let mut fast = [0; 256];
        let mut code = 0;
        let mut offset = 0;
        for (index, &count) in counts.iter().enumerate().take(8) {
            let length = index + 1;
            for _ in 0..count {
                let entry = ((length as u16) << 8) | u16::from(values[offset]);
                fast[code << (8 - length)..(code + 1) << (8 - length)].fill(entry);
                code += 1;
                offset += 1;
            }
            code <<= 1;
        }
        Ok(Self { counts, values: values.to_vec(), fast })
    }

    #[inline]
    fn symbol(&self, reader: &mut Reader<'_>) -> Result<u8> {
        if let Some(prefix) = reader.peek_byte() {
            let entry = self.fast[usize::from(prefix)];
            if entry != 0 {
                reader.bits.remaining -= (entry >> 8) as u8;
                return Ok(entry as u8);
            }
        }
        let mut code = 0;
        let mut first = 0;
        let mut offset = 0;
        for count in self.counts {
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
    let category = table.symbol(reader)?;
    *predictor = predictor.checked_add(reader.signed(category)?).ok_or(Error::Invalid)?;
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
    if *eob == 0 {
        while k <= end {
            let symbol = table.symbol(reader)?;
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
                reader.signed(size)?.checked_mul(bit).ok_or(Error::Invalid)?
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

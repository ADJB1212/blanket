// Row-aligned bilevel tasks target 128 KiB of unpacked samples.
const BILEVEL_CHUNK_BYTES: usize = 128 * 1024;
// Packing and unpacking use separate thresholds because their output sizes differ.
const PACK_PARALLEL_BYTES: usize = 512 * 1024;
const UNPACK_PARALLEL_BYTES: usize = 4 * 1024 * 1024;

const fn bilevel_bytes() -> [u64; 256] {
    let mut table = [0; 256];
    let mut value = 0;
    while value < 256 {
        let mut bit = 0;
        while bit < 8 {
            if value & (128 >> bit) != 0 {
                table[value] |= 255_u64 << (bit * 8);
            }
            bit += 1;
        }
        value += 1;
    }
    table
}

const BILEVEL_BYTES: [u64; 256] = bilevel_bytes();

pub fn unpack_bilevel(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    unpack_bilevel_bytes::<false>(data, width, height)
}

pub fn unpack_bilevel_inverted(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    unpack_bilevel_bytes::<true>(data, width, height)
}

fn unpack_bilevel_bytes<const INVERT: bool>(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let len = width * height;
    let mut pixels = Vec::with_capacity(len);
    if width == 0 {
        return pixels;
    }
    let row_bytes = width.div_ceil(8);
    let rows_per_chunk = (BILEVEL_CHUNK_BYTES / width).max(1);
    crate::parallel::chunks_mut_above(&mut pixels.spare_capacity_mut()[..len], rows_per_chunk * width, UNPACK_PARALLEL_BYTES, |chunk, rows| {
        for (row_index, row) in rows.chunks_mut(width).enumerate() {
            let y = chunk * rows_per_chunk + row_index;
            let packed = &data[y * row_bytes..(y + 1) * row_bytes];
            let (groups, tail) = row.as_chunks_mut::<8>();
            for (byte, dst) in packed.iter().zip(groups) {
                let value = BILEVEL_BYTES[*byte as usize];
                dst.write_copy_of_slice(&(if INVERT { !value } else { value }).to_le_bytes());
            }
            for (x, dst) in tail.iter_mut().enumerate() {
                dst.write(if (packed[width >> 3] & (128 >> x) != 0) != INVERT { 255 } else { 0 });
            }
        }
    });
    // SAFETY: Every reserved byte, including partial rows, was initialized above.
    unsafe { pixels.set_len(len) };
    pixels
}

pub fn pack_bilevel(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let row_bytes = width.div_ceil(8);
    let mut packed = vec![0; row_bytes * height];
    pack_bilevel_into(pixels, width, &mut packed);
    packed
}

pub fn pack_bilevel_into(pixels: &[u8], width: usize, packed: &mut [u8]) {
    if packed.is_empty() {
        return;
    }
    let row_bytes = width.div_ceil(8);
    let rows_per_chunk = (BILEVEL_CHUNK_BYTES / width).max(1);
    crate::parallel::chunks_mut_above(packed, rows_per_chunk * row_bytes, PACK_PARALLEL_BYTES, |chunk, rows| {
        for (row_index, row) in rows.chunks_mut(row_bytes).enumerate() {
            let y = chunk * rows_per_chunk + row_index;
            crate::pixels::pack_bilevel::<false>(&pixels[y * width..(y + 1) * width], row);
        }
    });
}

pub fn lab_raw_bytes(source: &[u8]) -> Vec<u8> {
    let mut output = source.to_vec();
    lab_raw_bytes_mut(&mut output);
    output
}

pub fn lab_raw_bytes_mut(pixels: &mut [u8]) {
    const MASK: [u8; 48] = {
        let mut mask = [128; 48];
        let mut i = 0;
        while i < 48 {
            mask[i] = 0;
            i += 3;
        }
        mask
    };
    let (blocks, tail) = pixels.as_chunks_mut::<48>();
    for block in blocks {
        *block = std::array::from_fn(|i| block[i] ^ MASK[i]);
    }
    for (value, mask) in tail.iter_mut().zip(MASK) {
        *value ^= mask;
    }
}

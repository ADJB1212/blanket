use blanket_codec_common::validate_dimensions;
use blanket_core::Image;
use libjpeg_turbo_rs::{ComponentCoefficients, JpegCoefficients, write_coefficients};

use super::entropy::{self, Bits, Error, Huffman, Reader, ZIGZAG};

const INPUT_LIMIT: usize = 65540;

struct Scan {
    components: Vec<(usize, usize, usize)>,
    start: u8,
    end: u8,
    ah: u8,
    al: u8,
    index: usize,
    width: usize,
    count: usize,
    predictors: [i16; 4],
    eob: u16,
    restart: u8,
    bits: Bits,
}

pub struct JpegStreamDecoder {
    input: Vec<u8> = Vec::new(),
    input_start: usize = 0,
    coefficients: Option<JpegCoefficients> = None,
    dc: [Option<Huffman>; 4] = [const { None }; 4],
    ac: [Option<Huffman>; 4] = [const { None }; 4],
    quant: [Option<[u16; 64]>; 4] = [None; 4],
    quant_slots: [usize; 4] = [0; 4],
    progression: [[i8; 64]; 4] = [[-1; 64]; 4],
    scan: Option<Scan> = None,
    staged: [(usize, usize, [i16; 64]); 10] = [(0, 0, [0; 64]); 10],
    progressive: bool = false,
    started: bool = false,
    seen_scan: bool = false,
    restart_interval: usize = 0,
    adobe: Option<u8> = None,
    jfif: bool = false,
    completed_color: (bool, Option<u8>) = (false, None),
    pixels: Option<Image> = None,
    failed: bool = false,
    pub complete: bool = false,
}

impl JpegStreamDecoder {
    pub fn new() -> Result<Self, String> {
        Ok(Self { input: Vec::with_capacity(INPUT_LIMIT), .. })
    }

    pub fn feed(&mut self, data: &[u8]) -> Result<(), String> {
        if self.failed {
            return Err("failed JPEG decoder".into());
        }
        let result = self.process_input(data);
        self.failed = result.is_err();
        result
    }

    fn process_input(&mut self, mut data: &[u8]) -> Result<(), String> {
        while !self.complete {
            self.input.drain(..self.input_start);
            self.input_start = 0;
            let count = data.len().min(INPUT_LIMIT - self.input.len());
            self.input.extend_from_slice(&data[..count]);
            data = &data[count..];
            loop {
                let consumed = if self.scan.is_some() { self.consume_scan()? } else { self.consume_marker()? };
                let Some(consumed) = consumed else {
                    break;
                };
                self.input_start += consumed;
                if self.complete {
                    self.input = Vec::new();
                    self.input_start = 0;
                    return Ok(());
                }
            }
            if self.input.len() - self.input_start == INPUT_LIMIT {
                return Err("JPEG input exceeds streaming buffer limit".into());
            }
            if data.is_empty() {
                self.input.drain(..self.input_start);
                self.input_start = 0;
                break;
            }
        }
        Ok(())
    }

    fn consume_marker(&mut self) -> Result<Option<usize>, String> {
        let input = &self.input[self.input_start..];
        if input.is_empty() {
            return Ok(None);
        }
        if input[0] != 255 {
            return Err("invalid JPEG marker".into());
        }
        let Some(offset) = input.iter().position(|&byte| byte != 255) else {
            return Ok(if input.len() > 1 { Some(input.len() - 1) } else { None });
        };
        let marker = input[offset];
        if !self.started {
            if marker != 0xd8 {
                return Err("invalid JPEG signature".into());
            }
            self.started = true;
            return Ok(Some(offset + 1));
        }
        if marker == 0xd9 {
            let coefficients = self.coefficients.as_ref().ok_or("JPEG stream has no frame")?;
            if !self.seen_scan || self.progression[..coefficients.components.len()].iter().any(|bits| bits[0] < 0) {
                return Err("JPEG stream has incomplete components".into());
            }
            self.complete = true;
            return Ok(Some(offset + 1));
        }
        if marker == 0 || marker == 0xd8 || (0xd0..=0xd7).contains(&marker) {
            return Err("unexpected JPEG marker".into());
        }
        if input.len() < offset + 3 {
            return Ok(None);
        }
        let length = usize::from(u16::from_be_bytes([input[offset + 1], input[offset + 2]]));
        if length < 2 {
            return Err("invalid JPEG marker length".into());
        }
        let end = offset + 1 + length;
        if input.len() < end {
            if offset > 1 {
                return Ok(Some(offset - 1));
            }
            return Ok(None);
        }
        // Move the bounded marker payload out so parsing can update decoder state.
        let payload = input[offset + 3..end].to_vec();
        match marker {
            0xc0..=0xc2 => self.frame(marker, &payload)?,
            0xc4 => self.huffman(&payload)?,
            0xdb => self.quantization(&payload)?,
            0xda => self.start_scan(&payload)?,
            0xdd => {
                if payload.len() != 2 {
                    return Err("invalid JPEG restart interval".into());
                }
                self.restart_interval = usize::from(u16::from_be_bytes([payload[0], payload[1]]));
            }
            0xee if payload.starts_with(b"Adobe") && payload.len() >= 12 => self.adobe = Some(payload[11]),
            0xe0 if payload.starts_with(b"JFIF\0") => self.jfif = true,
            0xe0..=0xef | 0xfe => (),
            _ => return Err("streaming JPEG supports 8-bit Huffman sequential and progressive images".into()),
        }
        Ok(Some(end))
    }

    fn frame(&mut self, marker: u8, data: &[u8]) -> Result<(), String> {
        if self.coefficients.is_some() || data.len() < 6 || data[0] != 8 {
            return Err("invalid or unsupported JPEG frame".into());
        }
        let height = u16::from_be_bytes([data[1], data[2]]);
        let width = u16::from_be_bytes([data[3], data[4]]);
        validate_dimensions(u32::from(width), u32::from(height))?;
        let count = usize::from(data[5]);
        if width == 0 || height == 0 || ![1, 3, 4].contains(&count) || data.len() != 6 + count * 3 {
            return Err("invalid JPEG frame dimensions or components".into());
        }
        let mut max_h = 1;
        let mut max_v = 1;
        let mut components = Vec::new();
        for component in data[6..].as_chunks::<3>().0 {
            let (mut h, mut v) = (component[1] >> 4, component[1] & 15);
            if !(1..=4).contains(&h)
                || !(1..=4).contains(&v)
                || component[2] > 3
                || components.iter().any(|c: &ComponentCoefficients| c.component_id == component[0])
            {
                return Err("invalid JPEG component".into());
            }
            if count == 1 {
                h = 1;
                v = 1;
            }
            max_h = max_h.max(h);
            max_v = max_v.max(v);
            let index = components.len();
            self.quant_slots[index] = usize::from(component[2]);
            components.push(ComponentCoefficients {
                blocks: Vec::new(),
                blocks_x: 0,
                blocks_y: 0,
                h_sampling: h,
                v_sampling: v,
                quant_table_index: index as u8,
                component_id: component[0],
            });
        }
        if count > 1 && components.iter().map(|c| usize::from(c.h_sampling) * usize::from(c.v_sampling)).sum::<usize>() > 10 {
            return Err("too many JPEG blocks per MCU".into());
        }
        let mcus_x = usize::from(width).div_ceil(usize::from(max_h) * 8);
        let mcus_y = usize::from(height).div_ceil(usize::from(max_v) * 8);
        for component in &mut components {
            component.blocks_x = mcus_x * usize::from(component.h_sampling);
            component.blocks_y = mcus_y * usize::from(component.v_sampling);
            component.blocks = vec![[0; 64]; component.blocks_x * component.blocks_y];
        }
        self.progressive = marker == 0xc2;
        self.coefficients = Some(JpegCoefficients {
            width,
            height,
            data_precision: 8,
            components,
            quant_tables: vec![[1; 64]; 4],
            restart_interval: 0,
            density_unit: 0,
            x_density: 1,
            y_density: 1,
            adobe_transform: None,
        });
        Ok(())
    }

    fn huffman(&mut self, mut data: &[u8]) -> Result<(), String> {
        while !data.is_empty() {
            if data.len() < 17 {
                return Err("invalid JPEG Huffman table".into());
            }
            let index = usize::from(data[0] & 15);
            let class = data[0] >> 4;
            let counts: [u8; 16] = data[1..17].try_into().unwrap();
            let count: usize = counts.iter().map(|&n| usize::from(n)).sum();
            if index > 3 || class > 1 || data.len() < 17 + count {
                return Err("invalid JPEG Huffman table".into());
            }
            let table = Huffman::new(counts, &data[17..17 + count]).map_err(|_| "invalid JPEG Huffman table")?;
            if class == 0 {
                self.dc[index] = Some(table);
            } else {
                self.ac[index] = Some(table);
            }
            data = &data[17 + count..];
        }
        Ok(())
    }

    fn quantization(&mut self, mut data: &[u8]) -> Result<(), String> {
        while !data.is_empty() {
            let precision = data[0] >> 4;
            let index = usize::from(data[0] & 15);
            let size = 1 + 64 * (usize::from(precision) + 1);
            if precision > 1 || index > 3 || data.len() < size {
                return Err("invalid JPEG quantization table".into());
            }
            let mut values = [0; 64];
            for (i, &natural) in ZIGZAG.iter().enumerate() {
                values[natural] = if precision == 0 { u16::from(data[1 + i]) } else { u16::from_be_bytes([data[1 + i * 2], data[2 + i * 2]]) };
                if values[natural] == 0 {
                    return Err("invalid JPEG quantization value".into());
                }
            }
            self.quant[index] = Some(values);
            data = &data[size..];
        }
        Ok(())
    }

    fn start_scan(&mut self, data: &[u8]) -> Result<(), String> {
        let coefficients = self.coefficients.as_mut().ok_or("JPEG scan before frame")?;
        if data.is_empty() {
            return Err("invalid JPEG scan".into());
        }
        let count = usize::from(data[0]);
        if count == 0 || count > coefficients.components.len() || data.len() != 4 + count * 2 {
            return Err("invalid JPEG scan".into());
        }
        let (start, end, ah, al) = (data[1 + count * 2], data[2 + count * 2], data[3 + count * 2] >> 4, data[3 + count * 2] & 15);
        if start > end
            || end > 63
            || ah > 13
            || al > 13
            || (ah != 0 && ah != al + 1)
            || (self.progressive && ((start == 0 && end != 0) || (start != 0 && count != 1)))
            || (!self.progressive && (start, end, ah, al) != (0, 63, 0, 0))
        {
            return Err("invalid JPEG spectral selection or approximation".into());
        }
        let mut components = Vec::new();
        for entry in data[1..1 + count * 2].as_chunks::<2>().0 {
            let index = coefficients.components.iter().position(|c| c.component_id == entry[0]).ok_or("unknown JPEG scan component")?;
            let (dc, ac) = (usize::from(entry[1] >> 4), usize::from(entry[1] & 15));
            if dc > 3
                || ac > 3
                || components.iter().any(|&(i, _, _)| i == index)
                || (start == 0 && ah == 0 && self.dc[dc].is_none())
                || (end > 0 && self.ac[ac].is_none())
            {
                return Err("invalid JPEG scan tables or components".into());
            }
            if start > 0 && self.progression[index][0] < 0 {
                return Err("JPEG AC scan before DC".into());
            }
            for bits in &mut self.progression[index][usize::from(start)..=usize::from(end)] {
                if *bits != if ah == 0 { -1 } else { ah as i8 } {
                    return Err("invalid JPEG scan progression".into());
                }
                *bits = al as i8;
            }
            // libjpeg latches a component's quantizer at its first scan.
            if start == 0 && ah == 0 {
                coefficients.quant_tables[index] = self.quant[self.quant_slots[index]].ok_or("missing JPEG quantization table")?;
            }
            components.push((index, dc, ac));
        }
        let max_h = coefficients.components.iter().map(|c| c.h_sampling).max().unwrap() as usize;
        let max_v = coefficients.components.iter().map(|c| c.v_sampling).max().unwrap() as usize;
        let (width, height) = if count == 1 {
            let component = &coefficients.components[components[0].0];
            (
                (usize::from(coefficients.width) * usize::from(component.h_sampling)).div_ceil(max_h * 8),
                (usize::from(coefficients.height) * usize::from(component.v_sampling)).div_ceil(max_v * 8),
            )
        } else {
            (usize::from(coefficients.width).div_ceil(max_h * 8), usize::from(coefficients.height).div_ceil(max_v * 8))
        };
        self.scan = Some(Scan {
            components,
            start,
            end,
            ah,
            al,
            index: 0,
            width,
            count: width * height,
            predictors: [0; 4],
            eob: 0,
            restart: 0,
            bits: Bits::default(),
        });
        Ok(())
    }

    fn consume_scan(&mut self) -> Result<Option<usize>, String> {
        let scan = self.scan.as_mut().unwrap();
        let coefficients = self.coefficients.as_mut().unwrap();
        let mut reader = Reader { data: &self.input[self.input_start..], pos: 0, bits: scan.bits };
        let mut predictors = scan.predictors;
        let mut eob = scan.eob;
        let mut restart = scan.restart;
        let blocks = &mut self.staged;
        let mut block_count = 0;
        let result = (|| -> entropy::Result<()> {
            if self.restart_interval > 0 && scan.index > 0 && scan.index.is_multiple_of(self.restart_interval) {
                if eob != 0 {
                    return Err(Error::Invalid);
                }
                reader.restart(restart)?;
                restart = (restart + 1) & 7;
                predictors = [0; 4];
            }
            for &(index, dc, ac) in &scan.components {
                let component = &coefficients.components[index];
                let (h, v) = if scan.components.len() == 1 { (1, 1) } else { (usize::from(component.h_sampling), usize::from(component.v_sampling)) };
                for y in 0..v {
                    for x in 0..h {
                        let block_index = (scan.index / scan.width * v + y) * component.blocks_x + scan.index % scan.width * h + x;
                        let mut block = component.blocks[block_index];
                        if scan.start == 0 {
                            if scan.ah == 0 {
                                entropy::dc(&mut reader, self.dc[dc].as_ref().unwrap(), &mut predictors[index], &mut block, scan.al)?;
                            } else if reader.read(1)? != 0 {
                                block[0] |= 1i16 << scan.al;
                            }
                        }
                        if scan.end > 0 {
                            entropy::ac(
                                &mut reader,
                                self.ac[ac].as_ref().unwrap(),
                                &mut block,
                                (scan.start, scan.end),
                                scan.al,
                                scan.ah != 0,
                                &mut eob,
                            )?;
                        }
                        if scan.start == 0 && !(-1024..=1023).contains(&block[0]) {
                            return Err(Error::Invalid);
                        }
                        blocks[block_count] = (index, block_index, block);
                        block_count += 1;
                    }
                }
            }
            Ok(())
        })();
        match result {
            Err(Error::NeedInput) => return Ok(None),
            Err(Error::Invalid) => return Err("invalid JPEG entropy data".into()),
            Ok(()) => (),
        }
        for &(index, block_index, block) in &blocks[..block_count] {
            coefficients.components[index].blocks[block_index] = block;
        }
        scan.bits = reader.bits;
        scan.predictors = predictors;
        scan.eob = eob;
        scan.restart = restart;
        scan.index += 1;
        let consumed = reader.pos;
        if scan.index == scan.count {
            if eob != 0 {
                return Err("JPEG end-of-block run exceeds scan".into());
            }
            self.scan = None;
            self.seen_scan = true;
            self.completed_color = (self.jfif, self.adobe);
            self.pixels = None;
        }
        Ok(Some(consumed))
    }

    fn render(&mut self) -> Result<(), String> {
        let mut previous = None;
        if let Some(scan) = &self.scan
            && scan.index > 0
        {
            let mut coefficients = self.coefficients.as_ref().unwrap().clone();
            // Undo only the active scan in a snapshot. Earlier scans remain
            // available without rendering or copying them during feed().
            for &(index, _, _) in &scan.components {
                for block in &mut coefficients.components[index].blocks {
                    for (k, value) in block.iter_mut().enumerate().take(usize::from(scan.end) + 1).skip(usize::from(scan.start)) {
                        *value = if scan.ah == 0 {
                            0
                        } else if k == 0 {
                            *value & !((1i16 << scan.ah) - 1)
                        } else {
                            *value / (1i16 << scan.ah) * (1i16 << scan.ah)
                        };
                    }
                }
            }
            previous = Some(coefficients);
        }
        let coefficients = previous.as_ref().unwrap_or_else(|| self.coefficients.as_ref().unwrap());
        // Reconstruct coefficients without requantization and use the eager
        // backend for identical IDCT, upsampling, and color conversion.
        let mut encoded = write_coefficients(coefficients).map_err(|error| error.to_string())?;
        if !self.completed_color.0 {
            let jfif_end = 4 + usize::from(u16::from_be_bytes([encoded[4], encoded[5]]));
            encoded.drain(2..jfif_end);
        }
        if let Some(transform) = self.completed_color.1 {
            let marker = [255, 238, 0, 14, b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, transform];
            encoded.splice(2..2, marker);
        }
        self.pixels = Some(crate::decode_jpeg(&encoded)?);
        Ok(())
    }

    pub fn image(&mut self) -> Result<Option<Image>, String> {
        if self.seen_scan && self.pixels.is_none() {
            self.render()?;
        }
        self.pixels
            .as_ref()
            .map(|image| {
                Image::from_pixels(
                    image.width,
                    image.height,
                    image.mode,
                    image.pixel_data().map_err(|e| e.to_string())?.to_vec(),
                    Some("JPEG".into()),
                )
                .map_err(|error| error.to_string())
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use blanket_codec_common::SaveOptions;
    use blanket_core::PixelMode;

    use super::*;
    use crate::{JpegCoding, decode_jpeg, encode_jpeg_variant};

    fn encoded(mode: PixelMode, progressive: bool) -> Vec<u8> {
        let pixels = (0..33 * 19 * mode.channels()).map(|i| (i * 71 + i / 7) as u8).collect();
        let image = Image::from_pixels(33, 19, mode, pixels, None).unwrap();
        encode_jpeg_variant(&image, SaveOptions { .. }, JpegCoding { progressive, .. }).unwrap()
    }

    #[test]
    fn chunks_match_eager_pixels_and_bound_input() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Cmyk] {
            for progressive in [false, true] {
                let data = encoded(mode, progressive);
                let expected = decode_jpeg(&data).unwrap();
                for chunk_size in [1, 7, 64, 65536] {
                    let mut decoder = JpegStreamDecoder::new().unwrap();
                    let mut preview = None;
                    for chunk in data.chunks(chunk_size) {
                        decoder.feed(chunk).unwrap();
                        assert!(decoder.input.len() <= INPUT_LIMIT);
                        assert!(decoder.input.capacity() <= INPUT_LIMIT);
                        if !decoder.complete && preview.is_none() {
                            preview = decoder.image().unwrap();
                        }
                    }
                    assert!(decoder.complete);
                    assert!(
                        decoder.image().unwrap().unwrap().pixel_data().unwrap() == expected.pixel_data().unwrap(),
                        "{mode:?}, progressive={progressive}, chunk={chunk_size}"
                    );
                    if progressive && chunk_size == 1 {
                        assert!(preview.is_some());
                    }
                    assert_eq!(decoder.input.capacity(), 0);
                }
            }
        }
    }

    #[test]
    fn metadata_does_not_accumulate_and_trailing_data_is_ignored() {
        let data = encoded(PixelMode::Rgb, true);
        let mut decoder = JpegStreamDecoder::new().unwrap();
        decoder.feed(&data[..2]).unwrap();
        let mut marker = vec![0xff, 0xe1, 0xff, 0xff];
        marker.resize(65537, 73);
        for _ in 0..32 {
            decoder.feed(&marker).unwrap();
            assert!(decoder.input.is_empty());
            assert!(decoder.input.capacity() <= INPUT_LIMIT);
        }
        decoder.feed(&data[2..]).unwrap();
        assert!(decoder.complete);
        decoder.feed(&vec![0; 1 << 20]).unwrap();
        assert_eq!(decoder.input.capacity(), 0);
        assert_eq!(decoder.image().unwrap().unwrap().pixel_data().unwrap(), decode_jpeg(&data).unwrap().pixel_data().unwrap());
    }

    #[test]
    fn corrupt_input_fails_and_cannot_resume() {
        for data in [b"invalid JPEG".as_slice(), &[255, 216, 255, 219, 0, 1], &[255, 216, 255, 217]] {
            let mut decoder = JpegStreamDecoder::new().unwrap();
            assert!(decoder.feed(data).is_err());
            assert!(decoder.feed(&[]).unwrap_err().contains("failed JPEG decoder"));
        }
    }

    #[test]
    fn large_progressive_input_is_consumed_through_a_fixed_window() {
        let mut seed = 74u32;
        let pixels = (0..513 * 257 * 3)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let image = Image::from_pixels(513, 257, PixelMode::Rgb, pixels, None).unwrap();
        let data = encode_jpeg_variant(&image, SaveOptions { .. }, JpegCoding { progressive: true, .. }).unwrap();
        assert!(data.len() > INPUT_LIMIT);
        for chunk_size in [4096, data.len()] {
            let mut decoder = JpegStreamDecoder::new().unwrap();
            for chunk in data.chunks(chunk_size) {
                decoder.feed(chunk).unwrap();
                assert!(decoder.input.capacity() <= INPUT_LIMIT);
            }
            assert!(decoder.complete);
            assert!(decoder.image().unwrap().unwrap().pixel_data().unwrap() == decode_jpeg(&data).unwrap().pixel_data().unwrap());
        }
    }

    #[test]
    fn ycck_color_interpretation_matches_eager() {
        for progressive in [false, true] {
            let pixels: Vec<u8> = (0..33 * 19 * 4).map(|i| (i * 73 + i / 5) as u8).collect();
            let mut encoder = turbojpeg::Compressor::new().unwrap();
            encoder.set_subsamp(turbojpeg::Subsamp::Sub2x2).unwrap();
            encoder.set_quality(90).unwrap();
            encoder.set_colorspace(turbojpeg::Colorspace::YCCK).unwrap();
            encoder.set_progressive(progressive).unwrap();
            let data = encoder
                .compress_to_vec(turbojpeg::Image {
                    pixels: pixels.as_slice(),
                    width: 33,
                    height: 19,
                    pitch: 33 * 4,
                    format: turbojpeg::PixelFormat::CMYK,
                })
                .unwrap();
            let mut decoder = JpegStreamDecoder::new().unwrap();
            for chunk in data.chunks(7) {
                decoder.feed(chunk).unwrap();
            }
            assert!(decoder.complete);
            assert!(decoder.image().unwrap().unwrap().pixel_data().unwrap() == decode_jpeg(&data).unwrap().pixel_data().unwrap());
        }
    }

    #[test]
    fn delayed_previews_match_snapshots_taken_before_the_active_scan() {
        for mode in [PixelMode::L, PixelMode::Rgb, PixelMode::Cmyk] {
            let data = encoded(mode, true);
            let mut reference = JpegStreamDecoder::new().unwrap();
            let mut checked = false;
            let mut checks = 0;
            for (offset, &byte) in data.iter().enumerate() {
                reference.feed(&[byte]).unwrap();
                let expected = reference.image().unwrap();
                if reference.scan.is_none() {
                    checked = false;
                }
                if !checked
                    && reference.scan.as_ref().is_some_and(|scan| scan.index > 0)
                    && let Some(expected) = expected
                {
                    let mut delayed = JpegStreamDecoder::new().unwrap();
                    delayed.feed(&data[..=offset]).unwrap();
                    assert_eq!(delayed.image().unwrap().unwrap().pixel_data().unwrap(), expected.pixel_data().unwrap());
                    checked = true;
                    checks += 1;
                    delayed.feed(&data[offset + 1..]).unwrap();
                    assert!(delayed.complete);
                    assert_eq!(delayed.image().unwrap().unwrap().pixel_data().unwrap(), decode_jpeg(&data).unwrap().pixel_data().unwrap());
                }
            }
            assert!(checks >= 5, "all initial and refinement scans must be exercised");
        }
    }
}

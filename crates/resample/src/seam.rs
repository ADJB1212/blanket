use blanket_core::buffers::{buffer, copy, output, reserved, reserved_buffer};
use blanket_core::{Image, PixelMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn ops_seam_carve(py: Python<'_>, image: &Image, size: (u32, u32)) -> PyResult<Image> {
    if image.palette().is_some() || !matches!(image.pixel_mode(), PixelMode::L | PixelMode::La | PixelMode::Rgb | PixelMode::Rgba) {
        return Err(PyValueError::new_err("seam carving supports L, LA, RGB, and RGBA images"));
    }
    if size.0 == 0 || size.1 == 0 || image.width() == 0 || image.height() == 0 {
        return Err(PyValueError::new_err("seam carving requires positive image dimensions"));
    }
    if size == image.size() {
        return image.copy(py);
    }
    // Validate the final allocation before starting the iterative work.
    let channels = image.pixel_mode().channels();
    let stride = image.raw_bytes_per_pixel();
    drop(reserved_buffer(size, stride)?);
    let mut raster = Raster {
        pixels: copy(image.raw_data()?)?,
        width: image.width() as usize,
        height: image.height() as usize,
        channels,
        sample_bytes: stride / channels,
        maximum: (1_u32 << image.bit_depth()) - 1,
    };
    py.detach(|| {
        raster.resize_width(size.0 as usize)?;
        if raster.height != size.1 as usize {
            raster.transpose()?;
            raster.resize_width(size.1 as usize)?;
            raster.transpose()?;
        }
        output(image, size, raster.pixels)
    })
}

struct Raster {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
    sample_bytes: usize,
    maximum: u32,
}

impl Raster {
    fn stride(&self) -> usize {
        self.channels * self.sample_bytes
    }

    fn sample(&self, pixel: usize, channel: usize) -> u32 {
        let offset = (pixel * self.channels + channel) * self.sample_bytes;
        if self.sample_bytes == 1 {
            u32::from(self.pixels[offset])
        } else {
            u32::from(u16::from_le_bytes([self.pixels[offset], self.pixels[offset + 1]]))
        }
    }

    fn distance(&self, a: usize, b: usize) -> u64 {
        let has_alpha = matches!(self.channels, 2 | 4);
        let colors = self.channels - usize::from(has_alpha);
        let alpha_a = if has_alpha { self.sample(a, colors) } else { self.maximum };
        let alpha_b = if has_alpha { self.sample(b, colors) } else { self.maximum };
        let mut distance = u64::from(alpha_a.abs_diff(alpha_b));
        for channel in 0..colors {
            let a = self.sample(a, channel) * alpha_a / self.maximum;
            let b = self.sample(b, channel) * alpha_b / self.maximum;
            distance += u64::from(a.abs_diff(b));
        }
        distance
    }

    fn seam(&self) -> PyResult<Vec<usize>> {
        let mut costs = reserved::<u64>(self.width)?;
        costs.resize(self.width, 0);
        let mut next = copy(&costs)?;
        let mut parents = reserved::<usize>(self.width * self.height)?;
        parents.resize(self.width * self.height, 0);
        for y in 0..self.height {
            for (x, cost) in next.iter_mut().enumerate() {
                let pixel = y * self.width + x;
                let energy = self.distance(pixel, y * self.width + x.saturating_sub(1))
                    + self.distance(pixel, y * self.width + (x + 1).min(self.width - 1))
                    + self.distance(pixel, y.saturating_sub(1) * self.width + x)
                    + self.distance(pixel, (y + 1).min(self.height - 1) * self.width + x);
                let mut parent = x;
                if x > 0 && costs[x - 1] < costs[parent] {
                    parent = x - 1;
                }
                if x + 1 < self.width && costs[x + 1] < costs[parent] {
                    parent = x + 1;
                }
                parents[pixel] = parent;
                *cost = energy + costs[parent];
            }
            std::mem::swap(&mut costs, &mut next);
        }
        let mut seam = reserved(self.height)?;
        seam.resize(self.height, 0);
        let mut x = costs.iter().enumerate().min_by_key(|&(_, cost)| cost).unwrap().0;
        for y in (0..self.height).rev() {
            seam[y] = x;
            x = parents[y * self.width + x];
        }
        Ok(seam)
    }

    fn resize_width(&mut self, target: usize) -> PyResult<()> {
        while self.width > target {
            let seam = self.seam()?;
            let stride = self.stride();
            remove_seam(&mut self.pixels, self.width, stride, &seam);
            self.width -= 1;
        }
        while self.width < target {
            self.insert_seams((target - self.width).min(self.width.saturating_sub(1).max(1)))?;
        }
        Ok(())
    }

    fn insert_seams(&mut self, count: usize) -> PyResult<()> {
        let stride = self.stride();
        let mut selected = reserved::<bool>(self.width * self.height)?;
        selected.resize(self.width * self.height, false);
        if self.width == 1 {
            selected.fill(true);
        } else {
            let mut indices = reserved(self.width * self.height)?;
            indices.extend((0..self.height).flat_map(|_| 0..self.width));
            let mut work = Self { pixels: copy(&self.pixels)?, ..*self };
            for _ in 0..count {
                let seam = work.seam()?;
                for (y, &x) in seam.iter().enumerate() {
                    selected[y * self.width + indices[y * work.width + x]] = true;
                }
                remove_seam(&mut indices, work.width, 1, &seam);
                remove_seam(&mut work.pixels, work.width, stride, &seam);
                work.width -= 1;
            }
        }
        let mut pixels = reserved_buffer(((self.width + count) as u32, self.height as u32), stride)?;
        for (pixel, &insert) in self.pixels.chunks_exact(stride).zip(&selected) {
            pixels.extend_from_slice(pixel);
            if insert {
                pixels.extend_from_slice(pixel);
            }
        }
        self.pixels = pixels;
        self.width += count;
        Ok(())
    }

    fn transpose(&mut self) -> PyResult<()> {
        let stride = self.stride();
        let mut pixels = buffer((self.height as u32, self.width as u32), stride)?;
        for y in 0..self.height {
            for x in 0..self.width {
                let source = (y * self.width + x) * stride;
                let dest = (x * self.height + y) * stride;
                pixels[dest..dest + stride].copy_from_slice(&self.pixels[source..source + stride]);
            }
        }
        self.pixels = pixels;
        std::mem::swap(&mut self.width, &mut self.height);
        Ok(())
    }
}

fn remove_seam<T: Copy>(pixels: &mut Vec<T>, width: usize, stride: usize, seam: &[usize]) {
    for (y, &x) in seam.iter().enumerate() {
        let source = y * width * stride;
        let dest = y * (width - 1) * stride;
        pixels.copy_within(source..source + x * stride, dest);
        pixels.copy_within(source + (x + 1) * stride..source + width * stride, dest + x * stride);
    }
    pixels.truncate((width - 1) * seam.len() * stride);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_size_shares_storage_until_mutation() {
        Python::initialize();
        Python::attach(|py| {
            for (mode, depth) in [(PixelMode::L, 8), (PixelMode::La, 8), (PixelMode::Rgb, 10), (PixelMode::Rgba, 16)] {
                let bytes = 6 * mode.channels() * if depth == 8 { 1 } else { 2 };
                let image = Image::from_raw_bytes(3, 2, mode, vec![0; bytes], depth).unwrap();
                let mut result = ops_seam_carve(py, &image, image.size()).unwrap();
                assert_eq!(image.raw_data().unwrap().as_ptr(), result.raw_data().unwrap().as_ptr());
                result.pixels_mut().unwrap()[0] = 1;
                assert_eq!(image.raw_data().unwrap()[0], 0);
            }
        });
    }
}

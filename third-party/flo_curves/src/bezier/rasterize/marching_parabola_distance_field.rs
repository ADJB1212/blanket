use std::ops::Range;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd};

use itertools::*;
use smallvec::*;

use super::marching_parabolas::*;
use super::path_contour::*;
use crate::bezier::path::*;
use crate::bezier::vectorize::*;
use crate::geo::*;

#[inline]
fn squared(val: f64) -> f64 {
    val * val
}

fn fill_column_from_intercepts(
    field: &mut [f64], x: usize, width: usize, height: usize, mut current: Range<f64>, mut rest: impl Iterator<Item = Range<f64>>,
) {
    let mut following = rest.next();
    let mut y = 0;
    while y < height {
        let ypos = y as f64;
        if ypos < current.start {
            let end = exclusive_below(current.start, height);
            fill_offset_squared(field, x, width, y, end, current.start);
            y = end;
        } else if ypos <= current.end {
            let end = exclusive_through(current.end, height);
            fill_inside_span(field, x, width, y, end, current.start, current.end);
            y = end;
        } else if let Some(next) = following.clone() {
            if squared(ypos - next.start) <= squared(ypos - current.end) {
                current = next;
                following = rest.next();
            } else {
                let end = first_y_closer_to_next(y, current.end, next.start, height);
                if end <= y {
                    current = next;
                    following = rest.next();
                } else {
                    fill_offset_squared(field, x, width, y, end, current.end);
                    y = end;
                }
            }
        } else {
            fill_offset_squared(field, x, width, y, height, current.end);
            y = height;
        }
    }
}

fn exclusive_below(start: f64, height: usize) -> usize {
    if !(start > 0.0) {
        0
    } else if start >= height as f64 {
        height
    } else {
        start.ceil() as usize
    }
}

fn exclusive_through(end: f64, height: usize) -> usize {
    if !(end >= 0.0) {
        0
    } else if end >= (height.saturating_sub(1)) as f64 {
        height
    } else {
        end.floor() as usize + 1
    }
}

fn first_y_closer_to_next(y: usize, end: f64, next_start: f64, height: usize) -> usize {
    if !(next_start > end) {
        return height;
    }
    let midpoint = (end + next_start) * 0.5;
    if y as f64 >= midpoint {
        y
    } else if midpoint >= height as f64 {
        height
    } else {
        midpoint.ceil() as usize
    }
}

fn fill_offset_squared(field: &mut [f64], x: usize, stride: usize, mut y: usize, end: usize, origin: f64) {
    while y + 4 <= end {
        let ys = Simd::<f64, 4>::from_array([y as f64, (y + 1) as f64, (y + 2) as f64, (y + 3) as f64]);
        let delta = ys - Simd::splat(origin);
        scatter_column(field, x, stride, y, delta * delta);
        y += 4;
    }
    while y < end {
        field[x + y * stride] = squared(y as f64 - origin);
        y += 1;
    }
}

fn fill_inside_span(field: &mut [f64], x: usize, stride: usize, mut y: usize, end: usize, start: f64, stop: f64) {
    while y + 4 <= end {
        let ys = Simd::<f64, 4>::from_array([y as f64, (y + 1) as f64, (y + 2) as f64, (y + 3) as f64]);
        let to_start = ys - Simd::splat(start);
        let to_end = ys - Simd::splat(stop);
        let nearest = (to_start * to_start).simd_min(to_end * to_end);
        scatter_column(field, x, stride, y, -nearest);
        y += 4;
    }
    while y < end {
        let ypos = y as f64;
        field[x + y * stride] = -(squared(ypos - start).min(squared(ypos - stop)));
        y += 1;
    }
}

fn scatter_column(field: &mut [f64], x: usize, stride: usize, y: usize, values: Simd<f64, 4>) {
    let values = values.to_array();
    for lane in 0..4 {
        field[x + (y + lane) * stride] = values[lane];
    }
}

fn apply_input_signs(distances: &mut [f64], original: &[f64]) {
    let (blocks, tail) = distances.as_chunks_mut::<4>();
    let (sign_blocks, sign_tail) = original.as_chunks::<4>();
    for (block, signs) in blocks.iter_mut().zip(sign_blocks) {
        let values = Simd::from_array(*block);
        let negative = Simd::from_slice(signs).simd_lt(Simd::splat(0.0));
        *block = negative.select(-values, values).to_array();
    }
    for (distance, sign) in tail.iter_mut().zip(sign_tail) {
        if *sign < 0.0 {
            *distance = -*distance;
        }
    }
}

///
/// Distance field that's computed by following the marching parabolas algorithm
///
pub struct MarchingParabolaDistanceField {
    /// Width of the distance field in pixels
    width: usize,

    /// Height of the distance field in pixels
    height: usize,

    /// width * height pixels indicating the squared distance
    squared_distance_field: Vec<f64>,
}

impl MarchingParabolaDistanceField {
    ///
    /// Computes a distance field using the marching parabolas algorithm given functions that calculate the X and Y intercepts against a
    /// shape.
    ///
    /// This takes the width and height of the distance fields and two functions. `intercepts_for_x(xpos)` takes an x position and returns
    /// where the shape intersects that column in ascending order, and `intercepts_for_y(ypos)` does the same except for a row along a y position.
    ///
    /// Error can be up to 1 pixel for this function but is usually much less as we can compute the parabolas to a higher precision than
    /// one pixel.
    ///
    pub fn from_intercepts<TXIterator, TYIterator>(
        width: usize, height: usize, intercepts_for_x: impl Fn(f64) -> TXIterator, intercepts_for_y: impl Fn(f64) -> TYIterator,
    ) -> Self
    where
        TXIterator: IntoIterator<Item = Range<f64>>,
        TYIterator: IntoIterator<Item = Range<f64>>,
    {
        // Create a 1D distance field for the columns
        // TODO: generate the signed distance field, not just the exterior field
        let mut x_distance_field = vec![f64::INFINITY; width * height];

        // Iterate over the columns to build the initial distance field
        for x in 0..width {
            let xpos = x as f64;
            let mut intercepts = intercepts_for_x(xpos).into_iter();

            if let Some(initial_intercept) = intercepts.next() {
                fill_column_from_intercepts(&mut x_distance_field, x, width, height, initial_intercept, intercepts);
            }
        }

        // Use the marching parabolas algorithm to fill in the remaining distance field
        let mut squared_distance_field = Vec::with_capacity(width * height);

        for y in 0..height {
            // Get the row that we sampled before
            let input_row = &x_distance_field[y * width..(y * width + width)];
            let y_intercepts = intercepts_for_y(y as _).into_iter();

            // Use the marching parabolas algorithm to generate the 2D distance field
            let row_start = squared_distance_field.len();
            MarchingParabolasIterator::new(
                input_row
                    .iter()
                    .enumerate()
                    .flat_map(|(x_pos, distance)| {
                        // <-- Distances from the first pass
                        if distance.is_infinite() { None } else { Some(Parabola { xpos: x_pos as f64, ypos: distance.abs() }) }
                    })
                    .merge_by(
                        // <-- Intercepts computed along a column
                        y_intercepts
                            .flat_map(|intercept| [Parabola { xpos: intercept.start, ypos: 0.0 }, Parabola { xpos: intercept.end, ypos: 0.0 }]),
                        |a, b| a.xpos < b.xpos,
                    ),
                (0..width).map(|x| x as f64),
            )
            .extend_squared(&mut squared_distance_field);
            apply_input_signs(&mut squared_distance_field[row_start..], input_row);
        }

        // Return this as the result
        MarchingParabolaDistanceField { width, height, squared_distance_field }
    }

    ///
    /// Creates a distance field from a path (within a particular region)
    ///
    pub fn from_path_region(
        x_origin: f64, y_origin: f64, width: usize, height: usize, path: Vec<impl 'static + BezierPath<Point = impl Coordinate + Coordinate2D>>,
    ) -> Self {
        let contour = PathContour::from_path(path, ContourSize(width, height));

        Self::from_intercepts(
            width,
            height,
            |x| contour.intercepts_on_column(x + y_origin).into_iter().map(|y| (y.start - y_origin)..(y.end - y_origin)),
            |y| contour.intercepts_on_line(y + x_origin).into_iter().map(|x| (x.start - x_origin)..(x.end - x_origin)),
        )
    }

    ///
    /// Creates a distance field from a path
    ///
    /// This will calculate the bounds of the path. The two f64 values are the x and y coordinates of the origin of the resulting distance field
    ///
    pub fn from_path(path: Vec<impl 'static + BezierPath<Point = impl Coordinate + Coordinate2D>>) -> (Self, Coord2) {
        // Compute the bounding box of the path
        let bounds =
            path.iter().map(|p| p.bounding_box::<Bounds<_>>()).reduce(|bounds1, bounds2| bounds1.union_bounds(bounds2)).unwrap_or(Bounds::empty());

        // Decide on the range to calculate a distance field for
        let origin_x = bounds.min().x() - 4.0;
        let origin_y = bounds.min().y() - 4.0;
        let width = (bounds.max().x() - origin_x).ceil() + 4.0;
        let height = (bounds.max().y() - origin_y).ceil() + 4.0;

        (Self::from_path_region(origin_x, origin_y, width as _, height as _, path), Coord2(origin_x, origin_y))
    }

    ///
    /// True if the specified point is inside in the contour
    ///
    #[inline]
    pub fn point_is_inside(&self, ContourPosition(x, y): ContourPosition) -> bool {
        self.squared_distance_field.get(x + y * self.width).map(|distance| *distance <= 0.0).unwrap_or(false)
    }
}

impl SampledSignedDistanceField for MarchingParabolaDistanceField {
    type Contour = Self;

    #[inline]
    fn field_size(&self) -> ContourSize {
        ContourSize(self.width, self.height)
    }

    #[inline]
    fn distance_at_point(&self, ContourPosition(x, y): ContourPosition) -> f64 {
        let x = if x >= self.width { self.width - 1 } else { x };
        let y = if y >= self.height { self.height - 1 } else { y };

        let distance_squared = self.squared_distance_field.get(x + y * self.width);

        if let Some(distance_squared) = distance_squared {
            if distance_squared >= &0.0 { distance_squared.sqrt() } else { -(-distance_squared).sqrt() }
        } else {
            f64::INFINITY
        }
    }

    fn as_contour(&self) -> &Self::Contour {
        self
    }
}

impl SampledContour for MarchingParabolaDistanceField {
    #[inline]
    fn contour_size(&self) -> ContourSize {
        ContourSize(self.width, self.height)
    }

    fn intercepts_on_line(&self, y: f64) -> SmallVec<[Range<f64>; 4]> {
        let width = self.width;
        let y = y.floor() as usize;
        if y >= self.height {
            return smallvec![];
        }

        let row = &self.squared_distance_field[y * width..(y + 1) * width];
        let mut ranges = smallvec![];
        let mut inside = None;
        let (blocks, tail) = row.as_chunks::<4>();
        for (block_index, block) in blocks.iter().enumerate() {
            let filled = Simd::<f64, 4>::from_slice(block).simd_le(Simd::splat(0.0));
            let bits = filled.to_bitmask();
            record_inside_bits(block_index * 4, bits, 4, &mut inside, &mut ranges);
        }
        for (offset, distance) in tail.iter().enumerate() {
            let x = blocks.len() * 4 + offset;
            match (inside, *distance <= 0.0) {
                (None, true) => inside = Some(x),
                (Some(start_x), false) => {
                    inside = None;
                    ranges.push((start_x as f64)..(x as f64));
                }
                _ => {}
            }
        }
        if let Some(start_x) = inside {
            ranges.push((start_x as f64)..(width as f64));
        }
        ranges
    }
}

fn record_inside_bits(x0: usize, bits: u64, lanes: usize, inside: &mut Option<usize>, ranges: &mut SmallVec<[Range<f64>; 4]>) {
    let full = (1u64 << lanes) - 1;
    if bits == 0 {
        if let Some(start_x) = inside.take() {
            ranges.push((start_x as f64)..(x0 as f64));
        }
        return;
    }
    if bits == full {
        if inside.is_none() {
            *inside = Some(x0);
        }
        return;
    }
    let mut was_inside = inside.is_some();
    for lane in 0..lanes {
        let is_inside = (bits >> lane) & 1 == 1;
        if is_inside == was_inside {
            continue;
        }
        let x = x0 + lane;
        if is_inside {
            *inside = Some(x);
        } else if let Some(start_x) = inside.take() {
            ranges.push((start_x as f64)..(x as f64));
        }
        was_inside = is_inside;
    }
}

impl ColumnSampledContour for MarchingParabolaDistanceField {
    fn intercepts_on_column(&self, x: f64) -> SmallVec<[Range<f64>; 4]> {
        let height = self.height;
        let x = x.floor() as usize;
        if x >= self.width {
            return smallvec![];
        }

        let mut ranges = smallvec![];
        let mut inside = None;
        let mut y = 0;
        while y + 4 <= height {
            let column = Simd::<f64, 4>::from_array(std::array::from_fn(|lane| self.squared_distance_field[x + (y + lane) * self.width]));
            record_inside_bits(y, column.simd_le(Simd::splat(0.0)).to_bitmask(), 4, &mut inside, &mut ranges);
            y += 4;
        }
        for y in y..height {
            match (inside, self.squared_distance_field[x + y * self.width] <= 0.0) {
                (None, true) => inside = Some(y),
                (Some(start_y), false) => {
                    inside = None;
                    ranges.push((start_y as f64)..(y as f64));
                }
                _ => {}
            }
        }
        if let Some(start_y) = inside {
            ranges.push((start_y as f64)..(height as f64));
        }
        ranges
    }
}

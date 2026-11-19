use std::ops::Range;
use std::simd::cmp::SimdPartialEq;
use std::simd::{Mask, Simd};

use smallvec::*;

use super::intercept_scan_edge_iterator::*;
use crate::geo::*;

///
/// The size of a bitmap contour (in the form width, height)
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContourSize(pub usize, pub usize);

///
/// An x,y coordinate within a contour bitmap
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContourPosition(pub usize, pub usize);

///
/// Represents a 'cell' in a contour bitmap, a 2x2 square of samples
///
/// The value stored in this cell is a value from 0-15, where each bit represents one of the four corners of the cell:
///
///  * Bit 0 = top-left
///  * Bit 1 = top-right
///  * Bit 2 = bottom-left
///  * Bit 3 = bottom-right
///
/// A y value of 0 is considered to be the top of the bitmap
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContourCell(pub(crate) u8);

///
/// Represents the midpoint of an edge in a contour bitmap.
///
/// Edges are represented as a number where the lowest bit indicates if it's a horizontal or vertical edge, then
/// counting from the top left, 'edge 0' is between samples (0,0) and (0,1), 'edge 1' between (0,1) and (0,2) and
/// so on.
///
/// For vertical edges, the coordinates are counted horizontally still, so 'edge 0' is (0,0) and (1, 0), edge 1 is
/// between (1,0) and (1,1) and so on.
///
/// Note that there is one less horizontal edge than there are samples across the contour, so there's one 'unused'
/// edge per row (similarly, the last set of horizontal edges is not followed by a set of vertical edges)
///
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContourEdge(pub(crate) usize);

///
/// Represents a contour stored as samples at integer coordinates, where each sample can either be within the shape (1) or outside of the shape (0)
///
pub trait SampledContour: Sized {
    ///
    /// The size of this contour
    ///
    fn contour_size(&self) -> ContourSize;

    ///
    /// Given a y coordinate returns ranges indicating the filled pixels on that line
    ///
    /// The ranges must be provided in ascending order, and must also not overlap.
    ///
    fn intercepts_on_line(&self, y: f64) -> SmallVec<[Range<f64>; 4]>;

    ///
    /// Returns an iterator that visits all of the cells that are on an edge (has at least one set and one unset bit in the ContourCell)
    /// starting from the top-left corner of the contour
    ///
    /// The position returned here is the position of the bottom-right corner of the cell containing the edge.
    ///
    #[inline]
    fn edge_cell_iterator<'a>(&'a self) -> InterceptScanEdgeIterator<ContourInterceptsIterator<'a, Self>> {
        InterceptScanEdgeIterator::from_iterator(ContourInterceptsIterator::new(self))
    }

    ///
    /// Retrieves the intercepts on a line, rounded to pixel positions
    ///
    #[inline]
    fn rounded_intercepts_on_line(&self, y: f64) -> SmallVec<[Range<usize>; 4]> {
        let intercepts = self
            .intercepts_on_line(y)
            .into_iter()
            .map(|intercept| {
                let min_x_ceil = intercept.start.ceil();
                let max_x_ceil = intercept.end.ceil();

                let min_x = min_x_ceil as usize;
                let max_x = max_x_ceil as usize;

                min_x..max_x
            })
            .filter(|intercept| intercept.start != intercept.end)
            .collect::<SmallVec<_>>();

        if intercepts.len() <= 1 { intercepts } else { merge_overlapping_intercepts(intercepts) }
    }
}

impl<T> SampledContour for &T
where T: SampledContour
{
    #[inline]
    fn contour_size(&self) -> ContourSize {
        (*self).contour_size()
    }
    #[inline]
    fn intercepts_on_line(&self, y: f64) -> SmallVec<[Range<f64>; 4]> {
        (*self).intercepts_on_line(y)
    }
    #[inline]
    fn rounded_intercepts_on_line(&self, y: f64) -> SmallVec<[Range<usize>; 4]> {
        (*self).rounded_intercepts_on_line(y)
    }
}

///
/// Merges any intercepts that are adjacent or overlapping in the range
///
#[inline]
pub(crate) fn merge_overlapping_intercepts(intercepts: SmallVec<[Range<usize>; 4]>) -> SmallVec<[Range<usize>; 4]> {
    let mut intercepts = intercepts;

    let mut idx = 0;
    while idx < intercepts.len() - 1 {
        if intercepts[idx].end >= intercepts[idx + 1].start {
            intercepts[idx].end = intercepts[idx + 1].end;
            intercepts.remove(idx + 1);
        } else {
            idx += 1;
        }
    }

    intercepts
}

impl ContourCell {
    ///
    /// Returns a cell made up of 4 corner values (top-left, top-right, bottom-left and bottom-right)
    ///
    #[inline]
    pub const fn from_corners(tl: bool, tr: bool, bl: bool, br: bool) -> ContourCell {
        let tl = if tl { 1 } else { 0 };
        let tr = if tr { 2 } else { 0 };
        let bl = if bl { 4 } else { 0 };
        let br = if br { 8 } else { 0 };

        ContourCell(tl | tr | bl | br)
    }

    ///
    /// True if this represents a cell on the edge of the shape
    ///
    #[inline]
    pub const fn is_on_edge(&self) -> bool {
        self.0 != 0 && self.0 != 15
    }

    ///
    /// Merge this cell with another cell to create a cell with all the corners selected
    ///
    #[inline]
    pub const fn merge(self, cell: ContourCell) -> ContourCell {
        ContourCell(self.0 | cell.0)
    }

    ///
    /// Returns this cell shifted one pixel to the left
    ///
    #[inline]
    pub const fn shift_left(self) -> ContourCell {
        ContourCell((self.0 >> 1) & !2)
    }

    ///
    /// Returns true if the cell is empty
    ///
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0 == 0
    }

    ///
    /// Returns true if the cell is full
    ///
    #[inline]
    pub const fn is_full(&self) -> bool {
        self.0 == 15
    }
}

impl ContourEdge {
    #[inline]
    pub(crate) const fn top() -> ContourEdge {
        ContourEdge(1)
    }
    #[inline]
    pub(crate) const fn left() -> ContourEdge {
        ContourEdge(0)
    }
    #[inline]
    pub(crate) const fn bottom() -> ContourEdge {
        ContourEdge(3)
    } // Assuming a 1x1 sample size
    #[inline]
    pub(crate) const fn right() -> ContourEdge {
        ContourEdge(2)
    } // Assuming a 1x1 sample size

    #[inline]
    pub(crate) const fn at_coordinates(self, size: ContourSize, pos: ContourPosition) -> ContourEdge {
        // Offset is calculated from the size and the position
        let edge_width = size.0 + 1;
        let offset = edge_width * pos.1 + pos.0;

        // This can either be the left or the right cell depending on the upper bit
        let offset = match self.0 {
            0 => offset,              // left
            1 => offset,              // top
            2 => offset + 1,          // right
            3 => offset + edge_width, // bottom
            _ => unreachable!(),
        };

        // This can be the horizontal or vertical edge depending on the lower bit
        let offset = (offset << 1) | (self.0 & 1);

        ContourEdge(offset)
    }

    ///
    /// True if this is a horizontal edgge
    ///
    #[inline]
    pub fn is_horizontal(self) -> bool {
        (self.0 & 1) == 1
    }

    ///
    /// True if this is a vertical edgge
    ///
    #[inline]
    pub fn is_vertical(self) -> bool {
        (self.0 & 1) != 1
    }

    ///
    /// Returns the coordinates of the samples in the original `SampledContour` for this edge
    ///
    #[inline]
    pub fn to_contour_coords(self, size: ContourSize) -> (ContourPosition, ContourPosition) {
        let edge_width = size.0 + 1;
        let x = (self.0 >> 1) % edge_width;
        let y = (self.0 >> 1) / edge_width;

        if self.is_horizontal() {
            // Horizontal edge
            (ContourPosition(x, y), ContourPosition(x + 1, y))
        } else {
            // Vertical edge
            (ContourPosition(x, y), ContourPosition(x, y + 1))
        }
    }

    ///
    /// Returns the coordinate of the center point of this edge
    ///
    #[inline]
    pub fn to_coords<TCoord>(self, size: ContourSize) -> TCoord
    where TCoord: Coordinate + Coordinate2D {
        let edge_width = size.0 + 1;
        let x = (self.0 >> 1) % edge_width;
        let y = (self.0 >> 1) / edge_width;

        if (self.0 & 1) == 1 {
            // Horizontal edge
            TCoord::from_components(&[x as f64 + 0.5, y as f64])
        } else {
            // Vertical edge
            TCoord::from_components(&[x as f64, y as f64 + 0.5])
        }
    }
}

impl ContourPosition {
    #[inline]
    pub fn x(&self) -> usize {
        self.0
    }

    #[inline]
    pub fn y(&self) -> usize {
        self.1
    }
}

impl ContourSize {
    #[inline]
    pub fn width(&self) -> usize {
        self.0
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.1
    }
}

///
/// Represents a contour sampled using boolean values that indicate whether or not each sample is in or out
///
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BoolSampledContour(pub ContourSize, pub Vec<bool>);

///
/// Represents a contour sampled using u8 values that are 0 for items outside the contour and 1 otherwise
///
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct U8SampledContour(pub ContourSize, pub Vec<u8>);

impl BoolSampledContour {
    ///
    /// Returns true if the specified point is inside the contour, or false if it's outside
    ///
    /// A y-value of 0 is considered to be the 'top' of the bitmap
    ///
    #[inline]
    pub fn point_is_inside(&self, pos: ContourPosition) -> bool {
        // Position as an offset into the vector array, without bounds checking
        let idx = pos.0 + (pos.1) * self.0.0;

        self.1[idx]
    }
}

impl U8SampledContour {}

impl SampledContour for BoolSampledContour {
    ///
    /// The size of this contour
    ///
    #[inline]
    fn contour_size(&self) -> ContourSize {
        self.0
    }

    ///
    /// Given a y coordinate returns ranges indicating the filled pixels on that line
    ///
    /// The ranges must be provided in ascending order, and must also not overlap.
    ///
    fn intercepts_on_line(&self, y: f64) -> SmallVec<[Range<f64>; 4]> {
        let width = self.contour_size().width();
        let y = y.floor() as usize;

        let mut ranges = smallvec![];
        let mut inside = None;
        let mut x = 0;
        while x + 32 <= width {
            let mut samples = [0u8; 32];
            for lane in 0..32 {
                samples[lane] = u8::from(self.point_is_inside(ContourPosition(x + lane, y)));
            }
            let filled = Simd::<u8, 32>::from_array(samples).simd_ne(Simd::splat(0));
            scan_filled_lanes_32(x, filled, &mut inside, &mut ranges);
            x += 32;
        }
        while x + 16 <= width {
            let mut samples = [0u8; 16];
            for lane in 0..16 {
                samples[lane] = u8::from(self.point_is_inside(ContourPosition(x + lane, y)));
            }
            let filled = Simd::<u8, 16>::from_array(samples).simd_ne(Simd::splat(0));
            scan_filled_lanes(x, filled, &mut inside, &mut ranges);
            x += 16;
        }
        for x in x..width {
            match (inside, self.point_is_inside(ContourPosition(x, y))) {
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

impl SampledContour for U8SampledContour {
    ///
    /// The size of this contour
    ///
    #[inline]
    fn contour_size(&self) -> ContourSize {
        self.0
    }

    ///
    /// Given a y coordinate returns ranges indicating the filled pixels on that line
    ///
    /// The ranges must be provided in ascending order, and must also not overlap.
    ///
    fn intercepts_on_line(&self, y: f64) -> SmallVec<[Range<f64>; 4]> {
        let width = self.contour_size().width();
        let y = y.floor() as usize;
        let row = &self.1[y * width..(y + 1) * width];

        let mut ranges = smallvec![];
        let mut inside = None;
        let (blocks32, rest) = row.as_chunks::<32>();
        for (block_index, block) in blocks32.iter().enumerate() {
            let filled = Simd::<u8, 32>::from_slice(block).simd_ne(Simd::splat(0));
            scan_filled_lanes_32(block_index * 32, filled, &mut inside, &mut ranges);
        }
        let tail_start32 = blocks32.len() * 32;
        let (blocks16, tail) = rest.as_chunks::<16>();
        for (block_index, block) in blocks16.iter().enumerate() {
            let filled = Simd::<u8, 16>::from_slice(block).simd_ne(Simd::splat(0));
            scan_filled_lanes(tail_start32 + block_index * 16, filled, &mut inside, &mut ranges);
        }
        let tail_start = tail_start32 + blocks16.len() * 16;
        for (offset, byte) in tail.iter().enumerate() {
            let x = tail_start + offset;
            match (inside, *byte != 0) {
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

fn scan_filled_lanes(x0: usize, filled: Mask<i8, 16>, inside: &mut Option<usize>, ranges: &mut SmallVec<[Range<f64>; 4]>) {
    let bits = filled.to_bitmask();
    if bits == 0 {
        if let Some(start_x) = inside.take() {
            ranges.push((start_x as f64)..(x0 as f64));
        }
        return;
    }
    if bits == u16::MAX as u64 {
        if inside.is_none() {
            *inside = Some(x0);
        }
        return;
    }

    let mut was_inside = inside.is_some();
    for lane in 0..16 {
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

fn scan_filled_lanes_32(x0: usize, filled: Mask<i8, 32>, inside: &mut Option<usize>, ranges: &mut SmallVec<[Range<f64>; 4]>) {
    let bits = filled.to_bitmask();
    if bits == 0 {
        if let Some(start_x) = inside.take() {
            ranges.push((start_x as f64)..(x0 as f64));
        }
        return;
    }
    if bits == u32::MAX as u64 {
        if inside.is_none() {
            *inside = Some(x0);
        }
        return;
    }

    let mut was_inside = inside.is_some();
    for lane in 0..32 {
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

///
/// Returns true if the specified point is considered to be 'inside' the shape represented by a `SampledContour`
///
/// The `SampledContour` interface uses a scan-conversion model rather than a point-by-point model. This function makes
/// it possible to test individual points without needing to handle the whole scanline, at the cost of poor performance.
///
/// In general, it is much more efficient to use the `intercepts_on_line()` function to find all of the 'inside' points
/// at a given y position in one go, so avoid using this function where possible. For non-performance critical code, this
/// can be a convenient way to check an individual point.
///
pub fn contour_point_is_inside(contour: &impl SampledContour, pos: ContourPosition) -> bool {
    // Convert the y position to a coordinate
    let size = contour.contour_size();
    let x = pos.x() as f64;
    let y = pos.y() as f64;
    let width = size.width() as f64;

    // Everything outside of the x-range is not inside in the contour
    if x >= width {
        return false;
    }

    for intercept in contour.intercepts_on_line(y) {
        if intercept.start <= x && intercept.end > x {
            return true;
        }

        if intercept.start > x {
            // Can give up early because the intercept function is assumed to return the intercepts in order
            return false;
        }
    }

    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn left_at_coordinate() {
        let size = ContourSize(80, 80);
        let left = ContourEdge::left();
        let at_coord = left.at_coordinates(size, ContourPosition(7, 8));
        let (start, end) = at_coord.to_contour_coords(size);

        assert!(start == ContourPosition(7, 8), "Start doesn't match {:?} {:?}", start, end);
        assert!(end == ContourPosition(7, 9), "End doesn't match {:?} {:?}", start, end);
    }

    #[test]
    fn right_at_coordinate() {
        let size = ContourSize(80, 80);
        let right = ContourEdge::right();
        let at_coord = right.at_coordinates(size, ContourPosition(7, 8));
        let (start, end) = at_coord.to_contour_coords(size);

        assert!(start == ContourPosition(8, 8), "Start doesn't match {:?} {:?}", start, end);
        assert!(end == ContourPosition(8, 9), "End doesn't match {:?} {:?}", start, end);
    }

    #[test]
    fn top_at_coordinate() {
        let size = ContourSize(80, 80);
        let top = ContourEdge::top();
        let at_coord = top.at_coordinates(size, ContourPosition(7, 8));
        let (start, end) = at_coord.to_contour_coords(size);

        assert!(start == ContourPosition(7, 8), "Start doesn't match {:?} {:?}", start, end);
        assert!(end == ContourPosition(8, 8), "End doesn't match {:?} {:?}", start, end);
    }

    #[test]
    fn u8_intercepts_cross_simd_lanes() {
        let mut samples = vec![0u8; 40];
        for x in [0, 1, 15, 16, 17, 31, 32, 39] {
            samples[x] = 1;
        }
        samples[20] = 1;
        samples[21] = 1;
        let contour = U8SampledContour(ContourSize(40, 1), samples);
        let ranges = contour.intercepts_on_line(0.0);
        assert_eq!(ranges.as_slice(), &[0.0..2.0, 15.0..18.0, 20.0..22.0, 31.0..33.0, 39.0..40.0]);
    }

    #[test]
    fn u8_intercepts_cross_32_lane_boundary() {
        // Tests that spans crossing the 32-wide SIMD boundary are merged correctly
        let mut samples = vec![0u8; 80];
        for x in [30, 31, 32, 33, 62, 63, 64, 65, 79] {
            samples[x] = 1;
        }
        let contour = U8SampledContour(ContourSize(80, 1), samples);
        let ranges = contour.intercepts_on_line(0.0);
        assert_eq!(ranges.as_slice(), &[30.0..34.0, 62.0..66.0, 79.0..80.0]);
    }

    fn bottom_at_coordinate() {
        let size = ContourSize(80, 80);
        let bottom = ContourEdge::bottom();
        let at_coord = bottom.at_coordinates(size, ContourPosition(7, 8));
        let (start, end) = at_coord.to_contour_coords(size);

        assert!(start == ContourPosition(7, 9), "Start doesn't match {:?} {:?}", start, end);
        assert!(end == ContourPosition(8, 9), "End doesn't match {:?} {:?}", start, end);
    }
}

use std::ops::*;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd, StdFloat};

use super::coordinate::*;

#[inline]
fn lanes(x: f64, y: f64) -> Simd<f64, 2> {
    Simd::from_array([x, y])
}

#[inline]
fn from_lanes(values: Simd<f64, 2>) -> Coord2 {
    let [x, y] = values.to_array();
    Coord2(x, y)
}

/// Represents a 2D point
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Coord2(pub f64, pub f64);

impl Coord2 {
    ///
    /// Creates a Coord2 from any other implementation of Coordinate2D
    ///
    #[inline]
    pub fn from_coordinate(coord: impl Coordinate2D) -> Coord2 {
        Coord2(coord.x(), coord.y())
    }
}

impl Coordinate2D for Coord2 {
    ///
    /// X component of this coordinate
    ///
    #[inline]
    fn x(&self) -> f64 {
        self.0
    }

    ///
    /// Y component of this coordinate
    ///
    #[inline]
    fn y(&self) -> f64 {
        self.1
    }
}

impl Add<Coord2> for Coord2 {
    type Output = Coord2;

    #[inline]
    fn add(self, rhs: Coord2) -> Coord2 {
        from_lanes(lanes(self.0, self.1) + lanes(rhs.0, rhs.1))
    }
}

impl Sub<Coord2> for Coord2 {
    type Output = Coord2;

    #[inline]
    fn sub(self, rhs: Coord2) -> Coord2 {
        from_lanes(lanes(self.0, self.1) - lanes(rhs.0, rhs.1))
    }
}

impl Mul<f64> for Coord2 {
    type Output = Coord2;

    #[inline]
    fn mul(self, rhs: f64) -> Coord2 {
        from_lanes(lanes(self.0, self.1) * Simd::splat(rhs))
    }
}

impl From<(f64, f64)> for Coord2 {
    fn from((x, y): (f64, f64)) -> Coord2 {
        Coord2(x, y)
    }
}

impl Into<(f64, f64)> for Coord2 {
    fn into(self) -> (f64, f64) {
        (self.0, self.1)
    }
}

impl From<(f32, f32)> for Coord2 {
    fn from((x, y): (f32, f32)) -> Coord2 {
        Coord2(x as _, y as _)
    }
}

impl Into<(f32, f32)> for Coord2 {
    fn into(self) -> (f32, f32) {
        (self.0 as _, self.1 as _)
    }
}

impl Coordinate for Coord2 {
    #[inline]
    fn from_components(components: &[f64]) -> Coord2 {
        Coord2(components[0], components[1])
    }

    #[inline]
    fn origin() -> Coord2 {
        Coord2(0.0, 0.0)
    }

    #[inline]
    fn len() -> usize {
        2
    }

    #[inline]
    fn get(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            _ => panic!("Coord2 only has two components"),
        }
    }

    fn from_biggest_components(p1: Coord2, p2: Coord2) -> Coord2 {
        let left = lanes(p1.0, p1.1);
        let right = lanes(p2.0, p2.1);
        from_lanes(left.simd_gt(right).select(left, right))
    }

    fn from_smallest_components(p1: Coord2, p2: Coord2) -> Coord2 {
        let left = lanes(p1.0, p1.1);
        let right = lanes(p2.0, p2.1);
        from_lanes(left.simd_lt(right).select(left, right))
    }

    #[inline]
    fn distance_to(&self, target: &Coord2) -> f64 {
        let delta = lanes(target.0, target.1) - lanes(self.0, self.1);
        (delta * delta).reduce_sum().sqrt()
    }

    #[inline]
    fn dot(&self, target: &Self) -> f64 {
        (lanes(self.0, self.1) * lanes(target.0, target.1)).reduce_sum()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        lanes(self.0, self.1).is_nan().any()
    }

    #[inline]
    fn round(self, accuracy: f64) -> Coord2 {
        let accuracy = Simd::splat(accuracy);
        from_lanes((lanes(self.0, self.1) / accuracy).round() * accuracy)
    }

    #[inline]
    fn is_near_to(&self, other: &Coord2, max_distance: f64) -> bool {
        let delta = lanes(self.0, self.1) - lanes(other.0, other.1);
        (delta * delta).reduce_sum() <= max_distance * max_distance
    }
}

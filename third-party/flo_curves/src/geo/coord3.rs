use std::ops::*;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::{Select, Simd, StdFloat};

use super::coord2::*;
use super::coordinate::*;

#[inline]
fn lanes(x: f64, y: f64, z: f64) -> Simd<f64, 4> {
    Simd::from_array([x, y, z, 0.0])
}

#[inline]
fn from_lanes(values: Simd<f64, 4>) -> Coord3 {
    let [x, y, z, _] = values.to_array();
    Coord3(x, y, z)
}

/// Represents a 2D point
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Coord3(pub f64, pub f64, pub f64);

impl Coordinate3D for Coord3 {
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

    ///
    /// Z component of this coordinate
    ///
    #[inline]
    fn z(&self) -> f64 {
        self.2
    }
}

impl Add<Coord3> for Coord3 {
    type Output = Coord3;

    #[inline]
    fn add(self, rhs: Coord3) -> Coord3 {
        from_lanes(lanes(self.0, self.1, self.2) + lanes(rhs.0, rhs.1, rhs.2))
    }
}

impl Sub<Coord3> for Coord3 {
    type Output = Coord3;

    #[inline]
    fn sub(self, rhs: Coord3) -> Coord3 {
        from_lanes(lanes(self.0, self.1, self.2) - lanes(rhs.0, rhs.1, rhs.2))
    }
}

impl Mul<f64> for Coord3 {
    type Output = Coord3;

    #[inline]
    fn mul(self, rhs: f64) -> Coord3 {
        from_lanes(lanes(self.0, self.1, self.2) * Simd::splat(rhs))
    }
}

impl From<(f64, f64, f64)> for Coord3 {
    fn from((x, y, z): (f64, f64, f64)) -> Coord3 {
        Coord3(x, y, z)
    }
}

impl From<(Coord2, f64)> for Coord3 {
    fn from((pos, z): (Coord2, f64)) -> Coord3 {
        Coord3(pos.0, pos.1, z)
    }
}

impl Into<(f64, f64, f64)> for Coord3 {
    fn into(self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}

impl From<(f32, f32, f32)> for Coord3 {
    fn from((x, y, z): (f32, f32, f32)) -> Coord3 {
        Coord3(x as _, y as _, z as _)
    }
}

impl Into<(f32, f32, f32)> for Coord3 {
    fn into(self) -> (f32, f32, f32) {
        (self.0 as _, self.1 as _, self.2 as _)
    }
}

impl Coordinate for Coord3 {
    #[inline]
    fn from_components(components: &[f64]) -> Coord3 {
        Coord3(components[0], components[1], components[2])
    }

    #[inline]
    fn origin() -> Coord3 {
        Coord3(0.0, 0.0, 0.0)
    }

    #[inline]
    fn len() -> usize {
        3
    }

    #[inline]
    fn get(&self, index: usize) -> f64 {
        match index {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            _ => panic!("Coord3 only has three components"),
        }
    }

    fn from_biggest_components(p1: Coord3, p2: Coord3) -> Coord3 {
        let left = lanes(p1.0, p1.1, p1.2);
        let right = lanes(p2.0, p2.1, p2.2);
        from_lanes(left.simd_gt(right).select(left, right))
    }

    fn from_smallest_components(p1: Coord3, p2: Coord3) -> Coord3 {
        let left = lanes(p1.0, p1.1, p1.2);
        let right = lanes(p2.0, p2.1, p2.2);
        from_lanes(left.simd_lt(right).select(left, right))
    }

    #[inline]
    fn distance_to(&self, target: &Coord3) -> f64 {
        let delta = lanes(target.0, target.1, target.2) - lanes(self.0, self.1, self.2);
        (delta * delta).reduce_sum().sqrt()
    }

    #[inline]
    fn dot(&self, target: &Self) -> f64 {
        (lanes(self.0, self.1, self.2) * lanes(target.0, target.1, target.2)).reduce_sum()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        // The padding lane is zero, so it never reports NaN
        lanes(self.0, self.1, self.2).is_nan().any()
    }

    #[inline]
    fn round(self, accuracy: f64) -> Coord3 {
        let accuracy = Simd::splat(accuracy);
        from_lanes((lanes(self.0, self.1, self.2) / accuracy).round() * accuracy)
    }

    #[inline]
    fn is_near_to(&self, other: &Coord3, max_distance: f64) -> bool {
        let delta = lanes(self.0, self.1, self.2) - lanes(other.0, other.1, other.2);
        (delta * delta).reduce_sum() <= max_distance * max_distance
    }
}

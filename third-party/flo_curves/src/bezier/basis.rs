use std::simd::Simd;

use smallvec::*;

use crate::geo::*;

#[inline]
fn bernstein_weights(t: f64) -> Simd<f64, 4> {
    let u = 1.0 - t;
    let uu = u * u;
    let tt = t * t;
    Simd::from_array([uu * u, 3.0 * uu * t, 3.0 * u * tt, tt * t])
}

///
/// Computes the bezier coefficients (A, B, C, D) for a bezier curve
///
pub fn bezier_coefficients<Point: Coordinate>(dimension: usize, w1: &Point, w2: &Point, w3: &Point, w4: &Point) -> (f64, f64, f64, f64) {
    let w1 = w1.get(dimension);
    let w2 = w2.get(dimension);
    let w3 = w3.get(dimension);
    let w4 = w4.get(dimension);

    (w4 - (3.0 * w3) + (3.0 * w2) - w1, (3.0 * w3) - (6.0 * w2) + 3.0 * w1, 3.0 * w2 - 3.0 * w1, w1)
}

///
/// The cubic bezier weighted basis function
///
#[inline]
pub fn basis<Point: Coordinate>(t: f64, w1: Point, w2: Point, w3: Point, w4: Point) -> Point {
    let [b0, b1, b2, b3] = bernstein_weights(t).to_array();
    w1 * b0 + w2 * b1 + w3 * b2 + w4 * b3
}

///
/// de Casteljau's algorithm for bezier curves of arbitrary degree
///
#[inline]
pub fn de_casteljau_n<Point: Coordinate, const N: usize>(t: f64, points: SmallVec<[Point; N]>) -> Point {
    let mut points = points;

    while points.len() > 1 {
        let mut next_points = smallvec![];

        for idx in 0..(points.len() - 1) {
            next_points.push(points[idx] * (1.0 - t) + points[idx + 1] * t);
        }

        points = next_points;
    }

    points[0]
}

///
/// de Casteljau's algorithm for cubic bezier curves
///
#[inline]
pub fn de_casteljau4<Point: Coordinate>(t: f64, w1: Point, w2: Point, w3: Point, w4: Point) -> Point {
    let wn1 = w1 * (1.0 - t) + w2 * t;
    let wn2 = w2 * (1.0 - t) + w3 * t;
    let wn3 = w3 * (1.0 - t) + w4 * t;

    de_casteljau3(t, wn1, wn2, wn3)
}

///
/// de Casteljau's algorithm for quadratic bezier curves
///
#[inline]
pub fn de_casteljau3<Point: Coordinate>(t: f64, w1: Point, w2: Point, w3: Point) -> Point {
    let wn1 = w1 * (1.0 - t) + w2 * t;
    let wn2 = w2 * (1.0 - t) + w3 * t;

    de_casteljau2(t, wn1, wn2)
}

///
/// de Casteljau's algorithm for lines
///
#[inline]
pub fn de_casteljau2<Point: Coordinate>(t: f64, w1: Point, w2: Point) -> Point {
    w1 * (1.0 - t) + w2 * t
}

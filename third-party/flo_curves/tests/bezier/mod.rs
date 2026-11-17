#![allow(clippy::all)] // Tests are lower priority to fix

use flo_curves::{bezier, *};

mod algorithms;
mod path;
mod rasterize;
mod vectorize;

mod basis;
mod bounds;
mod characteristics;
mod curve_intersection_clip_tests;
mod deform;
mod derivative;
mod distort;
mod fit_tests;
mod flatness_tests;
mod intersection;
mod length;
mod nearest_point_tests;
mod normal;
mod offset_tests;
mod overlaps_tests;
mod search;
mod section;
mod self_intersection;
mod solve;
mod subdivide;
mod tangent;
mod walk;

pub fn approx_equal(a: f64, b: f64) -> bool {
    f64::floor(f64::abs(a - b) * 10000.0) == 0.0
}

#[test]
fn read_curve_control_points() {
    let curve = bezier::Curve::from_points(Coord2(1.0, 1.0), (Coord2(3.0, 3.0), Coord2(4.0, 4.0)), Coord2(2.0, 2.0));

    assert!(curve.start_point() == Coord2(1.0, 1.0));
    assert!(curve.end_point() == Coord2(2.0, 2.0));
    assert!(curve.control_points() == (Coord2(3.0, 3.0), Coord2(4.0, 4.0)));
}

#[test]
fn read_curve_points() {
    let curve = bezier::Curve::from_points(Coord2(1.0, 1.0), (Coord2(3.0, 3.0), Coord2(4.0, 4.0)), Coord2(2.0, 2.0));

    for x in 0..100 {
        let t = (x as f64) / 100.0;

        let point = curve.point_at_pos(t);
        let another_point = bezier::de_casteljau4(t, Coord2(1.0, 1.0), Coord2(3.0, 3.0), Coord2(4.0, 4.0), Coord2(2.0, 2.0));

        assert!(point.distance_to(&another_point) < 0.001);
    }
}

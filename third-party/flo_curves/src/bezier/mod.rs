//!
//! # Routines for describing, querying and manipulating Bezier curves
//!
//! ```
//! # use flo_curves::*;
//! # use flo_curves::bezier::*;
//! #
//! let curve           = Curve::from_points(Coord2(1.0, 2.0), (Coord2(2.0, 0.0), Coord2(3.0, 5.0)), Coord2(4.0, 2.0));
//!
//! let mid_point       = curve.point_at_pos(0.5);
//! let all_points      = walk_curve_evenly(&curve, 1.0, 0.01).map(|section| section.point_at_pos(0.5)).collect::<Vec<_>>();
//! let fitted_curve    = fit_curve::<Curve<Coord2>>(&all_points, 0.1);
//! let intersections   = curve_intersects_ray(&curve, &(Coord2(1.0, 1.0), Coord2(2.0, 2.0)));
//! let offset_curve    = offset(&curve, 2.0, 2.0);
//! ```
//!
//! Anything that implements the `BezierCurve` trait can be manipulated by the functions in this crate. The `Curve` type
//! is provided as a basic implementation for defining bezier curves, but the trait can be defined on any type that
//! represents a bezier curve.
//!
//! The `BezierCurveFactory` trait extends the `BezierCurve` trait for use with functions that can build/return new curves.
//!
//! For routines that deal with paths made up of bezier curves, see the `path` namespace.
//!

mod basis;
mod bounds;
mod characteristics;
mod curve;
mod deform;
mod derivative;
mod distort;
mod fit;
mod intersection;
mod length;
mod nearest_point;
mod normal;
mod offset;
mod offset_lms;
mod offset_scaling;
mod offset_subdivision_lms;
mod overlaps;
mod search;
mod section;
mod solve;
mod subdivide;
mod tangent;
mod walk;

pub mod path;
pub mod rasterize;
pub mod roots;
pub mod vectorize;

pub use basis::*;
pub use bounds::*;
pub use characteristics::*;
pub use curve::*;
pub use deform::*;
pub use derivative::*;
pub use distort::*;
pub use fit::*;
pub use intersection::*;
pub use length::*;
pub use nearest_point::*;
pub use normal::*;
pub use offset::*;
pub use offset_lms::*;
pub use offset_scaling::*;
pub use offset_subdivision_lms::*;
pub use overlaps::*;
pub use search::*;
pub use section::*;
pub use solve::*;
pub use subdivide::*;
pub use tangent::*;
pub use walk::*;

pub use super::geo::*;

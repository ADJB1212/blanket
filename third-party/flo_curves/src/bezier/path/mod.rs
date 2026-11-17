//!
//! # Manipulates multiple Bezier curves joined into a path
//!
//! ```
//! # use flo_curves::*;
//! # use flo_curves::arc::*;
//! # use flo_curves::bezier;
//! # use flo_curves::bezier::path::*;
//! #
//! let rectangle = BezierPathBuilder::<SimpleBezierPath>::start(Coord2(1.0, 1.0))
//!     .line_to(Coord2(5.0, 1.0))
//!     .line_to(Coord2(5.0, 5.0))
//!     .line_to(Coord2(1.0, 5.0))
//!     .line_to(Coord2(1.0, 1.0))
//!     .build();
//! let circle = Circle::new(Coord2(3.0, 3.0), 1.0).to_path::<SimpleBezierPath>();
//!
//! let rectangle_with_hole = path_sub::<SimpleBezierPath>(&vec![rectangle], &vec![circle], 0.01);
//! ```
//!
//! Anything that implements the `BezierPath` trait can be treated as a path. The `SimpleBezierPath` type is provided
//! as a convenient default implementation of this trait. These paths represent a single perimeter of a region.
//!
//! The arithmetic operations such as `path_sub()`, `path_add()`, `path_intersect()` all work with collections of these
//! perimeters, stored in a `Vec`. A path with a hole in the middle will have two perimeters, for example.
//!
//! These perimeters must not be self-intersecting: `flo_curves` doesn't use a winding rule as such but instead considers
//! all edges to be exterior edges (which is very similar to an even-odd winding rule). A couple of methods are provided
//! for fixing paths with self-intersections: `path_remove_interior_points()` will find the outermost perimeter of a shape -
//! which is useful for tidying up the subpaths. `path_remove_overlapped_points()` will combine subpaths so that
//! there are no overlapping edges. These two functions provide much finer control than is possible through the traditional
//! idea of the winding rule.
//!
//! There are a few more advanced algorithms: for example, the `flood_fill_concave()` function provides a vector
//! implementation of the flood fill algorithm, returning a path that fills a space defined by a ray-casting function.
//!

pub mod algorithms;
mod arithmetic;
mod bounds;
mod graph_path;
mod intersection;
mod is_clockwise;
mod path;
mod path_builder;
mod point;
mod ray;
mod stroke;
mod to_curves;

pub use self::arithmetic::*;
pub use self::bounds::*;
pub use self::graph_path::*;
pub use self::intersection::*;
pub use self::is_clockwise::*;
pub use self::path::*;
pub use self::path_builder::*;
pub use self::point::*;
pub use self::stroke::*;
pub use self::to_curves::*;

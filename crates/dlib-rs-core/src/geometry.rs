//! Geometry port of dlib's `dlib/geometry/*.h`.
//!
//! Ported from (all paths relative to `dlib/geometry/`):
//! - `vector.h` / `vector_abstract.h` — [`Vector`], [`Point`], [`Dpoint`]
//! - `rectangle.h` / `rectangle_abstract.h` — [`Rectangle`] and free functions
//! - `drectangle.h` / `drectangle_abstract.h` — [`Drectangle`] and free functions
//! - `point_transforms.h` / `point_transforms_abstract.h` — [`PointRotator`],
//!   [`PointTransform`], [`PointTransformAffine`], [`RectangleTransform`],
//!   [`PointTransformProjective`], `find_affine_transform`,
//!   `find_similarity_transform`, `find_projective_transform`
//! - `line.h` / `line_abstract.h` — [`Line`] and line free functions
//! - `polygon.h` / `polygon_abstract.h` — [`Polygon`]
//! - `border_enumerator.h` / `border_enumerator_abstract.h` — [`BorderEnumerator`]
//!
//! Semantics (rounding, empty-rectangle conventions, area signs, operator
//! meanings) match the C++ headers exactly; each item cites its origin.

use num_traits::{Num, ToPrimitive};
use std::ops::{Add, AddAssign, BitAnd, Div, Mul, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
// Vector  (dlib/geometry/vector.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::vector<T, 2>` and `dlib::vector<T, 3>` (dlib/geometry/vector.h).
///
/// `N` is 2 or 3 (the only sizes dlib defines). `x()`/`y()` are always
/// available for `N >= 2`; `z()` returns `T::zero()` for `N == 2`, exactly like
/// `dlib::vector<T,2>::z()` which is defined to return the constant `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vector<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy> Vector<T, 2> {
    /// Port of `vector<T,2>(x, y)`.
    pub fn new(x: T, y: T) -> Self {
        Vector { data: [x, y] }
    }
}

impl<T: Copy> Vector<T, 3> {
    /// Port of `vector<T,3>(x, y, z)`.
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector { data: [x, y, z] }
    }
}

impl<T: Num + Copy + ToPrimitive, const N: usize> Vector<T, N> {
    /// Port of `vector::x()`.
    pub fn x(&self) -> T {
        self.data[0]
    }

    /// Port of `vector::y()`.
    pub fn y(&self) -> T {
        assert!(N >= 2, "y() requires a vector with at least 2 elements");
        self.data[1]
    }

    /// Port of `vector::z()`: the real component for `N == 3`, constant `0`
    /// for `N == 2` (dlib/geometry/vector.h, `vector<T,2>::z()`).
    pub fn z(&self) -> T {
        if N >= 3 {
            self.data[2]
        } else {
            T::zero()
        }
    }

    /// Port of `vector::dot(rhs)`.
    pub fn dot(&self, rhs: &Self) -> T {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.data[i] * rhs.data[i];
        }
        sum
    }

    /// Port of `vector::length()` — `sqrt((double)(x*x + y*y + z*z()))`.
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Port of `vector::length_squared()`.
    pub fn length_squared(&self) -> f64 {
        let mut sum = T::zero();
        for i in 0..N {
            sum = sum + self.data[i] * self.data[i];
        }
        sum.to_f64().unwrap_or(f64::NAN)
    }

    /// Port of `vector::normalize()` — always returns a double vector.
    pub fn normalize(&self) -> Vector<f64, N> {
        let tmp = self.length_squared().sqrt();
        Vector {
            data: std::array::from_fn(|i| self.data[i].to_f64().unwrap_or(f64::NAN) / tmp),
        }
    }
}

impl<T: Num + Copy + ToPrimitive> Vector<T, 2> {
    /// 2D cross product (the `z` component of the 3D cross product), matching
    /// the third row of `dlib::vector<T,2>::cross` in dlib/geometry/vector.h.
    pub fn cross(&self, rhs: &Self) -> T {
        self.x() * rhs.y() - self.y() * rhs.x()
    }
}

impl<T: Num + Copy + ToPrimitive> Vector<T, 3> {
    /// Port of `dlib::vector<T,3>::cross`.
    pub fn cross(&self, rhs: &Self) -> Self {
        Vector {
            data: [
                self.y() * rhs.z() - self.z() * rhs.y(),
                self.z() * rhs.x() - self.x() * rhs.z(),
                self.x() * rhs.y() - self.y() * rhs.x(),
            ],
        }
    }
}

// --- conversions -----------------------------------------------------------
//
// dlib/geometry/vector.h `vector_assign_helper`: converting from a floating
// point type to an integral type rounds with floor(x + 0.5); every other
// combination is a plain C++ `static_cast` (truncating `as` in Rust).

macro_rules! vec_conversions_round {
    ($src:ty => $($dst:ty),+) => {
        $(
        impl<const N: usize> From<Vector<$src, N>> for Vector<$dst, N> {
            fn from(v: Vector<$src, N>) -> Self {
                Vector {
                    data: std::array::from_fn(|i| (v.data[i] + 0.5).floor() as $dst),
                }
            }
        }
        )+
    };
}

macro_rules! vec_conversions_cast {
    ($src:ty => $($dst:ty),+) => {
        $(
        impl<const N: usize> From<Vector<$src, N>> for Vector<$dst, N> {
            fn from(v: Vector<$src, N>) -> Self {
                Vector {
                    data: std::array::from_fn(|i| v.data[i] as $dst),
                }
            }
        }
        )+
    };
}

vec_conversions_round!(f64 => i64, i32);
vec_conversions_round!(f32 => i64, i32);
vec_conversions_cast!(i64 => f64, f32, i32, u32);
vec_conversions_cast!(i32 => f64, f32, i64);
vec_conversions_cast!(u32 => f64, f32, i64);
vec_conversions_cast!(f64 => f32);
vec_conversions_cast!(f32 => f64);

// --- operators ------------------------------------------------------------

macro_rules! vec_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Num + Copy, const N: usize> $trait for Vector<T, N> {
            type Output = Vector<T, N>;
            fn $method(self, rhs: Self) -> Self {
                Vector {
                    data: std::array::from_fn(|i| self.data[i] $op rhs.data[i]),
                }
            }
        }
    };
}
vec_binop!(Add, add, +);
vec_binop!(Sub, sub, -);

impl<T: Num + Copy + Neg<Output = T>, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;
    fn neg(self) -> Self {
        Vector {
            data: std::array::from_fn(|i| -self.data[i]),
        }
    }
}

impl<T: Num + Copy, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn mul(self, s: T) -> Self {
        Vector {
            data: std::array::from_fn(|i| self.data[i] * s),
        }
    }
}

// The free `operator*(scalar, vector)` cannot be written generically in Rust
// (uncovered type parameter), so it is provided for the concrete scalar
// types dlib is used with.
macro_rules! scalar_mul_vector {
    ($t:ty) => {
        impl<const N: usize> Mul<Vector<$t, N>> for $t {
            type Output = Vector<$t, N>;
            fn mul(self, v: Vector<$t, N>) -> Vector<$t, N> {
                v * self
            }
        }
    };
}
scalar_mul_vector!(f64);
scalar_mul_vector!(f32);
scalar_mul_vector!(i64);
scalar_mul_vector!(i32);
scalar_mul_vector!(u32);

impl<T: Num + Copy, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;
    fn div(self, s: T) -> Self {
        Vector {
            data: std::array::from_fn(|i| self.data[i] / s),
        }
    }
}

impl<T: Num + Copy, const N: usize> AddAssign for Vector<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Num + Copy, const N: usize> SubAssign for Vector<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Port of `typedef dlib::vector<long,2> point` (dlib/geometry/vector.h).
pub type Point = Vector<i64, 2>;

/// Port of `typedef dlib::vector<double,2> dpoint` (dlib/geometry/vector.h).
pub type Dpoint = Vector<f64, 2>;

// --- serialization (dlib/geometry/vector.h serialize/deserialize) ----------

macro_rules! vec_serialize {
    ($t:ty, $w:ident, $r:ident, 2) => {
        impl Vector<$t, 2> {
            /// Port of `serialize(const vector<T,2>&, std::ostream&)`.
            pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
                out.$w(self.data[0]);
                out.$w(self.data[1]);
            }

            /// Port of `deserialize(vector<T,2>&, std::istream&)`.
            pub fn deserialize(
                inp: &mut crate::serialize::Deserializer<'_>,
            ) -> Result<Self, crate::serialize::SerializeError> {
                let x = inp.$r()?;
                let y = inp.$r()?;
                Ok(Vector {
                    data: [x as $t, y as $t],
                })
            }
        }
    };
    ($t:ty, $w:ident, $r:ident, 3) => {
        impl Vector<$t, 3> {
            /// Port of `serialize(const vector<T,3>&, std::ostream&)`.
            pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
                out.$w(self.data[0]);
                out.$w(self.data[1]);
                out.$w(self.data[2]);
            }

            /// Port of `deserialize(vector<T,3>&, std::istream&)`.
            pub fn deserialize(
                inp: &mut crate::serialize::Deserializer<'_>,
            ) -> Result<Self, crate::serialize::SerializeError> {
                let x = inp.$r()?;
                let y = inp.$r()?;
                let z = inp.$r()?;
                Ok(Vector {
                    data: [x as $t, y as $t, z as $t],
                })
            }
        }
    };
}

vec_serialize!(i64, write_i64, read_i64, 2);
vec_serialize!(i64, write_i64, read_i64, 3);
vec_serialize!(i32, write_i32, read_i32, 2);
vec_serialize!(i32, write_i32, read_i32, 3);
vec_serialize!(f32, write_f32, read_f32, 2);
vec_serialize!(f32, write_f32, read_f32, 3);
vec_serialize!(f64, write_f64, read_f64, 2);
vec_serialize!(f64, write_f64, read_f64, 3);

/// Port of `dlib::polygon_area` (dlib/geometry/vector.h) for any 2D points.
///
/// Shoelace formula; always returns `abs(val) / 2` (i.e. a non-negative area).
pub fn polygon_area(pts: &[Dpoint]) -> f64 {
    if pts.len() <= 2 {
        return 0.0;
    }
    let mut val = 0.0;
    for i in 1..pts.len() {
        val += pts[i].x() * pts[i - 1].y() - pts[i].y() * pts[i - 1].x();
    }
    let end = pts.len() - 1;
    val += pts[0].x() * pts[end].y() - pts[0].y() * pts[end].x();
    val.abs() / 2.0
}

/// Port of `dlib::is_convex_quadrilateral` (dlib/geometry/vector.h).
pub fn is_convex_quadrilateral(pts: &[Dpoint; 4]) -> bool {
    let orientation = |i: usize| -> f64 {
        let a = (i + 1) % 4;
        let b = (i + 3) % 4;
        (pts[a] - pts[i]).cross(&(pts[b] - pts[i]))
    };
    for p in pts {
        if p.x() == f64::INFINITY || p.y() == f64::INFINITY {
            return false;
        }
    }
    let s0 = orientation(0);
    let s1 = orientation(1);
    let s2 = orientation(2);
    let s3 = orientation(3);
    (s0 > 0.0 && s1 > 0.0 && s2 > 0.0 && s3 > 0.0) || (s0 < 0.0 && s1 < 0.0 && s2 < 0.0 && s3 < 0.0)
}

// ---------------------------------------------------------------------------
// Rectangle  (dlib/geometry/rectangle.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::rectangle` (dlib/geometry/rectangle.h).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rectangle {
    l: i64,
    t: i64,
    r: i64,
    b: i64,
}

impl Default for Rectangle {
    /// Port of `rectangle()`: `(0, 0, -1, -1)` (an empty rectangle).
    fn default() -> Self {
        Rectangle {
            l: 0,
            t: 0,
            r: -1,
            b: -1,
        }
    }
}

impl Rectangle {
    /// Port of `rectangle(long l, long t, long r, long b)`.
    pub fn new(l: i64, t: i64, r: i64, b: i64) -> Self {
        Rectangle { l, t, r, b }
    }

    /// Port of `rectangle(unsigned long width, unsigned long height)`.
    pub fn from_width_height(width: u64, height: u64) -> Self {
        debug_assert!(
            (width > 0 && height > 0) || (width == 0 && height == 0),
            "rectangle(width,height): width and height must be > 0 or both == 0"
        );
        Rectangle {
            l: 0,
            t: 0,
            r: width as i64 - 1,
            b: height as i64 - 1,
        }
    }

    /// Port of `rectangle(const point&)`.
    pub fn from_point(p: Point) -> Self {
        Rectangle {
            l: p.x(),
            t: p.y(),
            r: p.x(),
            b: p.y(),
        }
    }

    /// Port of `rectangle(const point&, const point&)` (union of two 1x1 rects).
    pub fn from_points(p1: Point, p2: Point) -> Self {
        Rectangle::from_point(p1) + Rectangle::from_point(p2)
    }

    /// Port of the templated `rectangle(const vector<T,2>&, const vector<T,2>&)`
    /// constructor: each dpoint is converted to a `point` first, which in dlib
    /// rounds with `floor(x + 0.5)` per component.
    pub fn from_dpoints(p1: Dpoint, p2: Dpoint) -> Self {
        Rectangle::from_points(Point::from(p1), Point::from(p2))
    }

    /// Port of `rectangle::left()`.
    pub fn left(&self) -> i64 {
        self.l
    }
    /// Port of `rectangle::top()`.
    pub fn top(&self) -> i64 {
        self.t
    }
    /// Port of `rectangle::right()`.
    pub fn right(&self) -> i64 {
        self.r
    }
    /// Port of `rectangle::bottom()`.
    pub fn bottom(&self) -> i64 {
        self.b
    }

    /// dlib shorthand `rectangle::l()`.
    pub fn l(&self) -> i64 {
        self.l
    }
    /// dlib shorthand `rectangle::t()`.
    pub fn t(&self) -> i64 {
        self.t
    }
    /// dlib shorthand `rectangle::r()`.
    pub fn r(&self) -> i64 {
        self.r
    }
    /// dlib shorthand `rectangle::b()`.
    pub fn b(&self) -> i64 {
        self.b
    }

    /// Port of `rectangle::set_left()`.
    pub fn set_left(&mut self, v: i64) {
        self.l = v;
    }
    /// Port of `rectangle::set_top()`.
    pub fn set_top(&mut self, v: i64) {
        self.t = v;
    }
    /// Port of `rectangle::set_right()`.
    pub fn set_right(&mut self, v: i64) {
        self.r = v;
    }
    /// Port of `rectangle::set_bottom()`.
    pub fn set_bottom(&mut self, v: i64) {
        self.b = v;
    }

    /// Port of `rectangle::tl_corner()`.
    pub fn tl_corner(&self) -> Point {
        Point::new(self.left(), self.top())
    }
    /// Port of `rectangle::bl_corner()`.
    pub fn bl_corner(&self) -> Point {
        Point::new(self.left(), self.bottom())
    }
    /// Port of `rectangle::tr_corner()`.
    pub fn tr_corner(&self) -> Point {
        Point::new(self.right(), self.top())
    }
    /// Port of `rectangle::br_corner()`.
    pub fn br_corner(&self) -> Point {
        Point::new(self.right(), self.bottom())
    }

    /// Port of `rectangle::width()`: `r - l + 1`, or 0 when empty.
    pub fn width(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            (self.r - self.l + 1) as u64
        }
    }

    /// Port of `rectangle::height()`: `b - t + 1`, or 0 when empty.
    pub fn height(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            (self.b - self.t + 1) as u64
        }
    }

    /// Port of `rectangle::area()`: `width() * height()`.
    pub fn area(&self) -> u64 {
        self.width() * self.height()
    }

    /// Port of `rectangle::is_empty()`: `t > b || l > r`.
    pub fn is_empty(&self) -> bool {
        self.t > self.b || self.l > self.r
    }

    /// Port of `rectangle::intersect(rhs)`. dlib does not canonicalize the
    /// result: intersecting disjoint rectangles yields a rectangle whose
    /// `left > right` (empty by the `is_empty()` convention).
    pub fn intersect(&self, rhs: &Rectangle) -> Rectangle {
        Rectangle::new(
            self.l.max(rhs.l),
            self.t.max(rhs.t),
            self.r.min(rhs.r),
            self.b.min(rhs.b),
        )
    }

    /// Port of `rectangle::contains(const point&)`.
    pub fn contains(&self, p: &Point) -> bool {
        !(p.x() < self.l || p.x() > self.r || p.y() < self.t || p.y() > self.b)
    }

    /// Port of `rectangle::contains(long x, long y)`.
    pub fn contains_xy(&self, x: i64, y: i64) -> bool {
        !(x < self.l || x > self.r || y < self.t || y > self.b)
    }

    /// Port of `rectangle::contains(const rectangle&)`:
    /// `rect + *this == *this` (true for any empty `rect`).
    pub fn contains_rect(&self, rect: &Rectangle) -> bool {
        *rect + *self == *self
    }

    /// Port of `serialize(const rectangle&, std::ostream&)`:
    /// left, top, right, bottom as packed longs in that order.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_i64(self.left());
        out.write_i64(self.top());
        out.write_i64(self.right());
        out.write_i64(self.bottom());
    }

    /// Port of `deserialize(rectangle&, std::istream&)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let l = inp.read_i64()?;
        let t = inp.read_i64()?;
        let r = inp.read_i64()?;
        let b = inp.read_i64()?;
        Ok(Rectangle::new(l, t, r, b))
    }
}

/// Port of `rectangle::operator+` (union; empty operands are neutral).
impl Add for Rectangle {
    type Output = Rectangle;
    fn add(self, rhs: Rectangle) -> Rectangle {
        if rhs.is_empty() {
            self
        } else if self.is_empty() {
            rhs
        } else {
            Rectangle::new(
                self.l.min(rhs.l),
                self.t.min(rhs.t),
                self.r.max(rhs.r),
                self.b.max(rhs.b),
            )
        }
    }
}

/// Port of `rectangle::operator+=`.
impl AddAssign for Rectangle {
    fn add_assign(&mut self, rhs: Rectangle) {
        *self = *self + rhs;
    }
}

/// Port of `rectangle::operator+= (const point&)` via the free
/// `operator+(rectangle, point)` in rectangle.h.
impl Add<Point> for Rectangle {
    type Output = Rectangle;
    fn add(self, p: Point) -> Rectangle {
        self + Rectangle::from_point(p)
    }
}

/// Port of the free `operator+(point, rectangle)`.
impl Add<Rectangle> for Point {
    type Output = Rectangle;
    fn add(self, r: Rectangle) -> Rectangle {
        r + Rectangle::from_point(self)
    }
}

/// Intersection via `&` (dlib semantics of `rectangle::intersect`).
impl BitAnd for Rectangle {
    type Output = Rectangle;
    fn bitand(self, rhs: Rectangle) -> Rectangle {
        self.intersect(&rhs)
    }
}

/// Port of `rectangle::operator<` (lexicographic in left/top/right/bottom).
impl PartialOrd for Rectangle {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.l
                .cmp(&other.l)
                .then(self.t.cmp(&other.t))
                .then(self.r.cmp(&other.r))
                .then(self.b.cmp(&other.b)),
        )
    }
}

// --- free rectangle functions (dlib/geometry/rectangle.h) ------------------

/// Port of `centered_rect(long x, long y, unsigned long width, height)`.
pub fn centered_rect(x: i64, y: i64, width: u64, height: u64) -> Rectangle {
    let mut result = Rectangle::default();
    result.set_left(x - width as i64 / 2);
    result.set_top(y - height as i64 / 2);
    result.set_right(result.left() + width as i64 - 1);
    result.set_bottom(result.top() + height as i64 - 1);
    result
}

/// Port of `centered_rect(const point&, width, height)`.
pub fn centered_rect_point(p: &Point, width: u64, height: u64) -> Rectangle {
    centered_rect(p.x(), p.y(), width, height)
}

/// Port of `centered_rect(const rectangle&, width, height)`.
pub fn centered_rect_rect(rect: &Rectangle, width: u64, height: u64) -> Rectangle {
    centered_rect(
        (rect.left() + rect.right()) / 2,
        (rect.top() + rect.bottom()) / 2,
        width,
        height,
    )
}

/// Port of the free `intersect(const rectangle&, const rectangle&)`.
pub fn intersect(a: &Rectangle, b: &Rectangle) -> Rectangle {
    a.intersect(b)
}

/// Port of the free `area(const rectangle&)`.
pub fn area(a: &Rectangle) -> u64 {
    a.area()
}

/// Port of `center(const rectangle&)`: computes
/// `(l + r + 1, t + b + 1)`, subtracting 1 from negative coordinates, then
/// dividing each component by 2 with integer (truncating) division.
pub fn center(rect: &Rectangle) -> Point {
    let mut temp = Point::new(
        rect.left() + rect.right() + 1,
        rect.top() + rect.bottom() + 1,
    );
    if temp.x() < 0 {
        temp = Point::new(temp.x() - 1, temp.y());
    }
    if temp.y() < 0 {
        temp = Point::new(temp.x(), temp.y() - 1);
    }
    temp / 2
}

/// Port of `dcenter(const rectangle&)`.
pub fn dcenter(rect: &Rectangle) -> Dpoint {
    Dpoint::new(
        (rect.left() + rect.right()) as f64 / 2.0,
        (rect.top() + rect.bottom()) as f64 / 2.0,
    )
}

/// Port of `distance_to_rect_edge(const rectangle&, const point&)`.
pub fn distance_to_rect_edge(rect: &Rectangle, p: &Point) -> i64 {
    let dist_x = (p.x() - rect.left())
        .abs()
        .min((p.x() - rect.right()).abs());
    let dist_y = (p.y() - rect.top())
        .abs()
        .min((p.y() - rect.bottom()).abs());
    if rect.contains(p) {
        dist_x.min(dist_y)
    } else if rect.left() <= p.x() && p.x() <= rect.right() {
        dist_y
    } else if rect.top() <= p.y() && p.y() <= rect.bottom() {
        dist_x
    } else {
        dist_x + dist_y
    }
}

/// Port of `nearest_point(const rectangle&, const vector<T,2>&)`.
pub fn nearest_point(rect: &Rectangle, p: &Dpoint) -> Dpoint {
    let mut temp = *p;
    let mut x = temp.x();
    if x < rect.left() as f64 {
        x = rect.left() as f64;
    } else if x > rect.right() as f64 {
        x = rect.right() as f64;
    }
    let mut y = temp.y();
    if y < rect.top() as f64 {
        y = rect.top() as f64;
    } else if y > rect.bottom() as f64 {
        y = rect.bottom() as f64;
    }
    temp = Dpoint::new(x, y);
    temp
}

/// Port of `shrink_rect(const rectangle&, long num)`.
pub fn shrink_rect(rect: &Rectangle, num: i64) -> Rectangle {
    Rectangle::new(
        rect.left() + num,
        rect.top() + num,
        rect.right() - num,
        rect.bottom() - num,
    )
}

/// Port of `grow_rect(const rectangle&, long num)`.
pub fn grow_rect(rect: &Rectangle, num: i64) -> Rectangle {
    shrink_rect(rect, -num)
}

/// Port of `shrink_rect(const rectangle&, long width, long height)`.
pub fn shrink_rect_wh(rect: &Rectangle, width: i64, height: i64) -> Rectangle {
    Rectangle::new(
        rect.left() + width,
        rect.top() + height,
        rect.right() - width,
        rect.bottom() - height,
    )
}

/// Port of `grow_rect(const rectangle&, long width, long height)`.
pub fn grow_rect_wh(rect: &Rectangle, width: i64, height: i64) -> Rectangle {
    shrink_rect_wh(rect, -width, -height)
}

/// Port of `scale_rect(const rectangle&, double scale)` (lround per edge).
pub fn scale_rect(rect: &Rectangle, scale: f64) -> Rectangle {
    assert!(scale > 0.0, "scale factor must be > 0");
    Rectangle::new(
        (rect.left() as f64 * scale).round() as i64,
        (rect.top() as f64 * scale).round() as i64,
        (rect.right() as f64 * scale).round() as i64,
        (rect.bottom() as f64 * scale).round() as i64,
    )
}

/// Port of `translate_rect(const rectangle&, const point&)`.
pub fn translate_rect(rect: &Rectangle, p: &Point) -> Rectangle {
    let mut result = Rectangle::default();
    result.set_top(rect.top() + p.y());
    result.set_bottom(rect.bottom() + p.y());
    result.set_left(rect.left() + p.x());
    result.set_right(rect.right() + p.x());
    result
}

/// Port of `move_rect(const rectangle&, const point&)`.
pub fn move_rect(rect: &Rectangle, p: &Point) -> Rectangle {
    Rectangle::new(
        p.x(),
        p.y(),
        p.x() + rect.width() as i64 - 1,
        p.y() + rect.height() as i64 - 1,
    )
}

/// Port of `move_rect(const rectangle&, long x, long y)`.
pub fn move_rect_xy(rect: &Rectangle, x: i64, y: i64) -> Rectangle {
    Rectangle::new(
        x,
        y,
        x + rect.width() as i64 - 1,
        y + rect.height() as i64 - 1,
    )
}

/// Port of `resize_rect(const rectangle&, unsigned long width, height)`.
pub fn resize_rect(rect: &Rectangle, width: u64, height: u64) -> Rectangle {
    Rectangle::new(
        rect.left(),
        rect.top(),
        rect.left() + width as i64 - 1,
        rect.top() + height as i64 - 1,
    )
}

/// Port of `set_rect_area(const rectangle&, unsigned long area)`.
pub fn set_rect_area(rect: &Rectangle, area: u64) -> Rectangle {
    assert!(area > 0);
    if rect.area() == 0 {
        let scale = (area as f64).sqrt().round() as u64;
        centered_rect_rect(rect, scale, scale)
    } else {
        let scale = (area as f64 / rect.area() as f64).sqrt();
        centered_rect_rect(
            rect,
            (rect.width() as f64 * scale).round() as u64,
            (rect.height() as f64 * scale).round() as u64,
        )
    }
}

/// Port of `set_aspect_ratio(const rectangle&, double ratio)`.
pub fn set_aspect_ratio(rect: &Rectangle, ratio: f64) -> Rectangle {
    assert!(ratio > 0.0);
    if ratio >= 1.0 {
        let h = (rect.area() as f64 / ratio).sqrt() + 0.5;
        let h = h as i64;
        let w = (h as f64 * ratio + 0.5) as i64;
        centered_rect_rect(rect, w as u64, h as u64)
    } else {
        let w = (rect.area() as f64 * ratio).sqrt() + 0.5;
        let w = w as i64;
        let h = (w as f64 / ratio + 0.5) as i64;
        centered_rect_rect(rect, w as u64, h as u64)
    }
}

// ---------------------------------------------------------------------------
// Drectangle  (dlib/geometry/drectangle.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::drectangle` (dlib/geometry/drectangle.h).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Drectangle {
    l: f64,
    t: f64,
    r: f64,
    b: f64,
}

impl Drectangle {
    /// Port of `drectangle(double l, t, r, b)`.
    pub fn new(l: f64, t: f64, r: f64, b: f64) -> Self {
        Drectangle { l, t, r, b }
    }

    /// Port of `drectangle(const vector<double,2>&)`.
    pub fn from_point(p: Dpoint) -> Self {
        Drectangle {
            l: p.x(),
            t: p.y(),
            r: p.x(),
            b: p.y(),
        }
    }

    /// Port of the templated two-point constructor (union of two 1x1 drects).
    pub fn from_points(p1: Dpoint, p2: Dpoint) -> Self {
        Drectangle::from_point(p1) + Drectangle::from_point(p2)
    }

    /// Port of `drectangle::left()` etc.
    pub fn left(&self) -> f64 {
        self.l
    }
    /// Port of `drectangle::top()`.
    pub fn top(&self) -> f64 {
        self.t
    }
    /// Port of `drectangle::right()`.
    pub fn right(&self) -> f64 {
        self.r
    }
    /// Port of `drectangle::bottom()`.
    pub fn bottom(&self) -> f64 {
        self.b
    }

    /// Port of `drectangle::set_left()`.
    pub fn set_left(&mut self, v: f64) {
        self.l = v;
    }
    /// Port of `drectangle::set_top()`.
    pub fn set_top(&mut self, v: f64) {
        self.t = v;
    }
    /// Port of `drectangle::set_right()`.
    pub fn set_right(&mut self, v: f64) {
        self.r = v;
    }
    /// Port of `drectangle::set_bottom()`.
    pub fn set_bottom(&mut self, v: f64) {
        self.b = v;
    }

    /// Port of `drectangle::tl_corner()`.
    pub fn tl_corner(&self) -> Dpoint {
        Dpoint::new(self.left(), self.top())
    }
    /// Port of `drectangle::bl_corner()`.
    pub fn bl_corner(&self) -> Dpoint {
        Dpoint::new(self.left(), self.bottom())
    }
    /// Port of `drectangle::tr_corner()`.
    pub fn tr_corner(&self) -> Dpoint {
        Dpoint::new(self.right(), self.top())
    }
    /// Port of `drectangle::br_corner()`.
    pub fn br_corner(&self) -> Dpoint {
        Dpoint::new(self.right(), self.bottom())
    }

    /// Port of `drectangle::width()`: `r - l + 1`, or 0 when empty.
    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.r - self.l + 1.0
        }
    }

    /// Port of `drectangle::height()`: `b - t + 1`, or 0 when empty.
    pub fn height(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.b - self.t + 1.0
        }
    }

    /// Port of `drectangle::area()`.
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Port of `drectangle::is_empty()`.
    pub fn is_empty(&self) -> bool {
        self.t > self.b || self.l > self.r
    }

    /// Port of `drectangle::intersect(rhs)` (no canonicalization, like
    /// `rectangle::intersect`).
    pub fn intersect(&self, rhs: &Drectangle) -> Drectangle {
        Drectangle::new(
            self.l.max(rhs.l),
            self.t.max(rhs.t),
            self.r.min(rhs.r),
            self.b.min(rhs.b),
        )
    }

    /// Port of `drectangle::contains(const vector<double,2>&)`.
    pub fn contains(&self, p: &Dpoint) -> bool {
        !(p.x() < self.l || p.x() > self.r || p.y() < self.t || p.y() > self.b)
    }

    /// Port of `drectangle::contains(const drectangle&)`.
    pub fn contains_rect(&self, rect: &Drectangle) -> bool {
        if rect.is_empty() {
            return true;
        }
        self.l <= rect.left()
            && self.r >= rect.right()
            && self.t <= rect.top()
            && self.b >= rect.bottom()
    }

    /// Port of `serialize(const drectangle&, std::ostream&)`.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_f64(self.left());
        out.write_f64(self.top());
        out.write_f64(self.right());
        out.write_f64(self.bottom());
    }

    /// Port of `deserialize(drectangle&, std::istream&)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let l = inp.read_f64()?;
        let t = inp.read_f64()?;
        let r = inp.read_f64()?;
        let b = inp.read_f64()?;
        Ok(Drectangle::new(l, t, r, b))
    }
}

/// Port of `drectangle::operator rectangle()` — `std::lround` per edge
/// (round half away from zero, like Rust's `f64::round`).
impl From<Drectangle> for Rectangle {
    fn from(rect: Drectangle) -> Rectangle {
        Rectangle::new(
            rect.l.round() as i64,
            rect.t.round() as i64,
            rect.r.round() as i64,
            rect.b.round() as i64,
        )
    }
}

/// Port of `drectangle(const rectangle&)` — plain widening conversion.
impl From<Rectangle> for Drectangle {
    fn from(rect: Rectangle) -> Drectangle {
        Drectangle::new(
            rect.left() as f64,
            rect.top() as f64,
            rect.right() as f64,
            rect.bottom() as f64,
        )
    }
}

/// Port of `drectangle::operator+` (union).
impl Add for Drectangle {
    type Output = Drectangle;
    fn add(self, rhs: Drectangle) -> Drectangle {
        if rhs.is_empty() {
            self
        } else if self.is_empty() {
            rhs
        } else {
            Drectangle::new(
                self.l.min(rhs.l),
                self.t.min(rhs.t),
                self.r.max(rhs.r),
                self.b.max(rhs.b),
            )
        }
    }
}

/// Port of the free `operator+(drectangle, vector<double,2>)`.
impl Add<Dpoint> for Drectangle {
    type Output = Drectangle;
    fn add(self, p: Dpoint) -> Drectangle {
        self + Drectangle::from_point(p)
    }
}

/// Port of `drectangle::operator*(const drectangle&, const double&)`:
/// scales the width/height about the rectangle center (empty passes through).
impl Mul<f64> for Drectangle {
    type Output = Drectangle;
    fn mul(self, scale: f64) -> Drectangle {
        if !self.is_empty() {
            let width = (self.right() - self.left()) * scale;
            let height = (self.bottom() - self.top()) * scale;
            let p = center_d(&self);
            Drectangle::new(
                p.x() - width / 2.0,
                p.y() - height / 2.0,
                p.x() + width / 2.0,
                p.y() + height / 2.0,
            )
        } else {
            self
        }
    }
}

/// Port of the free `operator*(const double&, const drectangle&)`.
impl Mul<Drectangle> for f64 {
    type Output = Drectangle;
    fn mul(self, rect: Drectangle) -> Drectangle {
        rect * self
    }
}

/// Port of `drectangle::operator/(const drectangle&, const double&)`.
impl Div<f64> for Drectangle {
    type Output = Drectangle;
    fn div(self, scale: f64) -> Drectangle {
        self * (1.0 / scale)
    }
}

// --- free drectangle functions (dlib/geometry/drectangle.h) ----------------

/// Port of `center(const drectangle&)`.
pub fn center_d(rect: &Drectangle) -> Dpoint {
    Dpoint::new(
        (rect.left() + rect.right()) / 2.0,
        (rect.top() + rect.bottom()) / 2.0,
    )
}

/// Port of `dcenter(const drectangle&)` (alias of `center`).
pub fn dcenter_d(rect: &Drectangle) -> Dpoint {
    center_d(rect)
}

/// Port of the free `intersect(const drectangle&, const drectangle&)`.
pub fn intersect_d(a: &Drectangle, b: &Drectangle) -> Drectangle {
    a.intersect(b)
}

/// Port of the free `area(const drectangle&)`.
pub fn area_d(a: &Drectangle) -> f64 {
    a.area()
}

/// Port of `centered_drect(const vector<double,2>&, double width, height)`.
pub fn centered_drect(p: &Dpoint, width: f64, height: f64) -> Drectangle {
    let width = width - 1.0;
    let height = height - 1.0;
    Drectangle::new(
        p.x() - width / 2.0,
        p.y() - height / 2.0,
        p.x() + width / 2.0,
        p.y() + height / 2.0,
    )
}

/// Port of `centered_drect(const drectangle&, double width, height)`.
pub fn centered_drect_rect(rect: &Drectangle, width: f64, height: f64) -> Drectangle {
    centered_drect(&dcenter_d(rect), width, height)
}

/// Port of `shrink_rect(const drectangle&, double num)`.
pub fn shrink_drect(rect: &Drectangle, num: f64) -> Drectangle {
    Drectangle::new(
        rect.left() + num,
        rect.top() + num,
        rect.right() - num,
        rect.bottom() - num,
    )
}

/// Port of `grow_rect(const drectangle&, double num)`.
pub fn grow_drect(rect: &Drectangle, num: f64) -> Drectangle {
    shrink_drect(rect, -num)
}

/// Port of `shrink_rect(const drectangle&, double width, height)`.
pub fn shrink_drect_wh(rect: &Drectangle, width: f64, height: f64) -> Drectangle {
    Drectangle::new(
        rect.left() + width,
        rect.top() + height,
        rect.right() - width,
        rect.bottom() - height,
    )
}

/// Port of `grow_rect(const drectangle&, double width, height)`.
pub fn grow_drect_wh(rect: &Drectangle, width: f64, height: f64) -> Drectangle {
    shrink_drect_wh(rect, -width, -height)
}

/// Port of `translate_rect(const drectangle&, const vector<T,2>&)`.
pub fn translate_drect(rect: &Drectangle, p: &Dpoint) -> Drectangle {
    let mut result = Drectangle::default();
    result.set_top(rect.top() + p.y());
    result.set_bottom(rect.bottom() + p.y());
    result.set_left(rect.left() + p.x());
    result.set_right(rect.right() + p.x());
    result
}

/// Port of `scale_rect(const drectangle&, double scale)`.
pub fn scale_drect(rect: &Drectangle, scale: f64) -> Drectangle {
    assert!(scale > 0.0, "scale factor must be > 0");
    Drectangle::new(
        rect.left() * scale,
        rect.top() * scale,
        rect.right() * scale,
        rect.bottom() * scale,
    )
}

/// Port of `set_rect_area(const drectangle&, double area)`.
pub fn set_rect_area_d(rect: &Drectangle, area: f64) -> Drectangle {
    assert!(area >= 0.0, "drectangle can't have a negative area.");
    if area == 0.0 {
        return Drectangle::from_point(dcenter_d(rect));
    }
    if rect.area() == 0.0 {
        let scale = area.sqrt();
        centered_drect_rect(rect, scale, scale)
    } else {
        let scale = (area / rect.area()).sqrt();
        centered_drect_rect(rect, rect.width() * scale, rect.height() * scale)
    }
}

/// Port of `set_aspect_ratio(const drectangle&, double ratio)`.
pub fn set_aspect_ratio_d(rect: &Drectangle, ratio: f64) -> Drectangle {
    assert!(ratio > 0.0);
    let h = (rect.area() / ratio).sqrt();
    let w = h * ratio;
    centered_drect_rect(rect, w, h)
}

// ---------------------------------------------------------------------------
// Point transforms  (dlib/geometry/point_transforms.h)
// ---------------------------------------------------------------------------

#[allow(clippy::needless_range_loop)]
fn mat2_mul(a: &[[f64; 2]; 2], b: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut out = [[0.0; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c];
        }
    }
    out
}

fn mat2_vec(a: &[[f64; 2]; 2], v: &Dpoint) -> Dpoint {
    Dpoint::new(
        a[0][0] * v.x() + a[0][1] * v.y(),
        a[1][0] * v.x() + a[1][1] * v.y(),
    )
}

fn mat2_det(a: &[[f64; 2]; 2]) -> f64 {
    a[0][0] * a[1][1] - a[0][1] * a[1][0]
}

fn mat2_inv(a: &[[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let d = mat2_det(a);
    [[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]]
}

#[allow(clippy::needless_range_loop)]
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

/// Port of `inv(const matrix<double,3,3>&)` (Gaussian elimination with
/// partial pivoting, same result as dlib's `inv` which uses `lu_decomposition`).
fn mat3_inv(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut a = *m;
    let mut inv = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for col in 0..3 {
        // partial pivot: largest absolute entry in this column at/below row `col`
        let mut pivot = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        assert!(
            a[pivot][col].abs() > 1e-300,
            "matrix is singular, cannot compute inverse"
        );
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for c in 0..3 {
            a[col][c] /= d;
            inv[col][c] /= d;
        }
        for r in 0..3 {
            if r != col {
                let f = a[r][col];
                if f != 0.0 {
                    for c in 0..3 {
                        a[r][c] -= f * a[col][c];
                        inv[r][c] -= f * inv[col][c];
                    }
                }
            }
        }
    }
    inv
}

/// Port of `point_rotator` (dlib/geometry/point_transforms.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointRotator {
    sin_angle: f64,
    cos_angle: f64,
}

impl Default for PointRotator {
    fn default() -> Self {
        PointRotator {
            sin_angle: 0.0,
            cos_angle: 1.0,
        }
    }
}

impl PointRotator {
    /// Port of `point_rotator(const double& angle)`.
    pub fn from_angle(angle: f64) -> Self {
        PointRotator {
            sin_angle: angle.sin(),
            cos_angle: angle.cos(),
        }
    }

    /// Port of `point_rotator::operator()` — always returns doubles.
    pub fn apply(&self, p: &Dpoint) -> Dpoint {
        Dpoint::new(
            self.cos_angle * p.x() - self.sin_angle * p.y(),
            self.sin_angle * p.x() + self.cos_angle * p.y(),
        )
    }

    /// Port of `point_rotator::get_m()`.
    pub fn get_m(&self) -> [[f64; 2]; 2] {
        [
            [self.cos_angle, -self.sin_angle],
            [self.sin_angle, self.cos_angle],
        ]
    }

    /// Port of `serialize(const point_rotator&, ...)`: sin then cos.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_f64(self.sin_angle);
        out.write_f64(self.cos_angle);
    }

    /// Port of `deserialize(point_rotator&, ...)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let sin_angle = inp.read_f64()?;
        let cos_angle = inp.read_f64()?;
        Ok(PointRotator {
            sin_angle,
            cos_angle,
        })
    }
}

/// Port of `point_transform` (rotation about origin followed by translation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTransform {
    sin_angle: f64,
    cos_angle: f64,
    translate: Dpoint,
}

impl Default for PointTransform {
    fn default() -> Self {
        PointTransform {
            sin_angle: 0.0,
            cos_angle: 1.0,
            translate: Dpoint::new(0.0, 0.0),
        }
    }
}

impl PointTransform {
    /// Port of `point_transform(const double& angle, const vector<double,2>&)`.
    pub fn new(angle: f64, translate: Dpoint) -> Self {
        PointTransform {
            sin_angle: angle.sin(),
            cos_angle: angle.cos(),
            translate,
        }
    }

    /// Port of `point_transform::operator()`.
    pub fn apply(&self, p: &Dpoint) -> Dpoint {
        self.apply_raw(p) + self.translate
    }

    fn apply_raw(&self, p: &Dpoint) -> Dpoint {
        Dpoint::new(
            self.cos_angle * p.x() - self.sin_angle * p.y(),
            self.sin_angle * p.x() + self.cos_angle * p.y(),
        )
    }

    /// Port of `point_transform::get_m()`.
    pub fn get_m(&self) -> [[f64; 2]; 2] {
        [
            [self.cos_angle, -self.sin_angle],
            [self.sin_angle, self.cos_angle],
        ]
    }

    /// Port of `point_transform::get_b()`.
    pub fn get_b(&self) -> Dpoint {
        self.translate
    }

    /// Port of `serialize(const point_transform&, ...)`: sin, cos, translate.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_f64(self.sin_angle);
        out.write_f64(self.cos_angle);
        self.translate.serialize(out);
    }

    /// Port of `deserialize(point_transform&, ...)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let sin_angle = inp.read_f64()?;
        let cos_angle = inp.read_f64()?;
        let translate = Dpoint::deserialize(inp)?;
        Ok(PointTransform {
            sin_angle,
            cos_angle,
            translate,
        })
    }
}

/// Port of `point_transform_affine` (dlib/geometry/point_transforms.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTransformAffine {
    m: [[f64; 2]; 2],
    b: Dpoint,
}

impl Default for PointTransformAffine {
    fn default() -> Self {
        PointTransformAffine {
            m: [[1.0, 0.0], [0.0, 1.0]],
            b: Dpoint::new(0.0, 0.0),
        }
    }
}

impl PointTransformAffine {
    /// Port of `point_transform_affine(const matrix<double,2,2>&, const vector<double,2>&)`.
    pub fn new(m: [[f64; 2]; 2], b: Dpoint) -> Self {
        PointTransformAffine { m, b }
    }

    /// Port of `point_transform_affine::operator()`: `m*p + b`.
    pub fn apply(&self, p: &Dpoint) -> Dpoint {
        mat2_vec(&self.m, p) + self.b
    }

    /// Port of `point_transform_affine::get_m()`.
    pub fn get_m(&self) -> &[[f64; 2]; 2] {
        &self.m
    }

    /// Port of `point_transform_affine::get_b()`.
    pub fn get_b(&self) -> &Dpoint {
        &self.b
    }

    /// Port of `serialize(const point_transform_affine&, ...)`: the 2x2
    /// matrix in dlib matrix serialization format (`-nr, -nc`, then elements
    /// row-major, see dlib/matrix/matrix.h) followed by the translation vector.
    #[allow(clippy::needless_range_loop)]
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_i64(-2);
        out.write_i64(-2);
        for r in 0..2 {
            for c in 0..2 {
                out.write_f64(self.m[r][c]);
            }
        }
        self.b.serialize(out);
    }

    /// Port of `deserialize(point_transform_affine&, ...)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let nr = -inp.read_i64()?;
        let nc = -inp.read_i64()?;
        if nr != 2 || nc != 2 {
            return Err(crate::serialize::SerializeError::Malformed(
                "point_transform_affine matrix must be 2x2",
            ));
        }
        let mut m = [[0.0; 2]; 2];
        for row in &mut m {
            for v in row {
                *v = inp.read_f64()?;
            }
        }
        let b = Dpoint::deserialize(inp)?;
        Ok(PointTransformAffine { m, b })
    }
}

/// Port of the free `operator*(point_transform_affine, point_transform_affine)`
/// (i.e. applying `rhs` first, then `lhs`).
#[allow(clippy::suspicious_arithmetic_impl)]
impl Mul for PointTransformAffine {
    type Output = PointTransformAffine;
    fn mul(self, rhs: PointTransformAffine) -> PointTransformAffine {
        PointTransformAffine::new(
            mat2_mul(&self.m, &rhs.m),
            mat2_vec(&self.m, &rhs.b) + self.b,
        )
    }
}

/// Port of `inv(const point_transform_affine&)`.
pub fn inv_affine(trans: &PointTransformAffine) -> PointTransformAffine {
    let im = mat2_inv(trans.get_m());
    PointTransformAffine::new(im, -(mat2_vec(&im, trans.get_b())))
}

/// Port of `rectangle_transform` (dlib/geometry/point_transforms.h).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RectangleTransform {
    tform: PointTransformAffine,
}

impl RectangleTransform {
    /// Port of `rectangle_transform(const point_transform_affine&)`.
    pub fn new(tform: PointTransformAffine) -> Self {
        RectangleTransform { tform }
    }

    /// Port of `rectangle_transform::operator()(const drectangle&)`.
    pub fn apply(&self, r: &Drectangle) -> Drectangle {
        let tl = r.tl_corner();
        let tr = r.tr_corner();
        let bl = r.bl_corner();
        let br = r.br_corner();
        let new_area = (1.0 + (self.tform.apply(&tl) - self.tform.apply(&tr)).length())
            * (1.0 + (self.tform.apply(&tl) - self.tform.apply(&bl)).length());

        let mut temp = Drectangle::default();
        temp = temp + self.tform.apply(&tl);
        temp = temp + self.tform.apply(&tr);
        temp = temp + self.tform.apply(&bl);
        temp = temp + self.tform.apply(&br);

        let scale = (new_area / temp.area()).sqrt();
        centered_drect_rect(&temp, temp.width() * scale, temp.height() * scale)
    }

    /// Port of `rectangle_transform::operator()(const rectangle&)`.
    pub fn apply_rect(&self, r: &Rectangle) -> Rectangle {
        let temp = self.apply(&Drectangle::from(*r));
        let c = Point::from(center_d(&temp));
        centered_rect(
            c.x(),
            c.y(),
            temp.width().round() as u64,
            temp.height().round() as u64,
        )
    }

    /// Port of `rectangle_transform::get_tform()`.
    pub fn get_tform(&self) -> &PointTransformAffine {
        &self.tform
    }

    /// Port of `serialize(const rectangle_transform&, ...)`.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        self.tform.serialize(out);
    }

    /// Port of `deserialize(rectangle_transform&, ...)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        Ok(RectangleTransform {
            tform: PointTransformAffine::deserialize(inp)?,
        })
    }
}

/// Port of `point_transform_projective` (dlib/geometry/point_transforms.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTransformProjective {
    m: [[f64; 3]; 3],
}

impl Default for PointTransformProjective {
    fn default() -> Self {
        PointTransformProjective {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

impl PointTransformProjective {
    /// Port of `point_transform_projective(const matrix<double,3,3>&)`.
    pub fn new(m: [[f64; 3]; 3]) -> Self {
        PointTransformProjective { m }
    }

    /// Port of `point_transform_projective(const point_transform_affine&)`.
    #[allow(clippy::needless_range_loop)]
    pub fn from_affine(tran: &PointTransformAffine) -> Self {
        let mut m = [[0.0; 3]; 3];
        for r in 0..2 {
            for c in 0..2 {
                m[r][c] = tran.get_m()[r][c];
            }
        }
        m[0][2] = tran.get_b().x();
        m[1][2] = tran.get_b().y();
        m[2][0] = 0.0;
        m[2][1] = 0.0;
        m[2][2] = 1.0;
        PointTransformProjective { m }
    }

    /// Port of `point_transform_projective::operator()`: homogeneous apply
    /// followed by a perspective divide (skipped when `z == 0`).
    #[allow(clippy::needless_range_loop)]
    pub fn apply(&self, p: &Dpoint) -> Dpoint {
        let mut temp = Vector::<f64, 3>::new(p.x(), p.y(), 1.0);
        let mut out = [0.0; 3];
        for r in 0..3 {
            out[r] = self.m[r][0] * temp.x() + self.m[r][1] * temp.y() + self.m[r][2] * temp.z();
        }
        temp = Vector::<f64, 3>::new(out[0], out[1], out[2]);
        if temp.z() != 0.0 {
            temp = temp / temp.z();
        }
        Dpoint::new(temp.x(), temp.y())
    }

    /// Port of `point_transform_projective::get_m()`.
    pub fn get_m(&self) -> &[[f64; 3]; 3] {
        &self.m
    }

    /// Port of `serialize(const point_transform_projective&, ...)`: the 3x3
    /// matrix in dlib matrix serialization format (row-major elements).
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_i64(-3);
        out.write_i64(-3);
        for r in 0..3 {
            for c in 0..3 {
                out.write_f64(self.m[r][c]);
            }
        }
    }

    /// Port of `deserialize(point_transform_projective&, ...)`.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer<'_>,
    ) -> Result<Self, crate::serialize::SerializeError> {
        let nr = -inp.read_i64()?;
        let nc = -inp.read_i64()?;
        if nr != 3 || nc != 3 {
            return Err(crate::serialize::SerializeError::Malformed(
                "point_transform_projective matrix must be 3x3",
            ));
        }
        let mut m = [[0.0; 3]; 3];
        for row in &mut m {
            for v in row {
                *v = inp.read_f64()?;
            }
        }
        Ok(PointTransformProjective { m })
    }
}

/// Port of the free `operator*(point_transform_projective, point_transform_projective)`.
impl Mul for PointTransformProjective {
    type Output = PointTransformProjective;
    fn mul(self, rhs: PointTransformProjective) -> PointTransformProjective {
        PointTransformProjective::new(mat3_mul(&self.m, &rhs.m))
    }
}

/// Port of `inv(const point_transform_projective&)`.
pub fn inv_projective(trans: &PointTransformProjective) -> PointTransformProjective {
    PointTransformProjective::new(mat3_inv(trans.get_m()))
}

/// Port of `rotate_point` (dlib/geometry/point_transforms.h).
pub fn rotate_point(center: &Dpoint, p: &Dpoint, angle: f64) -> Dpoint {
    let rot = PointRotator::from_angle(angle);
    rot.apply(&(*p - *center)) + *center
}

/// Port of `rotation_matrix(double angle)` (as a row-major 2x2 array).
pub fn rotation_matrix(angle: f64) -> [[f64; 2]; 2] {
    let ca = angle.cos();
    let sa = angle.sin();
    [[ca, -sa], [sa, ca]]
}

/// Port of `find_affine_transform` (dlib/geometry/point_transforms.h).
///
/// dlib computes `m = Q * pinv(P)` for `P = [x; y; 1]` (3xn) and `Q = [x; y]`
/// (2xn). `pinv` is the Moore-Penrose pseudo-inverse, so this is exactly the
/// least-squares solution `Q Pᵗ (P Pᵗ)⁻¹`; we solve the same normal equations
/// with a 3x3 inverse computed by Gaussian elimination with partial pivoting
/// (identical minimizer, same as dlib's `pinv` up to floating-point rounding).
pub fn find_affine_transform(from_points: &[Dpoint], to_points: &[Dpoint]) -> PointTransformAffine {
    assert!(
        from_points.len() == to_points.len() && from_points.len() >= 3,
        "find_affine_transform(): from_points.size() must equal to_points.size() and be >= 3"
    );

    // P * Pᵗ (3x3) and Q * Pᵗ (2x3)
    let mut ppt = [[0.0f64; 3]; 3];
    let mut qpt = [[0.0f64; 3]; 2];
    for i in 0..from_points.len() {
        let f = [from_points[i].x(), from_points[i].y(), 1.0];
        let q = [to_points[i].x(), to_points[i].y()];
        for a in 0..3 {
            for b in 0..3 {
                ppt[a][b] += f[a] * f[b];
            }
        }
        for a in 0..2 {
            for b in 0..3 {
                qpt[a][b] += q[a] * f[b];
            }
        }
    }

    let ppt_inv = mat3_inv(&ppt);
    // m = (Q Pᵗ) * (P Pᵗ)⁻¹, a 2x3 matrix
    let mut m = [[0.0f64; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            m[r][c] =
                qpt[r][0] * ppt_inv[0][c] + qpt[r][1] * ppt_inv[1][c] + qpt[r][2] * ppt_inv[2][c];
        }
    }
    PointTransformAffine::new(
        [[m[0][0], m[0][1]], [m[1][0], m[1][1]]],
        Dpoint::new(m[0][2], m[1][2]),
    )
}

/// Port of `find_similarity_transform` (dlib/geometry/point_transforms.h),
/// the Umeyama least-squares similarity estimate (equations 34-43 of his
/// paper). The 2x2 SVD of the covariance matrix is computed with a one-sided
/// Jacobi rotation; singular values are returned in descending order like
/// dlib's `svd`.
pub fn find_similarity_transform(
    from_points: &[Dpoint],
    to_points: &[Dpoint],
) -> PointTransformAffine {
    assert!(
        from_points.len() == to_points.len() && from_points.len() >= 2,
        "find_similarity_transform(): from_points.size() must equal to_points.size() and be >= 2"
    );
    let n = from_points.len() as f64;

    let mut mean_from = Dpoint::new(0.0, 0.0);
    let mut mean_to = Dpoint::new(0.0, 0.0);
    for i in 0..from_points.len() {
        mean_from += from_points[i];
        mean_to += to_points[i];
    }
    mean_from = mean_from / n;
    mean_to = mean_to / n;

    let mut sigma_from = 0.0;
    let mut _sigma_to = 0.0;
    let mut cov = [[0.0f64; 2]; 2];
    for i in 0..from_points.len() {
        let f = from_points[i] - mean_from;
        let t = to_points[i] - mean_to;
        sigma_from += f.length_squared();
        _sigma_to += t.length_squared();
        // cov += t * trans(f)
        cov[0][0] += t.x() * f.x();
        cov[0][1] += t.x() * f.y();
        cov[1][0] += t.y() * f.x();
        cov[1][1] += t.y() * f.y();
    }
    sigma_from /= n;
    _sigma_to /= n;
    cov[0][0] /= n;
    cov[0][1] /= n;
    cov[1][0] /= n;
    cov[1][1] /= n;

    let (u, d, v) = svd2(&cov);
    let mut s = [[1.0, 0.0], [0.0, 1.0]];
    let det_cov = mat2_det(&cov);
    let det_uv = mat2_det(&u) * mat2_det(&v);
    if det_cov < 0.0 || (det_cov == 0.0 && det_uv < 0.0) {
        if d[1] < d[0] {
            s[1][1] = -1.0;
        } else {
            s[0][0] = -1.0;
        }
    }
    // r = u * s * trans(v)
    let r = mat2_mul(&mat2_mul(&u, &s), &[[v[0][0], v[1][0]], [v[0][1], v[1][1]]]);

    let mut c = 1.0;
    if sigma_from != 0.0 {
        c = (d[0] * s[0][0] + d[1] * s[1][1]) / sigma_from;
    }
    let t = mean_to - mat2_vec(&r, &mean_from) * c;
    PointTransformAffine::new([[r[0][0] * c, r[0][1] * c], [r[1][0] * c, r[1][1] * c]], t)
}

/// One-sided Jacobi SVD of a 2x2 matrix: returns `(u, d, v)` with
/// `a = u * diag(d) * trans(v)` and `d` in descending order (dlib's `svd`
/// also returns descending singular values).
fn svd2(a: &[[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    // columns of the working matrix
    let mut col0 = [a[0][0], a[1][0]];
    let mut col1 = [a[0][1], a[1][1]];
    let mut v = [[1.0, 0.0], [0.0, 1.0]];

    let dot2 = |x: &[f64; 2], y: &[f64; 2]| x[0] * y[0] + x[1] * y[1];

    for _ in 0..60 {
        let alpha = dot2(&col0, &col0);
        let beta = dot2(&col1, &col1);
        let gamma = dot2(&col0, &col1);
        if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
            break;
        }
        let zeta = (beta - alpha) / (2.0 * gamma);
        let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;
        let new0 = [c * col0[0] - s * col1[0], c * col0[1] - s * col1[1]];
        let new1 = [s * col0[0] + c * col1[0], s * col0[1] + c * col1[1]];
        col0 = new0;
        col1 = new1;
        // rotate the COLUMNS of v (v's columns are the accumulated right
        // singular vectors)
        let newv00 = c * v[0][0] - s * v[0][1];
        let newv10 = c * v[1][0] - s * v[1][1];
        let newv01 = s * v[0][0] + c * v[0][1];
        let newv11 = s * v[1][0] + c * v[1][1];
        v = [[newv00, newv01], [newv10, newv11]];
    }

    let s0 = dot2(&col0, &col0).sqrt();
    let s1 = dot2(&col1, &col1).sqrt();
    if s0 >= s1 {
        (
            [[col0[0] / s0, col1[0] / s1], [col0[1] / s0, col1[1] / s1]],
            [s0, s1],
            v,
        )
    } else {
        (
            [[col1[0] / s1, col0[0] / s0], [col1[1] / s1, col0[1] / s0]],
            [s1, s0],
            [[v[0][1], v[0][0]], [v[1][1], v[1][0]]],
        )
    }
}

// --- find_projective_transform ---------------------------------------------

/// Port of `dlib::impl_proj::find_projective_transform_basic`
/// (dlib/geometry/point_transforms.h, "Method 3" of Zhang's paper).
///
/// Builds `accum += trans(B)*B` for the 2x9 constraint matrix `B` and returns
/// the minimizer of `trans(h)*accum*h`. dlib gets it as the right singular
/// vector for the smallest singular value of `accum` via `svd2`; since
/// `accum` is symmetric positive semi-definite that vector is exactly the
/// eigenvector for the smallest eigenvalue, which we compute with a symmetric
/// (cyclic Jacobi) eigendecomposition.
fn find_projective_transform_basic(
    from_points: &[Dpoint],
    to_points: &[Dpoint],
) -> PointTransformProjective {
    assert!(
        from_points.len() == to_points.len() && from_points.len() >= 4,
        "find_projective_transform_basic(): need >= 4 matching point pairs"
    );

    let mut accum = vec![0.0f64; 81];
    let mut add_outer = |b: &[f64; 9]| {
        for r in 0..9 {
            for c in 0..9 {
                accum[r * 9 + c] += b[r] * b[c];
            }
        }
    };

    for i in 0..from_points.len() {
        let f = [from_points[i].x(), from_points[i].y(), 1.0];
        let tx = to_points[i].x();
        let ty = to_points[i].y();
        // row0 of B: [ty * trans(f), -tx * trans(f), 0]
        // row1 of B: [trans(f),       0,            -tx * trans(f)]
        let row0 = [
            ty * f[0],
            ty * f[1],
            ty * f[2],
            -tx * f[0],
            -tx * f[1],
            -tx * f[2],
            0.0,
            0.0,
            0.0,
        ];
        let row1 = [
            f[0],
            f[1],
            f[2],
            0.0,
            0.0,
            0.0,
            -tx * f[0],
            -tx * f[1],
            -tx * f[2],
        ];
        add_outer(&row0);
        add_outer(&row1);
    }

    let (w, evecs) = jacobi_eigen_symm(9, &accum);
    // index_of_min(w): eigenvector of the smallest eigenvalue
    let mut j = 0;
    let mut min = f64::INFINITY;
    for (idx, val) in w.iter().enumerate() {
        if *val < min {
            min = *val;
            j = idx;
        }
    }

    // reshape(colm(u, j), 3, 3) is row-major in dlib (see op_reshape in
    // dlib/matrix/matrix_utilities.h: idx = r*cols + c)
    let mut h = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            h[r][c] = evecs[(r * 3 + c) * 9 + j];
        }
    }
    PointTransformProjective::new(h)
}

/// Symmetric Jacobi eigendecomposition of an `n x n` symmetric matrix given
/// row-major in `a` (destroyed). Returns `(eigenvalues, eigenvectors)` with
/// `eigenvalues[k]` paired with eigenvector column `k` of `eigenvectors`
/// (stored row-major, i.e. component `r` of eigenvector `k` is
/// `eigenvectors[k*n + r]`).
fn jacobi_eigen_symm(n: usize, a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        // off-diagonal Frobenius norm
        let mut off = 0.0;
        for r in 1..n {
            for c in 0..r {
                off += a[r * n + c] * a[r * n + c];
            }
        }
        if off <= 1e-30 {
            break;
        }
        let thresh = if _sweep < 4 {
            off / (n * n) as f64
        } else {
            0.0
        };
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= thresh.max(1e-300) {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let w: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (w, v)
}

/// Port of `dlib::impl_proj::obj` — the mean squared reprojection error of a
/// 3x3 projective matrix `p` (row-major, 9 entries).
fn projective_obj(from_points: &[Dpoint], to_points: &[Dpoint], p: &[f64; 9]) -> f64 {
    let tran =
        PointTransformProjective::new([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]]);
    let mut sum = 0.0;
    for i in 0..from_points.len() {
        sum += (tran.apply(&from_points[i]) - to_points[i]).length_squared();
    }
    sum
}

/// Port of `dlib::impl_proj::obj_der` — the derivative of [`projective_obj`].
fn projective_obj_der(from_points: &[Dpoint], to_points: &[Dpoint], p: &[f64; 9]) -> [f64; 9] {
    let mut grad = [0.0f64; 9];
    for i in 0..from_points.len() {
        let fx = from_points[i].x();
        let fy = from_points[i].y();
        let tx = to_points[i].x();
        let ty = to_points[i].y();

        let w = [
            p[0] * fx + p[1] * fy + p[2],
            p[3] * fx + p[4] * fy + p[5],
            p[6] * fx + p[7] * fy + p[8],
        ];
        let scale = if w[2] != 0.0 { 1.0 / w[2] } else { 1.0 };
        let w: Vec<f64> = w.iter().map(|x| x * scale).collect();
        let residual = [(w[0] - tx) * 2.0 * scale, (w[1] - ty) * 2.0 * scale];

        grad[0] += fx * residual[0];
        grad[1] += fy * residual[0];
        grad[2] += residual[0];

        grad[3] += fx * residual[1];
        grad[4] += fy * residual[1];
        grad[5] += residual[1];

        grad[6] += -(fx * w[0] * residual[0] + fx * w[1] * residual[1]);
        grad[7] += -(fy * w[0] * residual[0] + fy * w[1] * residual[1]);
        grad[8] += -(w[0] * residual[0] + w[1] * residual[1]);
    }
    grad
}

/// Compact BFGS with a strong-Wolfe line search, mirroring dlib's
/// `find_min(bfgs_search_strategy(), objective_delta_stop_strategy(1e-6, 100), ...)`
/// structure: the same default Wolfe constants (`rho = 0.01`,
/// `sigma = 0.9`) and the same stopping rule (stop when the change in the
/// objective drops below `1e-6`, at most 100 iterations).
fn bfgs_minimize(
    f: &dyn Fn(&[f64; 9]) -> f64,
    der: &dyn Fn(&[f64; 9]) -> [f64; 9],
    x: &mut [f64; 9],
) {
    const RHO: f64 = 0.01;
    const SIGMA: f64 = 0.9;
    const MIN_DELTA: f64 = 1e-6;
    const MAX_ITER: usize = 100;

    let n = x.len();
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        h[i * n + i] = 1.0;
    }
    let mut f_val = f(x);
    let mut grad = der(x).to_vec();

    for _ in 0..MAX_ITER {
        // direction = -h * grad
        let mut dir = vec![0.0; n];
        for r in 0..n {
            for c in 0..n {
                dir[r] -= h[r * n + c] * grad[c];
            }
        }
        let g0: f64 = (0..n).map(|i| grad[i] * dir[i]).sum();
        if g0 >= 0.0 || g0.abs() < 1e-300 {
            break;
        }

        // strong-Wolfe line search (Nocedal & Wright algorithm 3.5/3.6)
        let alpha = line_search_strong_wolfe(f, der, f_val, g0, x, &dir, RHO, SIGMA);
        if !alpha.is_finite() || alpha == 0.0 {
            break;
        }

        let mut x_new = *x;
        for i in 0..n {
            x_new[i] += alpha * dir[i];
        }
        let f_new = f(&x_new);
        let grad_new = der(&x_new).to_vec();

        // BFGS inverse-Hessian update
        let s: Vec<f64> = (0..n).map(|i| alpha * dir[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| grad_new[i] - grad[i]).collect();
        let sy: f64 = (0..n).map(|i| s[i] * y[i]).sum();
        if sy <= 1e-12 {
            break;
        }
        // h = (I - s yᵗ/sy) h (I - y sᵗ/sy) + s sᵗ/sy
        let hy: Vec<f64> = (0..n)
            .map(|r| (0..n).map(|c| h[r * n + c] * y[c]).sum())
            .collect();
        // (h y) sᵗ / sy is the second correction term's inner part
        let mut h_new = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                let first = h[r * n + c] - (s[r] * hy[c]) / sy;
                let second = (0..n).map(|k| first * y[k] * s[k] / sy).sum::<f64>();
                h_new[r * n + c] = first - second + (s[r] * s[c]) / sy;
            }
        }
        h = h_new;

        let delta = (f_val - f_new).abs();
        *x = x_new;
        f_val = f_new;
        grad = grad_new;
        if delta < MIN_DELTA {
            break;
        }
    }
}

/// Strong-Wolfe line search used by [`bfgs_minimize`] (bracket + zoom,
/// Nocedal & Wright algorithms 3.5 and 3.6). Directional derivatives are
/// computed from the exact gradient function `der`.
#[allow(clippy::too_many_arguments)]
fn line_search_strong_wolfe(
    f: &dyn Fn(&[f64; 9]) -> f64,
    der: &dyn Fn(&[f64; 9]) -> [f64; 9],
    f0: f64,
    g0: f64,
    x: &[f64; 9],
    dir: &[f64],
    c1: f64,
    c2: f64,
) -> f64 {
    let eval = |a: f64| -> (f64, f64) {
        let mut xa = [0.0; 9];
        for i in 0..9 {
            xa[i] = x[i] + a * dir[i];
        }
        let fa = f(&xa);
        let ga = der(&xa).iter().zip(dir).map(|(g, d)| g * d).sum::<f64>();
        (fa, ga)
    };

    let mut a_prev = 0.0;
    let mut f_prev = f0;
    let mut a = 1.0;
    let mut a_lo = 0.0;
    let mut f_lo = f0;

    for _ in 0..40 {
        let (fa, ga) = eval(a);
        if fa > f0 + c1 * a * g0 || (fa >= f_prev && a > 0.0) {
            return zoom(&eval, a_lo, a, f_lo, fa, f0, g0, c1, c2);
        }
        if ga.abs() <= -c2 * g0 {
            return a;
        }
        if ga >= 0.0 {
            return zoom(&eval, a, a_prev, fa, f_prev, f0, g0, c1, c2);
        }
        a_prev = a;
        f_prev = fa;
        f_lo = fa;
        a_lo = a;
        a *= 2.0;
    }
    a
}

#[allow(clippy::too_many_arguments)]
fn zoom(
    eval: &dyn Fn(f64) -> (f64, f64),
    mut a_lo: f64,
    mut a_hi: f64,
    mut f_lo: f64,
    mut _f_hi: f64,
    f0: f64,
    g0: f64,
    c1: f64,
    c2: f64,
) -> f64 {
    for _ in 0..40 {
        let a = 0.5 * (a_lo + a_hi);
        let (fa, ga) = eval(a);
        if fa > f0 + c1 * a * g0 || fa >= f_lo {
            a_hi = a;
            _f_hi = fa;
        } else {
            if ga.abs() <= -c2 * g0 {
                return a;
            }
            if ga * (a_hi - a_lo) >= 0.0 {
                a_hi = a;
                _f_hi = fa;
            }
            a_lo = a;
            f_lo = fa;
        }
        if (a_hi - a_lo).abs() < 1e-14 {
            break;
        }
    }
    a_lo
}

/// Port of `find_projective_transform` (dlib/geometry/point_transforms.h):
/// seed with the better of the direct (Zhang "Method 3") estimate and the
/// best affine transform, then refine with BFGS on the true mean squared
/// reprojection error.
pub fn find_projective_transform(
    from_points: &[Dpoint],
    to_points: &[Dpoint],
) -> PointTransformProjective {
    assert!(
        from_points.len() == to_points.len() && from_points.len() >= 4,
        "find_projective_transform(): from_points.size() must equal to_points.size() and be >= 4"
    );

    let tran1 = find_projective_transform_basic(from_points, to_points);
    let tran2 = find_affine_transform(from_points, to_points);
    let tran2p = PointTransformProjective::from_affine(&tran2);

    let mut error1 = 0.0;
    let mut error2 = 0.0;
    for i in 0..from_points.len() {
        error1 += (tran1.apply(&from_points[i]) - to_points[i]).length_squared();
        error2 += (tran2p.apply(&from_points[i]) - to_points[i]).length_squared();
    }

    let mut params: [f64; 9] = if error1 < error2 {
        let m = tran1.get_m();
        [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ]
    } else {
        let m = tran2p.get_m();
        [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ]
    };

    let from = from_points;
    let to = to_points;
    let obj = |p: &[f64; 9]| projective_obj(from, to, p);
    let der = |p: &[f64; 9]| projective_obj_der(from, to, p);
    bfgs_minimize(&obj, &der, &mut params);

    PointTransformProjective::new([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], params[8]],
    ])
}

// ---------------------------------------------------------------------------
// Line  (dlib/geometry/line.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::line` (dlib/geometry/line.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    end1: Dpoint,
    end2: Dpoint,
    normal_vector: Dpoint,
}

impl Default for Line {
    /// Port of `line() = default` (all-zero endpoints and normal).
    fn default() -> Self {
        Line {
            end1: Dpoint::new(0.0, 0.0),
            end2: Dpoint::new(0.0, 0.0),
            normal_vector: Dpoint::new(0.0, 0.0),
        }
    }
}

impl Line {
    /// Port of `line(const dpoint&, const dpoint&)`. The normal is
    /// `(end1 - end2).cross((0,0,1)).normalize()` projected back to 2D.
    pub fn new(a: Dpoint, b: Dpoint) -> Self {
        let normal_vector = {
            let d = a - b;
            // 2D x 3D cross in dlib gives (d.y()*1, -d.x()*1, 0); normalize
            let n = Dpoint::new(d.y(), -d.x());
            n.normalize()
        };
        Line {
            end1: a,
            end2: b,
            normal_vector,
        }
    }

    /// Port of `line::p1()`.
    pub fn p1(&self) -> &Dpoint {
        &self.end1
    }

    /// Port of `line::p2()`.
    pub fn p2(&self) -> &Dpoint {
        &self.end2
    }

    /// Port of `line::normal()`.
    pub fn normal(&self) -> &Dpoint {
        &self.normal_vector
    }

    /// Port of `signed_distance_to_line(const line&, const vector<U,2>&)`.
    pub fn signed_distance(&self, p: &Dpoint) -> f64 {
        (*p - *self.p1()).dot(self.normal())
    }

    /// Port of `distance_to_line(const line&, const vector<U,2>&)`.
    pub fn distance(&self, p: &Dpoint) -> f64 {
        self.signed_distance(p).abs()
    }
}

/// Port of `reverse(const line&)`.
pub fn reverse_line(l: &Line) -> Line {
    Line::new(*l.p2(), *l.p1())
}

/// Port of `intersect(const line&, const line&)`: returns
/// `(infinity, infinity)` for parallel lines, exactly like dlib.
pub fn intersect_lines(a: &Line, b: &Line) -> Dpoint {
    intersect_line_points(a.p1(), a.p2(), b.p1(), b.p2())
}

/// Checked variant of [`intersect_lines`] (dlib signals parallel lines with
/// an infinite point instead; this maps that case to `None`).
pub fn intersect_lines_checked(a: &Line, b: &Line) -> Option<Dpoint> {
    let p = intersect_lines(a, b);
    if p.x().is_infinite() || p.y().is_infinite() {
        None
    } else {
        Some(p)
    }
}

fn intersect_line_points(a1: &Dpoint, a2: &Dpoint, b1: &Dpoint, b2: &Dpoint) -> Dpoint {
    // convert to homogeneous coordinates and take cross products
    let h1 = Vector::<f64, 3>::new(a1.x(), a1.y(), 1.0);
    let h2 = Vector::<f64, 3>::new(a2.x(), a2.y(), 1.0);
    let h3 = Vector::<f64, 3>::new(b1.x(), b1.y(), 1.0);
    let h4 = Vector::<f64, 3>::new(b2.x(), b2.y(), 1.0);
    let l1 = h1.cross(&h2);
    let l2 = h3.cross(&h4);
    let p = l1.cross(&l2);
    if p.z() != 0.0 {
        Dpoint::new(p.x() / p.z(), p.y() / p.z())
    } else {
        Dpoint::new(f64::INFINITY, f64::INFINITY)
    }
}

/// Port of `count_points_on_side_of_line` (dlib/geometry/line.h).
pub fn count_points_on_side_of_line(
    l: &Line,
    reference_point: &Dpoint,
    pts: &[Dpoint],
    dist_thresh_min: f64,
    dist_thresh_max: f64,
) -> usize {
    let mut l = *l;
    if l.signed_distance(reference_point) < 0.0 {
        l = reverse_line(&l);
    }
    let mut cnt = 0;
    for p in pts {
        let dist = l.signed_distance(p);
        if dist_thresh_min <= dist && dist <= dist_thresh_max {
            cnt += 1;
        }
    }
    cnt
}

/// Port of `count_points_between_lines` (dlib/geometry/line.h).
pub fn count_points_between_lines(
    l1: &Line,
    l2: &Line,
    reference_point: &Dpoint,
    pts: &[Dpoint],
) -> usize {
    let mut l1 = *l1;
    let mut l2 = *l2;
    if l1.signed_distance(reference_point) < 0.0 {
        l1 = reverse_line(&l1);
    }
    if l2.signed_distance(reference_point) < 0.0 {
        l2 = reverse_line(&l2);
    }
    let mut cnt = 0;
    for p in pts {
        if l1.signed_distance(p) > 0.0 && l2.signed_distance(p) > 0.0 {
            cnt += 1;
        }
    }
    cnt
}

/// Port of `put_in_range(0.0, 1.0, x)`.
fn put_in_range(lo: f64, hi: f64, x: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Port of `angle_between_lines` (returns degrees).
pub fn angle_between_lines(a: &Line, b: &Line) -> f64 {
    let tmp = put_in_range(0.0, 1.0, a.normal().dot(b.normal()).abs());
    tmp.acos() * 180.0 / std::f64::consts::PI
}

/// Port of `dlib::no_convex_quadrilateral` (dlib/geometry/line.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Lines given to find_convex_quadrilateral() don't form any convex quadrilateral.")]
pub struct NoConvexQuadrilateral;

/// Port of `find_convex_quadrilateral` (dlib/geometry/line.h).
pub fn find_convex_quadrilateral(lines: &[Line; 4]) -> Result<[Dpoint; 4], NoConvexQuadrilateral> {
    let v01 = intersect_lines(&lines[0], &lines[1]);
    let v02 = intersect_lines(&lines[0], &lines[2]);
    let v03 = intersect_lines(&lines[0], &lines[3]);
    let v12 = intersect_lines(&lines[1], &lines[2]);
    let v13 = intersect_lines(&lines[1], &lines[3]);
    let v23 = intersect_lines(&lines[2], &lines[3]);
    let (v10, v20, v30, v21, v31, v32) = (v01, v02, v03, v12, v13, v23);

    if is_convex_quadrilateral(&[v01, v12, v23, v30]) {
        return Ok([v01, v12, v23, v30]);
    }
    if is_convex_quadrilateral(&[v01, v13, v32, v20]) {
        return Ok([v01, v13, v32, v20]);
    }
    if is_convex_quadrilateral(&[v02, v23, v31, v10]) {
        return Ok([v02, v23, v31, v10]);
    }
    if is_convex_quadrilateral(&[v02, v21, v13, v30]) {
        return Ok([v02, v21, v13, v30]);
    }
    if is_convex_quadrilateral(&[v03, v32, v21, v10]) {
        return Ok([v03, v32, v21, v10]);
    }
    if is_convex_quadrilateral(&[v03, v31, v12, v20]) {
        return Ok([v03, v31, v12, v20]);
    }
    Err(NoConvexQuadrilateral)
}

// ---------------------------------------------------------------------------
// Polygon  (dlib/geometry/polygon.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::polygon` (dlib/geometry/polygon.h).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Polygon {
    points: Vec<Point>,
}

impl Polygon {
    /// Port of `polygon(std::vector<point>)`.
    pub fn new(points: Vec<Point>) -> Self {
        Polygon { points }
    }

    /// Port of `polygon::size()`.
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// Port of `polygon::operator[]`.
    pub fn point(&self, idx: usize) -> &Point {
        &self.points[idx]
    }

    /// Port of `polygon::begin()/end()` iteration over the polygon's points.
    pub fn iter(&self) -> std::slice::Iter<'_, Point> {
        self.points.iter()
    }

    /// Port of `polygon::get_rect()`: the smallest rectangle containing all
    /// points.
    pub fn get_rect(&self) -> Rectangle {
        let mut rect = Rectangle::default();
        for p in &self.points {
            rect = rect + *p;
        }
        rect
    }

    /// Port of `polygon::area()`, i.e. of `dlib::polygon_area(points)`
    /// (dlib/geometry/vector.h): shoelace formula, result is always
    /// non-negative (`abs(val)/2`).
    pub fn area(&self) -> f64 {
        if self.points.len() <= 2 {
            return 0.0;
        }
        let mut val = 0.0f64;
        for i in 1..self.points.len() {
            val += self.points[i].x() as f64 * self.points[i - 1].y() as f64
                - self.points[i].y() as f64 * self.points[i - 1].x() as f64;
        }
        let end = self.points.len() - 1;
        val += self.points[0].x() as f64 * self.points[end].y() as f64
            - self.points[0].y() as f64 * self.points[end].x() as f64;
        val.abs() / 2.0
    }

    /// Even-odd (ray casting) point-in-polygon test. NOTE: dlib's
    /// `polygon` class does not define `contains`; this helper follows the
    /// boundary semantics of `get_left_and_right_bounds` (interior includes
    /// the boundary traced by the polygon edges).
    pub fn contains(&self, p: &Point) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let pi = self.points[i];
            let pj = self.points[j];
            if (pi.y() > p.y()) != (pj.y() > p.y()) {
                let intersect_x = (pj.x() - pi.x()) * (p.y() - pi.y()) / (pj.y() - pi.y()) + pi.x();
                if p.x() < intersect_x {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }
}

impl IntoIterator for Polygon {
    type Item = Point;
    type IntoIter = std::vec::IntoIter<Point>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter()
    }
}

impl<'a> IntoIterator for &'a Polygon {
    type Item = &'a Point;
    type IntoIter = std::slice::Iter<'a, Point>;
    fn into_iter(self) -> Self::IntoIter {
        self.points.iter()
    }
}

// ---------------------------------------------------------------------------
// BorderEnumerator  (dlib/geometry/border_enumerator.h)
// ---------------------------------------------------------------------------

/// Port of `dlib::border_enumerator` (dlib/geometry/border_enumerator.h),
/// including its exact state machine (left, top, right, bottom edge order and
/// `move_next()` advance logic).
#[derive(Clone, Debug)]
pub struct BorderEnumerator {
    p: Point,
    rect: Rectangle,
    inner_rect: Rectangle,
    mode: Emode,
    btop: Rectangle,
    bleft: Rectangle,
    bright: Rectangle,
    bbottom: Rectangle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum Emode {
    AtLeft,
    AtRight,
    AtBottom,
    AtTop,
}

impl Default for BorderEnumerator {
    fn default() -> Self {
        BorderEnumerator::new(Rectangle::default(), 0)
    }
}

impl BorderEnumerator {
    /// Port of `border_enumerator(const rectangle&, unsigned long border_size)`.
    pub fn new(rect: Rectangle, border_size: u64) -> Self {
        let mut e = BorderEnumerator {
            p: Point::new(0, 0),
            rect,
            inner_rect: shrink_rect(&rect, border_size as i64),
            mode: Emode::AtLeft,
            btop: Rectangle::default(),
            bleft: Rectangle::default(),
            bright: Rectangle::default(),
            bbottom: Rectangle::default(),
        };
        e.reset();
        e
    }

    /// Port of `border_enumerator(const rectangle&, const rectangle& non_border_region)`.
    pub fn new_with_non_border_region(rect: Rectangle, non_border_region: &Rectangle) -> Self {
        let mut e = BorderEnumerator {
            p: Point::new(0, 0),
            rect,
            inner_rect: non_border_region.intersect(&rect),
            mode: Emode::AtLeft,
            btop: Rectangle::default(),
            bleft: Rectangle::default(),
            bright: Rectangle::default(),
            bbottom: Rectangle::default(),
        };
        e.reset();
        e
    }

    /// Port of `border_enumerator::reset()`.
    pub fn reset(&mut self) {
        // make the four rectangles that surround inner_rect and intersect them
        // with rect.
        self.bleft = self.rect.intersect(&Rectangle::new(
            i64::MIN,
            i64::MIN,
            self.inner_rect.left() - 1,
            i64::MAX,
        ));

        self.bright = self.rect.intersect(&Rectangle::new(
            self.inner_rect.right() + 1,
            i64::MIN,
            i64::MAX,
            i64::MAX,
        ));

        self.btop = self.rect.intersect(&Rectangle::new(
            self.inner_rect.left(),
            i64::MIN,
            self.inner_rect.right(),
            self.inner_rect.top() - 1,
        ));

        self.bbottom = self.rect.intersect(&Rectangle::new(
            self.inner_rect.left(),
            self.inner_rect.bottom() + 1,
            self.inner_rect.right(),
            i64::MAX,
        ));

        self.p = self.bleft.tl_corner();
        self.p = Point::new(self.p.x() - 1, self.p.y());

        self.mode = Emode::AtLeft;
    }

    /// Port of `border_enumerator::at_start()`.
    pub fn at_start(&self) -> bool {
        let mut temp = self.bleft.tl_corner();
        temp = Point::new(temp.x() - 1, temp.y());
        temp == self.p
    }

    /// Port of `border_enumerator::current_element_valid()`.
    pub fn current_element_valid(&self) -> bool {
        self.rect.contains(&self.p)
    }

    /// Port of `border_enumerator::move_next()`.
    pub fn move_next(&mut self) -> bool {
        if self.mode == Emode::AtLeft {
            let bleft = self.bleft;
            if self.advance_point(bleft) {
                return true;
            }
            self.mode = Emode::AtTop;
            let mut p = self.btop.tl_corner();
            p = Point::new(p.x() - 1, p.y());
            self.p = p;
        }
        if self.mode == Emode::AtTop {
            let btop = self.btop;
            if self.advance_point(btop) {
                return true;
            }
            self.mode = Emode::AtRight;
            let mut p = self.bright.tl_corner();
            p = Point::new(p.x() - 1, p.y());
            self.p = p;
        }
        if self.mode == Emode::AtRight {
            let bright = self.bright;
            if self.advance_point(bright) {
                return true;
            }
            self.mode = Emode::AtBottom;
            let mut p = self.bbottom.tl_corner();
            p = Point::new(p.x() - 1, p.y());
            self.p = p;
        }

        let bbottom = self.bbottom;
        if self.advance_point(bbottom) {
            return true;
        }

        // put p outside rect since there are no more points to enumerate
        let mut p = self.rect.br_corner();
        p = Point::new(p.x() + 1, p.y());
        self.p = p;

        false
    }

    /// Port of `border_enumerator::size()`:
    /// `rect.area() - inner_rect.area()` (unsigned subtraction).
    pub fn size(&self) -> u64 {
        self.rect.area().wrapping_sub(self.inner_rect.area())
    }

    /// Port of `border_enumerator::element()` (as `(x, y)` row/column pair:
    /// dlib's `point` is `x() == column`, `y() == row`).
    pub fn element(&self) -> Point {
        self.p
    }

    /// Port of `border_enumerator::advance_point()`.
    fn advance_point(&mut self, r: Rectangle) -> bool {
        self.p = Point::new(self.p.x() + 1, self.p.y());
        if self.p.x() > r.right() {
            self.p = Point::new(r.left(), self.p.y() + 1);
        }
        r.contains(&self.p)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_vector_basics() {
        let a = Dpoint::new(3.0, 4.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.length_squared(), 25.0);
        let n = a.normalize();
        assert!(close(n.x(), 0.6, 1e-15) && close(n.y(), 0.8, 1e-15));

        let b = Dpoint::new(1.0, 2.0);
        assert_eq!(a.dot(&b), 11.0);
        assert_eq!(a - b, Dpoint::new(2.0, 2.0));
        assert_eq!(a + b, Dpoint::new(4.0, 6.0));
        assert_eq!(-a, Dpoint::new(-3.0, -4.0));
        assert_eq!(a * 2.0, Dpoint::new(6.0, 8.0));
        assert_eq!(2.0 * a, Dpoint::new(6.0, 8.0));
        assert_eq!(a / 2.0, Dpoint::new(1.5, 2.0));

        // z() of a 2D vector is 0, like dlib
        assert_eq!(Dpoint::new(1.0, 2.0).z(), 0.0);

        // integer division semantics for point / long
        let p = Point::new(7, -7);
        let q = p / 2;
        assert_eq!(q, Point::new(3, -3)); // C++ truncation toward zero
    }

    #[test]
    fn test_vector_cross() {
        // 2D cross: scalar z of the 3D cross product
        let a = Dpoint::new(1.0, 0.0);
        let b = Dpoint::new(0.0, 1.0);
        assert_eq!(a.cross(&b), 1.0);
        assert_eq!(b.cross(&a), -1.0);

        // 3D cross
        let u = Vector::<f64, 3>::new(1.0, 0.0, 0.0);
        let v = Vector::<f64, 3>::new(0.0, 1.0, 0.0);
        assert_eq!(u.cross(&v), Vector::<f64, 3>::new(0.0, 0.0, 1.0));
        assert_eq!(v.cross(&u), Vector::<f64, 3>::new(0.0, 0.0, -1.0));

        // integer cross
        let p = Point::new(2, 3);
        let q = Point::new(-1, 4);
        assert_eq!(p.cross(&q), 2 * 4 + 3);
    }

    #[test]
    fn test_vector_conversion_rounding() {
        // float -> integral rounds with floor(x + 0.5)
        let p = Point::from(Dpoint::new(0.5, 1.49));
        assert_eq!(p, Point::new(1, 1));
        let p = Point::from(Dpoint::new(-0.5, -1.51));
        assert_eq!(p, Point::new(0, -2));
        // integral -> float is a plain cast
        let d = Dpoint::from(Point::new(3, -4));
        assert_eq!(d, Dpoint::new(3.0, -4.0));
    }

    #[test]
    fn test_rectangle_union_intersect_conventions() {
        let a = Rectangle::new(0, 0, 10, 10);
        let b = Rectangle::new(5, 5, 15, 15);
        let u = a + b;
        assert_eq!(u, Rectangle::new(0, 0, 15, 15));
        let i = a.intersect(&b);
        assert_eq!(i, Rectangle::new(5, 5, 10, 10));

        // Disjoint intersection: NOT canonicalized; left > right, so is_empty
        let c = Rectangle::new(20, 20, 30, 30);
        let i = a.intersect(&c);
        assert_eq!(i, Rectangle::new(20, 20, 10, 10));
        assert!(i.is_empty());
        assert_eq!(i.width(), 0);
        assert_eq!(i.height(), 0);
        assert_eq!(i.area(), 0);

        // union with empty returns the other operand
        let e = Rectangle::default();
        assert_eq!(e + a, a);
        assert_eq!(a + e, a);

        // BitAnd is intersect
        assert_eq!(a & b, a.intersect(&b));
    }

    #[test]
    fn test_rectangle_area_and_contains() {
        let r = Rectangle::new(2, 3, 5, 7);
        assert_eq!(r.width(), 4);
        assert_eq!(r.height(), 5);
        assert_eq!(r.area(), 20); // (right-left+1)*(bottom-top+1)
        assert!(r.contains(&Point::new(2, 3)));
        assert!(r.contains(&Point::new(5, 7)));
        assert!(!r.contains(&Point::new(6, 7)));
        assert!(r.contains_xy(5, 7));
        assert!(!r.contains_xy(1, 3));
        assert!(r.contains_rect(&Rectangle::new(3, 4, 4, 6)));
        assert!(!r.contains_rect(&Rectangle::new(3, 4, 4, 8)));
        // any empty rect is contained
        assert!(r.contains_rect(&Rectangle::new(100, 100, 90, 90)));

        // 1x1 and default rectangle
        let one = Rectangle::from_point(Point::new(4, 9));
        assert_eq!((one.width(), one.height(), one.area()), (1, 1, 1));
        assert!(Rectangle::default().is_empty());

        // from_width_height
        assert_eq!(
            Rectangle::from_width_height(5, 3),
            Rectangle::new(0, 0, 4, 2)
        );
        assert_eq!(Rectangle::from_width_height(0, 0), Rectangle::default());
    }

    #[test]
    fn test_rectangle_from_dpoints_rounding() {
        // each dpoint rounds to a point (floor(x+0.5)), then union of 1x1s
        let r = Rectangle::from_dpoints(Dpoint::new(0.6, -0.4), Dpoint::new(3.2, 2.6));
        // (0.6,-0.4) -> (1, 0); (3.2, 2.6) -> (3, 3)
        assert_eq!(r, Rectangle::new(1, 0, 3, 3));
        assert_eq!(r.width(), 3);
        assert_eq!(r.height(), 4);

        // point-based constructor
        assert_eq!(
            Rectangle::from_points(Point::new(5, 6), Point::new(1, 2)),
            Rectangle::new(1, 2, 5, 6)
        );
    }

    #[test]
    fn test_rectangle_center() {
        // center(): (l+r+1, t+b+1)/2 with the negative-coordinate adjustment
        assert_eq!(center(&Rectangle::new(0, 0, 9, 9)), Point::new(5, 5));
        assert_eq!(
            center(&Rectangle::new(-10, -10, -1, -1)),
            Point::new(-5, -5)
        );
        // (-10 + -1 + 1) = -10 -> subtract 1 -> -11; -11/2 = -5 (trunc)
        assert_eq!(center(&Rectangle::new(-10, 0, -1, 0)), Point::new(-5, 0));
        assert_eq!(dcenter(&Rectangle::new(0, 0, 9, 9)), Dpoint::new(4.5, 4.5));
    }

    #[test]
    fn test_rectangle_free_functions() {
        let r = Rectangle::new(0, 0, 9, 9);
        assert_eq!(
            centered_rect(5, 5, 4, 4),
            Rectangle::new(5 - 2, 5 - 2, 5 - 2 + 3, 5 - 2 + 3)
        );
        assert_eq!(shrink_rect(&r, 1), Rectangle::new(1, 1, 8, 8));
        assert_eq!(grow_rect(&r, 2), Rectangle::new(-2, -2, 11, 11));
        assert_eq!(
            translate_rect(&r, &Point::new(1, 2)),
            Rectangle::new(1, 2, 10, 11)
        );
        assert_eq!(
            move_rect(&r, &Point::new(10, 10)),
            Rectangle::new(10, 10, 19, 19)
        );
        assert_eq!(resize_rect(&r, 3, 2), Rectangle::new(0, 0, 2, 1));
        assert_eq!(scale_rect(&r, 2.0), Rectangle::new(0, 0, 18, 18));
        assert_eq!(set_rect_area(&r, 100).area(), 100);
        // 10x10 -> ratio 1 keeps area ~100
        assert_eq!(
            set_aspect_ratio(&r, 2.0).width() as f64 / set_aspect_ratio(&r, 2.0).height() as f64,
            2.0
        );
        assert_eq!(
            nearest_point(&r, &Dpoint::new(-5.0, 50.0)),
            Dpoint::new(0.0, 9.0)
        );
        assert_eq!(distance_to_rect_edge(&r, &Point::new(20, 5)), 11);
        assert_eq!(distance_to_rect_edge(&r, &Point::new(20, 20)), 22);
        assert_eq!(distance_to_rect_edge(&r, &Point::new(2, 2)), 2);
    }

    #[test]
    fn test_drectangle() {
        let r = Drectangle::new(0.0, 0.0, 9.0, 9.0);
        assert_eq!(r.width(), 10.0); // +1 convention like rectangle
        assert_eq!(r.height(), 10.0);
        assert_eq!(r.area(), 100.0);
        assert_eq!(center_d(&r), Dpoint::new(4.5, 4.5));

        // scaling about the center: operator*(drectangle, double)
        let s = r * 2.0;
        assert_eq!(s, Drectangle::new(-4.5, -4.5, 13.5, 13.5));
        assert_eq!(s.width(), 19.0);
        let s2 = s / 2.0;
        assert!(close(s2.left(), r.left(), 1e-12) && close(s2.right(), r.right(), 1e-12));

        // conversion to rectangle rounds each edge (lround semantics)
        let dr = Drectangle::new(0.4, 0.5, 9.4, 9.6);
        let rr = Rectangle::from(dr);
        assert_eq!(rr, Rectangle::new(0, 1, 9, 10)); // 0.5 rounds away from zero -> 1
        let back = Drectangle::from(rr);
        assert_eq!(back, Drectangle::new(0.0, 1.0, 9.0, 10.0));

        assert_eq!(
            intersect_d(
                &Drectangle::new(0.0, 0.0, 5.0, 5.0),
                &Drectangle::new(4.0, 4.0, 9.0, 9.0)
            ),
            Drectangle::new(4.0, 4.0, 5.0, 5.0)
        );
        assert_eq!(
            centered_drect(&Dpoint::new(0.0, 0.0), 4.0, 4.0),
            Drectangle::new(-1.5, -1.5, 1.5, 1.5)
        );
    }

    #[test]
    fn test_rotators_and_transforms() {
        let rot = PointRotator::from_angle(std::f64::consts::FRAC_PI_2);
        let p = rot.apply(&Dpoint::new(1.0, 0.0));
        assert!(close(p.x(), 0.0, 1e-15) && close(p.y(), 1.0, 1e-15));

        let t = PointTransform::new(std::f64::consts::FRAC_PI_2, Dpoint::new(1.0, 1.0));
        let q = t.apply(&Dpoint::new(1.0, 0.0));
        assert!(close(q.x(), 1.0, 1e-15) && close(q.y(), 2.0, 1e-15));

        let affine = PointTransformAffine::new([[2.0, 0.0], [0.0, 3.0]], Dpoint::new(1.0, -1.0));
        assert_eq!(affine.apply(&Dpoint::new(1.0, 1.0)), Dpoint::new(3.0, 2.0));

        // composition: (lhs*rhs)(p) == lhs(rhs(p))
        let rhs = PointTransformAffine::new([[1.0, 1.0], [0.0, 1.0]], Dpoint::new(1.0, 0.0));
        let lhs = PointTransformAffine::new([[2.0, 0.0], [0.0, 2.0]], Dpoint::new(0.0, 5.0));
        let comp = lhs * rhs;
        let p = Dpoint::new(3.0, -2.0);
        let expected = lhs.apply(&rhs.apply(&p));
        assert!(close(comp.apply(&p).x(), expected.x(), 1e-12));
        assert!(close(comp.apply(&p).y(), expected.y(), 1e-12));

        // inverse
        let inv = inv_affine(&affine);
        let roundtrip = inv.apply(&affine.apply(&p));
        assert!(close(roundtrip.x(), p.x(), 1e-12) && close(roundtrip.y(), p.y(), 1e-12));

        // projective identity and from_affine
        let proj = PointTransformProjective::from_affine(&affine);
        assert_eq!(proj.apply(&p), affine.apply(&p));
        let ip = inv_projective(&proj);
        let rt = ip.apply(&proj.apply(&p));
        assert!(close(rt.x(), p.x(), 1e-12) && close(rt.y(), p.y(), 1e-12));

        // rotate_point
        let rp = rotate_point(
            &Dpoint::new(1.0, 1.0),
            &Dpoint::new(2.0, 1.0),
            std::f64::consts::PI,
        );
        assert!(close(rp.x(), 0.0, 1e-12) && close(rp.y(), 1.0, 1e-12));

        // rotation_matrix
        let m = rotation_matrix(std::f64::consts::FRAC_PI_2);
        assert!(close(m[0][0], 0.0, 1e-15) && close(m[0][1], -1.0, 1e-15));
    }

    #[test]
    fn test_rectangle_transform() {
        let t = RectangleTransform::new(PointTransformAffine::new(
            rotation_matrix(std::f64::consts::FRAC_PI_4),
            Dpoint::new(0.0, 0.0),
        ));
        let r = Rectangle::new(0, 0, 9, 9);
        let out = t.apply_rect(&r);
        // a rotated square keeps its area (up to rounding)
        assert!((out.area() as i64 - 100).abs() <= 8, "area: {}", out.area());
        let d = t.apply(&Drectangle::new(0.0, 0.0, 9.0, 9.0));
        // a rotation is area-preserving: the corrected box keeps area ~100
        assert!(close(d.area(), 100.0, 1e-9), "area: {}", d.area());
    }

    #[test]
    fn test_find_affine_transform_exact_on_three_points() {
        let from = vec![
            Dpoint::new(0.0, 0.0),
            Dpoint::new(1.0, 0.0),
            Dpoint::new(0.0, 1.0),
        ];
        let to = vec![
            Dpoint::new(2.0, 3.0),
            Dpoint::new(4.0, 3.5),
            Dpoint::new(1.5, 6.0),
        ];
        let t = find_affine_transform(&from, &to);
        for i in 0..3 {
            let got = t.apply(&from[i]);
            assert!(
                close(got.x(), to[i].x(), 1e-12) && close(got.y(), to[i].y(), 1e-12),
                "point {}: got ({}, {}), want ({}, {})",
                i,
                got.x(),
                got.y(),
                to[i].x(),
                to[i].y()
            );
        }

        // overdetermined least squares: identity data recovers identity
        let from4: Vec<Dpoint> = vec![
            Dpoint::new(0.0, 0.0),
            Dpoint::new(1.0, 0.0),
            Dpoint::new(0.0, 1.0),
            Dpoint::new(1.0, 1.0),
        ];
        let t = find_affine_transform(&from4, &from4);
        for p in &from4 {
            let got = t.apply(p);
            assert!(close(got.x(), p.x(), 1e-10) && close(got.y(), p.y(), 1e-10));
        }
    }

    #[test]
    fn test_find_similarity_transform() {
        // build a known similarity: rotate 30 degrees, scale 2, translate
        let theta = std::f64::consts::PI / 6.0;
        let m = rotation_matrix(theta);
        let c = 2.0;
        let b = Dpoint::new(3.0, -7.0);
        let from: Vec<Dpoint> = vec![
            Dpoint::new(0.0, 0.0),
            Dpoint::new(1.0, 0.0),
            Dpoint::new(0.0, 1.0),
            Dpoint::new(2.5, -1.25),
            Dpoint::new(-3.0, 4.0),
        ];
        let to: Vec<Dpoint> = from
            .iter()
            .map(|p| {
                Dpoint::new(
                    (m[0][0] * p.x() + m[0][1] * p.y()) * c + b.x(),
                    (m[1][0] * p.x() + m[1][1] * p.y()) * c + b.y(),
                )
            })
            .collect();
        let t = find_similarity_transform(&from, &to);
        for i in 0..from.len() {
            let got = t.apply(&from[i]);
            assert!(
                close(got.x(), to[i].x(), 1e-9) && close(got.y(), to[i].y(), 1e-9),
                "point {}: got ({}, {}), want ({}, {})",
                i,
                got.x(),
                got.y(),
                to[i].x(),
                to[i].y()
            );
        }
    }

    #[test]
    fn test_find_projective_transform() {
        // exact homography on 4 point pairs
        let h = [[1.5f64, 0.2, 3.0], [-0.1, 1.3, -2.0], [0.001, -0.002, 1.0]];
        let ttrue = PointTransformProjective::new(h);
        let from: Vec<Dpoint> = vec![
            Dpoint::new(0.0, 0.0),
            Dpoint::new(10.0, 0.0),
            Dpoint::new(0.0, 10.0),
            Dpoint::new(10.0, 10.0),
        ];
        let to: Vec<Dpoint> = from.iter().map(|p| ttrue.apply(p)).collect();
        let t = find_projective_transform(&from, &to);
        for i in 0..from.len() {
            let got = t.apply(&from[i]);
            assert!(
                close(got.x(), to[i].x(), 1e-6) && close(got.y(), to[i].y(), 1e-6),
                "point {}: got ({}, {}), want ({}, {})",
                i,
                got.x(),
                got.y(),
                to[i].x(),
                to[i].y()
            );
        }
    }

    #[test]
    fn test_line() {
        let l = Line::new(Dpoint::new(0.0, 0.0), Dpoint::new(2.0, 0.0));
        // normal is perpendicular to the line direction
        assert!(close(l.normal().dot(&(Dpoint::new(2.0, 0.0))), 0.0, 1e-12));
        // signed distance of a point above the x axis line
        let d = l.signed_distance(&Dpoint::new(0.0, 3.0));
        assert!(d.abs() == 3.0);
        assert_eq!(l.distance(&Dpoint::new(5.0, -4.0)), 4.0);

        // intersection of two non-parallel lines
        let a = Line::new(Dpoint::new(0.0, 0.0), Dpoint::new(4.0, 4.0));
        let b = Line::new(Dpoint::new(0.0, 4.0), Dpoint::new(4.0, 0.0));
        let p = intersect_lines(&a, &b);
        assert!(close(p.x(), 2.0, 1e-12) && close(p.y(), 2.0, 1e-12));
        assert_eq!(intersect_lines_checked(&a, &b), Some(p));

        // parallel lines -> (inf, inf), per dlib
        let c = Line::new(Dpoint::new(0.0, 1.0), Dpoint::new(4.0, 5.0));
        let q = intersect_lines(&a, &c);
        assert!(q.x().is_infinite() && q.y().is_infinite());
        assert_eq!(intersect_lines_checked(&a, &c), None);

        // angle_between_lines: perpendicular lines are 90 degrees
        assert!(close(angle_between_lines(&a, &b), 90.0, 1e-10));
        // same-direction lines are 0 degrees
        assert!(close(
            angle_between_lines(&a, &Line::new(Dpoint::new(1.0, 1.0), Dpoint::new(9.0, 9.0))),
            0.0,
            1e-5
        ));

        // count_points_on_side_of_line / between lines
        let pts = vec![
            Dpoint::new(0.0, 1.0),
            Dpoint::new(0.0, 2.0),
            Dpoint::new(0.0, 5.0),
            Dpoint::new(0.0, -1.0),
        ];
        let xaxis = Line::new(Dpoint::new(-10.0, 0.0), Dpoint::new(10.0, 0.0));
        assert_eq!(
            count_points_on_side_of_line(&xaxis, &Dpoint::new(0.0, 1.0), &pts, 0.0, 3.0),
            2
        );
        let l1 = Line::new(Dpoint::new(-10.0, 1.0), Dpoint::new(10.0, 1.0));
        let l2 = Line::new(Dpoint::new(-10.0, 4.0), Dpoint::new(10.0, 4.0));
        // only y=2 lies strictly between the lines (y=1 sits on l1 itself)
        assert_eq!(
            count_points_between_lines(&l1, &l2, &Dpoint::new(0.0, 2.0), &pts),
            1
        );
    }

    #[test]
    fn test_find_convex_quadrilateral() {
        // four lines forming a square
        let lines = [
            Line::new(Dpoint::new(0.0, 0.0), Dpoint::new(10.0, 0.0)), // bottom
            Line::new(Dpoint::new(10.0, 0.0), Dpoint::new(10.0, 10.0)), // right
            Line::new(Dpoint::new(10.0, 10.0), Dpoint::new(0.0, 10.0)), // top
            Line::new(Dpoint::new(0.0, 10.0), Dpoint::new(0.0, 0.0)), // left
        ];
        let quad = find_convex_quadrilateral(&lines).expect("square lines form a quadrilateral");
        let mut xs: Vec<f64> = quad.iter().map(|p| p.x()).collect();
        let mut ys: Vec<f64> = quad.iter().map(|p| p.y()).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(close(xs[0], 0.0, 1e-9) && close(xs[3], 10.0, 1e-9));
        assert!(close(ys[0], 0.0, 1e-9) && close(ys[3], 10.0, 1e-9));

        // four parallel lines form nothing
        let par = [
            Line::new(Dpoint::new(0.0, 0.0), Dpoint::new(1.0, 0.0)),
            Line::new(Dpoint::new(0.0, 1.0), Dpoint::new(1.0, 1.0)),
            Line::new(Dpoint::new(0.0, 2.0), Dpoint::new(1.0, 2.0)),
            Line::new(Dpoint::new(0.0, 3.0), Dpoint::new(1.0, 3.0)),
        ];
        assert!(find_convex_quadrilateral(&par).is_err());
    }

    #[test]
    fn test_polygon_area_sign_convention() {
        // counter-clockwise square: positive shoelace sum, result is abs/2
        let ccw = Polygon::new(vec![
            Point::new(0, 0),
            Point::new(4, 0),
            Point::new(4, 4),
            Point::new(0, 4),
        ]);
        assert_eq!(ccw.area(), 16.0);
        // clockwise square: negative raw shoelace sum, same (absolute) area
        let cw = Polygon::new(vec![
            Point::new(0, 0),
            Point::new(0, 4),
            Point::new(4, 4),
            Point::new(4, 0),
        ]);
        assert_eq!(cw.area(), 16.0);
        // triangles
        let tri = Polygon::new(vec![Point::new(0, 0), Point::new(10, 0), Point::new(0, 10)]);
        assert_eq!(tri.area(), 50.0);
        // degenerate
        assert_eq!(
            Polygon::new(vec![Point::new(1, 1), Point::new(2, 2)]).area(),
            0.0
        );

        // free polygon_area on dpoints
        let pts = vec![
            Dpoint::new(0.0, 0.0),
            Dpoint::new(4.0, 0.0),
            Dpoint::new(4.0, 4.0),
            Dpoint::new(0.0, 4.0),
        ];
        assert_eq!(polygon_area(&pts), 16.0);

        // get_rect / size / iteration
        assert_eq!(ccw.get_rect(), Rectangle::new(0, 0, 4, 4));
        assert_eq!(ccw.size(), 4);
        assert_eq!(ccw.iter().count(), 4);
        assert_eq!(*ccw.point(1), Point::new(4, 0));

        // contains
        assert!(ccw.contains(&Point::new(2, 2)));
        assert!(!ccw.contains(&Point::new(10, 2)));
    }

    #[test]
    fn test_border_enumerator_3x5() {
        // rect covering columns 0..4 and rows 0..2 (a "3x5" rect), border 1
        let rect = Rectangle::new(0, 0, 4, 2); // width 5, height 3
        let mut e = BorderEnumerator::new(rect, 1);
        assert_eq!(e.size(), rect.area() - Rectangle::new(1, 1, 3, 1).area());

        let mut pts = Vec::new();
        assert!(!e.current_element_valid());
        while e.move_next() {
            assert!(e.current_element_valid());
            pts.push(e.element());
        }
        assert!(!e.current_element_valid());
        // perimeter of the 5x3 rect: 5*3 - 3*1 = 12 cells
        assert_eq!(pts.len(), 12);
        // all enumerated points lie on the border of rect
        for p in &pts {
            let on_border = p.x() == rect.left()
                || p.x() == rect.right()
                || p.y() == rect.top()
                || p.y() == rect.bottom();
            assert!(on_border, "point {:?} not on border", p);
            assert!(rect.contains(p));
        }
        // every border cell is enumerated exactly once
        let mut expected = Vec::new();
        for y in 0..3 {
            for x in 0..5 {
                if x == 0 || x == 4 || y == 0 || y == 2 {
                    expected.push(Point::new(x, y));
                }
            }
        }
        let mut sorted_pts = pts.clone();
        sorted_pts.sort();
        let mut sorted_expected = expected.clone();
        sorted_expected.sort();
        assert_eq!(sorted_pts, sorted_expected);

        // reset + at_start behavior
        e.reset();
        assert!(e.at_start());
        assert!(e.move_next());
        assert!(!e.at_start());
    }

    #[test]
    fn test_border_enumerator_non_border_region_ctor() {
        let rect = Rectangle::new(0, 0, 9, 9);
        let inner = Rectangle::new(2, 2, 7, 7);
        let mut e = BorderEnumerator::new_with_non_border_region(rect, &inner);
        assert_eq!(e.size(), 100 - 36);
        let mut count = 0;
        while e.move_next() {
            assert!(rect.contains(&e.element()));
            count += 1;
        }
        assert_eq!(count, 64);
    }

    #[test]
    fn test_serialize_roundtrips() {
        use crate::serialize::{Deserializer, Serializer};

        // Point / Dpoint
        let mut out = Serializer::new();
        let p = Point::new(-3, 7);
        p.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(Point::deserialize(&mut inp).unwrap(), p);

        let mut out = Serializer::new();
        let d = Dpoint::new(1.5, -2.25);
        d.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(Dpoint::deserialize(&mut inp).unwrap(), d);

        // Vector<f64,3>
        let mut out = Serializer::new();
        let v = Vector::<f64, 3>::new(1.0, 2.0, 3.0);
        v.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(Vector::<f64, 3>::deserialize(&mut inp).unwrap(), v);

        // Rectangle
        let mut out = Serializer::new();
        let r = Rectangle::new(-10, 20, 30, -40);
        r.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(Rectangle::deserialize(&mut inp).unwrap(), r);

        // Drectangle
        let mut out = Serializer::new();
        let dr = Drectangle::new(0.5, 1.5, 2.5, 3.5);
        dr.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(Drectangle::deserialize(&mut inp).unwrap(), dr);

        // PointTransformAffine / PointTransformProjective
        let mut out = Serializer::new();
        let t = PointTransformAffine::new([[1.0, 2.0], [3.0, 4.0]], Dpoint::new(5.0, 6.0));
        t.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(PointTransformAffine::deserialize(&mut inp).unwrap(), t);

        let mut out = Serializer::new();
        let pj = PointTransformProjective::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        pj.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(PointTransformProjective::deserialize(&mut inp).unwrap(), pj);

        // PointRotator / PointTransform
        let mut out = Serializer::new();
        let rot = PointRotator::from_angle(0.25);
        rot.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(PointRotator::deserialize(&mut inp).unwrap(), rot);

        let mut out = Serializer::new();
        let tr = PointTransform::new(0.5, Dpoint::new(1.0, 2.0));
        tr.serialize(&mut out);
        let mut inp = Deserializer::new(out.as_bytes());
        assert_eq!(PointTransform::deserialize(&mut inp).unwrap(), tr);
    }
}

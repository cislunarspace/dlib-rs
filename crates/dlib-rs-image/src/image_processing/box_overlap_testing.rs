//! Box overlap testing, ported from
//! `dlib/image_processing/box_overlap_testing.h`.

use dlib_rs_core::geometry::Rectangle;
use dlib_rs_core::serialize::{Deserializer, SerializeError, Serializer};

/// Port of `box_intersection_over_union(const rectangle&, const rectangle&)`
/// (dlib/image_processing/box_overlap_testing.h). Areas are computed as
/// doubles via `drectangle`.
pub fn box_intersection_over_union(a: &Rectangle, b: &Rectangle) -> f64 {
    let inner = a.intersect(b).area() as f64;
    if inner == 0.0 {
        return 0.0;
    }
    let outer = a.area() as f64 + b.area() as f64 - inner;
    inner / outer
}

/// Port of `box_percent_covered(const rectangle&, const rectangle&)`.
pub fn box_percent_covered(a: &Rectangle, b: &Rectangle) -> f64 {
    let inner = a.intersect(b).area() as f64;
    if inner == 0.0 {
        return 0.0;
    }
    (inner / a.area() as f64).max(inner / b.area() as f64)
}

/// Port of the `test_box_overlap` functor class
/// (dlib/image_processing/box_overlap_testing.h). Default thresholds are
/// `iou_thresh = 0.5` and `percent_covered_thresh = 1.0`.
#[derive(Clone, Copy, Debug)]
pub struct TestBoxOverlap {
    iou_thresh: f64,
    percent_covered_thresh: f64,
}

impl Default for TestBoxOverlap {
    fn default() -> Self {
        TestBoxOverlap {
            iou_thresh: 0.5,
            percent_covered_thresh: 1.0,
        }
    }
}

impl TestBoxOverlap {
    /// Port of `test_box_overlap()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `test_box_overlap(iou_thresh, percent_covered_thresh)`.
    pub fn new_with_params(iou_thresh: f64, percent_covered_thresh: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&iou_thresh) && (0.0..=1.0).contains(&percent_covered_thresh),
            "invalid test_box_overlap thresholds"
        );
        TestBoxOverlap {
            iou_thresh,
            percent_covered_thresh,
        }
    }

    /// Port of `test_box_overlap::operator()(a, b)`.
    pub fn overlaps(&self, a: &Rectangle, b: &Rectangle) -> bool {
        let inner = a.intersect(b).area() as f64;
        if inner == 0.0 {
            return false;
        }

        let outer = (*a + *b).area() as f64;
        inner / outer > self.iou_thresh
            || inner / a.area() as f64 > self.percent_covered_thresh
            || inner / b.area() as f64 > self.percent_covered_thresh
    }

    /// Port of `test_box_overlap::get_iou_thresh()`.
    pub fn get_iou_thresh(&self) -> f64 {
        self.iou_thresh
    }

    /// Port of `test_box_overlap::get_percent_covered_thresh()`.
    pub fn get_percent_covered_thresh(&self) -> f64 {
        self.percent_covered_thresh
    }

    /// Port of `serialize(const test_box_overlap&, std::ostream&)`.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_f64(self.get_iou_thresh());
        out.write_f64(self.get_percent_covered_thresh());
    }

    /// Port of `deserialize(test_box_overlap&, std::istream&)`.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        let iou_thresh = inp.read_f64()?;
        let percent_covered_thresh = inp.read_f64()?;
        Ok(TestBoxOverlap {
            iou_thresh,
            percent_covered_thresh,
        })
    }
}

/// Port of `test_box_overlap()(a, b)` with the default thresholds
/// (IoU 0.5, percent-covered 1.0).
pub fn test_box_overlap(a: &Rectangle, b: &Rectangle) -> bool {
    TestBoxOverlap::new().overlaps(a, b)
}

/// Port of `test_box_overlap(iou_thresh, percent_covered_thresh)(a, b)`.
pub fn test_box_overlap_with_params(
    a: &Rectangle,
    b: &Rectangle,
    intersect_thresh: f64,
    covered_thresh: f64,
) -> bool {
    TestBoxOverlap::new_with_params(intersect_thresh, covered_thresh).overlaps(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(l: i64, t: i64, r: i64, b: i64) -> Rectangle {
        Rectangle::new(l, t, r, b)
    }

    #[test]
    fn disjoint_boxes_do_not_overlap() {
        let a = rect(0, 0, 9, 9);
        let b = rect(20, 20, 29, 29);
        assert!(!test_box_overlap(&a, &b));
        assert_eq!(box_intersection_over_union(&a, &b), 0.0);
        assert_eq!(box_percent_covered(&a, &b), 0.0);
    }

    #[test]
    fn identical_boxes_overlap() {
        let a = rect(0, 0, 9, 9);
        assert!(test_box_overlap(&a, &a));
        assert_eq!(box_intersection_over_union(&a, &a), 1.0);
        assert_eq!(box_percent_covered(&a, &a), 1.0);
    }

    #[test]
    fn touching_edges_do_not_overlap() {
        // Edge-adjacent boxes have an empty intersection (dlib rectangles
        // are inclusive on both ends), so inner == 0 -> no overlap.
        let a = rect(0, 0, 9, 9);
        let b = rect(10, 0, 19, 9);
        assert!(!test_box_overlap(&a, &b));
        assert!(!test_box_overlap(&b, &a));
    }

    #[test]
    fn half_overlap_triggers_default_iou() {
        // a: 10x10 = 100, b shifted by 5: intersection 25 (5x5), union 175.
        let a = rect(0, 0, 9, 9);
        let b = rect(5, 5, 14, 14);
        // iou = 25/175 < 0.5, pcov = 25/100 < 1.0 -> no overlap.
        assert!(!test_box_overlap(&a, &b));
        // With a small enough iou threshold it does overlap.
        assert!(test_box_overlap_with_params(&a, &b, 0.1, 1.0));
    }

    #[test]
    fn containment_triggers_percent_covered() {
        // b fully contains a; iou = 100/400 = 0.25 < 0.5, but pcov = 1.0.
        let a = rect(0, 0, 9, 9);
        let b = rect(0, 0, 19, 19);
        // Default percent_covered_thresh = 1.0 and inner/area == 1.0 is not
        // strictly greater, so the default tester says no.
        assert!(!test_box_overlap(&a, &b));
        // Lowering the covered threshold to 0.9 catches the containment.
        assert!(test_box_overlap_with_params(&a, &b, 0.5, 0.9));
        // Also the IoU branch with a lowered iou threshold.
        assert!(test_box_overlap_with_params(&a, &b, 0.2, 1.0));
    }

    #[test]
    fn exact_iou_boundary_is_strict() {
        // iou == 0.5 exactly must NOT count as overlap (dlib uses >).
        let a = rect(0, 0, 9, 9); // 100 px
        let b = rect(0, 0, 9, 19); // 200 px, intersection 100, union 200
        assert!((box_intersection_over_union(&a, &b) - 0.5).abs() < 1e-12);
        assert!(!test_box_overlap(&a, &b));
    }

    #[test]
    fn serialize_roundtrip() {
        let t = TestBoxOverlap::new_with_params(0.3, 0.8);
        let mut ser = Serializer::new();
        t.serialize(&mut ser);
        let mut de = Deserializer::new(ser.as_bytes());
        let t2 = TestBoxOverlap::deserialize(&mut de).unwrap();
        assert_eq!(t2.get_iou_thresh(), 0.3);
        assert_eq!(t2.get_percent_covered_thresh(), 0.8);
        assert_eq!(de.remaining(), 0);
    }
}

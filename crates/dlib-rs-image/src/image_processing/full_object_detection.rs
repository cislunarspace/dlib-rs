//! Port of `dlib/image_processing/full_object_detection.h`: the
//! `full_object_detection` type that couples an object's bounding box with its
//! landmark parts, plus the `OBJECT_PART_NOT_PRESENT` sentinel.

use dlib_rs_core::geometry::{Dpoint, Point, Rectangle};
use dlib_rs_core::serialize::{Deserializer, SerializeError, Serializer};

/// Port of `dlib::OBJECT_PART_NOT_PRESENT` (a const `dpoint` in
/// `full_object_detection.h`, spelled here as a function because Rust cannot
/// build a `const` `Dpoint` from another crate's non-const constructor).
pub fn object_part_not_present() -> Dpoint {
    Dpoint::new(0x7_FFFF_FFFF_FFFF_u64 as f64, 0x7_FFFF_FFFF_FFFF_u64 as f64)
}

/// Port of `class dlib::full_object_detection`
/// (`dlib/image_processing/full_object_detection.h`): a rectangle plus one
/// landmark point per part.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FullObjectDetection {
    /// Bounding box of the detected object (`rect`).
    pub rect: Rectangle,
    /// The object's landmark parts (`parts`), stored as `dpoint`s.
    pub parts: Vec<Dpoint>,
}

impl FullObjectDetection {
    /// Port of `full_object_detection(rect, parts)` taking `dpoint` parts.
    pub fn new(rect: Rectangle, parts: Vec<Dpoint>) -> Self {
        FullObjectDetection { rect, parts }
    }

    /// Port of `full_object_detection(rect, parts)` taking integer `point`
    /// parts (each is widened to a `dpoint`, exactly like the C++ iterator
    /// conversion).
    pub fn from_points(rect: Rectangle, parts: Vec<Point>) -> Self {
        FullObjectDetection {
            rect,
            parts: parts
                .iter()
                .map(|p| Dpoint::new(p.x() as f64, p.y() as f64))
                .collect(),
        }
    }

    /// Port of the explicit `full_object_detection(rect)` constructor.
    pub fn from_rect(rect: Rectangle) -> Self {
        FullObjectDetection {
            rect,
            parts: Vec::new(),
        }
    }

    /// Port of `get_rect()`.
    pub fn get_rect(&self) -> &Rectangle {
        &self.rect
    }

    /// Port of `num_parts()`.
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Port of `part(idx)` (the read accessor; the index is bounds-checked
    /// like `DLIB_ASSERT`).
    pub fn part(&self, idx: usize) -> &Dpoint {
        assert!(
            idx < self.num_parts(),
            "full_object_detection::part(): idx {} out of range (num_parts {})",
            idx,
            self.num_parts()
        );
        &self.parts[idx]
    }

    /// Port of `part(idx)` (the mutable accessor).
    pub fn part_mut(&mut self, idx: usize) -> &mut Dpoint {
        assert!(
            idx < self.num_parts(),
            "full_object_detection::part(): idx {} out of range (num_parts {})",
            idx,
            self.num_parts()
        );
        &mut self.parts[idx]
    }

    /// Port of `serialize(const full_object_detection&, std::ostream&)`:
    /// version int 2, the rectangle, then the parts as
    /// `std::vector<vector<double,2>>`.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_i32(2);
        self.rect.serialize(out);
        out.write_u64(self.parts.len() as u64);
        for p in &self.parts {
            p.serialize(out);
        }
    }

    /// Port of `deserialize(full_object_detection&, std::istream&)`; accepts
    /// version 1 (legacy `vector<point>` parts) and version 2
    /// (`vector<dpoint>` parts).
    pub fn deserialize(inp: &mut Deserializer<'_>) -> Result<Self, SerializeError> {
        let version = inp.read_i32()?;
        if version != 1 && version != 2 {
            return Err(SerializeError::Malformed(
                "Unexpected version encountered while deserializing dlib::full_object_detection.",
            ));
        }
        let rect = Rectangle::deserialize(inp)?;
        let len = read_count(inp)?;
        let mut parts = Vec::with_capacity(len);
        for _ in 0..len {
            if version == 1 {
                // Legacy support: read vector<point, 2> and cast to dpoint.
                let p = Point::deserialize(inp)?;
                parts.push(Dpoint::new(p.x() as f64, p.y() as f64));
            } else {
                parts.push(Dpoint::deserialize(inp)?);
            }
        }
        Ok(FullObjectDetection { rect, parts })
    }
}

/// Reads a `std::vector` element count, rejecting absurd lengths early so a
/// corrupt stream cannot request a huge allocation (each serialized element
/// consumes at least one input byte).
fn read_count(inp: &mut Deserializer<'_>) -> Result<usize, SerializeError> {
    let len = inp.read_u64()?;
    if len > inp.remaining() as u64 {
        return Err(SerializeError::Malformed(
            "vector length exceeds remaining input",
        ));
    }
    Ok(len as usize)
}

/// Port of `all_parts_in_rect(obj)` (`full_object_detection.h`).
pub fn all_parts_in_rect(obj: &FullObjectDetection) -> bool {
    for i in 0..obj.num_parts() {
        let p = obj.part(i);
        let contained = obj.rect.contains(&Point::new(p.x() as i64, p.y() as i64));
        if !contained && *p != object_part_not_present() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_v2() {
        let d = FullObjectDetection::new(
            Rectangle::new(1, 2, 30, 40),
            vec![Dpoint::new(3.5, 4.5), Dpoint::new(-1.0, 12.0)],
        );
        let mut ser = Serializer::new();
        d.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let d2 = FullObjectDetection::deserialize(&mut de).unwrap();
        assert_eq!(d, d2);
        assert_eq!(de.remaining(), 0);
    }

    #[test]
    fn test_legacy_v1_points() {
        // version 1: rect + vector<point> (i64 x/y pairs)
        let mut ser = Serializer::new();
        ser.write_i32(1);
        Rectangle::new(0, 0, 9, 9).serialize(&mut ser);
        ser.write_u64(2);
        Point::new(1, 2).serialize(&mut ser);
        Point::new(3, 4).serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let d = FullObjectDetection::deserialize(&mut de).unwrap();
        assert_eq!(d.num_parts(), 2);
        assert_eq!(*d.part(0), Dpoint::new(1.0, 2.0));
        assert_eq!(*d.part(1), Dpoint::new(3.0, 4.0));
    }

    #[test]
    fn test_bad_version_rejected() {
        let mut ser = Serializer::new();
        ser.write_i32(7);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        assert!(FullObjectDetection::deserialize(&mut de).is_err());
    }

    #[test]
    fn test_sentinel_and_all_parts_in_rect() {
        assert_eq!(object_part_not_present().x(), 0x7_FFFF_FFFF_FFFFu64 as f64);
        let d = FullObjectDetection::new(
            Rectangle::new(0, 0, 9, 9),
            vec![Dpoint::new(5.0, 5.0), object_part_not_present()],
        );
        assert!(all_parts_in_rect(&d));
        let bad =
            FullObjectDetection::new(Rectangle::new(0, 0, 9, 9), vec![Dpoint::new(50.0, 5.0)]);
        assert!(!all_parts_in_rect(&bad));
    }
}

//! Frontal face detector, ported from
//! `dlib/image_processing/frontal_face_detector.h`: the embedded serialized
//! `object_detector<scan_fhog_pyramid<pyramid_down<6>>>` model blob
//! (`get_serialized_frontal_faces()`), deserialized on demand with the
//! byte-exact dlib serialization format.

use dlib_rs_core::geometry::Rectangle;
use dlib_rs_core::serialize::{Deserializer, SerializeError};

use crate::array2d::GenericImage;
use crate::pixel::Pixel;

use super::object_detector::ObjectDetector;

/// The exact byte stream of dlib's `get_serialized_frontal_faces()`
/// (frontal_face_detector.h).
pub fn frontal_face_detector_bytes() -> &'static [u8] {
    include_bytes!("frontal_faces.dat")
}

/// Alias matching dlib's `frontal_face_detector` typedef.
pub type FrontalFaceDetector = ObjectDetector;

/// Port of `frontal_face_detector()`: deserializes the embedded model blob
/// into an [`ObjectDetector`].
pub fn frontal_face_detector() -> Result<ObjectDetector, SerializeError> {
    let mut de = Deserializer::new(frontal_face_detector_bytes());
    ObjectDetector::deserialize(&mut de)
}

/// Convenience wrapper: `detector(img)` in dlib, i.e. `run(img, 0.0)`.
pub fn detect_faces<S: GenericImage>(det: &mut ObjectDetector, img: &S) -> Vec<Rectangle>
where
    S::PixelType: Pixel,
{
    det.run(img, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn frontal_face_detector_loads() {
        let det = frontal_face_detector().unwrap();
        assert_eq!(det.num_detectors(), 5);
        assert_eq!(det.get_scanner().get_cell_size(), 8);
    }

    #[test]
    fn detect_faces_runs() {
        let mut det = frontal_face_detector().unwrap();
        let mut img: Array2D<u8> = Array2D::zeros(120, 120);
        for r in 0..120 {
            for c in 0..120 {
                *img.pixel_mut(r, c) = ((r + c) % 256) as u8;
            }
        }
        let dets = detect_faces(&mut det, &img);
        for d in &dets {
            assert!(d.width() > 0 && d.height() > 0);
        }
    }
}

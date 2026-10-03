//! Object detector, ported from `dlib/image_processing/object_detector.h`
//! specialized to the FHOG pyramid scanner (`scan_fhog_pyramid<pyramid_down<6>>`).
//!
//! The stored form is dlib's serialization version 2: scanner, overlap
//! tester, packed detector count, then each weight vector (a plain
//! `matrix<double,0,1>` of `num_dimensions + 1` elements — a *linear*
//! decision function whose bias is the last element, not a kernelized
//! `decision_function`).

use dlib_rs_core::geometry::Rectangle;
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, SerializeError};

use crate::array2d::GenericImage;
use crate::pixel::Pixel;

use super::box_overlap_testing::TestBoxOverlap;
use super::scan_fhog_pyramid::{FhogFilterbank, ScanFhogPyramid};

/// Port of `dlib::rect_detection` (object_detector.h).
#[derive(Clone, Debug)]
pub struct RectDetection {
    pub detection_confidence: f64,
    pub weight_index: usize,
    pub rect: Rectangle,
}

/// Port of `dlib::object_detector<scan_fhog_pyramid<pyramid_down<6>>>`.
#[derive(Clone)]
pub struct ObjectDetector {
    boxes_overlap: TestBoxOverlap,
    /// The raw weight vectors (processed_weight_vector::w), each of length
    /// `scanner.get_num_dimensions() + 1` with the bias/threshold last.
    w: Vec<Matrix<f64>>,
    /// The processed weight vectors (processed_weight_vector::fb), built
    /// from `w` at deserialization time.
    fbs: Vec<FhogFilterbank>,
    scanner: ScanFhogPyramid,
}

impl ObjectDetector {
    /// Port of `deserialize(object_detector<scanner>&, std::istream&)`.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        let version = inp.read_i32()?;
        let mut scanner = ScanFhogPyramid::new();
        let boxes_overlap;
        let w;
        match version {
            1 => {
                scanner.load(inp)?;
                w = vec![Matrix::<f64>::deserialize(inp)?];
                boxes_overlap = TestBoxOverlap::deserialize(inp)?;
            }
            2 => {
                scanner.load(inp)?;
                boxes_overlap = TestBoxOverlap::deserialize(inp)?;
                let num_detectors = inp.read_u64()? as usize;
                if num_detectors > inp.remaining() {
                    return Err(SerializeError::Eof);
                }
                let mut v = Vec::with_capacity(num_detectors);
                for _ in 0..num_detectors {
                    v.push(Matrix::<f64>::deserialize(inp)?);
                }
                w = v;
            }
            _ => {
                return Err(SerializeError::Malformed(
                    "Unexpected version encountered while deserializing a dlib::object_detector object.",
                ))
            }
        }

        let fbs: Vec<FhogFilterbank> = w
            .iter()
            .map(|wv| scanner.build_fhog_filterbank(wv))
            .collect();

        Ok(ObjectDetector {
            boxes_overlap,
            w,
            fbs,
            scanner,
        })
    }

    /// Port of `object_detector::num_detectors()`.
    pub fn num_detectors(&self) -> usize {
        self.w.len()
    }

    /// Port of `object_detector::get_w(idx)`.
    pub fn get_w(&self, idx: usize) -> &Matrix<f64> {
        &self.w[idx]
    }

    /// The per-detector decision thresholds, i.e. `w[i](num_dimensions)`
    /// (the bias element of each linear weight vector).
    pub fn filter_thresholds(&self) -> Vec<f64> {
        let dims = self.scanner.get_num_dimensions() as usize;
        self.w.iter().map(|wv| wv[(dims, 0)]).collect()
    }

    /// Port of `object_detector::get_overlap_tester()`.
    pub fn get_overlap_tester(&self) -> &TestBoxOverlap {
        &self.boxes_overlap
    }

    /// Port of `object_detector::get_scanner()`.
    pub fn get_scanner(&self) -> &ScanFhogPyramid {
        &self.scanner
    }

    /// Port of `object_detector::operator()(img, final_dets,
    /// adjust_threshold)` for `rect_detection` outputs: run every weight
    /// vector over the scanner's detections and apply non-max suppression
    /// (keep the highest confidence detection of each overlapping group,
    /// sorted by descending confidence when several weight vectors exist).
    pub fn run_rect_detections<S: GenericImage>(
        &mut self,
        img: &S,
        adjust_threshold: f64,
        final_dets: &mut Vec<RectDetection>,
    ) where
        S::PixelType: Pixel,
    {
        self.scanner.load_image(img);
        let dims = self.scanner.get_num_dimensions() as usize;

        let mut dets: Vec<(f64, Rectangle)> = Vec::new();
        let mut dets_accum: Vec<RectDetection> = Vec::new();
        for i in 0..self.w.len() {
            let thresh = self.w[i][(dims, 0)];
            self.scanner
                .detect(&self.fbs[i], &mut dets, thresh + adjust_threshold);
            for &(score, rect) in &dets {
                dets_accum.push(RectDetection {
                    detection_confidence: score - thresh,
                    weight_index: i,
                    rect,
                });
            }
        }

        // Do non-max suppression
        final_dets.clear();
        if self.w.len() > 1 {
            dets_accum.sort_by(|a, b| {
                b.detection_confidence
                    .partial_cmp(&a.detection_confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for det in dets_accum {
            if self.overlaps_any_box(final_dets, &det.rect) {
                continue;
            }
            final_dets.push(det);
        }
    }

    /// Port of `object_detector::operator()(img, adjust_threshold)`:
    /// returns the final detection rectangles.
    pub fn run<S: GenericImage>(&mut self, img: &S, adjust_threshold: f64) -> Vec<Rectangle>
    where
        S::PixelType: Pixel,
    {
        let mut dets = Vec::new();
        self.run_rect_detections(img, adjust_threshold, &mut dets);
        dets.into_iter().map(|d| d.rect).collect()
    }

    /// Port of `object_detector::overlaps_any_box` (no weight-index filter;
    /// the object_detector's own suppression suppresses across all weight
    /// vectors).
    fn overlaps_any_box(&self, rects: &[RectDetection], rect: &Rectangle) -> bool {
        rects
            .iter()
            .any(|d| self.boxes_overlap.overlaps(&d.rect, rect))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn deserialize_embedded_frontal_face_model() {
        let mut de = Deserializer::new(
            crate::image_processing::frontal_face_detector::frontal_face_detector_bytes(),
        );
        let det = ObjectDetector::deserialize(&mut de).unwrap();

        // The frontal face detector is 5 linear detectors over the 3100-dim
        // (10x10 cells x 31 channels) fhog feature space.
        assert_eq!(det.num_detectors(), 5);
        let dims = det.get_scanner().get_num_dimensions() as usize;
        assert_eq!(dims, 3100);
        assert!(!det.get_scanner().is_loaded_with_image());
        for i in 0..det.num_detectors() {
            assert_eq!(det.get_w(i).size(), dims + 1);
        }
        let thresholds = det.filter_thresholds();
        assert_eq!(thresholds.len(), 5);
        assert!(thresholds.iter().all(|t| t.is_finite() && *t != 0.0));
        // All weights are consumed by the deserializer.
        assert!(de.remaining() < 8);
    }

    #[test]
    fn run_on_synthetic_image_smoke() {
        let mut de = Deserializer::new(
            crate::image_processing::frontal_face_detector::frontal_face_detector_bytes(),
        );
        let mut det = ObjectDetector::deserialize(&mut de).unwrap();

        let mut img: Array2D<u8> = Array2D::zeros(100, 100);
        for r in 0..100 {
            for c in 0..100 {
                *img.pixel_mut(r, c) = ((r * 2 + c * 7) % 256) as u8;
            }
        }
        // No face in a diagonal gradient: this exercises the full pipeline
        // (pyramid, fhog, separable filtering, NMS) without expecting hits.
        let dets = det.run(&img, 0.0);
        // A pure gradient typically yields no detections; the important part
        // is that the pipeline runs and produces rectangles if any.
        for d in &dets {
            assert!(d.width() > 0 && d.height() > 0);
        }
    }
}

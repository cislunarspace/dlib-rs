//! Face-detection pipeline port of `dlib/image_processing/*.h`:
//! `object_detector`, `scan_fhog_pyramid`, box-overlap testing, the embedded
//! `frontal_face_detector` model and `shape_predictor`.

pub mod box_overlap_testing;
pub mod frontal_face_detector;
pub mod full_object_detection;
pub mod object_detector;
pub mod scan_fhog_pyramid;
pub mod shape_predictor;
pub mod shape_predictor_trainer;

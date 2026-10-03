//! Keypoint detectors and descriptors, ported from dlib's
//! `image_keypoint/` headers (`surf.h`, `hessian_pyramid.h`) and
//! `image_transforms/fhog.h`.

pub mod fhog;
pub mod surf;

pub use fhog::{extract_fhog_features, HogImage};
pub use surf::{get_surf_points, integral_image, SurfPoint};

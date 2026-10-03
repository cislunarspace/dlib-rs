//! DNG saving: re-export of dlib's own DNG format implementation which lives
//! in `crate::image_loader::dng` (see `dlib/image_saver/image_saver.h`,
//! `save_dng` helpers and `dng_shared.h`).

pub use crate::image_loader::dng::{save_dng, save_dng_file};

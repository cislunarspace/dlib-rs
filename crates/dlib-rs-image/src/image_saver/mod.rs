//! Image saving (port of dlib `image_saver/`).
//!
//! - BMP: hand-ported from `dlib/image_saver/image_saver.h` (`save_bmp`).
//! - PNG: via the pure-Rust `png` crate (`save_png.h` shape).
//! - JPEG: via `jpeg-encoder` (`save_jpeg.h` shape, default quality 75).
//! - DNG: dlib's own format, re-exported from [`dng`].

pub mod bmp;
pub mod dng;
pub mod jpeg;
pub mod png;

pub use bmp::{save_bmp, save_bmp_file};
pub use dng::{save_dng, save_dng_file};
pub use jpeg::{save_jpeg, save_jpeg_file, save_jpeg_with_quality};
pub use png::{save_png, save_png_file};

/// Error type for image encoding failures (dlib `image_save_error`).
#[derive(thiserror::Error, Debug)]
pub enum ImageSaveError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode error: {0}")]
    Encode(String),
}

//! Image loading (port of dlib `image_loader/`, `image_saver/` codec APIs).
//!
//! Mirrors dlib's `load_image.h` dispatcher: the actual file type is
//! determined from the leading magic bytes, then the matching loader is
//! invoked.
//!
//! - BMP: hand-ported from `dlib/image_loader/image_loader.h` (`load_bmp`).
//! - PNG: via the pure-Rust `png` crate (`png_loader.h` shape).
//! - JPEG: via `zune-jpeg` (`jpeg_loader.h` shape).
//! - WebP: via `image-webp` (`webp_loader.h` shape).
//! - DNG: dlib's own format (`dng_shared.h`), ported in [`dng`].

use std::fs::File;
use std::io::Read;

use crate::array2d::Array2D;
use crate::pixel::Pixel;

pub mod bmp;
pub mod dng;
pub mod jpeg;
pub mod png;
pub mod webp;

pub use bmp::{load_bmp, load_bmp_file};
pub use dng::{load_dng, load_dng_file};
pub use jpeg::{load_jpeg, load_jpeg_file};
pub use png::{load_png, load_png_file};
pub use webp::{load_webp, load_webp_file};

/// Error type for image decoding failures (dlib `image_load_error`).
#[derive(thiserror::Error, Debug)]
pub enum ImageLoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported image format: {0}")]
    UnsupportedFormat(String),
    #[error("corrupt image data: {0}")]
    Corrupt(String),
}

/// File type detected from magic bytes (dlib `image_file_type::type`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageFileType {
    Bmp,
    Jpg,
    Png,
    Dng,
    Webp,
    Unknown,
}

/// Determine the image type from the leading bytes of a file
/// (port of dlib `image_file_type::read_type`, magic-bytes dispatch).
pub fn read_type(bytes: &[u8]) -> ImageFileType {
    if bytes.len() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff {
        ImageFileType::Jpg
    } else if bytes.len() >= 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a] {
        ImageFileType::Png
    } else if bytes.len() >= 2 && bytes[0] == b'B' && bytes[1] == b'M' {
        ImageFileType::Bmp
    } else if bytes.len() >= 3 && bytes[0] == b'D' && bytes[1] == b'N' && bytes[2] == b'G' {
        ImageFileType::Dng
    } else if bytes.len() >= 12
        && bytes[0] == b'R'
        && bytes[1] == b'I'
        && bytes[2] == b'F'
        && bytes[3] == b'F'
        && bytes[8] == b'W'
        && bytes[9] == b'E'
        && bytes[10] == b'B'
        && bytes[11] == b'P'
    {
        ImageFileType::Webp
    } else {
        ImageFileType::Unknown
    }
}

fn dispatch<P: Pixel>(bytes: &[u8]) -> Result<Array2D<P>, ImageLoadError> {
    match read_type(bytes) {
        ImageFileType::Bmp => load_bmp(&mut &bytes[..]),
        ImageFileType::Dng => {
            dng::load_dng(&mut &bytes[..]).map_err(|e| ImageLoadError::Corrupt(e.to_string()))
        }
        ImageFileType::Png => load_png(&mut &bytes[..]),
        ImageFileType::Jpg => load_jpeg(&mut &bytes[..]),
        ImageFileType::Webp => load_webp(&mut &bytes[..]),
        ImageFileType::Unknown => Err(ImageLoadError::UnsupportedFormat(
            "unknown image file format".to_string(),
        )),
    }
}

/// Load an image from a file, dispatching on the file's magic bytes
/// (port of dlib `load_image` from `load_image.h`).
pub fn load_image<P: Pixel>(path: &str) -> Result<Array2D<P>, ImageLoadError> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    load_image_from_memory(&bytes)
}

/// Load an image from an in-memory buffer, dispatching on magic bytes only
/// (memory counterpart of dlib `load_image`).
pub fn load_image_from_memory<P: Pixel>(bytes: &[u8]) -> Result<Array2D<P>, ImageLoadError> {
    dispatch(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{assign_pixel, RgbPixel};

    fn write_temp(name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("dlib_rs_load_image_test_{}", name));
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn test_read_type() {
        assert_eq!(read_type(b"BM\x00\x00"), ImageFileType::Bmp);
        assert_eq!(
            read_type(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
            ImageFileType::Png
        );
        assert_eq!(read_type(&[0xff, 0xd8, 0xff, 0xe0]), ImageFileType::Jpg);
        assert_eq!(read_type(b"DNG\x00"), ImageFileType::Dng);
        assert_eq!(
            read_type(b"RIFF\x00\x00\x00\x00WEBPVP8 "),
            ImageFileType::Webp
        );
        assert_eq!(read_type(b"hello world!"), ImageFileType::Unknown);
    }

    #[test]
    fn test_dispatch_bmp_and_png_via_files() {
        // 2x2 rgb image
        let mut img = Array2D::<RgbPixel>::zeros(2, 2);
        for i in 0..4 {
            let p = RgbPixel {
                r: (i * 10) as u8,
                g: (i * 20 + 3) as u8,
                b: (255 - i * 30) as u8,
            };
            *img.get_mut(i / 2, i % 2) = p;
        }
        let mut bmp_bytes = Vec::new();
        crate::image_saver::save_bmp(&img, &mut bmp_bytes).unwrap();
        let bmp_path = write_temp("a.bmp", &bmp_bytes);
        let loaded: Array2D<RgbPixel> =
            load_image(bmp_path.to_str().unwrap()).expect("bmp dispatch");
        assert_eq!(loaded.nr(), 2);
        assert_eq!(loaded.nc(), 2);
        for i in 0..4 {
            assert_eq!(*loaded.get(i / 2, i % 2), *img.get(i / 2, i % 2));
        }

        let mut png_bytes = Vec::new();
        crate::image_saver::save_png(&img, &mut png_bytes).unwrap();
        let png_path = write_temp("a.png", &png_bytes);
        let loaded: Array2D<RgbPixel> =
            load_image(png_path.to_str().unwrap()).expect("png dispatch");
        assert_eq!(loaded.nr(), 2);
        assert_eq!(loaded.nc(), 2);
        for i in 0..4 {
            assert_eq!(*loaded.get(i / 2, i % 2), *img.get(i / 2, i % 2));
        }
        std::fs::remove_file(&bmp_path).ok();
        std::fs::remove_file(&png_path).ok();
    }

    #[test]
    fn test_load_image_from_memory_unknown() {
        let err = load_image_from_memory::<RgbPixel>(b"not an image at all").unwrap_err();
        assert!(matches!(err, ImageLoadError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_load_bmp_into_gray_target() {
        // gray round trip through 8-bit palette BMP
        let mut img = Array2D::<u8>::zeros(3, 5);
        for r in 0..3 {
            for c in 0..5 {
                *img.get_mut(r, c) = (r * 50 + c * 7) as u8;
            }
        }
        let mut bytes = Vec::new();
        crate::image_saver::save_bmp(&img, &mut bytes).unwrap();
        let loaded = load_bmp::<u8, _>(&mut &bytes[..]).unwrap();
        assert_eq!(loaded.nr(), 3);
        assert_eq!(loaded.nc(), 5);
        for r in 0..3 {
            for c in 0..5 {
                assert_eq!(*loaded.get(r, c), *img.get(r, c));
            }
        }
        // also load into rgb target through assign_pixel
        let rgb: Array2D<RgbPixel> = load_bmp(&mut &bytes[..]).unwrap();
        let mut expect = RgbPixel::default();
        assign_pixel(&mut expect, img.get(1, 1));
        assert_eq!(*rgb.get(1, 1), expect);
    }
}

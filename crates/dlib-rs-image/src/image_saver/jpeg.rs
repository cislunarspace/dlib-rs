//! JPEG saving via the pure-Rust `jpeg-encoder` crate, mirroring the API
//! shape of `dlib/image_saver/save_jpeg.h` (`save_jpeg`, default quality 75).
//!
//! dlib semantics: grayscale-like images are encoded as grayscale JPEGs,
//! rgb_alpha as RGBA, and every other color pixel type is converted to rgb
//! via `assign_pixel` and encoded as an RGB JPEG.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::array2d::Array2D;
use crate::image_saver::ImageSaveError;
use crate::pixel::{assign_pixel, Pixel, RgbAlphaPixel, RgbPixel};

fn encode<P: Pixel, W: Write>(
    img: &Array2D<P>,
    out: &mut W,
    quality: u8,
) -> Result<(), ImageSaveError> {
    let nr = img.nr();
    let nc = img.nc();
    if nc > u16::MAX as usize || nr > u16::MAX as usize {
        return Err(ImageSaveError::Encode(
            "image too large for jpeg (max 65535x65535)".to_string(),
        ));
    }

    let (color_type, data): (jpeg_encoder::ColorType, Vec<u8>) = if P::is_rgb_alpha() {
        let mut data = Vec::with_capacity(nr * nc * 4);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = RgbAlphaPixel::default();
                assign_pixel(&mut p, img.get(r, c));
                data.extend_from_slice(&[p.r, p.g, p.b, p.a]);
            }
        }
        (jpeg_encoder::ColorType::Rgba, data)
    } else if P::is_gray() {
        let mut data = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = 0u8;
                assign_pixel(&mut p, img.get(r, c));
                data.push(p);
            }
        }
        (jpeg_encoder::ColorType::Luma, data)
    } else {
        let mut data = Vec::with_capacity(nr * nc * 3);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = RgbPixel::default();
                assign_pixel(&mut p, img.get(r, c));
                data.extend_from_slice(&[p.r, p.g, p.b]);
            }
        }
        (jpeg_encoder::ColorType::Rgb, data)
    };

    let enc = jpeg_encoder::Encoder::new(out, quality);
    enc.encode(&data, nc as u16, nr as u16, color_type)
        .map_err(|e| ImageSaveError::Encode(format!("jpeg encode error: {e}")))
}

/// Save `img` as a JPEG to a writer with the given quality (0-100)
/// (port of dlib `save_jpeg(image, filename, quality)`).
pub fn save_jpeg_with_quality<P: Pixel, W: Write>(
    img: &Array2D<P>,
    out: &mut W,
    quality: u8,
) -> Result<(), ImageSaveError> {
    encode(img, out, quality)
}

/// Save `img` as a JPEG to a writer with dlib's default quality of 75.
pub fn save_jpeg<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), ImageSaveError> {
    encode(img, out, 75)
}

/// Save `img` as a JPEG file with the given quality.
pub fn save_jpeg_file<P: Pixel>(
    img: &Array2D<P>,
    path: &str,
    quality: u8,
) -> Result<(), ImageSaveError> {
    let mut f = File::create(Path::new(path))?;
    encode(img, &mut f, quality)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_jpeg_magic() {
        let img = Array2D::<RgbPixel>::zeros(4, 4);
        let mut bytes = Vec::new();
        save_jpeg(&img, &mut bytes).unwrap();
        assert_eq!(&bytes[..3], &[0xff, 0xd8, 0xff]);
    }
}

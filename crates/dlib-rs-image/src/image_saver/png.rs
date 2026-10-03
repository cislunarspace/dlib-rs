//! PNG saving via the pure-Rust `png` crate, mirroring the API shape of
//! `dlib/image_saver/save_png.h` (`save_png`).
//!
//! dlib semantics: grayscale-like images are written as 8-bit grayscale PNGs,
//! rgb_alpha images as RGBA8, and every other color pixel type is converted
//! to rgb via `assign_pixel` and written as RGB8.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::array2d::Array2D;
use crate::image_saver::ImageSaveError;
use crate::pixel::{assign_pixel, Pixel, RgbAlphaPixel, RgbPixel};

/// Save `img` as a PNG to a writer (port of dlib `save_png`).
pub fn save_png<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), ImageSaveError> {
    let nr = img.nr();
    let nc = img.nc();

    let (color, data): (png::ColorType, Vec<u8>) = if P::is_rgb_alpha() {
        let mut data = Vec::with_capacity(nr * nc * 4);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = RgbAlphaPixel::default();
                assign_pixel(&mut p, img.get(r, c));
                data.extend_from_slice(&[p.r, p.g, p.b, p.a]);
            }
        }
        (png::ColorType::Rgba, data)
    } else if P::is_gray() {
        let mut data = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = 0u8;
                assign_pixel(&mut p, img.get(r, c));
                data.push(p);
            }
        }
        (png::ColorType::Grayscale, data)
    } else {
        // rgb-like (rgb, hsi, lab, ...)
        let mut data = Vec::with_capacity(nr * nc * 3);
        for r in 0..nr {
            for c in 0..nc {
                let mut p = RgbPixel::default();
                assign_pixel(&mut p, img.get(r, c));
                data.extend_from_slice(&[p.r, p.g, p.b]);
            }
        }
        (png::ColorType::Rgb, data)
    };

    let mut enc = png::Encoder::new(out, nc as u32, nr as u32);
    enc.set_color(color);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc
        .write_header()
        .map_err(|e| ImageSaveError::Encode(format!("png encode error: {e}")))?;
    writer
        .write_image_data(&data)
        .map_err(|e| ImageSaveError::Encode(format!("png encode error: {e}")))
}

/// Save `img` as a PNG file (dlib `save_png` from a filename).
pub fn save_png_file<P: Pixel>(img: &Array2D<P>, path: &str) -> Result<(), ImageSaveError> {
    let mut f = File::create(Path::new(path))?;
    save_png(img, &mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_png_header() {
        let img = Array2D::<RgbPixel>::zeros(2, 2);
        let mut bytes = Vec::new();
        save_png(&img, &mut bytes).unwrap();
        assert_eq!(
            &bytes[..8],
            &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
        );
    }
}

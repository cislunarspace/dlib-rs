//! WebP loading via the pure-Rust `image-webp` crate, mirroring the API
//! shape of `dlib/image_loader/webp_loader.h` (`load_webp`).

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use image_webp::WebPDecoder;

use crate::array2d::Array2D;
use crate::image_loader::ImageLoadError;
use crate::pixel::{assign_pixel, Pixel, RgbAlphaPixel, RgbPixel};

/// Load a WebP image from a reader, converting into any pixel type via
/// `assign_pixel` (port of dlib `load_webp`). Decodes to RGB or RGBA
/// depending on whether the file has an alpha channel.
pub fn load_webp<P: Pixel, R: Read>(input: &mut R) -> Result<Array2D<P>, ImageLoadError> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    let mut decoder = WebPDecoder::new(Cursor::new(bytes))
        .map_err(|e| ImageLoadError::Corrupt(format!("webp decode error: {e}")))?;
    let (width, height) = decoder.dimensions();
    let width = width as usize;
    let height = height as usize;
    let has_alpha = decoder.has_alpha();

    let mut buf = vec![0u8; decoder.output_buffer_size().unwrap_or(0)];
    decoder
        .read_image(&mut buf)
        .map_err(|e| ImageLoadError::Corrupt(format!("webp decode error: {e}")))?;

    let mut out = Array2D::zeros(height, width);
    if has_alpha {
        for r in 0..height {
            for c in 0..width {
                let i = (r * width + c) * 4;
                let p = RgbAlphaPixel {
                    r: buf[i],
                    g: buf[i + 1],
                    b: buf[i + 2],
                    a: buf[i + 3],
                };
                assign_pixel(out.get_mut(r, c), &p);
            }
        }
    } else {
        for r in 0..height {
            for c in 0..width {
                let i = (r * width + c) * 3;
                let p = RgbPixel {
                    r: buf[i],
                    g: buf[i + 1],
                    b: buf[i + 2],
                };
                assign_pixel(out.get_mut(r, c), &p);
            }
        }
    }
    Ok(out)
}

/// Load a WebP image from a file (dlib `load_webp` from a filename).
pub fn load_webp_file<P: Pixel>(path: &str) -> Result<Array2D<P>, ImageLoadError> {
    let mut f = File::open(Path::new(path))?;
    load_webp(&mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_webp_lossless_roundtrip() {
        // encode a tiny lossless webp in-memory, then decode it
        let w = 8u32;
        let h = 4u32;
        let mut data = Vec::new();
        for r in 0..h {
            for c in 0..w {
                data.push((r * 30) as u8);
                data.push((c * 25) as u8);
                data.push((r * 10 + c * 5) as u8);
            }
        }
        let mut webp = Vec::new();
        let enc = image_webp::WebPEncoder::new(&mut webp);
        enc.encode(&data, w, h, image_webp::ColorType::Rgb8)
            .unwrap();
        let img = load_webp::<RgbPixel, _>(&mut &webp[..]).unwrap();
        assert_eq!((img.nr(), img.nc()), (4, 8));

        // lossless: exact roundtrip
        for r in 0..h as usize {
            for c in 0..w as usize {
                let i = (r * w as usize + c) * 3;
                assert_eq!(
                    *img.get(r, c),
                    RgbPixel {
                        r: data[i],
                        g: data[i + 1],
                        b: data[i + 2]
                    }
                );
            }
        }
    }

    #[test]
    fn test_load_webp_rejects_garbage() {
        let err = load_webp::<RgbPixel, _>(&mut &b"not a webp"[..]).unwrap_err();
        assert!(matches!(err, ImageLoadError::Corrupt(_)));
    }
}

//! JPEG loading via the pure-Rust `zune-jpeg` crate, mirroring the API shape
//! of `dlib/image_loader/jpeg_loader.h` (`load_jpeg`).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::JpegDecoder;

use crate::array2d::Array2D;
use crate::image_loader::ImageLoadError;
use crate::pixel::{assign_pixel, Pixel, RgbPixel};

/// Load a JPEG image from a reader, converting into any pixel type via
/// `assign_pixel` (port of dlib `load_jpeg`). Grayscale JPEGs decode to
/// grayscale pixels, everything else decodes to RGB.
pub fn load_jpeg<P: Pixel, R: Read>(input: &mut R) -> Result<Array2D<P>, ImageLoadError> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    let mut decoder = JpegDecoder::new(bytes);
    let pixels = decoder
        .decode()
        .map_err(|e| ImageLoadError::Corrupt(format!("jpeg decode error: {e}")))?;
    let info = decoder
        .info()
        .ok_or_else(|| ImageLoadError::Corrupt("jpeg decode error: no info".to_string()))?;
    let width = info.width as usize;
    let height = info.height as usize;
    let colorspace = decoder.get_output_colorspace();

    let mut out = Array2D::zeros(height, width);
    match colorspace {
        Some(ColorSpace::Luma) => {
            for r in 0..height {
                for c in 0..width {
                    assign_pixel(out.get_mut(r, c), &pixels[r * width + c]);
                }
            }
        }
        Some(ColorSpace::RGB) => {
            for r in 0..height {
                for c in 0..width {
                    let i = (r * width + c) * 3;
                    let p = RgbPixel {
                        r: pixels[i],
                        g: pixels[i + 1],
                        b: pixels[i + 2],
                    };
                    assign_pixel(out.get_mut(r, c), &p);
                }
            }
        }
        other => {
            return Err(ImageLoadError::Corrupt(format!(
                "jpeg decode error: unsupported output colorspace {other:?}"
            )))
        }
    }
    Ok(out)
}

/// Load a JPEG image from a file (dlib `load_jpeg` from a filename).
pub fn load_jpeg_file<P: Pixel>(path: &str) -> Result<Array2D<P>, ImageLoadError> {
    let mut f = File::open(Path::new(path))?;
    load_jpeg(&mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_jpeg_roundtrip_dims() {
        // encode a tiny 8x6 rgb gradient jpeg in-memory, then decode it
        let w = 8u16;
        let h = 6u16;
        let mut data = Vec::new();
        for r in 0..h {
            for c in 0..w {
                data.push((r * 30) as u8);
                data.push((c * 25) as u8);
                data.push((r * 10 + c * 5) as u8);
            }
        }
        let mut jfif = Vec::new();
        let enc = jpeg_encoder::Encoder::new(&mut jfif, 90);
        enc.encode(&data, w, h, jpeg_encoder::ColorType::Rgb)
            .unwrap();

        let img = load_jpeg::<RgbPixel, _>(&mut &jfif[..]).unwrap();
        assert_eq!(img.nr(), 6);
        assert_eq!(img.nc(), 8);
        // sanity: decode is roughly the same gradient
        let tl = img.get(0, 0);
        assert!(tl.r < 60);
        assert!(tl.b < 40);
    }

    #[test]
    fn test_load_jpeg_rejects_garbage() {
        let err = load_jpeg::<RgbPixel, _>(&mut &b"garbage"[..]).unwrap_err();
        assert!(matches!(err, ImageLoadError::Corrupt(_)));
    }
}

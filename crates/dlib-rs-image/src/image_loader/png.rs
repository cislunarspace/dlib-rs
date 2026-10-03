//! PNG loading via the pure-Rust `png` crate, mirroring the API shape of
//! `dlib/image_loader/png_loader.h` (`load_png`).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::array2d::Array2D;
use crate::image_loader::ImageLoadError;
use crate::pixel::{assign_pixel, Pixel, RgbAlphaPixel, RgbPixel};

/// Load a PNG image from a reader, converting into any pixel type via
/// `assign_pixel` (port of dlib `load_png`).
pub fn load_png<P: Pixel, R: Read>(input: &mut R) -> Result<Array2D<P>, ImageLoadError> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes)?;
    let input = &mut std::io::Cursor::new(bytes);
    let mut decoder = png::Decoder::new(input);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|e| ImageLoadError::Corrupt(format!("png decode error: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| ImageLoadError::Corrupt(format!("png decode error: {e}")))?;
    let width = info.width as usize;
    let height = info.height as usize;

    let (color_type, bit_depth) = reader.output_color_type();
    if bit_depth != png::BitDepth::Eight {
        return Err(ImageLoadError::Corrupt(
            "png decode error: unexpected output bit depth".to_string(),
        ));
    }

    let mut out = Array2D::zeros(height, width);
    match color_type {
        png::ColorType::Grayscale => {
            for r in 0..height {
                for c in 0..width {
                    assign_pixel(out.get_mut(r, c), &buf[r * width + c]);
                }
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for r in 0..height {
                for c in 0..width {
                    let g = buf[(r * width + c) * 2];
                    let a = buf[(r * width + c) * 2 + 1];
                    assign_pixel(out.get_mut(r, c), &RgbAlphaPixel { r: g, g, b: g, a });
                }
            }
        }
        png::ColorType::Rgb => {
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
        png::ColorType::Rgba => {
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
        }
        _ => {
            return Err(ImageLoadError::Corrupt(
                "png decode error: unsupported color type".to_string(),
            ))
        }
    }
    Ok(out)
}

/// Load a PNG image from a file (dlib `load_png` from a filename).
pub fn load_png_file<P: Pixel>(path: &str) -> Result<Array2D<P>, ImageLoadError> {
    let mut f = File::open(Path::new(path))?;
    load_png(&mut f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_png_roundtrip_rgb_and_gray() {
        // rgb
        let mut img = Array2D::<RgbPixel>::zeros(4, 3);
        for r in 0..4 {
            for c in 0..3 {
                *img.get_mut(r, c) = RgbPixel {
                    r: (r * 60 + c * 9) as u8,
                    g: (r * 25 + c * 40 + 7) as u8,
                    b: (200 - r * 30 - c * 5) as u8,
                };
            }
        }
        let mut bytes = Vec::new();
        crate::image_saver::save_png(&img, &mut bytes).unwrap();
        let loaded = load_png::<RgbPixel, _>(&mut &bytes[..]).unwrap();
        assert_eq!((loaded.nr(), loaded.nc()), (4, 3));
        for r in 0..4 {
            for c in 0..3 {
                assert_eq!(*loaded.get(r, c), *img.get(r, c));
            }
        }

        // gray
        let mut g = Array2D::<u8>::zeros(3, 7);
        for r in 0..3 {
            for c in 0..7 {
                *g.get_mut(r, c) = (r * 77 + c * 13) as u8;
            }
        }
        let mut bytes = Vec::new();
        crate::image_saver::save_png(&g, &mut bytes).unwrap();
        let loaded = load_png::<u8, _>(&mut &bytes[..]).unwrap();
        for r in 0..3 {
            for c in 0..7 {
                assert_eq!(*loaded.get(r, c), *g.get(r, c));
            }
        }

        // rgba roundtrip through png
        let mut a = Array2D::<RgbAlphaPixel>::zeros(2, 2);
        for i in 0..4 {
            *a.get_mut(i / 2, i % 2) = RgbAlphaPixel {
                r: i as u8 * 10,
                g: 255 - i as u8 * 20,
                b: i as u8 * 30,
                a: 128 + i as u8,
            };
        }
        let mut bytes = Vec::new();
        crate::image_saver::save_png(&a, &mut bytes).unwrap();
        let loaded = load_png::<RgbAlphaPixel, _>(&mut &bytes[..]).unwrap();
        for i in 0..4 {
            assert_eq!(*loaded.get(i / 2, i % 2), *a.get(i / 2, i % 2));
        }
    }

    #[test]
    fn test_png_rejects_garbage() {
        let err = load_png::<RgbPixel, _>(&mut &b"not a png"[..]).unwrap_err();
        assert!(matches!(err, ImageLoadError::Corrupt(_)));
    }
}

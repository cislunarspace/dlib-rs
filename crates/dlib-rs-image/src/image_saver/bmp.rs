//! BMP saving, hand-ported from `dlib/image_saver/image_saver.h`
//! (`save_bmp_helper`).
//!
//! dlib semantics: grayscale images are written as 8-bit paletted BMPs with
//! an identity gray palette; every other pixel type (including rgb_alpha,
//! hsi, lab) is written as a 24-bit BGR BMP with colors converted via
//! `assign_pixel` (alpha is dropped). This dlib version has no 32-bit BMP
//! output path.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::array2d::Array2D;
use crate::image_saver::ImageSaveError;
use crate::pixel::{assign_pixel, Pixel, RgbPixel};

/// Write a little-endian u32.
fn put_u32<W: Write>(out: &mut W, v: u32) -> std::io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

/// Write a little-endian u16.
fn put_u16<W: Write>(out: &mut W, v: u16) -> std::io::Result<()> {
    out.write_all(&v.to_le_bytes())
}

/// Save `img` as a BMP to a writer (port of dlib `save_bmp(image, ostream)`).
pub fn save_bmp<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), ImageSaveError> {
    if P::is_gray() {
        save_bmp_gray(img, out)
    } else {
        save_bmp_color(img, out)
    }
}

/// Save `img` as a BMP file (dlib `save_bmp(image, filename)`).
pub fn save_bmp_file<P: Pixel>(img: &Array2D<P>, path: &str) -> Result<(), ImageSaveError> {
    let mut f = File::create(Path::new(path))?;
    save_bmp(img, &mut f)
}

fn write_bmp_header<W: Write>(
    out: &mut W,
    bf_size: u32,
    bf_off_bits: u32,
    width: u32,
    height: u32,
    bit_count: u16,
) -> std::io::Result<()> {
    out.write_all(b"BM")?;
    put_u32(out, bf_size)?;
    put_u32(out, 0)?; // reserved
    put_u32(out, bf_off_bits)?;
    put_u32(out, 40)?; // biSize
    put_u32(out, width)?;
    put_u32(out, height)?;
    put_u16(out, 1)?; // biPlanes
    put_u16(out, bit_count)?;
    put_u32(out, 0)?; // biCompression
    put_u32(out, 0)?; // biSizeImage
    put_u32(out, 0)?; // biXPelsPerMeter
    put_u32(out, 0)?; // biYPelsPerMeter
    put_u32(out, 0)?; // biClrUsed
    put_u32(out, 0) // biClrImportant
}

fn save_bmp_color<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), ImageSaveError> {
    // we are going to write out a 24bit color image.
    let pad = (4 - (img.nc() as u64 * 3) % 4) % 4;
    let bf_size = (14 + 40 + (img.nc() as u64 * 3 + pad) * img.nr() as u64) as u32;
    write_bmp_header(out, bf_size, 14 + 40, img.nc() as u32, img.nr() as u32, 24)?;

    for row in (0..img.nr()).rev() {
        for col in 0..img.nc() {
            let mut p = RgbPixel::default();
            assign_pixel(&mut p, img.get(row, col));
            out.write_all(&[p.b, p.g, p.r])?;
        }
        // write out some zeros so that this line is a multiple of 4 bytes
        for _ in 0..pad {
            out.write_all(&[0u8])?;
        }
    }
    Ok(())
}

fn save_bmp_gray<P: Pixel, W: Write>(img: &Array2D<P>, out: &mut W) -> Result<(), ImageSaveError> {
    // we are going to write out an 8bit color image.
    let pad = (4 - img.nc() as u64 % 4) % 4;
    let bf_size = (14 + 40 + (img.nc() as u64 + pad) * img.nr() as u64 + 256 * 4) as u32;
    write_bmp_header(
        out,
        bf_size,
        14 + 40 + 256 * 4,
        img.nc() as u32,
        img.nr() as u32,
        8,
    )?;

    // write out the color palette
    for i in 0u16..=255 {
        let b = i.to_le_bytes();
        out.write_all(&[b[0], b[0], b[0], 0])?;
    }

    for row in (0..img.nr()).rev() {
        for col in 0..img.nc() {
            let mut p = 0u8;
            assign_pixel(&mut p, img.get(row, col));
            out.write_all(&[p])?;
        }
        for _ in 0..pad {
            out.write_all(&[0u8])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_loader::load_bmp;

    #[test]
    fn test_bmp_roundtrip_rgb() {
        let mut img = Array2D::<RgbPixel>::zeros(3, 2);
        for r in 0..3 {
            for c in 0..2 {
                *img.get_mut(r, c) = RgbPixel {
                    r: (r * 80 + c * 7) as u8,
                    g: (r * 30 + c * 11 + 5) as u8,
                    b: (255 - r * 60 - c * 3) as u8,
                };
            }
        }
        let mut bytes = Vec::new();
        save_bmp(&img, &mut bytes).unwrap();
        let loaded = load_bmp::<RgbPixel, _>(&mut &bytes[..]).unwrap();
        assert_eq!((loaded.nr(), loaded.nc()), (3, 2));
        for r in 0..3 {
            for c in 0..2 {
                assert_eq!(*loaded.get(r, c), *img.get(r, c));
            }
        }
    }

    #[test]
    fn test_bmp_roundtrip_gray() {
        let mut img = Array2D::<u8>::zeros(2, 5);
        for r in 0..2 {
            for c in 0..5 {
                *img.get_mut(r, c) = (r * 100 + c * 17 + 3) as u8;
            }
        }
        let mut bytes = Vec::new();
        save_bmp(&img, &mut bytes).unwrap();
        let loaded = load_bmp::<u8, _>(&mut &bytes[..]).unwrap();
        for r in 0..2 {
            for c in 0..5 {
                assert_eq!(*loaded.get(r, c), *img.get(r, c));
            }
        }
    }

    #[test]
    fn test_bmp_header_fields() {
        let img = Array2D::<RgbPixel>::zeros(4, 2); // row = 6 bytes -> pad 2
        let mut bytes = Vec::new();
        save_bmp(&img, &mut bytes).unwrap();
        assert_eq!(&bytes[0..2], b"BM");
        let bf_size = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        assert_eq!(bf_size as usize, bytes.len());
        let bf_off_bits = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
        assert_eq!(bf_off_bits, 54);
        assert_eq!(
            u32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]),
            2
        ); // width
        assert_eq!(
            u32::from_le_bytes([bytes[22], bytes[23], bytes[24], bytes[25]]),
            4
        ); // height
        assert_eq!(u16::from_le_bytes([bytes[28], bytes[29]]), 24);
    }
}

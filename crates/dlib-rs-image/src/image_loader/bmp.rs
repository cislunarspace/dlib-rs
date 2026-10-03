//! BMP loading, hand-ported from `dlib/image_loader/image_loader.h`
//! (`load_bmp`).
//!
//! Supports exactly what dlib supports: 1/4/8-bit paletted (uncompressed,
//! plus the RLE compression of the 8-bit branch), and 24-bit true color;
//! 16/32-bit BMPs are rejected just like dlib does.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::array2d::Array2D;
use crate::image_loader::ImageLoadError;
use crate::pixel::{assign_pixel, Pixel, RgbPixel};

fn read_exact_or<R: Read>(input: &mut R, n: usize) -> Result<Vec<u8>, ImageLoadError> {
    let mut buf = vec![0u8; n];
    input.read_exact(&mut buf)?;
    Ok(buf)
}

fn seek_to_pixel_data<R: Read>(
    input: &mut R,
    mut bytes_read_so_far: u64,
    bf_off_bits: u64,
) -> Result<(), ImageLoadError> {
    while bytes_read_so_far != bf_off_bits {
        let to_read = std::cmp::min(bf_off_bits - bytes_read_so_far, 100) as usize;
        read_exact_or(input, to_read)?;
        bytes_read_so_far += to_read as u64;
    }
    Ok(())
}

fn le32(b: &[u8]) -> u64 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64
}

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// Load a BMP image from a reader, converting into any pixel type via
/// `assign_pixel` (port of dlib `load_bmp(image, istream)`).
pub fn load_bmp<P: Pixel, R: Read>(input: &mut R) -> Result<Array2D<P>, ImageLoadError> {
    let rgb = load_bmp_rgb(input)?;
    let mut out = Array2D::zeros(rgb.nr(), rgb.nc());
    for r in 0..rgb.nr() {
        for c in 0..rgb.nc() {
            assign_pixel(out.get_mut(r, c), rgb.get(r, c));
        }
    }
    Ok(out)
}

/// Load a BMP image from a file (dlib `load_bmp(image, filename)`).
pub fn load_bmp_file<P: Pixel>(path: &str) -> Result<Array2D<P>, ImageLoadError> {
    let mut f = File::open(Path::new(path))?;
    load_bmp(&mut f)
}

fn load_bmp_rgb<R: Read>(input: &mut R) -> Result<Array2D<RgbPixel>, ImageLoadError> {
    let corrupt = |m: &str| ImageLoadError::Corrupt(m.to_string());

    let mut bytes_read_so_far: u64 = 0;

    let buf = read_exact_or(input, 2)?;
    bytes_read_so_far += 2;
    if buf[0] != b'B' || buf[1] != b'M' {
        return Err(corrupt("bmp load error 2: header error"));
    }

    // BITMAPFILEHEADER (rest of it)
    let buf = read_exact_or(input, 12)?;
    bytes_read_so_far += 12;
    let bf_size = le32(&buf[0..4]);
    let bf_off_bits = le32(&buf[8..12]);

    // BITMAPINFOHEADER
    let buf = read_exact_or(input, 40)?;
    bytes_read_so_far += 40;
    let bi_size = le32(&buf[0..4]);
    let bi_width = le32(&buf[4..8]);
    let bi_height_raw = i32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let bottom_up = bi_height_raw < 0;
    let bi_height = (bi_height_raw.unsigned_abs()) as u64;
    let bi_bit_count = le16(&buf[14..16]);
    let bi_compression = le32(&buf[16..20]);

    if bi_size != 40 {
        return Err(corrupt("bmp load error 6: header too small"));
    }

    if bi_width == 0 || bi_height == 0 || bi_width > (1 << 28) || bi_height > (1 << 28) {
        return Err(corrupt("bmp load error: bad dimensions"));
    }
    let width = bi_width as usize;
    let height = bi_height as usize;

    let mut image = Array2D::<RgbPixel>::zeros(height, width);

    // row in the file -> row in the image (dlib's bottomUp handling)
    let dest_row = |row: i64| -> usize {
        if bottom_up {
            (height as i64 - row - 1) as usize
        } else {
            row as usize
        }
    };

    match bi_bit_count {
        1 | 4 | 8 => {
            let palette_size = match bi_bit_count {
                1 => 2usize,
                4 => 16,
                _ => 256,
            };
            let mut red = vec![0u8; palette_size];
            let mut green = vec![0u8; palette_size];
            let mut blue = vec![0u8; palette_size];
            for i in 0..palette_size {
                let entry = read_exact_or(input, 4)?;
                bytes_read_so_far += 4;
                blue[i] = entry[0];
                green[i] = entry[1];
                red[i] = entry[2];
            }

            // figure out how the pixels are packed (dlib logic preserved)
            let bpp_bits = bi_bit_count as u64;
            let row_bytes_exact = width as u64 * height as u64 * bpp_bits / 8;
            let mut padding: i64 = if bf_size.wrapping_sub(bf_off_bits) == row_bytes_exact {
                0
            } else {
                let row_bytes = match bi_bit_count {
                    1 => (width as u64).div_ceil(8),
                    4 => (width as u64).div_ceil(2),
                    _ => width as u64,
                };
                4 - (row_bytes % 4) as i64
            };

            if bi_bit_count == 8 || bi_bit_count == 24 {
                // some BMP writers screw up the files so we have to do this
                if (height as i64) * (width as i64 + padding)
                    > (bf_size.wrapping_sub(bf_off_bits)) as i64
                {
                    padding = 0;
                }
            }

            seek_to_pixel_data(input, bytes_read_so_far, bf_off_bits)?;

            if bi_bit_count == 8 && bi_compression != 0 {
                // RLE-compressed 8-bit BMP (dlib handles RLE in the 8bpp branch
                // whenever biCompression != 0).
                load_rle8(input, &mut image, &red, &green, &blue, bottom_up, padding)?;
                return Ok(image);
            }

            let pixels_per_byte = 8 / bi_bit_count as usize;
            let mask: u8 = (palette_size - 1) as u8;

            for row in (0..height as i64).rev() {
                let dr = dest_row(row);
                let mut col = 0usize;
                while col < width {
                    let byte = read_exact_or(input, 1)?[0];
                    for sub in 0..pixels_per_byte {
                        if col >= width {
                            break;
                        }
                        let shift = 8 - bi_bit_count as usize * (sub + 1);
                        let idx = ((byte >> shift) & mask) as usize;
                        let p = RgbPixel {
                            r: red[idx],
                            g: green[idx],
                            b: blue[idx],
                        };
                        *image.get_mut(dr, col) = p;
                        col += 1;
                    }
                }
                if padding > 0 {
                    read_exact_or(input, padding as usize)?;
                }
            }
        }
        24 => {
            let row_bytes_exact = width as u64 * height as u64 * 3;
            let mut padding: i64 = if bf_size.wrapping_sub(bf_off_bits) == row_bytes_exact {
                0
            } else {
                4 - ((width as u64 * 3) % 4) as i64
            };
            if (height as i64) * (width as i64 * 3 + padding)
                > (bf_size.wrapping_sub(bf_off_bits)) as i64
            {
                padding = 0;
            }

            seek_to_pixel_data(input, bytes_read_so_far, bf_off_bits)?;

            for row in (0..height as i64).rev() {
                let dr = dest_row(row);
                for col in 0..width {
                    let px = read_exact_or(input, 3)?;
                    let p = RgbPixel {
                        b: px[0],
                        g: px[1],
                        r: px[2],
                    };
                    *image.get_mut(dr, col) = p;
                }
                if padding > 0 {
                    read_exact_or(input, padding as usize)?;
                }
            }
        }
        16 => return Err(corrupt("16 bit BMP images not supported")),
        32 => return Err(corrupt("32 bit BMP images not supported")),
        _ => return Err(corrupt("bmp load error 10: unknown color depth")),
    }

    Ok(image)
}

/// The "psychotic RLE used by BMP files", ported verbatim from the 8-bit
/// branch of dlib's `load_bmp`.
fn load_rle8<R: Read>(
    input: &mut R,
    image: &mut Array2D<RgbPixel>,
    red: &[u8],
    green: &[u8],
    blue: &[u8],
    bottom_up: bool,
    padding: i64,
) -> Result<(), ImageLoadError> {
    let corrupt = |m: &str| ImageLoadError::Corrupt(m.to_string());
    let height = image.nr() as i64;
    let width = image.nc() as i64;

    // RLE sometimes jumps over pixels and assumes the image is zeroed.
    *image = Array2D::zeros(image.nr(), image.nc());

    let dest_row = |row: i64| -> i64 {
        if bottom_up {
            height - row - 1
        } else {
            row
        }
    };

    let mut row: i64 = height - 1;
    let mut col: i64 = 0;
    loop {
        let pair = read_exact_or(input, 2)?;
        let count = pair[0];
        let command = pair[1];

        if count == 0 && command == 0 {
            // escape: go to the next row
            row -= 1;
            col = 0;
            continue;
        } else if count == 0 && command == 1 {
            // end of image
            break;
        } else if count == 0 && command == 2 {
            // jump to a new part of the image relative to here
            let delta = read_exact_or(input, 2)?;
            col += delta[0] as i64;
            row -= delta[1] as i64;
            continue;
        } else if count == 0 {
            // escape: run of uncompressed bytes
            if row < 0 || col + command as i64 > width {
                // just some padding bytes at the end then ignore them
                if row >= 0 && col + count as i64 <= width + padding {
                    continue;
                }
                return Err(corrupt("bmp load error 21.2: file data corrupt"));
            }
            let dr = dest_row(row);
            for _ in 0..command {
                let b = read_exact_or(input, 1)?[0];
                let p = RgbPixel {
                    r: red[b as usize],
                    g: green[b as usize],
                    b: blue[b as usize],
                };
                *image.get_mut(dr as usize, col as usize) = p;
                col += 1;
            }
            // if we read an uneven number of bytes then we need to read and
            // discard the next byte (dlib behavior preserved)
            if (command & 1) != 1 {
                read_exact_or(input, 1)?;
            }
            continue;
        }

        if row < 0 || col + count as i64 > width {
            // just some padding bytes at the end then ignore them
            if row >= 0 && col + count as i64 <= width + padding {
                continue;
            }
            return Err(corrupt("bmp load error 21.5: file data corrupt"));
        }

        let dr = dest_row(row);
        let p = RgbPixel {
            r: red[command as usize],
            g: green[command as usize],
            b: blue[command as usize],
        };
        for _ in 0..count {
            *image.get_mut(dr as usize, col as usize) = p;
            col += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- hand-built BMP test vectors -------------------------------------

    /// Build a paletted BMP (uncompressed, bottom-up).
    /// palette: list of (b,g,r) entries; indices given row-major top-down.
    fn build_paletted_bmp(
        width: u32,
        height: u32,
        bpp: u32,
        palette: &[[u8; 3]],
        indices: &[u8],
    ) -> Vec<u8> {
        let palette_bytes = palette.len() as u64 * 4;
        let row_bytes = (width as u64 * bpp as u64).div_ceil(8) as usize;
        let pad = (4 - row_bytes % 4) % 4;
        let data_len = (row_bytes + pad) * height as usize;
        let bf_off_bits = 14u64 + 40 + palette_bytes;
        let bf_size = bf_off_bits + data_len as u64;

        let mut v = Vec::new();
        v.extend_from_slice(b"BM");
        v.extend_from_slice(&(bf_size as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&(bf_off_bits as u32).to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&width.to_le_bytes());
        v.extend_from_slice(&height.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&(bpp as u16).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes()); // compression
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        for (b, g, r) in palette.iter().map(|x| (x[0], x[1], x[2])) {
            v.extend_from_slice(&[b, g, r, 0]);
        }
        // rows bottom-up, indices given top-down
        for row in (0..height as usize).rev() {
            let row_slice = &indices[row * width as usize..(row + 1) * width as usize];
            match bpp {
                1 => {
                    let mut acc = 0u8;
                    let mut n = 0;
                    for &ix in row_slice {
                        acc = (acc << 1) | (ix & 1);
                        n += 1;
                        if n == 8 {
                            v.push(acc);
                            acc = 0;
                            n = 0;
                        }
                    }
                    if n > 0 {
                        v.push(acc << (8 - n));
                    }
                }
                4 => {
                    let mut even = false;
                    let mut acc = 0u8;
                    for &ix in row_slice {
                        if even {
                            v.push((acc << 4) | (ix & 0xf));
                        } else {
                            acc = ix & 0xf;
                        }
                        even = !even;
                    }
                    if even {
                        v.push(acc << 4);
                    }
                }
                8 => v.extend_from_slice(row_slice),
                _ => unreachable!(),
            }
            v.extend(std::iter::repeat_n(0u8, pad));
        }
        v
    }

    #[test]
    fn test_load_1bpp() {
        // 8x2, palette: black, white; row0 = 10101010, row1 = 11001100
        let bmp = build_paletted_bmp(
            8,
            2,
            1,
            &[[0, 0, 0], [255, 255, 255]],
            &[
                1, 0, 1, 0, 1, 0, 1, 0, //
                1, 1, 0, 0, 1, 1, 0, 0,
            ],
        );
        let img = load_bmp::<RgbPixel, _>(&mut &bmp[..]).unwrap();
        assert_eq!(img.nr(), 2);
        assert_eq!(img.nc(), 8);
        assert_eq!(
            *img.get(0, 0),
            RgbPixel {
                r: 255,
                g: 255,
                b: 255
            }
        );
        assert_eq!(*img.get(0, 1), RgbPixel { r: 0, g: 0, b: 0 });
        assert_eq!(*img.get(1, 2), RgbPixel { r: 0, g: 0, b: 0 });
        assert_eq!(
            *img.get(1, 4),
            RgbPixel {
                r: 255,
                g: 255,
                b: 255
            }
        );
    }

    #[test]
    fn test_load_4bpp() {
        // 3x2 with two palette entries, row padded to 4 bytes
        let palette: Vec<[u8; 3]> = (0..16).map(|i| [i * 3, i * 5, i * 7]).collect();
        let bmp = build_paletted_bmp(3, 2, 4, &palette, &[1, 2, 3, 15, 0, 14]);
        let img = load_bmp::<RgbPixel, _>(&mut &bmp[..]).unwrap();
        assert_eq!((img.nr(), img.nc()), (2, 3));
        assert_eq!(*img.get(0, 0), RgbPixel { r: 7, g: 5, b: 3 });
        assert_eq!(*img.get(0, 2), RgbPixel { r: 21, g: 15, b: 9 });
        assert_eq!(
            *img.get(1, 0),
            RgbPixel {
                r: 105,
                g: 75,
                b: 45
            }
        );
        assert_eq!(
            *img.get(1, 2),
            RgbPixel {
                r: 98,
                g: 70,
                b: 42
            }
        );
    }

    #[test]
    fn test_load_8bpp_and_rle8() {
        let palette: Vec<[u8; 3]> = (0..256)
            .map(|i| {
                [
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                ]
            })
            .collect();
        // uncompressed 8bpp, 4x2 (multiple of 4 so no padding)
        let bmp = build_paletted_bmp(4, 2, 8, &palette, &[10, 20, 30, 40, 50, 60, 70, 80]);
        let img = load_bmp::<RgbPixel, _>(&mut &bmp[..]).unwrap();
        assert_eq!(
            *img.get(0, 0),
            RgbPixel {
                r: 30,
                g: 20,
                b: 10
            }
        );
        assert_eq!(
            *img.get(1, 3),
            RgbPixel {
                r: 240,
                g: 160,
                b: 80
            }
        );

        // RLE8: encoded by hand. 4x2 bottom-up means first encoded row is the
        // bottom (image row 1). Sequence:
        //   run 3x idx 7; escape end-of-row; run 2x idx 9; run 1x idx 11;
        //   run 1x idx 12; end-of-image.
        let mut v = bmp_header_rle(4, 2);
        v.extend_from_slice(&[3, 7]);
        v.extend_from_slice(&[0, 0]); // next row
        v.extend_from_slice(&[2, 9]);
        v.extend_from_slice(&[1, 11]);
        v.extend_from_slice(&[1, 12]);
        v.extend_from_slice(&[0, 1]); // end of image

        let img = load_bmp::<RgbPixel, _>(&mut &v[..]).unwrap();
        assert_eq!((img.nr(), img.nc()), (2, 4));
        // row 1 (bottom): 7,7,7,0 (zeroed since RLE skips pixels)
        assert_eq!(*img.get(1, 0), RgbPixel { r: 21, g: 14, b: 7 });
        assert_eq!(*img.get(1, 1), RgbPixel { r: 21, g: 14, b: 7 });
        assert_eq!(*img.get(1, 2), RgbPixel { r: 21, g: 14, b: 7 });
        assert_eq!(*img.get(1, 3), RgbPixel { r: 0, g: 0, b: 0 });
        // row 0: 9,9,11,12
        assert_eq!(*img.get(0, 0), RgbPixel { r: 27, g: 18, b: 9 });
        assert_eq!(*img.get(0, 1), RgbPixel { r: 27, g: 18, b: 9 });
        assert_eq!(
            *img.get(0, 2),
            RgbPixel {
                r: 33,
                g: 22,
                b: 11
            }
        );
        assert_eq!(
            *img.get(0, 3),
            RgbPixel {
                r: 36,
                g: 24,
                b: 12
            }
        );
    }

    fn bmp_header_rle(width: u32, height: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"BM");
        v.extend_from_slice(&1000u32.to_le_bytes()); // bfSize (fake)
        v.extend_from_slice(&0u32.to_le_bytes());
        let bf_off_bits: u32 = 14 + 40 + 256 * 4;
        v.extend_from_slice(&bf_off_bits.to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&width.to_le_bytes());
        v.extend_from_slice(&height.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&8u16.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes()); // BI_RLE8 (dlib treats any nonzero compression as RLE)
        for _ in 0..5 {
            v.extend_from_slice(&0u32.to_le_bytes());
        }
        // NOTE: header above is 14+40=54 bytes then palette must be 256 entries;
        // the loop after adds the palette.
        let palette: Vec<[u8; 3]> = (0..256)
            .map(|i| {
                [
                    (i % 256) as u8,
                    ((i * 2) % 256) as u8,
                    ((i * 3) % 256) as u8,
                ]
            })
            .collect();
        for (b, g, r) in palette.iter().map(|x| (x[0], x[1], x[2])) {
            v.extend_from_slice(&[b, g, r, 0]);
        }
        assert_eq!(v.len(), bf_off_bits as usize);
        v
    }

    #[test]
    fn test_load_24bpp_top_down() {
        // 2x2, top-down (negative height), 24bpp
        let mut v = Vec::new();
        v.extend_from_slice(b"BM");
        let row = 2 * 3 + 2; // padded to 8
        let bf_off_bits = 14 + 40;
        v.extend_from_slice(&((bf_off_bits + row * 2) as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&(bf_off_bits as u32).to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&(-2i32).to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&24u16.to_le_bytes());
        for _ in 0..6 {
            v.extend_from_slice(&0u32.to_le_bytes());
        }
        // file rows in order map to image rows 0,1 for top-down
        let pixels: [[u8; 3]; 4] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
        for row_px in 0..2 {
            for col in 0..2 {
                let (b, g, r) = (
                    pixels[row_px * 2 + col][0],
                    pixels[row_px * 2 + col][1],
                    pixels[row_px * 2 + col][2],
                );
                v.extend_from_slice(&[b, g, r]);
            }
            v.extend_from_slice(&[0, 0]);
        }
        let img = load_bmp::<RgbPixel, _>(&mut &v[..]).unwrap();
        assert_eq!(*img.get(0, 0), RgbPixel { r: 3, g: 2, b: 1 });
        assert_eq!(*img.get(0, 1), RgbPixel { r: 6, g: 5, b: 4 });
        assert_eq!(*img.get(1, 0), RgbPixel { r: 9, g: 8, b: 7 });
        assert_eq!(
            *img.get(1, 1),
            RgbPixel {
                r: 12,
                g: 11,
                b: 10
            }
        );
    }

    #[test]
    fn test_reject_16bpp() {
        let mut v = Vec::new();
        v.extend_from_slice(b"BM");
        v.extend_from_slice(&100u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&54u32.to_le_bytes());
        v.extend_from_slice(&40u32.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        for _ in 0..6 {
            v.extend_from_slice(&0u32.to_le_bytes());
        }
        let err = load_bmp::<RgbPixel, _>(&mut &v[..]).unwrap_err();
        assert!(err.to_string().contains("16 bit"));
    }
}

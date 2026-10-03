//! Port of `dlib/image_transforms/colormaps.h`: `jet`, `heatmap`,
//! `colormap_jet`, `colormap_heat` and `randomly_color_image` (the latter uses
//! dlib's `murmur_hash3_2` from `general_hash/murmur_hash3.h`, not rand).

use crate::array2d::GenericImage;
use crate::pixel::{get_pixel_intensity, Pixel, RgbPixel};

/// Port of dlib's `put_in_range(a, b, val)` for doubles: clamps `val` to the
/// closer end of the (possibly reversed) range `[a, b]`.
fn put_in_range(a: f64, b: f64, val: f64) -> f64 {
    if a < b {
        if val < a {
            a
        } else if val > b {
            b
        } else {
            val
        }
    } else if val < b {
        b
    } else if val > a {
        a
    } else {
        val
    }
}

/// Port of `dlib::murmur_hash3_2(v1, v2)` from `general_hash/murmur_hash3.h`.
pub fn murmur_hash3_2(v1: u32, v2: u32) -> u32 {
    fn fmix(mut h: u32) -> u32 {
        h ^= h >> 16;
        h = h.wrapping_mul(0x85eb_ca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2_ae35);
        h ^= h >> 16;
        h
    }

    let mut h1 = v2;

    let c1 = 0xcc9e_2d51u32;
    let c2 = 0x1b87_3593u32;

    let mut k1 = v1;

    k1 = k1.wrapping_mul(c1);
    k1 = k1.rotate_left(15);
    k1 = k1.wrapping_mul(c2);

    h1 ^= k1;
    h1 = h1.rotate_left(13);
    h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);

    // finalization
    h1 ^= 4; // ^= by length in bytes
    h1 = fmix(h1);

    h1
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::colormap_jet(value, min_val, max_val)`: dlib's piecewise jet
/// colormap (slope 1/2, `+0.5` rounding before the `unsigned char` cast).
pub fn colormap_jet(value: f64, min_val: f64, max_val: f64) -> RgbPixel {
    // scale the gray value into the range [0, 8]
    let gray = 8.0 * put_in_range(0.0, 1.0, (value - min_val) / (max_val - min_val));
    let mut pix = RgbPixel::default();
    // s is the slope of color change
    let s = 1.0 / 2.0;

    if gray <= 1.0 {
        pix.r = 0;
        pix.g = 0;
        pix.b = ((gray + 1.0) * s * 255.0 + 0.5) as u8;
    } else if gray <= 3.0 {
        pix.r = 0;
        pix.g = ((gray - 1.0) * s * 255.0 + 0.5) as u8;
        pix.b = 255;
    } else if gray <= 5.0 {
        pix.r = ((gray - 3.0) * s * 255.0 + 0.5) as u8;
        pix.g = 255;
        pix.b = ((5.0 - gray) * s * 255.0 + 0.5) as u8;
    } else if gray <= 7.0 {
        pix.r = 255;
        pix.g = ((7.0 - gray) * s * 255.0 + 0.5) as u8;
        pix.b = 0;
    } else {
        pix.r = ((9.0 - gray) * s * 255.0 + 0.5) as u8;
        pix.g = 0;
        pix.b = 0;
    }

    pix
}

/// Port of `dlib::jet(img, max_val, min_val = 0)` (eager evaluation of dlib's
/// lazy matrix expression into an RGB image).
pub fn jet_with_range<S, D>(src: &S, dst: &mut D, max_val: f64, min_val: f64)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage<PixelType = RgbPixel>,
{
    dst.set_image_size(src.num_rows(), src.num_columns());
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            *dst.pixel_mut(r, c) =
                colormap_jet(get_pixel_intensity(src.pixel(r, c)), min_val, max_val);
        }
    }
}

/// Port of `dlib::jet(img)`: scales with the min/max pixel intensity of the
/// image.
pub fn jet<S, D>(src: &S, dst: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage<PixelType = RgbPixel>,
{
    if src.num_rows() * src.num_columns() != 0 {
        let mut max_val = f64::NEG_INFINITY;
        let mut min_val = f64::INFINITY;
        for r in 0..src.num_rows() {
            for c in 0..src.num_columns() {
                let v = get_pixel_intensity(src.pixel(r, c));
                max_val = max_val.max(v);
                min_val = min_val.min(v);
            }
        }
        jet_with_range(src, dst, max_val, min_val);
    } else {
        jet_with_range(src, dst, 0.0, 0.0);
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::colormap_heat(value, min_val, max_val)`.
pub fn colormap_heat(value: f64, min_val: f64, max_val: f64) -> RgbPixel {
    // scale the gray value into the range [0, 1]
    let gray = put_in_range(0.0, 1.0, (value - min_val) / (max_val - min_val));
    let mut pix = RgbPixel {
        r: ((gray / 0.4).min(1.0) * 255.0 + 0.5) as u8,
        ..Default::default()
    };

    if gray > 0.4 {
        pix.g = (((gray - 0.4) / 0.4).min(1.0) * 255.0 + 0.5) as u8;
    }
    if gray > 0.8 {
        pix.b = (((gray - 0.8) / 0.2).min(1.0) * 255.0 + 0.5) as u8;
    }

    pix
}

/// Port of `dlib::heatmap(img, max_val, min_val = 0)`.
pub fn heatmap_with_range<S, D>(src: &S, dst: &mut D, max_val: f64, min_val: f64)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage<PixelType = RgbPixel>,
{
    dst.set_image_size(src.num_rows(), src.num_columns());
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            *dst.pixel_mut(r, c) =
                colormap_heat(get_pixel_intensity(src.pixel(r, c)), min_val, max_val);
        }
    }
}

/// Port of `dlib::heatmap(img)`: scales with the min/max pixel intensity of
/// the image.
pub fn heatmap<S, D>(src: &S, dst: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage<PixelType = RgbPixel>,
{
    if src.num_rows() * src.num_columns() != 0 {
        let mut max_val = f64::NEG_INFINITY;
        let mut min_val = f64::INFINITY;
        for r in 0..src.num_rows() {
            for c in 0..src.num_columns() {
                let v = get_pixel_intensity(src.pixel(r, c));
                max_val = max_val.max(v);
                min_val = min_val.min(v);
            }
        }
        heatmap_with_range(src, dst, max_val, min_val);
    } else {
        heatmap_with_range(src, dst, 0.0, 0.0);
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::randomly_color_image(img)`.
///
/// Despite the name this uses no random numbers in dlib: each non-zero
/// intensity `gray` is mapped through `murmur_hash3_2(gray, 0)` and
/// `red/green/blue = (h & 0xff, h >> 8 & 0xff, h >> 16 & 0xff) % 200 + 55`;
/// black pixels stay black. The result is therefore deterministic.
pub fn randomly_color_image<S, D>(src: &S, dst: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage<PixelType = RgbPixel>,
{
    dst.set_image_size(src.num_rows(), src.num_columns());
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            let gray = get_pixel_intensity(src.pixel(r, c)) as u32;
            if gray != 0 {
                let h = murmur_hash3_2(gray, 0);
                let pix = RgbPixel {
                    r: (h as u8) % 200 + 55,
                    g: ((h >> 8) as u8) % 200 + 55,
                    b: ((h >> 16) as u8) % 200 + 55,
                };
                *dst.pixel_mut(r, c) = pix;
            } else {
                // keep black pixels black
                *dst.pixel_mut(r, c) = RgbPixel::default();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn jet_endpoints_match_dlib_formula() {
        // jet(0) with range [0,255]: gray=0 -> blue=(0+1)*0.5*255+0.5 = 128
        let p = colormap_jet(0.0, 0.0, 255.0);
        assert_eq!((p.r, p.g, p.b), (0, 0, 128));
        // jet(255): gray=8 -> red=(9-8)*0.5*255+0.5 = 128
        let p = colormap_jet(255.0, 0.0, 255.0);
        assert_eq!((p.r, p.g, p.b), (128, 0, 0));
        // midpoint: gray=4 -> full green, red=blue=(0.5)*127.5+0.5=128
        let p = colormap_jet(127.5, 0.0, 255.0);
        assert_eq!((p.r, p.g, p.b), (128, 255, 128));
    }

    #[test]
    fn heat_endpoints_match_dlib_formula() {
        let p = colormap_heat(0.0, 0.0, 255.0);
        assert_eq!((p.r, p.g, p.b), (0, 0, 0));
        let p = colormap_heat(255.0, 0.0, 255.0);
        assert_eq!((p.r, p.g, p.b), (255, 255, 255));
    }

    #[test]
    fn jet_image_end_to_end() {
        let mut src: Array2D<u8> = Array2D::zeros(1, 2);
        src[(0, 0)] = 0;
        src[(0, 1)] = 255;
        let mut dst: Array2D<RgbPixel> = Array2D::zeros(1, 1);
        jet(&src, &mut dst);
        assert_eq!(dst[(0, 0)], RgbPixel { r: 0, g: 0, b: 128 });
        assert_eq!(dst[(0, 1)], RgbPixel { r: 128, g: 0, b: 0 });
    }

    #[test]
    fn randomly_color_image_is_deterministic_and_keeps_black() {
        let mut src: Array2D<u8> = Array2D::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                src[(r, c)] = ((r * 3 + c) * 37 % 256) as u8;
            }
        }
        src[(1, 1)] = 0;
        let mut a: Array2D<RgbPixel> = Array2D::zeros(1, 1);
        let mut b: Array2D<RgbPixel> = Array2D::zeros(1, 1);
        randomly_color_image(&src, &mut a);
        randomly_color_image(&src, &mut b);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(a[(r, c)], b[(r, c)]);
            }
        }
        // spot-check against the hash formula
        let h = murmur_hash3_2(37, 0);
        let expect = RgbPixel {
            r: (h as u8) % 200 + 55,
            g: ((h >> 8) as u8) % 200 + 55,
            b: ((h >> 16) as u8) % 200 + 55,
        };
        assert_eq!(a[(0, 1)], expect);
    }

    #[test]
    fn murmur_hash3_2_known_vector() {
        // exact values produced by dlib's murmur_hash3_2 (C++ reference run)
        assert_eq!(murmur_hash3_2(37, 0), 4148071971);
        assert_eq!(murmur_hash3_2(0, 0), 593689054);
        assert_eq!(murmur_hash3_2(255, 0), 3962596441);
    }
}

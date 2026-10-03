//! Port of `dlib/image_transforms/edge_detector.h`: `edge_orientation`,
//! `sobel_edge_detector`, `suppress_non_maximum_edges` and
//! `normalize_image_gradients`.

use dlib_rs_core::matrix::Matrix;

use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, assign_pixel_intensity, get_pixel_intensity, Pixel};

/// Port of `dlib::edge_orientation(x, y)`: classifies a gradient direction as
/// one of `'|'`, `'-'`, `'/'`, `'\\'` using dlib's quantization
/// (`|x|*128/|y|` compared against 309 and 53).
pub fn edge_orientation(x_: f64, y_: f64) -> u8 {
    // if this is a perfectly horizontal gradient then return right away
    if x_ == 0.0 {
        return b'|';
    } else if y_ == 0.0 {
        // if this is a perfectly vertical gradient then return right away
        return b'-';
    }

    let mut x = x_;
    let mut y = y_;

    if x < 0.0 {
        x = -x;
        if y < 0.0 {
            y = -y;
            let temp = x * 128.0 / y;
            if temp > 309.0 {
                b'-'
            } else if temp > 53.0 {
                b'/'
            } else {
                b'|'
            }
        } else {
            let temp = x * 128.0 / y;
            if temp > 309.0 {
                b'-'
            } else if temp > 53.0 {
                b'\\'
            } else {
                b'|'
            }
        }
    } else if y < 0.0 {
        y = -y;
        let temp = x * 128.0 / y;
        if temp > 309.0 {
            b'-'
        } else if temp > 53.0 {
            b'\\'
        } else {
            b'|'
        }
    } else {
        let temp = x * 128.0 / y;
        if temp > 309.0 {
            b'-'
        } else if temp > 53.0 {
            b'/'
        } else {
            b'|'
        }
    }
}

/// Port of `dlib::sobel_edge_detector(in_img, horz, vert)`.
///
/// Applies dlib's sobel masks
/// `horz = [[-1,0,1],[-2,0,2],[-1,0,1]]` and
/// `vert = [[-1,-2,-1],[0,0,0],[1,2,1]]`
/// over pixel intensities, leaving a 1-pixel zero border
/// (`assign_border_pixels(out, 1, 1, 0)`). Returns `(horz, vert)` as
/// `Matrix<f64>` (dlib's "horz" is the x-gradient, "vert" the y-gradient).
pub fn sobel_edge_detector<S>(img: &S) -> (Matrix<f64>, Matrix<f64>)
where
    S: GenericImage,
    S::PixelType: Pixel,
{
    const VERT_FILTER: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];
    const HORZ_FILTER: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];

    let nr = img.num_rows();
    let nc = img.num_columns();
    let mut horz: Matrix<f64> = Matrix::zeros(nr, nc);
    let mut vert: Matrix<f64> = Matrix::zeros(nr, nc);

    // figure out the range that we should apply the filter to
    let first_row = 1usize;
    let first_col = 1usize;
    let last_row = nr - 1;
    let last_col = nc - 1;

    // apply the filter to the image
    for r in first_row..last_row {
        for c in first_col..last_col {
            let mut horz_temp: f64 = 0.0;
            let mut vert_temp: f64 = 0.0;
            for m in 0..3usize {
                for n in 0..3usize {
                    // pull out the current pixel and put it into p
                    let p = get_pixel_intensity(img.pixel(r - 1 + m, c - 1 + n));
                    horz_temp += p * HORZ_FILTER[m][n] as f64;
                    vert_temp += p * VERT_FILTER[m][n] as f64;
                }
            }
            horz[(r, c)] = horz_temp;
            vert[(r, c)] = vert_temp;
        }
    }

    (horz, vert)
}

/// Port of `dlib::suppress_non_maximum_edges(horz, vert, out_img)`.
///
/// Uses `edge_orientation` to quantize each gradient's angle and zeroes any
/// pixel whose (squared) gradient magnitude is strictly smaller than one of
/// its two neighbors along the gradient direction; the 1-pixel border is
/// zeroed (`zero_border_pixels(out, 1, 1)`).
pub fn suppress_non_maximum_edges<O>(horz: &Matrix<f64>, vert: &Matrix<f64>, out: &mut O)
where
    O: GenericImage,
    O::PixelType: Pixel,
{
    assert!(
        horz.nr() == vert.nr() && horz.nc() == vert.nc(),
        "suppress_non_maximum_edges: horz and vert must be the same size"
    );

    // if there isn't any input image then don't do anything
    if horz.nr() * horz.nc() == 0 {
        out.set_image_size(0, 0);
        return;
    }

    out.set_image_size(horz.nr(), horz.nc());

    // zero_border_pixels(out, 1, 1)
    let nr = horz.nr();
    let nc = horz.nc();
    for c in 0..nc {
        assign_pixel(out.pixel_mut(0, c), &0.0f64);
        assign_pixel(out.pixel_mut(nr - 1, c), &0.0f64);
    }
    for r in 0..nr {
        assign_pixel(out.pixel_mut(r, 0), &0.0f64);
        assign_pixel(out.pixel_mut(r, nc - 1), &0.0f64);
    }

    let sq = |m: &Matrix<f64>, r: usize, c: usize| m[(r, c)] * m[(r, c)];

    // figure out the range that we should apply the filter to
    let first_row = 1usize;
    let first_col = 1usize;
    let last_row = nr - 1;
    let last_col = nc - 1;

    // apply the filter to the image
    for r in first_row..last_row {
        for c in first_col..last_col {
            let y = horz[(r, c)];
            let x = vert[(r, c)];

            let val = sq(horz, r, c) + sq(vert, r, c);

            let ori = edge_orientation(x, y);
            match ori {
                b'-' => {
                    if sq(horz, r - 1, c) + sq(vert, r - 1, c) > val
                        || sq(horz, r + 1, c) + sq(vert, r + 1, c) > val
                    {
                        assign_pixel(out.pixel_mut(r, c), &0.0f64);
                    } else {
                        assign_pixel(out.pixel_mut(r, c), &val.sqrt());
                    }
                }
                b'|' => {
                    if sq(horz, r, c - 1) + sq(vert, r, c - 1) > val
                        || sq(horz, r, c + 1) + sq(vert, r, c + 1) > val
                    {
                        assign_pixel(out.pixel_mut(r, c), &0.0f64);
                    } else {
                        assign_pixel(out.pixel_mut(r, c), &val.sqrt());
                    }
                }
                b'/' => {
                    if sq(horz, r - 1, c - 1) + sq(vert, r - 1, c - 1) > val
                        || sq(horz, r + 1, c + 1) + sq(vert, r + 1, c + 1) > val
                    {
                        assign_pixel(out.pixel_mut(r, c), &0.0f64);
                    } else {
                        assign_pixel(out.pixel_mut(r, c), &val.sqrt());
                    }
                }
                b'\\' => {
                    if sq(horz, r + 1, c - 1) + sq(vert, r + 1, c - 1) > val
                        || sq(horz, r - 1, c + 1) + sq(vert, r - 1, c + 1) > val
                    {
                        assign_pixel(out.pixel_mut(r, c), &0.0f64);
                    } else {
                        assign_pixel(out.pixel_mut(r, c), &val.sqrt());
                    }
                }
                _ => {}
            }
        }
    }
}

/// Port of `dlib::normalize_image_gradients(img1, img2)` from
/// `edge_detector.h`: divides both gradient images by the gradient magnitude,
/// leaving zero vectors untouched.
pub fn normalize_image_gradients<I>(img1: &mut I, img2: &mut I)
where
    I: GenericImage,
    I::PixelType: Pixel,
{
    assert!(
        img1.num_rows() == img2.num_rows() && img1.num_columns() == img2.num_columns(),
        "normalize_image_gradients: images must be the same size"
    );

    // normalize all the gradients
    for r in 0..img1.num_rows() {
        for c in 0..img1.num_columns() {
            let v1 = *img1.pixel(r, c);
            let v2 = *img2.pixel(r, c);
            let g1 = get_pixel_intensity(&v1);
            let g2 = get_pixel_intensity(&v2);
            if g1 != 0.0 || g2 != 0.0 {
                let len = (g1 * g1 + g2 * g2).sqrt();
                assign_pixel_intensity(img1.pixel_mut(r, c), g1 / len);
                assign_pixel_intensity(img2.pixel_mut(r, c), g2 / len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn edge_orientation_quantization() {
        assert_eq!(edge_orientation(0.0, 5.0), b'|');
        assert_eq!(edge_orientation(5.0, 0.0), b'-');
        assert_eq!(edge_orientation(5.0, 5.0), b'/'); // ratio 128*5/5=128 -> '/'
        assert_eq!(edge_orientation(5.0, -5.0), b'\\');
        assert_eq!(edge_orientation(-5.0, 5.0), b'\\');
        assert_eq!(edge_orientation(-5.0, -5.0), b'/');
        assert_eq!(edge_orientation(1000.0, 1.0), b'-'); // 128000 > 309
        assert_eq!(edge_orientation(1.0, 1000.0), b'|'); // small ratio
    }

    #[test]
    fn sobel_horizontal_ramp() {
        // intensity = column index: perfect horizontal gradient
        let mut img: Array2D<u8> = Array2D::zeros(5, 6);
        for r in 0..5 {
            for c in 0..6 {
                img[(r, c)] = c as u8;
            }
        }
        let (horz, vert) = sobel_edge_detector(&img);
        // interior: ((c+1)-(c-1))*(1+2+1) = 8
        for r in 1..4 {
            for c in 1..5 {
                assert_eq!(horz[(r, c)], 8.0, "horz at ({r},{c})");
                assert_eq!(vert[(r, c)], 0.0, "vert at ({r},{c})");
            }
        }
        // borders are zero per dlib's assign_border_pixels(..., 1, 1, 0)
        for c in 0..6 {
            assert_eq!(horz[(0, c)], 0.0);
            assert_eq!(horz[(4, c)], 0.0);
        }
        for r in 0..5 {
            assert_eq!(horz[(r, 0)], 0.0);
            assert_eq!(horz[(r, 5)], 0.0);
        }
    }

    #[test]
    fn sobel_vertical_ramp() {
        let mut img: Array2D<u8> = Array2D::zeros(6, 5);
        for r in 0..6 {
            for c in 0..5 {
                img[(r, c)] = r as u8;
            }
        }
        let (horz, vert) = sobel_edge_detector(&img);
        for r in 1..5 {
            for c in 1..4 {
                assert_eq!(vert[(r, c)], 8.0);
                assert_eq!(horz[(r, c)], 0.0);
            }
        }
    }

    #[test]
    fn suppress_keeps_ridge_and_drops_slopes() {
        // vertical edge ridge: strong gradient in a single column
        let mut img: Array2D<u8> = Array2D::zeros(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                img[(r, c)] = if c >= 2 { 100 } else { 0 };
            }
        }
        let (horz, vert) = sobel_edge_detector(&img);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        suppress_non_maximum_edges(&horz, &vert, &mut out);
        // The single wide step has no interior maximum along '-' direction
        // neighbors, so only non-suppressed magnitudes survive; borders are 0.
        for r in 0..5 {
            assert_eq!(out[(r, 0)], 0);
            assert_eq!(out[(r, 4)], 0);
        }
        // thin ridge: strong center column between weaker columns
        let mut img2: Array2D<u8> = Array2D::zeros(5, 5);
        for r in 0..5 {
            img2[(r, 2)] = 200;
            img2[(r, 1)] = 100;
            img2[(r, 3)] = 100;
        }
        let (h2, v2) = sobel_edge_detector(&img2);
        let mut out2: Array2D<u8> = Array2D::zeros(1, 1);
        suppress_non_maximum_edges(&h2, &v2, &mut out2);
        // horizontal gradient => '|' orientation => compare columns c-1/c+1:
        // the weaker ridge center (magnitude 400) sits between the two strong
        // 800-magnitude columns and is suppressed; the strong columns survive.
        for r in 1..4 {
            assert_eq!(out2[(r, 2)], 0, "ridge center not suppressed at row {r}");
            assert!(out2[(r, 1)] > 0, "strong column suppressed at row {r}");
            assert!(out2[(r, 3)] > 0, "strong column suppressed at row {r}");
        }
    }

    #[test]
    fn normalize_gradients_unit_length() {
        let mut img1: Array2D<f64> = Array2D::zeros(2, 2);
        let mut img2: Array2D<f64> = Array2D::zeros(2, 2);
        img1[(0, 0)] = 3.0;
        img2[(0, 0)] = 4.0;
        img1[(1, 1)] = 0.0;
        img2[(1, 1)] = 0.0;
        normalize_image_gradients(&mut img1, &mut img2);
        assert!((img1[(0, 0)] - 0.6).abs() < 1e-12);
        assert!((img2[(0, 0)] - 0.8).abs() < 1e-12);
        assert_eq!(img1[(1, 1)], 0.0);
        assert_eq!(img2[(1, 1)], 0.0);
    }
}

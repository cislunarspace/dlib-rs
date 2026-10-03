//! Port of `dlib/image_transforms/assign_image.h` (`assign_image`,
//! `assign_image_scaled`, `assign_all_pixels`, `assign_border_pixels`,
//! `zero_border_pixels`).

use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, get_pixel_intensity, Pixel, PixelValue};

/// Port of `assign_image(dest, src)` (dlib/image_transforms/assign_image.h):
/// resizes `dest` to the source dimensions and converts each pixel with
/// `assign_pixel`.
pub fn assign_image<D, S>(dest: &mut D, src: &S)
where
    D: GenericImage,
    D::PixelType: Pixel,
    S: GenericImage,
    S::PixelType: Pixel,
{
    dest.set_image_size(src.num_rows(), src.num_columns());
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            assign_pixel(dest.pixel_mut(r, c), src.pixel(r, c));
        }
    }
}

/// dlib `pixel_traits<P>::max()` / `min()` as doubles (basic pixel range).
fn traits_range<P: Pixel>() -> (f64, f64) {
    match P::max_pixel().to_value() {
        PixelValue::U8(_) => (0.0, 255.0),
        PixelValue::U16(_) => (0.0, 65535.0),
        PixelValue::I32(v) => (f64::from(-v), f64::from(v)),
        _ => (0.0, 1.0),
    }
}

/// Port of `assign_image_scaled(dest, src, thresh = 4)`
/// (dlib/image_transforms/assign_image.h).
///
/// If the destination dynamic range covers the source pixel range (or, for
/// integer sources, the observed intensity range), this is a plain
/// `assign_image`. Otherwise the intensities are linearly remapped from
/// `[lower, upper]` (with `upper/lower = mean +/- thresh*stddev` clamped to the
/// observed min/max) onto the destination range.
pub fn assign_image_scaled<D, S>(dest: &mut D, src: &S, thresh: f64)
where
    D: GenericImage,
    D::PixelType: Pixel,
    S: GenericImage,
    S::PixelType: Pixel,
{
    assert!(thresh > 0.0, "assign_image_scaled(): thresh must be > 0");

    let (dest_min, dest_max) = traits_range::<D::PixelType>();
    let (src_min, src_max) = traits_range::<S::PixelType>();

    // If the destination has a dynamic range big enough to contain the source
    // image data then just do a regular assign_image().
    if dest_max >= src_max && dest_min <= src_min {
        assign_image(dest, src);
        return;
    }

    dest.set_image_size(src.num_rows(), src.num_columns());

    let size = src.num_rows() * src.num_columns();
    if size == 0 || size == 1 {
        assign_image(dest, src);
        return;
    }

    // gather image statistics (dlib running_stats<double>: sum/sum_sq based)
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            let v = get_pixel_intensity(src.pixel(r, c));
            sum += v;
            sum_sq += v * v;
            min = min.min(v);
            max = max.max(v);
        }
    }
    let n = size as f64;
    let mean = sum / n;
    let variance = ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0);
    let stddev = variance.sqrt();

    if !<S::PixelType as Pixel>::is_float() && dest_max >= max && dest_min <= min {
        assign_image(dest, src);
        return;
    }

    // Don't let huge outliers dictate the whole range.
    let upper = (mean + thresh * stddev).min(max);
    let lower = (mean - thresh * stddev).max(min);

    let scale = if upper != lower {
        (dest_max - dest_min) / (upper - lower)
    } else {
        0.0
    };

    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            let val = get_pixel_intensity(src.pixel(r, c)) - lower;
            let out = scale * val + dest_min;
            assign_pixel(dest.pixel_mut(r, c), &out);
        }
    }
}

/// Port of `assign_image_scaled(dest, src)` with dlib's default `thresh = 4`.
pub fn assign_image_scaled_default<D, S>(dest: &mut D, src: &S)
where
    D: GenericImage,
    D::PixelType: Pixel,
    S: GenericImage,
    S::PixelType: Pixel,
{
    assign_image_scaled(dest, src, 4.0);
}

/// Port of `assign_all_pixels(dest_img, src_pixel)`
/// (dlib/image_transforms/assign_image.h).
pub fn assign_all_pixels<D, P>(dest: &mut D, pixel: &P)
where
    D: GenericImage,
    D::PixelType: Pixel,
    P: Pixel,
{
    for r in 0..dest.num_rows() {
        for c in 0..dest.num_columns() {
            assign_pixel(dest.pixel_mut(r, c), pixel);
        }
    }
}

/// Port of `assign_border_pixels(img, x_border_size, y_border_size, p)`
/// (dlib/image_transforms/assign_image.h). `x_border_size` trims columns,
/// `y_border_size` trims rows; both are clamped to `dim/2 + 1` like the C++.
pub fn assign_border_pixels<D>(
    img: &mut D,
    x_border_size: usize,
    y_border_size: usize,
    p: &D::PixelType,
) where
    D: GenericImage,
    D::PixelType: Pixel,
{
    let nr = img.num_rows();
    let nc = img.num_columns();
    let y_border_size = y_border_size.min(nr / 2 + 1);
    let x_border_size = x_border_size.min(nc / 2 + 1);

    // assign the top border
    for r in 0..y_border_size {
        for c in 0..nc {
            *img.pixel_mut(r, c) = *p;
        }
    }

    // assign the bottom border
    for r in nr - y_border_size..nr {
        for c in 0..nc {
            *img.pixel_mut(r, c) = *p;
        }
    }

    // now assign the two sides
    for r in y_border_size..nr - y_border_size {
        for c in 0..x_border_size {
            *img.pixel_mut(r, c) = *p;
        }
        for c in nc - x_border_size..nc {
            *img.pixel_mut(r, c) = *p;
        }
    }
}

/// Port of `zero_border_pixels(img, x_border_size, y_border_size)`
/// (dlib/image_transforms/assign_image.h).
pub fn zero_border_pixels<D>(img: &mut D, x_border_size: usize, y_border_size: usize)
where
    D: GenericImage,
    D::PixelType: Pixel,
{
    let mut zero_pixel = D::PixelType::default();
    assign_pixel(&mut zero_pixel, &0i32);
    assign_border_pixels(img, x_border_size, y_border_size, &zero_pixel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn test_assign_image_copies_and_converts() {
        let mut src: Array2D<u16> = Array2D::zeros(2, 3);
        for r in 0..2 {
            for c in 0..3 {
                *src.get_mut(r, c) = (1000 + r * 10 + c) as u16;
            }
        }
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        assign_image(&mut dst, &src);
        assert_eq!(dst.nr(), 2);
        assert_eq!(dst.nc(), 3);
        // u16 -> u8 assign_pixel clamps values > 255
        assert_eq!(*dst.get(0, 0), 255);
    }

    #[test]
    fn test_assign_image_scaled_maps_min_max_to_dest_range() {
        // linear ramp 0..65535 in u16; dest u8 can't hold it
        let nr = 16;
        let nc = 16;
        let mut src: Array2D<u16> = Array2D::zeros(nr, nc);
        for r in 0..nr {
            for c in 0..nc {
                *src.get_mut(r, c) = ((r * nc + c) * 65535 / (nr * nc - 1)) as u16;
            }
        }
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        assign_image_scaled_default(&mut dst, &src);
        assert_eq!(dst.nr(), nr);
        assert_eq!(dst.nc(), nc);
        assert_eq!(*dst.get(0, 0), 0);
        assert_eq!(*dst.get(nr - 1, nc - 1), 255);
        // monotone mapping
        let first = *dst.get(0, 1);
        let last = *dst.get(nr - 1, nc - 2);
        assert!(first <= last);
    }

    #[test]
    fn test_assign_image_scaled_plain_when_range_fits() {
        let mut src: Array2D<u8> = Array2D::zeros(2, 2);
        *src.get_mut(0, 0) = 10;
        *src.get_mut(0, 1) = 20;
        *src.get_mut(1, 0) = 30;
        *src.get_mut(1, 1) = 240;
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        assign_image_scaled_default(&mut dst, &src);
        assert_eq!(*dst.get(0, 0), 10);
        assert_eq!(*dst.get(1, 1), 240);
    }

    #[test]
    fn test_assign_all_and_border_pixels() {
        let mut img: Array2D<u8> = Array2D::zeros(5, 7);
        assign_all_pixels(&mut img, &9u8);
        for r in 0..5 {
            for c in 0..7 {
                assert_eq!(*img.get(r, c), 9);
            }
        }
        assign_border_pixels(&mut img, 2, 1, &1u8);
        assert_eq!(*img.get(0, 0), 1);
        assert_eq!(*img.get(0, 6), 1);
        assert_eq!(*img.get(4, 6), 1);
        assert_eq!(*img.get(2, 2), 9);
        assert_eq!(*img.get(2, 1), 1);
        zero_border_pixels(&mut img, 3, 2);
        assert_eq!(*img.get(0, 0), 0);
        assert_eq!(*img.get(2, 2), 0);
        // y border clamped to nr/2+1 = 3, so rows 0..3 and 2..5 -> all rows
        assert_eq!(*img.get(4, 6), 0);
    }
}

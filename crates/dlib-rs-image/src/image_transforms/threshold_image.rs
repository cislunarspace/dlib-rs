//! Port of `dlib/image_transforms/thresholding.h`: `threshold_image`,
//! `partition_pixels` and `auto_threshold_image` (dlib has no
//! `threshold_image.h`; the logic lives in `thresholding.h`).

use crate::array2d::GenericImage;
use crate::image_transforms::equalize_hist::get_histogram;
use crate::pixel::{assign_pixel, get_pixel_intensity, Pixel};

/// dlib's `on_pixel` constant.
pub const ON_PIXEL: u8 = 255;
/// dlib's `off_pixel` constant.
pub const OFF_PIXEL: u8 = 0;

/// Port of `dlib::threshold_image(in_img, out_img, thresh)`.
///
/// dlib's comparison is `get_pixel_intensity(in[r][c]) >= thresh` → `on_pixel`
/// (255), else `off_pixel` (0). The threshold is taken in the intensity
/// domain, so a pixel whose intensity equals `thresh` is ON.
pub fn threshold_image<S, D>(src: &S, dst: &mut D, thresh: f64)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    // if there isn't any input image then don't do anything
    if src.num_rows() * src.num_columns() == 0 {
        dst.set_image_size(0, 0);
        return;
    }

    dst.set_image_size(src.num_rows(), src.num_columns());

    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            if get_pixel_intensity(src.pixel(r, c)) >= thresh {
                assign_pixel(dst.pixel_mut(r, c), &ON_PIXEL);
            } else {
                assign_pixel(dst.pixel_mut(r, c), &OFF_PIXEL);
            }
        }
    }
}

/// Port of `dlib::threshold_image(in_img, out_img)` (auto threshold via
/// `partition_pixels`).
pub fn threshold_image_auto<S, D>(src: &S, dst: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let thresh = partition_pixels(src);
    threshold_image(src, dst, thresh as f64);
}

// ----------------------------------------------------------------------------------------

/// `total_abs(begin, thresh)` from dlib's `partition_pixels` histogram
/// overload: the sum of absolute deviations of each pixel from the mean of its
/// group if pixels were split at histogram bin `thresh`.
fn total_abs(cum_hist: &[f64], cum_histi: &[f64], begin: usize, thresh: usize) -> f64 {
    let histsum = |b: usize, e: usize| cum_hist[e] - cum_hist[b];
    let histsumi = |b: usize, e: usize| cum_histi[e] - cum_histi[b];

    let mut left_avg = histsumi(begin, thresh);
    let mut tmp = histsum(begin, thresh);
    if tmp != 0.0 {
        left_avg /= tmp;
    }
    let mut right_avg = histsumi(thresh, cum_hist.len() - 1);
    tmp = histsum(thresh, cum_hist.len() - 1);
    if tmp != 0.0 {
        right_avg /= tmp;
    }

    let left_idx = left_avg.ceil() as i64;
    let right_idx = right_avg.ceil() as i64;

    let mut score = 0.0;
    score += left_avg * histsum(begin, left_idx as usize) - histsumi(begin, left_idx as usize);
    score -= left_avg * histsum(left_idx as usize, thresh) - histsumi(left_idx as usize, thresh);
    score += right_avg * histsum(thresh, right_idx as usize) - histsumi(thresh, right_idx as usize);
    score -= right_avg * histsum(right_idx as usize, cum_hist.len() - 1)
        - histsumi(right_idx as usize, cum_hist.len() - 1);
    score
}

/// Port of `dlib::partition_pixels(img)` (histogram overload, used for images
/// whose pixels are unsigned and at most 16 bits): finds the intensity value
/// minimizing the sum of absolute deviations of a two-group split.
pub fn partition_pixels<S>(img: &S) -> u64
where
    S: GenericImage,
    S::PixelType: Pixel,
{
    let hist = get_histogram(img);

    // create integral histograms
    let n = hist.len();
    let mut cum_hist = vec![0.0f64; n + 1];
    let mut cum_histi = vec![0.0f64; n + 1];
    for i in 0..n {
        cum_hist[i + 1] = cum_hist[i] + hist[i] as f64;
        cum_histi[i + 1] = cum_histi[i] + hist[i] as f64 * i as f64;
    }

    // impl::partition_pixels_work with begin = 0, end = hist.size()
    let begin = 0usize;
    let end = n;
    let mut int_thresh = begin;
    let mut min_sad = f64::INFINITY;
    for i in begin..end {
        let sad = total_abs(&cum_hist, &cum_histi, begin, i);
        if sad <= min_sad {
            min_sad = sad;
            int_thresh = i;
        }
    }

    // dlib assigns pix_thresh = int_thresh (the histogram bin index)
    int_thresh as u64
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::auto_threshold_image(in_img, out_img)` from
/// `thresholding.h`: finds the two dominant intensity modes with k-means
/// (k = 2) on the histogram using dlib's exact integer arithmetic and puts the
/// threshold between the two means. No random numbers are involved.
pub fn auto_threshold_image<S, D>(src: &S, dst: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    // if there isn't any input image then don't do anything
    if src.num_rows() * src.num_columns() == 0 {
        dst.set_image_size(0, 0);
        return;
    }

    let hist = get_histogram(src);

    // Start our two means (a and b) out at the ends of the histogram
    let mut a: i64 = 0;
    let mut b: i64 = hist.len() as i64 - 1;
    let mut moved_a = true;
    let mut moved_b = true;
    while moved_a || moved_b {
        moved_a = false;
        moved_b = false;

        // catch the degenerate case where the histogram is empty
        if a >= b {
            break;
        }

        if hist[a as usize] == 0 {
            a += 1;
            moved_a = true;
        }

        if hist[b as usize] == 0 {
            b -= 1;
            moved_b = true;
        }
    }

    // now do k-means clustering with k = 2 on the histogram.
    moved_a = true;
    moved_b = true;
    while moved_a || moved_b {
        moved_a = false;
        moved_b = false;

        let mut a_hits: i64 = 0;
        let mut b_hits: i64 = 0;
        let mut a_mass: i64 = 0;
        let mut b_mass: i64 = 0;

        for i in 0..hist.len() as i64 {
            // if i is closer to a
            if (i - a).abs() < (i - b).abs() {
                a_mass += hist[i as usize] as i64 * i;
                a_hits += hist[i as usize] as i64;
            } else {
                // if i is closer to b
                b_mass += hist[i as usize] as i64 * i;
                b_hits += hist[i as usize] as i64;
            }
        }

        let new_a = (a_mass + a_hits / 2) / a_hits;
        let new_b = (b_mass + b_hits / 2) / b_hits;

        if new_a != a {
            moved_a = true;
            a = new_a;
        }

        if new_b != b {
            moved_b = true;
            b = new_b;
        }
    }

    // put the threshold between the two means we found
    let thresh = (a + b) / 2;

    // now actually apply the threshold
    threshold_image(src, dst, thresh as f64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn threshold_boundary_value_is_on() {
        // dlib uses >= : intensity equal to the threshold goes to on_pixel
        let mut src: Array2D<u8> = Array2D::zeros(1, 3);
        src[(0, 0)] = 99;
        src[(0, 1)] = 100;
        src[(0, 2)] = 101;
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        threshold_image(&src, &mut dst, 100.0);
        assert_eq!(dst[(0, 0)], 0);
        assert_eq!(dst[(0, 1)], 255);
        assert_eq!(dst[(0, 2)], 255);
    }

    #[test]
    fn auto_threshold_splits_two_modes() {
        let mut src: Array2D<u8> = Array2D::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                src[(r, c)] = if (r * 4 + c) % 3 == 0 { 20 } else { 200 };
            }
        }
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        auto_threshold_image(&src, &mut dst);
        let mut expected: Array2D<u8> = Array2D::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                expected[(r, c)] = if (r * 4 + c) % 3 == 0 { 0 } else { 255 };
            }
        }
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dst[(r, c)], expected[(r, c)]);
            }
        }
    }

    #[test]
    fn partition_pixels_two_value_image() {
        let mut src: Array2D<u8> = Array2D::zeros(2, 4);
        for r in 0..2 {
            for c in 0..4 {
                src[(r, c)] = if (r + c) % 2 == 0 { 10 } else { 90 };
            }
        }
        // split between the two modes; exact value follows dlib's minimum-SAD
        let t = partition_pixels(&src);
        assert!((10..=90).contains(&t));
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        threshold_image_auto(&src, &mut dst);
        for r in 0..2 {
            for c in 0..4 {
                assert_eq!(dst[(r, c)], if (r + c) % 2 == 0 { 0 } else { 255 });
            }
        }
    }
}

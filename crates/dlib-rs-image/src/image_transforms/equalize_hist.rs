//! Port of `dlib/image_transforms/equalize_histogram.h`: `get_histogram` and
//! `equalize_histogram` with dlib's exact integer/float scaling math.

use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, assign_pixel_intensity, get_pixel_intensity, Pixel};

/// Port of `dlib::get_histogram(in_img, hist)`: histogram of pixel intensities
/// over `[0, max_pixel]` of the input pixel type (256 bins for u8 images,
/// 65536 for u16, etc.).
pub fn get_histogram<S>(img: &S) -> Vec<u64>
where
    S: GenericImage,
    S::PixelType: Pixel,
{
    let hist_size = get_pixel_intensity(&S::PixelType::max_pixel()) as usize + 1;
    let mut hist = vec![0u64; hist_size];
    for r in 0..img.num_rows() {
        for c in 0..img.num_columns() {
            let p = get_pixel_intensity(img.pixel(r, c));
            hist[p as usize] += 1;
        }
    }
    hist
}

/// Port of `dlib::equalize_histogram(in_img, out_img)`.
///
/// dlib's exact algorithm:
/// 1. build the intensity histogram (black pixels are excluded afterwards by
///    zeroing `hist[0]` and dividing by `size - hist[0]`);
/// 2. turn it into a cumulative histogram;
/// 3. scale each bin with `scale = out_max / (size - hist[0])` truncated to
///    `unsigned long` (`as u64`);
/// 4. map each pixel by copying it and then assigning
///    `assign_pixel_intensity(out, hist[intensity(in)])`.
pub fn equalize_hist<S, D>(src: &S, dst: &mut D)
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

    let mut histogram = get_histogram(src);

    let size = (src.num_rows() * src.num_columns()) as u64;
    let mut scale = get_pixel_intensity(&D::PixelType::max_pixel());
    if size > histogram[0] {
        scale /= (size - histogram[0]) as f64;
    } else {
        scale = 0.0;
    }

    // make the black pixels remain black in the output image
    histogram[0] = 0;

    // compute the transform function
    for i in 1..histogram.len() {
        histogram[i] += histogram[i - 1];
    }
    // scale so that it is in the range [0, out max]
    for h in histogram.iter_mut() {
        *h = (*h as f64 * scale) as u64;
    }

    // now do the transform
    for r in 0..src.num_rows() {
        for c in 0..src.num_columns() {
            let p = histogram[get_pixel_intensity(src.pixel(r, c)) as usize];
            assign_pixel(dst.pixel_mut(r, c), src.pixel(r, c));
            assign_pixel_intensity(dst.pixel_mut(r, c), p as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    #[test]
    fn two_value_image_maps_to_0_and_255() {
        let mut src: Array2D<u8> = Array2D::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                src[(r, c)] = if (r + c) % 2 == 0 { 0 } else { 200 };
            }
        }
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        equalize_hist(&src, &mut dst);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dst[(r, c)], if (r + c) % 2 == 0 { 0 } else { 255 });
            }
        }
    }

    #[test]
    fn constant_nonzero_image_maps_to_max() {
        let mut src: Array2D<u8> = Array2D::zeros(3, 5);
        for r in 0..3 {
            for c in 0..5 {
                src[(r, c)] = 77;
            }
        }
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        equalize_hist(&src, &mut dst);
        // size == hist[77], so hist[0] == 0 pixels... all pixels are in bin 77,
        // size > hist[0] (=0), cumulative at 77 = size -> full scale.
        for r in 0..3 {
            for c in 0..5 {
                assert_eq!(dst[(r, c)], 255);
            }
        }
    }

    #[test]
    fn all_black_stays_black() {
        let src: Array2D<u8> = Array2D::zeros(3, 3);
        let mut dst: Array2D<u8> = Array2D::zeros(1, 1);
        equalize_hist(&src, &mut dst);
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(dst[(r, c)], 0);
            }
        }
    }

    #[test]
    fn histogram_counts_intensities() {
        let mut src: Array2D<u8> = Array2D::zeros(2, 3);
        let vals = [0u8, 1, 1, 2, 2, 255];
        let mut i = 0;
        for r in 0..2 {
            for c in 0..3 {
                src[(r, c)] = vals[i];
                i += 1;
            }
        }
        let hist = get_histogram(&src);
        assert_eq!(hist.len(), 256);
        assert_eq!(hist[0], 1);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 2);
        assert_eq!(hist[255], 1);
    }
}

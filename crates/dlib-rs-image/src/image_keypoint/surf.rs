//! SURF keypoint extraction, ported from `dlib/image_keypoint/surf.h`
//! (`get_surf_points`, `surf_point`, `compute_dominant_angle`,
//! `compute_surf_descriptor`), `dlib/image_keypoint/hessian_pyramid.h`
//! (`hessian_pyramid`, `interest_point`, `get_interest_points`) and
//! `dlib/image_transforms/integral_image.h` (`integral_image_generic`,
//! `haar_x`, `haar_y`).

use crate::array2d::Array2D;
use crate::array2d::GenericImage;
use crate::pixel::{get_pixel_intensity, Pixel, PixelValue};
use dlib_rs_core::geometry::{centered_rect, Dpoint, Point, PointRotator, Rectangle};
use dlib_rs_core::matrix::Matrix;

/// Port of `interest_point` (dlib/image_keypoint/hessian_pyramid.h).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterestPoint {
    /// Location in image coordinates.
    pub center: Dpoint,
    /// Scale of the interest point.
    pub scale: f64,
    /// Determinant-of-Hessian detection score.
    pub score: f64,
    /// Sign of the Laplacian (+1.0 or -1.0).
    pub laplacian: f64,
}

/// Port of `surf_point` (dlib/image_keypoint/surf.h): an `interest_point`
/// plus a dominant angle and a 64-dimensional descriptor. Field names follow
/// the assignment contract: `response` is dlib's `p.score` and `laplacian`
/// (i64) is the sign of dlib's `p.laplacian`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfPoint {
    /// dlib `p.center`.
    pub center: Dpoint,
    /// dlib `p.scale`.
    pub scale: f64,
    /// dlib `angle` (dominant orientation).
    pub angle: f64,
    /// dlib `des`: the length-normalized 64-dim SURF descriptor.
    pub vector: [f64; 64],
    /// dlib `p.score` (determinant of Hessian).
    pub response: f64,
    /// Sign of dlib `p.laplacian` (+1 or -1).
    pub laplacian: i64,
}

/// Port of `integral_image_generic<T>::load` (dlib/image_transforms/
/// integral_image.h) with 64-bit accumulation. Pixels are converted with dlib
/// `assign_pixel` semantics: grayscale pixels are used as-is (truncating
/// floating point values like the C++ static_cast) and rgb pixels contribute
/// their `(r+g+b)/3` (integer arithmetic, as in dlib's pixel.h).
#[derive(Clone, Debug, Default)]
pub struct IntegralImage {
    nr: usize,
    nc: usize,
    data: Vec<i64>,
}

impl IntegralImage {
    /// Port of `integral_image_generic::load`: first row/col accumulation
    /// then cumulative rows.
    pub fn load<S: GenericImage>(img: &S) -> Self
    where
        S::PixelType: Pixel,
    {
        let nr = img.num_rows();
        let nc = img.num_columns();
        let mut data = vec![0i64; nr * nc];

        // compute the first row of the integral image
        let mut temp = 0i64;
        for (cc, dc) in data.iter_mut().enumerate() {
            temp += assign_pixel_to_i64(img.pixel(0, cc));
            *dc = temp;
        }

        // now compute the rest of the integral image
        for r in 1..nr {
            let mut temp = 0i64;
            for c in 0..nc {
                temp += assign_pixel_to_i64(img.pixel(r, c));
                data[r * nc + c] = temp + data[(r - 1) * nc + c];
            }
        }

        IntegralImage { nr, nc, data }
    }

    /// Number of rows.
    pub fn nr(&self) -> usize {
        self.nr
    }

    /// Number of columns.
    pub fn nc(&self) -> usize {
        self.nc
    }

    /// The rectangle covered by this image (dlib `get_rect`).
    pub fn get_rect(&self) -> Rectangle {
        Rectangle::new(0, 0, self.nc as i64 - 1, self.nr as i64 - 1)
    }

    /// Port of `integral_image_generic::get_sum_of_area`.
    pub fn get_sum_of_area(&self, rect: &Rectangle) -> i64 {
        let br = self.data[rect.bottom() as usize * self.nc + rect.right() as usize];
        let mut tl = 0i64;
        let mut bl = 0i64;
        let mut tr = 0i64;
        if rect.left() > 0 && rect.top() > 0 {
            tl = self.data[(rect.top() - 1) as usize * self.nc + (rect.left() - 1) as usize];
            bl = self.data[rect.bottom() as usize * self.nc + (rect.left() - 1) as usize];
            tr = self.data[(rect.top() - 1) as usize * self.nc + rect.right() as usize];
        } else if rect.left() > 0 {
            bl = self.data[rect.bottom() as usize * self.nc + (rect.left() - 1) as usize];
        } else if rect.top() > 0 {
            tr = self.data[(rect.top() - 1) as usize * self.nc + rect.right() as usize];
        }
        br - bl - tr + tl
    }
}

/// dlib `assign_pixel(long, pixel)` semantics used by the integral image.
fn assign_pixel_to_i64<P: Pixel>(p: &P) -> i64 {
    if <P as Pixel>::is_rgb() {
        match p.to_value() {
            PixelValue::Rgb(v) => (v.r as i64 + v.g as i64 + v.b as i64) / 3,
            PixelValue::Rgba(v) => (v.r as i64 + v.g as i64 + v.b as i64) / 3,
            _ => unreachable!("is_rgb pixel without rgb value"),
        }
    } else {
        get_pixel_intensity(p) as i64
    }
}

/// Port of `integral_image_generic<double>::load` returned as a `Matrix<f64>`
/// (row-major accumulation identical to the integer version).
pub fn integral_image<S: GenericImage>(img: &S) -> Matrix<f64>
where
    S::PixelType: Pixel,
{
    let ii = IntegralImage::load(img);
    let vals: Vec<f64> = ii.data.iter().map(|&v| v as f64).collect();
    Matrix::from_row_vec(ii.nr, ii.nc, &vals)
}
/// Port of `haar_x` (dlib/image_transforms/integral_image.h): difference
/// between the right and left halves of a `width` x `width` square centered
/// (in dlib's sense) on point `p`.
pub fn haar_x(img: &IntegralImage, p: &Point, width: i64) -> i64 {
    let left = Rectangle::new(
        p.x() - width / 2,
        p.y() - width / 2,
        p.x() - 1,
        p.y() - width / 2 + width - 1,
    );
    let right = Rectangle::new(p.x(), left.top(), left.left() + width - 1, left.bottom());
    img.get_sum_of_area(&right) - img.get_sum_of_area(&left)
}

/// Port of `haar_y` (dlib/image_transforms/integral_image.h): difference
/// between the bottom and top halves of a `width` x `width` square centered
/// (in dlib's sense) on point `p`.
pub fn haar_y(img: &IntegralImage, p: &Point, width: i64) -> i64 {
    let top = Rectangle::new(
        p.x() - width / 2,
        p.y() - width / 2,
        p.x() - width / 2 + width - 1,
        p.y() - 1,
    );
    let bottom = Rectangle::new(top.left(), p.y(), top.right(), top.top() + width - 1);
    img.get_sum_of_area(&bottom) - img.get_sum_of_area(&top)
}

/// dlib `gaussian(x, y, sig)` (dlib/image_keypoint/surf.h).
fn gaussian(x: f64, y: f64, sig: f64) -> f64 {
    const SQRT_2_PI: f64 = 2.5066282746310002416123552393401041626930;
    1.0 / (sig * SQRT_2_PI) * (-(x * x + y * y) / (2.0 * sig * sig)).exp()
}

/// Conversion dlib uses from `vector<double,2>` to `point`:
/// `floor(v + 0.5)` per component.
fn round_to_point(p: Dpoint) -> Point {
    Point::new((p.x() + 0.5).floor() as i64, (p.y() + 0.5).floor() as i64)
}

// --------------------------------------------------------------------------------

/// Port of `hessian_pyramid` (dlib/image_keypoint/hessian_pyramid.h).
#[derive(Clone, Debug, Default)]
pub struct HessianPyramid {
    num_octaves: i64,
    num_intervals: i64,
    initial_step_size: i64,
    pyramid: Vec<Array2D<f64>>,
}

impl HessianPyramid {
    /// Port of `hessian_pyramid::build_pyramid`.
    pub fn build_pyramid(
        &mut self,
        img: &IntegralImage,
        num_octaves: i64,
        num_intervals: i64,
        initial_step_size: i64,
    ) {
        assert!(num_octaves > 0 && num_intervals > 0 && initial_step_size > 0);
        self.num_octaves = num_octaves;
        self.num_intervals = num_intervals;
        self.initial_step_size = initial_step_size;

        let img_nr = img.nr() as i64;
        let img_nc = img.nc() as i64;

        // allocate space for the pyramid
        self.pyramid = Vec::new();
        for o in 0..num_octaves {
            let step_size = self.get_step_size(o);
            for _i in 0..num_intervals {
                self.pyramid.push(Array2D::zeros(
                    (img_nr / step_size) as usize,
                    (img_nc / step_size) as usize,
                ));
            }
        }

        // now fill out the pyramid with data
        for o in 0..num_octaves {
            let step_size = self.get_step_size(o);
            for i in 0..num_intervals {
                let border_size = self.get_border_size(i) * step_size;
                let lobe_size = 2f64.powi((o + 1) as i32).round() as i64 * (i + 1) + 1;
                let area_inv = 1.0 / (3.0 * lobe_size as f64).powi(2);

                let lobe_offset = lobe_size / 2 + 1;
                let tl = Point::new(-lobe_offset, -lobe_offset);
                let tr = Point::new(lobe_offset, -lobe_offset);
                let bl = Point::new(-lobe_offset, lobe_offset);
                let br = Point::new(lobe_offset, lobe_offset);

                let mut r = border_size;
                while r < img_nr - border_size {
                    let mut c = border_size;
                    while c < img_nc - border_size {
                        let p = Point::new(c, r);

                        let mut dxx = img.get_sum_of_area(&centered_rect(
                            p.x(),
                            p.y(),
                            (lobe_size * 3) as u64,
                            (2 * lobe_size - 1) as u64,
                        )) as f64
                            - 3.0
                                * img.get_sum_of_area(&centered_rect(
                                    p.x(),
                                    p.y(),
                                    lobe_size as u64,
                                    (2 * lobe_size - 1) as u64,
                                )) as f64;

                        let mut dyy = img.get_sum_of_area(&centered_rect(
                            p.x(),
                            p.y(),
                            (2 * lobe_size - 1) as u64,
                            (lobe_size * 3) as u64,
                        )) as f64
                            - 3.0
                                * img.get_sum_of_area(&centered_rect(
                                    p.x(),
                                    p.y(),
                                    (2 * lobe_size - 1) as u64,
                                    lobe_size as u64,
                                )) as f64;

                        let corner_rect = |dp: &Point| {
                            centered_rect(
                                p.x() + dp.x(),
                                p.y() + dp.y(),
                                lobe_size as u64,
                                lobe_size as u64,
                            )
                        };
                        let mut dxy = img.get_sum_of_area(&corner_rect(&bl)) as f64
                            + img.get_sum_of_area(&corner_rect(&tr)) as f64
                            - img.get_sum_of_area(&corner_rect(&tl)) as f64
                            - img.get_sum_of_area(&corner_rect(&br)) as f64;

                        // now we normalize the filter responses
                        dxx *= area_inv;
                        dyy *= area_inv;
                        dxy *= area_inv;

                        let sign_of_laplacian = if dxx + dyy < 0.0 { -1.0 } else { 1.0 };

                        let mut determinant = dxx * dyy - 0.81 * dxy * dxy;
                        // If the determinant is negative then just blank it out.
                        if determinant < 0.0 {
                            determinant = 0.0;
                        }

                        // Save the determinant of the Hessian, packing the
                        // laplacian sign into the value.
                        *self.pyramid[(o * num_intervals + i) as usize]
                            .get_mut((r / step_size) as usize, (c / step_size) as usize) =
                            sign_of_laplacian * determinant;

                        c += step_size;
                    }
                    r += step_size;
                }
            }
        }
    }

    /// Port of `hessian_pyramid::get_border_size`.
    pub fn get_border_size(&self, interval: i64) -> i64 {
        let lobe_size = 2.0 * (interval + 1) as f64 + 1.0;
        let filter_size = 3.0 * lobe_size;
        (filter_size / 2.0).ceil() as i64
    }

    /// Port of `hessian_pyramid::get_step_size`.
    pub fn get_step_size(&self, octave: i64) -> i64 {
        self.initial_step_size * 2f64.powi(octave as i32).round() as i64
    }

    /// Port of `hessian_pyramid::nr(octave)`.
    pub fn nr(&self, octave: i64) -> i64 {
        self.pyramid[(self.num_intervals * octave) as usize].nr() as i64
    }

    /// Port of `hessian_pyramid::nc(octave)`.
    pub fn nc(&self, octave: i64) -> i64 {
        self.pyramid[(self.num_intervals * octave) as usize].nc() as i64
    }

    /// Port of `hessian_pyramid::get_value`.
    pub fn get_value(&self, octave: i64, interval: i64, r: i64, c: i64) -> f64 {
        self.pyramid[(self.num_intervals * octave + interval) as usize]
            .get(r as usize, c as usize)
            .abs()
    }

    /// Port of `hessian_pyramid::get_laplacian`.
    pub fn get_laplacian(&self, octave: i64, interval: i64, r: i64, c: i64) -> f64 {
        if *self.pyramid[(self.num_intervals * octave + interval) as usize]
            .get(r as usize, c as usize)
            > 0.0
        {
            1.0
        } else {
            -1.0
        }
    }

    /// Port of `hessian_pyramid::octaves`.
    pub fn octaves(&self) -> i64 {
        self.num_octaves
    }

    /// Port of `hessian_pyramid::intervals`.
    pub fn intervals(&self) -> i64 {
        self.num_intervals
    }
}

// --------------------------------------------------------------------------------

/// Port of `hessian_pyramid_helpers::is_maximum_in_region`.
fn is_maximum_in_region(pyr: &HessianPyramid, o: i64, i: i64, r: i64, c: i64) -> bool {
    // First check if this point is near the edge of the octave.  If it is
    // then we say it isn't a maximum as these points are not as reliable.
    if i <= 0 || i + 1 >= pyr.intervals() {
        return false;
    }

    let val = pyr.get_value(o, i, r, c);

    for ii in i - 1..=i + 1 {
        for rr in r - 1..=r + 1 {
            for cc in c - 1..=c + 1 {
                if pyr.get_value(o, ii, rr, cc) > val {
                    return false;
                }
            }
        }
    }
    true
}

/// Solves the 3x3 system `a * x = b` (dlib uses `inv(hess) * grad`).
fn solve3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    // Gaussian elimination with partial pivoting on an augmented matrix.
    let mut m = [a[0], a[1], a[2]];
    let mut rhs = *b;
    for col in 0..3 {
        // pivot
        let mut piv = col;
        for r in col + 1..3 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        rhs.swap(col, piv);
        let mcol = m[col];
        let d = mcol[col];
        for r in col + 1..3 {
            let f = m[r][col] / d;
            for c in col..3 {
                m[r][c] -= f * mcol[c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut x = [0.0f64; 3];
    for r in (0..3).rev() {
        let mut s = rhs[r];
        for c in r + 1..3 {
            s -= m[r][c] * x[c];
        }
        x[r] = s / m[r][r];
    }
    x
}

/// Port of `hessian_pyramid_helpers::interpolate_point`.
fn interpolate_point(pyr: &HessianPyramid, o: i64, i: i64, r: i64, c: i64) -> InterestPoint {
    let val = pyr.get_value(o, i, r, c);
    let dx = (pyr.get_value(o, i, r, c + 1) - pyr.get_value(o, i, r, c - 1)) / 2.0;
    let dy = (pyr.get_value(o, i, r + 1, c) - pyr.get_value(o, i, r - 1, c)) / 2.0;
    let ds = (pyr.get_value(o, i + 1, r, c) - pyr.get_value(o, i - 1, r, c)) / 2.0;

    let dxx = (pyr.get_value(o, i, r, c + 1) + pyr.get_value(o, i, r, c - 1)) - 2.0 * val;
    let dyy = (pyr.get_value(o, i, r + 1, c) + pyr.get_value(o, i, r - 1, c)) - 2.0 * val;
    let dss = (pyr.get_value(o, i + 1, r, c) + pyr.get_value(o, i - 1, r, c)) - 2.0 * val;
    let dxy = (pyr.get_value(o, i, r + 1, c + 1) + pyr.get_value(o, i, r - 1, c - 1)
        - pyr.get_value(o, i, r - 1, c + 1)
        - pyr.get_value(o, i, r + 1, c - 1))
        / 4.0;
    let dxs = (pyr.get_value(o, i + 1, r, c + 1) + pyr.get_value(o, i - 1, r, c - 1)
        - pyr.get_value(o, i - 1, r, c + 1)
        - pyr.get_value(o, i + 1, r, c - 1))
        / 4.0;
    let dys = (pyr.get_value(o, i + 1, r + 1, c) + pyr.get_value(o, i - 1, r - 1, c)
        - pyr.get_value(o, i - 1, r + 1, c)
        - pyr.get_value(o, i + 1, r - 1, c))
        / 4.0;

    let hess = [[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]];
    let ip = solve3(&hess, &[-dx, -dy, -ds]);

    let mut temp = InterestPoint {
        center: Dpoint::new(0.0, 0.0),
        scale: 0.0,
        score: 0.0,
        laplacian: 0.0,
    };
    if ip[0].abs() < 0.5 && ip[1].abs() < 0.5 && ip[2].abs() < 0.5 {
        let p = Dpoint::new(
            (c as f64 + ip[0]) * pyr.get_step_size(o) as f64,
            (r as f64 + ip[1]) * pyr.get_step_size(o) as f64,
        );
        let lobe_size = 2f64.powi((o + 1) as i32) * (i as f64 + ip[2] + 1.0) + 1.0;
        let filter_size = 3.0 * lobe_size;
        let scale = 1.2 / 9.0 * filter_size;

        temp.center = p;
        temp.scale = scale;
        temp.score = val;
        temp.laplacian = pyr.get_laplacian(o, i, r, c);
    } else {
        // this indicates to the caller that no interest point was found.
        temp.score = -1.0;
    }
    temp
}

/// Port of `get_interest_points` (dlib/image_keypoint/hessian_pyramid.h):
/// non-maximum suppression over the pyramid followed by 3D quadratic
/// interpolation of position/scale.
pub fn get_interest_points(pyr: &HessianPyramid, threshold: f64) -> Vec<InterestPoint> {
    assert!(threshold >= 0.0);
    let mut result_points = Vec::new();

    for o in 0..pyr.octaves() {
        let nr = pyr.nr(o);
        let nc = pyr.nc(o);

        // do non-maximum suppression on all the intervals in the current
        // octave and accumulate the results in result_points
        for i in 1..pyr.intervals() - 1 {
            let border_size = pyr.get_border_size(i + 1);
            for r in border_size + 1..nr - border_size - 1 {
                for c in border_size + 1..nc - border_size - 1 {
                    let max_val = pyr.get_value(o, i, r, c);

                    // If the max point we found is really a maximum in its
                    // own region and is big enough then add it to the results.
                    if max_val >= threshold && is_maximum_in_region(pyr, o, i, r, c) {
                        let sp = interpolate_point(pyr, o, i, r, c);
                        if sp.score >= threshold {
                            result_points.push(sp);
                        }
                    }
                }
            }
        }
    }

    result_points
}

// --------------------------------------------------------------------------------

/// Port of `compute_dominant_angle` (dlib/image_keypoint/surf.h).
fn compute_dominant_angle(img: &IntegralImage, center: &Dpoint, scale: f64) -> f64 {
    let sc = scale.round() as i64;

    let mut ang: Vec<f64> = Vec::new();
    let mut samples: Vec<Dpoint> = Vec::new();

    // accumulate a bunch of angle and vector samples
    for r in -6i64..=6 {
        for c in -6i64..=6 {
            if r * r + c * c < 36 {
                // compute a Gaussian weighted gradient and the gradient's angle.
                let gauss = gaussian(c as f64, r as f64, 2.5);
                let p = round_to_point(Dpoint::new(
                    center.x() + (sc * c) as f64,
                    center.y() + (sc * r) as f64,
                ));
                let vect = Dpoint::new(
                    gauss * haar_x(img, &p, 4 * sc) as f64,
                    gauss * haar_y(img, &p, 4 * sc) as f64,
                );
                samples.push(vect);
                ang.push(vect.y().atan2(vect.x()));
            }
        }
    }

    // now find the dominant direction
    let mut max_length = 0.0f64;
    let mut best_ang = 0.0f64;
    // look at a bunch of pie shaped slices of a circle
    let slices = 45i64;
    let ang_step = 2.0 * std::f64::consts::PI / slices as f64;
    for ang_i in 0..slices {
        // compute the bounding angles
        let ang1 = ang_step * ang_i as f64 - std::f64::consts::PI;
        let ang2 = ang1 + std::f64::consts::PI / 3.0;

        // compute sum of all vectors that are within the above two angles
        let mut vx = 0.0f64;
        let mut vy = 0.0f64;
        for (a, s) in ang.iter().zip(samples.iter()) {
            if (ang1 <= *a && *a <= ang2)
                || (ang2 > std::f64::consts::PI
                    && (*a >= ang1 || *a <= (-2.0 * std::f64::consts::PI + ang2)))
            {
                vx += s.x();
                vy += s.y();
            }
        }

        // record the angle of the best vectors
        if vx * vx + vy * vy > max_length {
            max_length = vx * vx + vy * vy;
            best_ang = vy.atan2(vx);
        }
    }

    best_ang
}

/// Port of `compute_surf_descriptor` (dlib/image_keypoint/surf.h).
fn compute_surf_descriptor(
    img: &IntegralImage,
    center: &Dpoint,
    scale: f64,
    angle: f64,
) -> [f64; 64] {
    let rot = PointRotator::from_angle(angle);
    let inv_rot = PointRotator::from_angle(-angle);

    let sc = scale.round() as i64;
    let mut des = [0.0f64; 64];
    let mut count = 0usize;

    // loop over the 4x4 grid of histogram buckets
    let mut r = -10i64;
    while r < 10 {
        let mut c = -10i64;
        while c < 10 {
            let mut vect = Dpoint::new(0.0, 0.0);
            let mut abs_vect = Dpoint::new(0.0, 0.0);

            // now loop over 25 points in this bucket and sum their features.
            // Note that we include 1 pixels worth of padding around the
            // outside of each 5x5 cell.
            for y in r - 1..r + 5 + 1 {
                if !(-10..10).contains(&y) {
                    continue;
                }
                for x in c - 1..c + 5 + 1 {
                    if !(-10..10).contains(&x) {
                        continue;
                    }

                    // get the rotated point for this extraction point
                    let rotated = rot.apply(&Dpoint::new(x as f64 * scale, y as f64 * scale));
                    let p = round_to_point(Dpoint::new(
                        rotated.x() + center.x(),
                        rotated.y() + center.y(),
                    ));

                    // Give points farther from the center of the bucket a
                    // lower weight.
                    let center_r = r + 2;
                    let center_c = c + 2;
                    let weight = 1.0 / (4 + (center_r - y).abs() + (center_c - x).abs()) as f64;

                    let tx = weight * haar_x(img, &p, 2 * sc) as f64;
                    let ty = weight * haar_y(img, &p, 2 * sc) as f64;

                    // rotate this vector into alignment with the surf
                    // descriptor box
                    let temp = inv_rot.apply(&Dpoint::new(tx, ty));

                    vect = Dpoint::new(vect.x() + temp.x(), vect.y() + temp.y());
                    abs_vect =
                        Dpoint::new(abs_vect.x() + temp.x().abs(), abs_vect.y() + temp.y().abs());
                }
            }

            des[count] = vect.x();
            des[count + 1] = vect.y();
            des[count + 2] = abs_vect.x();
            des[count + 3] = abs_vect.y();
            count += 4;

            c += 5;
        }
        r += 5;
    }

    // Return the length normalized descriptor.  Add a small number
    // to guard against division by zero.
    let len = des.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-7;
    for v in des.iter_mut() {
        *v /= len;
    }
    des
}

/// Port of `get_surf_points` with dlib's defaults
/// (`max_points = 10000`, `detection_threshold = 30.0`, pyramid built with 4
/// octaves, 6 intervals, initial step size 2).
pub fn get_surf_points<S: GenericImage>(img: &S) -> Vec<SurfPoint>
where
    S::PixelType: Pixel,
{
    get_surf_points_opts(img, 10000, 30.0)
}

/// Port of `get_surf_points` (dlib/image_keypoint/surf.h) with explicit
/// `max_points` and `detection_threshold`.
pub fn get_surf_points_opts<S: GenericImage>(
    img: &S,
    max_points: i64,
    detection_threshold: f64,
) -> Vec<SurfPoint>
where
    S::PixelType: Pixel,
{
    assert!(max_points > 0 && detection_threshold >= 0.0);

    // make an integral image first
    let int_img = IntegralImage::load(img);

    // now make a hessian pyramid
    let mut pyr = HessianPyramid::default();
    pyr.build_pyramid(&int_img, 4, 6, 2);

    // now get all the interest points from the hessian pyramid
    let mut points = get_interest_points(&pyr, detection_threshold);
    let mut spoints = Vec::new();

    // sort all the points by how strong their detect is
    points.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // now extract SURF descriptors for the points
    for point in points.iter().take(max_points as usize) {
        // ignore points that are close to the edge of the image
        let border = 32.0f64;
        let border_size = (border * point.scale) as u64;
        let rect = centered_rect(
            round_to_point(point.center).x(),
            round_to_point(point.center).y(),
            border_size,
            border_size,
        );
        if int_img.get_rect().contains_rect(&rect) {
            let angle = compute_dominant_angle(&int_img, &point.center, point.scale);
            let vector = compute_surf_descriptor(&int_img, &point.center, point.scale, angle);
            spoints.push(SurfPoint {
                center: point.center,
                scale: point.scale,
                angle,
                vector,
                response: point.score,
                laplacian: if point.laplacian > 0.0 { 1 } else { -1 },
            });
        }
    }

    spoints
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    // (c) integral_image on a known 3x3 image.
    #[test]
    fn integral_image_3x3() {
        let mut img = Array2D::<u8>::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                *img.get_mut(r, c) = (3 * r + c + 1) as u8;
            }
        }
        let ii = integral_image(&img);
        assert_eq!(ii.nr(), 3);
        assert_eq!(ii.nc(), 3);
        let expected = [[1.0, 3.0, 6.0], [5.0, 12.0, 21.0], [12.0, 27.0, 45.0]];
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(ii[(r, c)], expected[r][c]);
            }
        }
    }

    // (d) A bright blob in the middle of the image is detected at its center.
    #[test]
    fn surf_detects_blob_center() {
        let nr = 100usize;
        let nc = 100usize;
        let mut img = Array2D::<u8>::zeros(nr, nc);
        for r in 0..nr {
            for c in 0..nc {
                let dr = r as f64 - 50.0;
                let dc = c as f64 - 50.0;
                let v = 255.0 * (-(dr * dr + dc * dc) / (2.0 * 4.0 * 4.0)).exp();
                *img.get_mut(r, c) = v as u8;
            }
        }
        let pts = get_surf_points(&img);
        assert!(!pts.is_empty(), "expected at least one surf point");
        let best = pts
            .iter()
            .max_by(|a, b| a.response.partial_cmp(&b.response).unwrap())
            .unwrap();
        assert!(best.response > 0.0);
        assert!(
            (best.center.x() - 50.0).abs() <= 1.0 && (best.center.y() - 50.0).abs() <= 1.0,
            "center: {:?}",
            best.center
        );
        // descriptor is length normalized (up to the 1e-7 guard)
        let norm: f64 = best.vector.iter().map(|v| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-6, "norm: {norm}");
        assert!(best.scale > 0.0);
    }

    // (e) Deterministic across calls.
    #[test]
    fn surf_deterministic() {
        let mut img = Array2D::<u8>::zeros(90, 90);
        for r in 0..90 {
            for c in 0..90 {
                let dr = r as f64 - 45.0;
                let dc = c as f64 - 45.0;
                let v = 200.0 * (-(dr * dr + dc * dc) / (2.0 * 7.0 * 7.0)).exp();
                *img.get_mut(r, c) = v as u8;
            }
        }
        let a = get_surf_points(&img);
        let b = get_surf_points(&img);
        assert_eq!(a, b);
    }
}

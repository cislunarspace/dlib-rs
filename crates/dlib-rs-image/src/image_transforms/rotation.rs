//! Port of the rotation/flip functions from `dlib/image_transforms/interpolation.h`
//! (`rotate_image`, `flip_image_left_right`, `flip_image_up_down`). dlib has no
//! separate rotation.h; these live in interpolation.h.

use dlib_rs_core::geometry::{
    center, dcenter, find_affine_transform, inv_affine, Dpoint, Point, PointTransformAffine,
    Rectangle,
};

use crate::array2d::GenericImage;
use crate::image_transforms::interpolation::{
    dlib_round, transform_image_black, Interpolate, InterpolateQuadratic,
};
use crate::pixel::{assign_pixel, Pixel};

/// dlib `rotate_point(center, p, angle)` instantiated with integer vectors:
/// the rotated offset is computed in doubles and stored back into a
/// `vector<long,2>`, which rounds with `floor(v + 0.5)`.
fn rotate_point_long(cx: i64, cy: i64, px: i64, py: i64, angle: f64) -> Point {
    let ca = angle.cos();
    let sa = angle.sin();
    let rx = ca * (px - cx) as f64 - sa * (py - cy) as f64;
    let ry = sa * (px - cx) as f64 + ca * (py - cy) as f64;
    Point::new(dlib_round(rx) + cx, dlib_round(ry) + cy)
}

/// Port of `rotate_image(in_img, out_img, angle, interp)`
/// (dlib/image_transforms/interpolation.h).
///
/// The output size is the bounding box of the input rectangle's four corners
/// rotated by `-angle` about the (integer) input center; each output pixel is
/// mapped back through the affine transform
/// `trans = point_transform_affine(R, -R*dcenter(out_rect) + dcenter(in_rect))`
/// with `R = rotation_matrix(angle)`, and pixels whose interpolation falls
/// outside the input get black. Returns `inv(trans)` (input -> output coords).
pub fn rotate_image_interpolate<S, D, I>(
    img: &S,
    angle: f64,
    out: &mut D,
    interp: &I,
) -> PointTransformAffine
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
    I: Interpolate,
{
    let rimg = Rectangle::new(
        0,
        0,
        img.num_columns() as i64 - 1,
        img.num_rows() as i64 - 1,
    );

    // figure out bounding box for rotated rectangle
    let c = center(&rimg);
    let (cx, cy) = (c.x(), c.y());
    let corners = [
        rimg.tl_corner(),
        rimg.tr_corner(),
        rimg.bl_corner(),
        rimg.br_corner(),
    ];
    let mut min_x = i64::MAX;
    let mut min_y = i64::MAX;
    let mut max_x = i64::MIN;
    let mut max_y = i64::MIN;
    for p in &corners {
        let q = rotate_point_long(cx, cy, p.x(), p.y(), -angle);
        min_x = min_x.min(q.x());
        min_y = min_y.min(q.y());
        max_x = max_x.max(q.x());
        max_y = max_y.max(q.y());
    }
    let rect = Rectangle::new(min_x, min_y, max_x, max_y);
    out.set_image_size(rect.height() as usize, rect.width() as usize);

    let ca = angle.cos();
    let sa = angle.sin();
    // R = rotation_matrix(angle)
    let m = [[ca, -sa], [sa, ca]];
    // b = -R*dcenter(get_rect(out_img)) + dcenter(rimg)
    let out_rect = Rectangle::new(
        0,
        0,
        out.num_columns() as i64 - 1,
        out.num_rows() as i64 - 1,
    );
    let oc = dcenter(&out_rect);
    let ic = dcenter(&rimg);
    let r_oc = Dpoint::new(
        m[0][0] * oc.x() + m[0][1] * oc.y(),
        m[1][0] * oc.x() + m[1][1] * oc.y(),
    );
    let b = Dpoint::new(ic.x() - r_oc.x(), ic.y() - r_oc.y());

    let trans = PointTransformAffine::new(m, b);
    transform_image_black(img, out, interp, |p| trans.apply(&p));
    inv_affine(&trans)
}

/// Port of `rotate_image(in_img, out_img, angle)` — dlib's default
/// `rotate_image(angle)` uses `interpolate_quadratic()`.
pub fn rotate_image<S, D>(img: &S, angle: f64, out: &mut D) -> PointTransformAffine
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    rotate_image_interpolate(img, angle, out, &InterpolateQuadratic)
}

/// Port of `flip_image_left_right(in_img, out_img)`
/// (dlib/image_transforms/interpolation.h): mirrors columns and returns the
/// affine map found by `find_affine_transform` on the corner correspondences.
pub fn flip_image_left_right<S, D>(img: &S, out: &mut D) -> PointTransformAffine
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let nr = img.num_rows();
    let nc = img.num_columns();
    out.set_image_size(nr, nc);
    for r in 0..nr {
        for c in 0..nc {
            assign_pixel(out.pixel_mut(r, c), img.pixel(r, nc - 1 - c));
        }
    }

    let rect = Rectangle::new(0, 0, nc as i64 - 1, nr as i64 - 1);
    let tl = rect.tl_corner();
    let tr = rect.tr_corner();
    let bl = rect.bl_corner();
    let br = rect.br_corner();
    let from = [
        Dpoint::new(tl.x() as f64, tl.y() as f64),
        Dpoint::new(bl.x() as f64, bl.y() as f64),
        Dpoint::new(tr.x() as f64, tr.y() as f64),
        Dpoint::new(br.x() as f64, br.y() as f64),
    ];
    let to = [
        Dpoint::new(tr.x() as f64, tr.y() as f64),
        Dpoint::new(br.x() as f64, br.y() as f64),
        Dpoint::new(tl.x() as f64, tl.y() as f64),
        Dpoint::new(bl.x() as f64, bl.y() as f64),
    ];
    find_affine_transform(&from, &to)
}

/// Port of `flip_image_up_down(in_img, out_img)`
/// (dlib/image_transforms/interpolation.h): mirrors rows.
pub fn flip_image_up_down<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let nr = img.num_rows();
    let nc = img.num_columns();
    out.set_image_size(nr, nc);
    for r in 0..nr {
        for c in 0..nc {
            assign_pixel(out.pixel_mut(r, c), img.pixel(nr - 1 - r, c));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    fn mkimg(nr: usize, nc: usize, f: impl Fn(usize, usize) -> u8) -> Array2D<u8> {
        let mut img = Array2D::zeros(nr, nc);
        for r in 0..nr {
            for c in 0..nc {
                *img.get_mut(r, c) = f(r, c);
            }
        }
        img
    }

    #[test]
    fn test_rotate_zero_angle_is_identity_interior() {
        // quadratic interpolation needs a 3x3 neighborhood, so the outermost
        // ring maps to the black background even at angle 0 (same as dlib).
        let mut img: Array2D<f64> = Array2D::zeros(4, 5);
        for r in 0..4 {
            for c in 0..5 {
                *img.get_mut(r, c) = (r * 7 + c * 11 + 3) as f64;
            }
        }
        let mut out: Array2D<f64> = Array2D::zeros(1, 1);
        let tform = rotate_image(&img, 0.0, &mut out);
        assert_eq!(out.nr(), 4);
        assert_eq!(out.nc(), 5);
        for r in 1..3 {
            for c in 1..4 {
                assert!((*out.get(r, c) - *img.get(r, c)).abs() < 1e-6);
            }
        }
        assert_eq!(*out.get(0, 0), 0.0);
        // identity transform
        let p = tform.apply(&Dpoint::new(2.0, 3.0));
        assert!((p.x() - 2.0).abs() < 1e-9 && (p.y() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_rotate_90_constant_image() {
        // 7 wide x 5 tall constant image; a 90 degree rotation transposes the
        // bounding box (5 wide, 7 tall) exactly like dlib.
        let img = mkimg(5, 7, |_, _| 42);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        rotate_image(&img, std::f64::consts::FRAC_PI_2, &mut out);
        assert_eq!(out.nr(), 7);
        assert_eq!(out.nc(), 5);
        // the center maps back onto the input center: constant survives
        assert_eq!(*out.get(3, 2), 42);
        let mut constant = 0;
        for r in 0..7 {
            for c in 0..5 {
                let v = *out.get(r, c);
                assert!(v == 42 || v == 0, "unexpected pixel {v} at ({r},{c})");
                if v == 42 {
                    constant += 1;
                }
            }
        }
        assert!(constant >= 10, "constant pixel count {constant}");
    }

    #[test]
    fn test_rotate_small_angle_constant_stays_constant_inside() {
        let img = mkimg(10, 10, |_, _| 200);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        rotate_image(&img, 0.01, &mut out);
        // interior must be exactly the constant; only some border pixels may
        // be black where the mapping leaves the image
        for r in 1..out.nr() - 1 {
            for c in 1..out.nc() - 1 {
                assert_eq!(*out.get(r, c), 200, "at ({r},{c})");
            }
        }
        for r in 0..out.nr() {
            for c in 0..out.nc() {
                let v = *out.get(r, c);
                assert!(v == 200 || v == 0);
            }
        }
    }

    #[test]
    fn test_flip_left_right() {
        let img = mkimg(3, 4, |r, c| (r * 10 + c) as u8);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        let tform = flip_image_left_right(&img, &mut out);
        assert_eq!(out.nr(), 3);
        assert_eq!(out.nc(), 4);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(out.get(r, c), img.get(r, 3 - c));
            }
        }
        // returned affine maps tl -> tr
        let p = tform.apply(&Dpoint::new(0.0, 0.0));
        assert!((p.x() - 3.0).abs() < 1e-9 && p.y().abs() < 1e-9);
    }

    #[test]
    fn test_flip_up_down() {
        let img = mkimg(3, 4, |r, c| (r * 10 + c) as u8);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        flip_image_up_down(&img, &mut out);
        assert_eq!(out.nr(), 3);
        assert_eq!(out.nc(), 4);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(out.get(r, c), img.get(2 - r, c));
            }
        }
    }
}

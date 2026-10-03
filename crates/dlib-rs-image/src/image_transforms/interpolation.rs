//! Port of `dlib/image_transforms/interpolation.h` (nearest/bilinear/quadratic
//! interpolation, `transform_image`, pyramids via `image_pyramid.h`) plus the
//! pyramid types from `dlib/image_transforms/image_pyramid.h`.
//!
//! All rounding, clamping and weight arithmetic mirrors the C++ code exactly.

use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, Pixel, PixelValue, RgbPixel};
use dlib_rs_core::geometry::{Dpoint, Drectangle};

// ----------------------------------------------------------------------------------------

/// dlib rounds double -> integral `vector` coordinates with
/// `static_cast<T>(std::floor(v + 0.5))` (dlib/geometry/vector.h).
#[inline]
pub(crate) fn dlib_round(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// `pixel_to_vector<double>` equivalent: expands a pixel into its cartesian
/// channel values. Returns the channel count (1 for grayscale, 3 for rgb).
/// dlib's interpolators statically reject alpha and non-cartesian (hsi/lab)
/// pixels; we do the same at run time.
pub(crate) fn pixel_to_channels<P: Pixel>(p: &P, ch: &mut [f64; 4]) -> usize {
    match p.to_value() {
        PixelValue::U8(v) => {
            ch[0] = f64::from(v);
            1
        }
        PixelValue::U16(v) => {
            ch[0] = f64::from(v);
            1
        }
        PixelValue::I32(v) => {
            ch[0] = f64::from(v);
            1
        }
        PixelValue::F32(v) => {
            ch[0] = f64::from(v);
            1
        }
        PixelValue::F64(v) => {
            ch[0] = v;
            1
        }
        PixelValue::Rgb(v) => {
            ch[0] = f64::from(v.r);
            ch[1] = f64::from(v.g);
            ch[2] = f64::from(v.b);
            3
        }
        PixelValue::Rgba(_) | PixelValue::Hsi(_) | PixelValue::Lab(_) => {
            panic!("interpolation supports only cartesian non-alpha pixels")
        }
    }
}

/// `vector_to_pixel` equivalent for a pixel of the same type described by
/// `proto`: each channel goes through a truncating `static_cast` exactly like
/// dlib's `vector_to_pixel_helper` (dlib/matrix/matrix_utilities.h).
pub(crate) fn pixel_from_channels<P: Pixel>(proto: &P, ch: &[f64; 4]) -> P {
    let value = match proto.to_value() {
        PixelValue::U8(_) => PixelValue::U8(ch[0] as u8),
        PixelValue::U16(_) => PixelValue::U16(ch[0] as u16),
        PixelValue::I32(_) => PixelValue::I32(ch[0] as i32),
        PixelValue::F32(_) => PixelValue::F32(ch[0] as f32),
        PixelValue::F64(_) => PixelValue::F64(ch[0]),
        PixelValue::Rgb(_) => PixelValue::Rgb(RgbPixel {
            r: ch[0] as u8,
            g: ch[1] as u8,
            b: ch[2] as u8,
        }),
        PixelValue::Rgba(_) | PixelValue::Hsi(_) | PixelValue::Lab(_) => {
            panic!("interpolation supports only cartesian non-alpha pixels")
        }
    };
    let mut out = P::default();
    out.assign_from_value(&value);
    out
}

/// Truncating static-cast of a clamped f64 into the integer range described by
/// `proto` (used to reproduce `vector_to_pixel` after dlib's `clamp`).
pub(crate) fn clamp_to_proto_range<P: Pixel>(proto: &P, v: f64) -> f64 {
    let (lo, hi) = match proto.to_value() {
        PixelValue::U8(_) => (0.0, 255.0),
        PixelValue::U16(_) => (0.0, 65535.0),
        PixelValue::I32(_) => (i32::MIN as f64, i32::MAX as f64),
        _ => (f64::MIN, f64::MAX),
    };
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

// ----------------------------------------------------------------------------------------

/// Trait shared by the interpolator functors, mirroring the
/// `interpolation_type` template parameter of `transform_image`
/// (dlib/image_transforms/interpolation.h).
pub trait Interpolate {
    /// Writes the interpolated pixel at `p` into `result`, returning `false`
    /// when `p` falls outside the safely interpolatable region.
    fn interpolate_image<S, P>(&self, img: &S, p: Dpoint, result: &mut P) -> bool
    where
        S: GenericImage,
        S::PixelType: Pixel,
        P: Pixel;
}

/// Port of `interpolate_nearest_neighbor`
/// (dlib/image_transforms/interpolation.h).
#[derive(Clone, Copy, Debug, Default)]
pub struct InterpolateNearestNeighbor;

impl InterpolateNearestNeighbor {
    /// Integer-coordinate form matching dlib's `operator()(img, point, result)`.
    pub fn interpolate_at<S, P>(&self, img: &S, x: i64, y: i64, result: &mut P) -> bool
    where
        S: GenericImage,
        S::PixelType: Pixel,
        P: Pixel,
    {
        if x >= 0
            && y >= 0
            && (x as u64) < img.num_columns() as u64
            && (y as u64) < img.num_rows() as u64
        {
            assign_pixel(result, img.pixel(y as usize, x as usize));
            true
        } else {
            false
        }
    }
}

impl Interpolate for InterpolateNearestNeighbor {
    fn interpolate_image<S, P>(&self, img: &S, p: Dpoint, result: &mut P) -> bool
    where
        S: GenericImage,
        S::PixelType: Pixel,
        P: Pixel,
    {
        // dlib::point from dpoint rounds each coordinate (floor(v + 0.5)).
        let x = dlib_round(p.x());
        let y = dlib_round(p.y());
        self.interpolate_at(img, x, y, result)
    }
}

/// Port of `nearest_interpolate` style free-function access.
pub fn nearest_interpolate<S, P>(img: &S, x: i64, y: i64, result: &mut P) -> bool
where
    S: GenericImage,
    S::PixelType: Pixel,
    P: Pixel,
{
    InterpolateNearestNeighbor.interpolate_at(img, x, y, result)
}

/// Port of `interpolate_bilinear`
/// (dlib/image_transforms/interpolation.h).
#[derive(Clone, Copy, Debug, Default)]
pub struct InterpolateBilinear;

impl Interpolate for InterpolateBilinear {
    fn interpolate_image<S, P>(&self, img: &S, p: Dpoint, result: &mut P) -> bool
    where
        S: GenericImage,
        S::PixelType: Pixel,
        P: Pixel,
    {
        let in_nr = img.num_rows() as i64;
        let in_nc = img.num_columns() as i64;

        let left = p.x().floor() as i64;
        let top = p.y().floor() as i64;
        let right = left + 1;
        let bottom = top + 1;

        // if the interpolation goes outside img
        if !(left >= 0 && top >= 0 && right < in_nc && bottom < in_nr) {
            return false;
        }

        let lr_frac = p.x() - left as f64;
        let tb_frac = p.y() - top as f64;

        let (li, ri, ti, bi) = (left as usize, right as usize, top as usize, bottom as usize);
        let tl = img.pixel(ti, li);
        let tr = img.pixel(ti, ri);
        let bl = img.pixel(bi, li);
        let br = img.pixel(bi, ri);

        let mut tch = [0.0; 4];
        let n = pixel_to_channels(tl, &mut tch);
        let mut rch = [0.0; 4];
        pixel_to_channels(tr, &mut rch);
        let mut bch = [0.0; 4];
        pixel_to_channels(bl, &mut bch);
        let mut brch = [0.0; 4];
        pixel_to_channels(br, &mut brch);

        let mut out_ch = [0.0f64; 4];
        for i in 0..n {
            out_ch[i] = (1.0 - tb_frac) * ((1.0 - lr_frac) * tch[i] + lr_frac * rch[i])
                + tb_frac * ((1.0 - lr_frac) * bch[i] + lr_frac * brch[i]);
        }
        let temp = pixel_from_channels(tl, &out_ch);
        assign_pixel(result, &temp);
        true
    }
}

/// Free-function form of [`InterpolateBilinear`].
pub fn bilinear_interpolate<S, P>(img: &S, p: Dpoint, result: &mut P) -> bool
where
    S: GenericImage,
    S::PixelType: Pixel,
    P: Pixel,
{
    InterpolateBilinear.interpolate_image(img, p, result)
}

/// Port of `interpolate_quadratic`
/// (dlib/image_transforms/interpolation.h).
#[derive(Clone, Copy, Debug, Default)]
pub struct InterpolateQuadratic;

impl InterpolateQuadratic {
    /// dlib fits a quadratic to the 3x3 neighborhood and evaluates it at
    /// `p - pp`; the weight polynomial below is copied verbatim from the C++.
    #[allow(clippy::too_many_arguments)]
    fn interpolate9(
        &self,
        x: f64,
        y: f64,
        tl: f64,
        tm: f64,
        tr: f64,
        ml: f64,
        mm: f64,
        mr: f64,
        bl: f64,
        bm: f64,
        br: f64,
    ) -> f64 {
        let w0 = (tr + mr + br - tl - ml - bl) * 0.16666666666; // x
        let w1 = (bl + bm + br - tl - tm - tr) * 0.16666666666; // y
        let w2 = (tl + tr + ml + mr + bl + br) * 0.16666666666 - (tm + mm + bm) * 0.333333333; // x^2
        let w3 = (tl - tr - bl + br) * 0.25; // x*y
        let w4 = (tl + tm + tr + bl + bm + br) * 0.16666666666 - (ml + mm + mr) * 0.333333333; // y^2
        let w5 =
            (tm + ml + mr + bm) * 0.222222222 - (tl + tr + bl + br) * 0.11111111 + mm * 0.55555556; // 1

        w0 * x + w1 * y + w2 * x * x + w3 * x * y + w4 * y * y + w5
    }
}

impl Interpolate for InterpolateQuadratic {
    fn interpolate_image<S, P>(&self, img: &S, p: Dpoint, result: &mut P) -> bool
    where
        S: GenericImage,
        S::PixelType: Pixel,
        P: Pixel,
    {
        let in_nr = img.num_rows() as i64;
        let in_nc = img.num_columns() as i64;

        let px = dlib_round(p.x());
        let py = dlib_round(p.y());

        // get_rect(img).contains(grow_rect(pp,1))
        if !(px >= 1 && py >= 1 && px + 1 < in_nc && py + 1 < in_nr) {
            return false;
        }

        let r = py as usize;
        let c = px as usize;
        let center = img.pixel(r, c);

        let mut ch = [[0.0f64; 4]; 9];
        let ns = [
            (r - 1, c - 1),
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
            (r + 1, c + 1),
        ];
        let mut n = 0usize;
        for (i, (rr, cc)) in ns.iter().enumerate() {
            n = pixel_to_channels(img.pixel(*rr, *cc), &mut ch[i]);
        }

        let dx = p.x() - px as f64;
        let dy = p.y() - py as f64;
        let mut out_ch = [0.0f64; 4];
        for i in 0..n {
            // clamp to the source pixel type's range, then truncate through
            // vector_to_pixel, exactly like the C++.
            let v = self.interpolate9(
                dx, dy, ch[0][i], ch[1][i], ch[2][i], ch[3][i], ch[4][i], ch[5][i], ch[6][i],
                ch[7][i], ch[8][i],
            );
            out_ch[i] = clamp_to_proto_range(center, v);
        }
        let temp = pixel_from_channels(center, &out_ch);
        assign_pixel(result, &temp);
        true
    }
}

/// Free-function form of [`InterpolateQuadratic`].
pub fn quadratic_interpolate<S, P>(img: &S, p: Dpoint, result: &mut P) -> bool
where
    S: GenericImage,
    S::PixelType: Pixel,
    P: Pixel,
{
    InterpolateQuadratic.interpolate_image(img, p, result)
}

// ----------------------------------------------------------------------------------------

/// Port of `black_background` (dlib/image_transforms/interpolation.h).
pub fn black_background<P: Pixel>(p: &mut P) {
    assign_pixel(p, &0i32);
}

/// Port of `white_background` (dlib/image_transforms/interpolation.h).
pub fn white_background<P: Pixel>(p: &mut P) {
    assign_pixel(p, &255i32);
}

// ----------------------------------------------------------------------------------------

/// Port of `transform_image(in_img, out_img, interp, map_point, set_background)`
/// (dlib/image_transforms/interpolation.h): for every pixel of `out_img`, map
/// its coordinates through `map_point` and interpolate in `in_img`; pixels
/// that fall outside get the background treatment.
#[allow(clippy::too_many_arguments)]
pub fn transform_image<S, D, I, M, B>(
    in_img: &S,
    out_img: &mut D,
    interp: &I,
    map_point: M,
    set_background: B,
) where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
    I: Interpolate,
    M: Fn(Dpoint) -> Dpoint,
    B: Fn(&mut D::PixelType),
{
    for r in 0..out_img.num_rows() {
        for c in 0..out_img.num_columns() {
            let p = map_point(Dpoint::new(c as f64, r as f64));
            let dst = out_img.pixel_mut(r, c);
            if !interp.interpolate_image(in_img, p, dst) {
                set_background(dst);
            }
        }
    }
}

/// Port of `transform_image(in, out, interp, map_point)` with black background.
pub fn transform_image_black<S, D, I, M>(in_img: &S, out_img: &mut D, interp: &I, map_point: M)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
    I: Interpolate,
    M: Fn(Dpoint) -> Dpoint,
{
    transform_image(in_img, out_img, interp, map_point, black_background);
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// Pyramids (dlib/image_transforms/image_pyramid.h)
// ----------------------------------------------------------------------------------------

/// Port of `pyramid_down<N>` (dlib/image_transforms/image_pyramid.h).
///
/// - `N == 1` is `pyramid_disable` (output collapses to 0x0).
/// - `N == 2` and `N == 3` use dlib's hand-written 5x5 / 3x3 gaussian filters
///   with decimation (`pyramid_down_2_1` / `pyramid_down_3_2`).
/// - `N >= 4` is the generic template: resize the image to
///   `((N-1)*rows)/N x ((N-1)*cols)/N` (integer division, as in dlib's
///   `std::lround(((N-1)*num_rows(original))/N)` with long arithmetic) with
///   bilinear interpolation (rate `(N-1)/N`; `pyramid_down<6>` is the
///   face-detector default).
#[derive(Clone, Copy, Debug, Default)]
pub struct PyramidDown<const N: usize>;

impl<const N: usize> PyramidDown<N> {
    pub fn new() -> Self {
        PyramidDown
    }

    /// Port of `pyramid_rate(pyramid_down<N>&)`: `(N-1.0)/N`.
    pub fn rate() -> f64 {
        (N as f64 - 1.0) / N as f64
    }

    /// Output row count of one downsampling step (matches
    /// `find_pyramid_down_output_image_size`).
    pub fn nr(rows: usize) -> usize {
        Self::dim(rows)
    }

    /// Output column count of one downsampling step.
    pub fn nc(cols: usize) -> usize {
        Self::dim(cols)
    }

    fn dim(n: usize) -> usize {
        match N {
            1 => 0,
            2 => (((n as i64) - 3) / 2).max(0) as usize,
            3 => (2 * ((n as i64) - 2) / 3).max(0) as usize,
            _ => {
                // dlib: std::lround(((N-1)*num_rows(img))/N) where the
                // numerator is INTEGER arithmetic (long * long), so the
                // division truncates before lround ever sees it.
                (((N as i64 - 1) * (n as i64)) / N as i64).max(0) as usize
            }
        }
    }

    /// Port of `pyramid_down<N>::point_down`: maps a point in the input image
    /// to the corresponding point in the downsampled image.
    pub fn point_down(&self, p: &Dpoint) -> Dpoint {
        match N {
            1 => Dpoint::new(0.0, 0.0),
            2 => Dpoint::new(p.x() / 2.0 - 1.25, p.y() / 2.0 - 0.75),
            3 => {
                let ratio = 2.0 / 3.0;
                Dpoint::new(p.x() * ratio - 1.0, p.y() * ratio - 1.0)
            }
            _ => {
                let ratio = (N as f64 - 1.0) / N as f64;
                Dpoint::new((p.x() - 0.3) * ratio, (p.y() - 0.3) * ratio)
            }
        }
    }

    /// Port of `pyramid_down<N>::point_up`.
    pub fn point_up(&self, p: &Dpoint) -> Dpoint {
        match N {
            1 => Dpoint::new(0.0, 0.0),
            2 => Dpoint::new((p.x() + 1.25) * 2.0, (p.y() + 0.75) * 2.0),
            3 => {
                let ratio = 3.0 / 2.0;
                Dpoint::new(p.x() * ratio + ratio, p.y() * ratio + ratio)
            }
            _ => {
                let ratio = N as f64 / (N as f64 - 1.0);
                Dpoint::new(p.x() * ratio + 0.3, p.y() * ratio + 0.3)
            }
        }
    }

    /// Port of `pyramid_down<N>::rect_down` (single level).
    pub fn rect_down(&self, rect: &Drectangle) -> Drectangle {
        let tl = self.point_down(&rect.tl_corner());
        let br = self.point_down(&rect.br_corner());
        Drectangle::new(tl.x(), tl.y(), br.x(), br.y())
    }

    /// Port of `pyramid_down<N>::rect_up` (single level).
    pub fn rect_up(&self, rect: &Drectangle) -> Drectangle {
        let tl = self.point_up(&rect.tl_corner());
        let br = self.point_up(&rect.br_corner());
        Drectangle::new(tl.x(), tl.y(), br.x(), br.y())
    }

    /// Port of `pyramid_down<N>::operator()(in, out)` (`apply`).
    pub fn apply<S, D>(&self, img: &S, out: &mut D)
    where
        S: GenericImage,
        S::PixelType: Pixel,
        D: GenericImage,
        D::PixelType: Pixel,
    {
        match N {
            1 => out.set_image_size(0, 0),
            2 => pyramid_down_2_1_apply(img, out),
            3 => pyramid_down_3_2_apply(img, out),
            _ => {
                out.set_image_size(Self::nr(img.num_rows()), Self::nc(img.num_columns()));
                crate::image_transforms::resize_image::resize_image_into(img, out);
            }
        }
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `pyramid_up(in_img, out_img, pyr, interpolate_bilinear())`
/// (dlib/image_transforms/interpolation.h).
pub fn pyramid_up_with<S, D, const N: usize>(img: &S, out: &mut D, pyr: &PyramidDown<N>)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    if img.num_rows() * img.num_columns() == 0 {
        out.set_image_size(0, 0);
        return;
    }

    let rect = Drectangle::new(
        0.0,
        0.0,
        (img.num_columns() - 1) as f64,
        (img.num_rows() - 1) as f64,
    );
    let uprect = pyr.rect_up(&rect);
    if uprect.is_empty() {
        out.set_image_size(0, 0);
        return;
    }
    // dlib assigns the drectangle to a rectangle (lround per edge) before
    // set_image_size(bottom()+1, right()+1).
    let bottom = (uprect.bottom() + 0.5).floor() as i64;
    let right = (uprect.right() + 0.5).floor() as i64;
    out.set_image_size((bottom + 1) as usize, (right + 1) as usize);

    crate::image_transforms::resize_image::resize_image_into(img, out);
}

/// Port of `pyramid_up(img)` / `pyramid_up(in_img, out_img)`: dlib's default
/// uses `pyramid_down<2>` and bilinear interpolation.
pub fn pyramid_up<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    pyramid_up_with(img, out, &PyramidDown::<2>::new());
}

// ----------------------------------------------------------------------------------------
// pyramid_down<2> filter (dlib::impl::pyramid_down_2_1)
// ----------------------------------------------------------------------------------------

/// Reads one input pixel as the accumulated type: integer inputs go through
/// dlib's `promote` (int32) via `assign_pixel`, float inputs stay float.
#[inline]
fn accum_value<S: GenericImage>(img: &S, r: usize, c: usize, is_float: bool) -> f64
where
    S::PixelType: Pixel,
{
    if is_float {
        let mut v = 0f64;
        assign_pixel(&mut v, img.pixel(r, c));
        v
    } else {
        let mut v = 0i32;
        assign_pixel(&mut v, img.pixel(r, c));
        f64::from(v)
    }
}

/// Integer-division-preserving division: dlib divides the promoted integer
/// accumulator by `scale` with C++ integer division (truncation toward zero);
/// float accumulators divide normally.
#[inline]
fn accum_div(v: f64, scale: i64, is_float: bool) -> f64 {
    if is_float {
        v / scale as f64
    } else {
        (v as i64 / scale) as f64
    }
}

/// Writes an accumulated (possibly fractional) value into the output pixel the
/// same way dlib's `assign_pixel(down[r][c], ptype_value)` does.
fn accum_assign<D: GenericImage>(out: &mut D, r: usize, c: usize, v: f64, is_float: bool)
where
    D::PixelType: Pixel,
{
    if is_float {
        let src = v;
        assign_pixel(out.pixel_mut(r, c), &src);
    } else {
        let src = v as i32;
        assign_pixel(out.pixel_mut(r, c), &src);
    }
}

#[allow(clippy::needless_range_loop)]
fn pyramid_down_2_1_apply<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    if img.num_rows() <= 8 || img.num_columns() <= 8 {
        out.set_image_size(0, 0);
        return;
    }

    let is_float = <S::PixelType as Pixel>::is_float();
    let is_rgb_pair = <S::PixelType as Pixel>::is_rgb() && <D::PixelType as Pixel>::is_rgb();

    let nr = img.num_rows() as i64;
    let nc = img.num_columns() as i64;
    // temp_img: nr x (nc-3)/2 ; down: (nr-3)/2 x (nc-3)/2
    let temp_nc = ((nc - 3) / 2) as usize;
    let down_nr = ((nr - 3) / 2) as usize;
    let down_nc = temp_nc;
    out.set_image_size(down_nr, down_nc);

    if is_rgb_pair {
        // per-channel uint16/uint32 arithmetic, exactly like the C++ RGB overload.
        let mut temp: Vec<Vec<[i64; 3]>> = vec![vec![[0i64; 3]; temp_nc]; nr as usize];
        for r in 0..nr as usize {
            let mut oc = 0usize;
            for c in 0..temp_nc {
                let mut acc = [0i64; 3];
                for k in 0..5 {
                    let p = img.pixel(r, oc + k);
                    if let PixelValue::Rgb(v) = p.to_value() {
                        let w = match k {
                            1 | 3 => 4,
                            2 => 6,
                            _ => 1,
                        };
                        acc[0] += i64::from(v.r) * w;
                        acc[1] += i64::from(v.g) * w;
                        acc[2] += i64::from(v.b) * w;
                    } else {
                        unreachable!("is_rgb_pair checked above");
                    }
                }
                temp[r][c] = acc;
                oc += 2;
            }
        }
        // column filter
        let mut dr = 0usize;
        let mut r = 2i64;
        while r < nr - 2 {
            for c in 0..temp_nc {
                let ru = r as usize;
                let mut acc = [0i64; 3];
                let weights = [1i64, 4, 6, 4, 1];
                for (k, w) in weights.iter().enumerate() {
                    let row = match k {
                        0 => ru - 2,
                        1 => ru - 1,
                        2 => ru,
                        3 => ru + 1,
                        _ => ru + 2,
                    };
                    for ch in 0..3 {
                        acc[ch] += temp[row][c][ch] * w;
                    }
                }
                let dst = out.pixel_mut(dr, c);
                let value = PixelValue::Rgb(RgbPixel {
                    r: (acc[0] / 256) as u8,
                    g: (acc[1] / 256) as u8,
                    b: (acc[2] / 256) as u8,
                });
                dst.assign_from_value(&value);
            }
            dr += 1;
            r += 2;
        }
        return;
    }

    // generic (grayscale-promoted) path
    let mut temp: Vec<Vec<f64>> = vec![vec![0.0; temp_nc]; nr as usize];
    for r in 0..nr as usize {
        let mut oc = 0usize;
        for c in 0..temp_nc {
            let p1 = accum_value(img, r, oc, is_float);
            let p2 = accum_value(img, r, oc + 1, is_float) * 4.0;
            let p3 = accum_value(img, r, oc + 2, is_float) * 6.0;
            let p4 = accum_value(img, r, oc + 3, is_float) * 4.0;
            let p5 = accum_value(img, r, oc + 4, is_float);
            temp[r][c] = p1 + p2 + p3 + p4 + p5;
            oc += 2;
        }
    }

    let mut dr = 0usize;
    let mut r = 2i64;
    while r < nr - 2 {
        for c in 0..temp_nc {
            let ru = r as usize;
            let v = temp[ru - 2][c]
                + temp[ru - 1][c] * 4.0
                + temp[ru][c] * 6.0
                + temp[ru + 1][c] * 4.0
                + temp[ru + 2][c];
            let v = accum_div(v, 256, is_float);
            accum_assign(out, dr, c, v, is_float);
        }
        dr += 1;
        r += 2;
    }
}

// ----------------------------------------------------------------------------------------
// pyramid_down<3> filter (dlib::impl::pyramid_down_3_2)
// ----------------------------------------------------------------------------------------

/// `separable_3x3_filter_block_grayscale(block, img, r, c, 2, 12, 2)`:
/// applies [2,12,2] horizontally then vertically (no normalization).
#[allow(clippy::needless_range_loop)]
fn filter_block_3x3<S: GenericImage>(
    img: &S,
    r: usize,
    c: usize,
    nr_blk: usize,
    nc_blk: usize,
    is_float: bool,
) -> [[f64; 3]; 3]
where
    S::PixelType: Pixel,
{
    // row_filt has NR+2 rows / NC cols; horizontal [2,12,2] then vertical.
    let mut row_filt = [[0.0f64; 3]; 5];
    for rr in 0..nr_blk + 2 {
        for cc in 0..nc_blk {
            row_filt[rr][cc] = accum_value(img, r + rr - 1, c + cc - 1, is_float) * 2.0
                + accum_value(img, r + rr - 1, c + cc, is_float) * 12.0
                + accum_value(img, r + rr - 1, c + cc + 1, is_float) * 2.0;
        }
    }
    let mut block = [[0.0f64; 3]; 3];
    for rr in 0..nr_blk {
        for cc in 0..nc_blk {
            block[rr][cc] =
                row_filt[rr][cc] * 2.0 + row_filt[rr + 1][cc] * 12.0 + row_filt[rr + 2][cc] * 2.0;
        }
    }
    block
}

#[allow(clippy::needless_range_loop)]
fn pyramid_down_3_2_apply<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    if img.num_rows() <= 8 || img.num_columns() <= 8 {
        out.set_image_size(0, 0);
        return;
    }

    let is_float = <S::PixelType as Pixel>::is_float();

    let nr = img.num_rows() as i64;
    let nc = img.num_columns() as i64;
    let full_nr = 2 * ((nr - 2) / 3);
    let part_nr = (2 * (nr - 2)) / 3;
    let full_nc = 2 * ((nc - 2) / 3);
    let part_nc = (2 * (nc - 2)) / 3;
    out.set_image_size(part_nr as usize, part_nc as usize);

    // bi-linear interpolation of each filtered 3x3 block; weights (9,3,3,1)/16
    // followed by the fixed point /256 of the [2,12,2]^2 filter: (16*256).

    let mut rr = 1usize;
    let mut r = 0usize;
    while r < full_nr as usize {
        let mut cc = 1usize;
        let mut c = 0usize;
        while c < full_nc as usize {
            let b = filter_block_3x3(img, rr, cc, 3, 3, is_float);
            let w = |b: &[[f64; 3]; 3],
                     i0: usize,
                     j0: usize,
                     i1: usize,
                     j1: usize,
                     i2: usize,
                     j2: usize,
                     i3: usize,
                     j3: usize| {
                (b[i0][j0] * 9.0 + b[i1][j1] * 3.0 + b[i2][j2] * 3.0 + b[i3][j3]) / 4096.0
            };
            let v = accum_div(w(&b, 0, 0, 1, 0, 0, 1, 1, 1), 1, is_float);
            accum_assign(out, r, c, v, is_float);
            let v = accum_div(w(&b, 0, 2, 1, 2, 0, 1, 1, 1), 1, is_float);
            accum_assign(out, r, c + 1, v, is_float);
            let v = accum_div(w(&b, 2, 0, 1, 0, 2, 1, 1, 1), 1, is_float);
            accum_assign(out, r + 1, c, v, is_float);
            let v = accum_div(w(&b, 2, 2, 1, 2, 2, 1, 1, 1), 1, is_float);
            accum_assign(out, r + 1, c + 1, v, is_float);
            cc += 3;
            c += 2;
        }
        if part_nc - full_nc == 1 {
            let b = filter_block_3x3(img, rr, cc, 3, 2, is_float);
            let v = accum_div(
                (b[0][0] * 9.0 + b[1][0] * 3.0 + b[0][1] * 3.0 + b[1][1]) / 4096.0,
                1,
                is_float,
            );
            accum_assign(out, r, c, v, is_float);
            let v = accum_div(
                (b[2][0] * 9.0 + b[1][0] * 3.0 + b[2][1] * 3.0 + b[1][1]) / 4096.0,
                1,
                is_float,
            );
            accum_assign(out, r + 1, c, v, is_float);
        }
        rr += 3;
        r += 2;
    }
    if part_nr - full_nr == 1 {
        let mut cc = 1usize;
        let mut c = 0usize;
        while c < full_nc as usize {
            let b = filter_block_3x3(img, rr, cc, 2, 3, is_float);
            let v = accum_div(
                (b[0][0] * 9.0 + b[1][0] * 3.0 + b[0][1] * 3.0 + b[1][1]) / 4096.0,
                1,
                is_float,
            );
            accum_assign(out, r, c, v, is_float);
            let v = accum_div(
                (b[0][2] * 9.0 + b[1][2] * 3.0 + b[0][1] * 3.0 + b[1][1]) / 4096.0,
                1,
                is_float,
            );
            accum_assign(out, r, c + 1, v, is_float);
            cc += 3;
            c += 2;
        }
        if part_nc - full_nc == 1 {
            let b = filter_block_3x3(img, rr, cc, 2, 2, is_float);
            let v = accum_div(
                (b[0][0] * 9.0 + b[1][0] * 3.0 + b[0][1] * 3.0 + b[1][1]) / 4096.0,
                1,
                is_float,
            );
            accum_assign(out, r, c, v, is_float);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    fn mkimg<T: Clone + Default>(nr: usize, nc: usize, data: Vec<T>) -> Array2D<T> {
        let mut img = Array2D::zeros(nr, nc);
        for (i, v) in data.into_iter().enumerate() {
            *img.get_mut(i / nc, i % nc) = v;
        }
        img
    }

    #[test]
    fn test_nearest_rounding_and_bounds() {
        let img: Array2D<u8> = mkimg(2, 2, vec![10, 20, 30, 40]);
        let mut out = 0u8;
        assert!(nearest_interpolate(&img, 0, 0, &mut out));
        assert_eq!(out, 10);
        // dlib rounds 0.5 up
        let mut out = 0u8;
        assert!(InterpolateNearestNeighbor.interpolate_image(
            &img,
            Dpoint::new(0.5, 0.4),
            &mut out
        ));
        assert_eq!(out, 20);
        assert!(!nearest_interpolate(&img, 2, 0, &mut out));
        assert!(!nearest_interpolate(&img, 0, -1, &mut out));
    }

    #[test]
    fn test_bilinear_corners_and_center() {
        let img: Array2D<f64> = mkimg(2, 2, vec![0.0, 10.0, 20.0, 30.0]);
        let mut out = 0.0;
        assert!(bilinear_interpolate(&img, Dpoint::new(0.0, 0.0), &mut out));
        assert_eq!(out, 0.0);
        assert!(bilinear_interpolate(&img, Dpoint::new(0.5, 0.5), &mut out));
        assert_eq!(out, 15.0);
        // outside -> false
        assert!(!bilinear_interpolate(&img, Dpoint::new(1.0, 1.0), &mut out));
    }

    #[test]
    fn test_quadratic_constant_and_center() {
        let img: Array2D<f64> = mkimg(4, 4, vec![7.0; 16]);
        let mut out = 0.0f64;
        assert!(quadratic_interpolate(&img, Dpoint::new(1.5, 1.5), &mut out));
        assert!((out - 7.0).abs() < 1e-6);
        // needs a 3x3 neighborhood: (0.5,0.5) rounds to (1,1)... still ok;
        // (0.2,0.2) rounds to (0,0) -> outside
        assert!(!quadratic_interpolate(
            &img,
            Dpoint::new(0.2, 0.2),
            &mut out
        ));
    }

    #[test]
    fn test_pyramid_down_dims() {
        // dlib uses long arithmetic: ((N-1)*n)/N truncated (45/6 = 7).
        assert_eq!(PyramidDown::<6>::nr(9), 7);
        assert_eq!(PyramidDown::<6>::nc(9), 7);
        assert_eq!(PyramidDown::<6>::nr(12), 10);
        assert_eq!(PyramidDown::<6>::nr(375), 312);
        assert_eq!(PyramidDown::<6>::nr(500), 416);
        assert_eq!(PyramidDown::<2>::nr(21), 9);
        assert_eq!(PyramidDown::<3>::nr(12), 6);
        assert_eq!(PyramidDown::<1>::nr(100), 0);
    }

    #[test]
    fn test_pyramid_down_6_matches_direct_bilinear() {
        // gradient image
        let nr = 9;
        let nc = 9;
        let data: Vec<u8> = (0..nr * nc).map(|i| ((i * 7) % 251) as u8).collect();
        let img: Array2D<u8> = mkimg(nr, nc, data.clone());
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        PyramidDown::<6>::new().apply(&img, &mut out);
        assert_eq!(out.nr(), 7);
        assert_eq!(out.nc(), 7);

        // out is 7x7 now; out(4,4): x_scale = (9-1)/(7-1) = 8/6
        let scale = 8.0f64 / 6.0;
        let y = 4.0 * scale;
        let top = y.floor() as usize;
        let bottom = (top + 1).min(nr - 1);
        let tb = y - top as f64;
        let x = 4.0 * scale;
        let left = x.floor() as usize;
        let right = (left + 1).min(nc - 1);
        let lr = x - left as f64;
        let tl = f64::from(data[top * nc + left]);
        let tr = f64::from(data[top * nc + right]);
        let bl = f64::from(data[bottom * nc + left]);
        let br = f64::from(data[bottom * nc + right]);
        // gray→gray SIMD path: f32 math then static_cast<T>(v + 0.5)
        let s = (((1.0 - tb) * ((1.0 - lr) * tl + lr * tr) + tb * ((1.0 - lr) * bl + lr * br))
            as f32
            + 0.5f32) as u8;
        assert_eq!(*out.get(4, 4), s);
    }

    #[test]
    fn test_pyramid_up_dims() {
        let img: Array2D<u8> = mkimg(5, 5, vec![100; 25]);
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        pyramid_up(&img, &mut out);
        // pyramid_down<2>::point_up: tl=(0,0)->(2.5,1.5), br=(4,4)->(10.5,9.5);
        // drectangle -> rectangle uses lround per edge, so size is
        // (lround(9.5)+1, lround(10.5)+1) = (11, 12)
        assert_eq!(out.nr(), 11);
        assert_eq!(out.nc(), 12);
    }

    #[test]
    fn test_pyramid_down_2_filter_center() {
        // 10x10 image so the >8 requirement holds; out is (10-3)/2 = 3
        let nr = 12;
        let nc = 12;
        let data: Vec<u8> = (0..nr * nc).map(|i| ((i * 13 + 5) % 251) as u8).collect();
        let img: Array2D<u8> = mkimg(nr, nc, data.clone());
        let mut out: Array2D<u8> = Array2D::zeros(1, 1);
        PyramidDown::<2>::new().apply(&img, &mut out);
        assert_eq!(out.nr(), 4);
        assert_eq!(out.nc(), 4);

        // direct computation of out(1,1):
        // row filter at r=4 (temp row), oc=4
        // temp column 1 corresponds to oc = 2*1 = 2
        let row = |r: usize, oc: usize| -> i64 {
            let mut s = 0i64;
            for k in 0..5 {
                let w = match k {
                    1 | 3 => 4,
                    2 => 6,
                    _ => 1,
                };
                s += i64::from(data[r * nc + oc + k]) * w;
            }
            s
        };
        let tr4 = row(4, 2);
        let tr3 = row(3, 2);
        let tr2 = row(2, 2);
        let tr5 = row(5, 2);
        let tr6 = row(6, 2);
        let v = tr2 + tr3 * 4 + tr4 * 6 + tr5 * 4 + tr6;
        let expected = (v / 256) as u8;
        assert_eq!(*out.get(1, 1), expected);
    }
}

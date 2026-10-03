//! Port of `dlib/image_transforms/interpolation.h` `resize_image` (bilinear).
//!
//! dlib has three bilinear resize code paths with observable numeric
//! differences, all replicated here:
//! - rgb→rgb: 4-wide SIMD f32 bulk loop + f64 scalar tail (`vector_to_pixel`
//!   truncation, no rounding).
//! - grayscale→grayscale with identical pixel types: 4-wide SIMD f32 bulk
//!   loop (`+0.5` rounding for integer outputs) + f32 scalar tail
//!   (`assign_pixel` from float, i.e. truncation).
//! - everything else: a generic per-pixel f64 path via pixel intensities.
//!
//! The SIMD loops accumulate the source x coordinate as 4 f32 lanes advanced
//! by `+= 4*x_scale` per iteration; that f32 drift is observable in dlib's
//! own output, so it is reproduced exactly.
use crate::array2d::GenericImage;
use crate::pixel::{assign_pixel, get_pixel_intensity, Pixel, PixelValue};

#[inline]
fn f32_of(v: f64) -> f32 {
    v as f32
}

/// `simd4i(simd4f)` conversion: f32 -> i32 truncation.
#[inline]
fn trunc_i32(v: f32) -> i32 {
    v as i32
}

/// The rgb→rgb SIMD path of `dlib::resize_image(in, out, interpolate_bilinear)`.
fn resize_rgb_simd<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let in_nr = img.num_rows() as i64;
    let in_nc = img.num_columns() as i64;
    let out_nr = out.num_rows() as i64;
    let out_nc = out.num_columns() as i64;
    if out_nr * out_nc == 0 || in_nr * in_nc == 0 {
        return;
    }

    let x_scale = (in_nc - 1) as f64 / (out_nc - 1).max(1) as f64;
    let y_scale = (in_nr - 1) as f64 / (out_nr - 1).max(1) as f64;

    let chan = |p: &<S as GenericImage>::PixelType| match p.to_value() {
        PixelValue::Rgb(c) => [c.r as f32, c.g as f32, c.b as f32],
        PixelValue::Rgba(c) => [c.r as f32, c.g as f32, c.b as f32],
        _ => unreachable!("rgb resize path requires an rgb-like source"),
    };

    let mut y = -y_scale;
    for r in 0..out_nr as usize {
        y += y_scale;
        let top = y.floor() as i64;
        let bottom = (top + 1).min(in_nr - 1) as usize;
        let top = top as usize;
        let tb_frac = y - top as f64;

        let tb_frac_f = f32_of(tb_frac);
        let inv_tb_frac = f32_of(1.0 - tb_frac);
        let x_scale4 = f32_of(4.0 * x_scale);
        let x0 = -4.0 * x_scale;
        let mut lx = [
            f32_of(x0),
            f32_of(x0 + x_scale),
            f32_of(x0 + 2.0 * x_scale),
            f32_of(x0 + 3.0 * x_scale),
        ];
        let mut c: i64 = 0;
        loop {
            for v in lx.iter_mut() {
                *v += x_scale4;
            }
            let left = [
                trunc_i32(lx[0]),
                trunc_i32(lx[1]),
                trunc_i32(lx[2]),
                trunc_i32(lx[3]),
            ];
            let lr_frac = [
                lx[0] - left[0] as f32,
                lx[1] - left[1] as f32,
                lx[2] - left[2] as f32,
                lx[3] - left[3] as f32,
            ];
            let inv_lr_frac = [
                1.0f32 - lr_frac[0],
                1.0f32 - lr_frac[1],
                1.0f32 - lr_frac[2],
                1.0f32 - lr_frac[3],
            ];
            let right = [left[0] + 1, left[1] + 1, left[2] + 1, left[3] + 1];
            if right[3] as i64 >= in_nc {
                break;
            }
            for k in 0..4 {
                let tlfk = inv_tb_frac * inv_lr_frac[k];
                let trfk = inv_tb_frac * lr_frac[k];
                let blfk = tb_frac_f * inv_lr_frac[k];
                let brfk = tb_frac_f * lr_frac[k];

                let tl = chan(img.pixel(top, left[k] as usize));
                let tr = chan(img.pixel(top, right[k] as usize));
                let bl = chan(img.pixel(bottom, left[k] as usize));
                let br = chan(img.pixel(bottom, right[k] as usize));
                let mut outc = [0u8; 3];
                for ch in 0..3 {
                    // simd4i(tlf*tl + trf*tr + blf*bl + brf*br): f32 sum,
                    // left-to-right, then a C cast (no clamping).
                    let s = tlfk * tl[ch] + trfk * tr[ch] + blfk * bl[ch] + brfk * br[ch];
                    outc[ch] = trunc_i32(s) as u8;
                }
                let mut px = <D as GenericImage>::PixelType::default();
                px.assign_from_value(&PixelValue::Rgb(crate::pixel::RgbPixel {
                    r: outc[0],
                    g: outc[1],
                    b: outc[2],
                }));
                *out.pixel_mut(r, (c as usize) + k) = px;
            }
            c += 4;
        }
        // scalar tail in f64 with clamped right neighbor
        let mut x = -x_scale + c as f64 * x_scale;
        while c < out_nc {
            x += x_scale;
            let left = x.floor() as i64;
            let right = (left + 1).min(in_nc - 1) as usize;
            let left = left as usize;
            let lr_frac = x - left as f64;

            let tl = chan(img.pixel(top, left));
            let tr = chan(img.pixel(top, right));
            let bl = chan(img.pixel(bottom, left));
            let br = chan(img.pixel(bottom, right));
            let mut outc = [0u8; 3];
            for ch in 0..3 {
                let s = (1.0 - tb_frac)
                    * ((1.0 - lr_frac) * f64::from(tl[ch]) + lr_frac * f64::from(tr[ch]))
                    + tb_frac * ((1.0 - lr_frac) * f64::from(bl[ch]) + lr_frac * f64::from(br[ch]));
                outc[ch] = s as i64 as u8;
            }
            let mut px = <D as GenericImage>::PixelType::default();
            px.assign_from_value(&PixelValue::Rgb(crate::pixel::RgbPixel {
                r: outc[0],
                g: outc[1],
                b: outc[2],
            }));
            *out.pixel_mut(r, c as usize) = px;
            c += 1;
        }
    }
}

/// The grayscale→grayscale (same pixel type) SIMD path of
/// `dlib::resize_image(in, out, interpolate_bilinear)`.
fn resize_gray_simd<S, D>(img: &S, out: &mut D, out_is_integral: bool)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let in_nr = img.num_rows() as i64;
    let in_nc = img.num_columns() as i64;
    let out_nr = out.num_rows() as i64;
    let out_nc = out.num_columns() as i64;
    if out_nr * out_nc == 0 || in_nr * in_nc == 0 {
        return;
    }

    let x_scale = (in_nc - 1) as f64 / (out_nc - 1).max(1) as f64;
    let y_scale = (in_nr - 1) as f64 / (out_nr - 1).max(1) as f64;

    let gray_f32 = |p: &<S as GenericImage>::PixelType| f32_of(get_pixel_intensity(p));

    let mut y = -y_scale;
    for r in 0..out_nr as usize {
        y += y_scale;
        let top = y.floor() as i64;
        let bottom = (top + 1).min(in_nr - 1) as usize;
        let top = top as usize;
        let tb_frac = y - top as f64;

        let tb_frac_f = f32_of(tb_frac);
        let inv_tb_frac = f32_of(1.0 - tb_frac);
        let x_scale4 = f32_of(4.0 * x_scale);
        let x0 = -4.0 * x_scale;
        let mut lx = [
            f32_of(x0),
            f32_of(x0 + x_scale),
            f32_of(x0 + 2.0 * x_scale),
            f32_of(x0 + 3.0 * x_scale),
        ];
        let mut c: i64 = 0;
        loop {
            for v in lx.iter_mut() {
                *v += x_scale4;
            }
            let left = [
                trunc_i32(lx[0]),
                trunc_i32(lx[1]),
                trunc_i32(lx[2]),
                trunc_i32(lx[3]),
            ];
            let lr_frac = [
                lx[0] - left[0] as f32,
                lx[1] - left[1] as f32,
                lx[2] - left[2] as f32,
                lx[3] - left[3] as f32,
            ];
            let inv_lr_frac = [
                1.0f32 - lr_frac[0],
                1.0f32 - lr_frac[1],
                1.0f32 - lr_frac[2],
                1.0f32 - lr_frac[3],
            ];
            let right = [left[0] + 1, left[1] + 1, left[2] + 1, left[3] + 1];
            if right[3] as i64 >= in_nc {
                break;
            }
            for k in 0..4 {
                let tlfk = inv_tb_frac * inv_lr_frac[k];
                let trfk = inv_tb_frac * lr_frac[k];
                let blfk = tb_frac_f * inv_lr_frac[k];
                let brfk = tb_frac_f * lr_frac[k];
                let tl = gray_f32(img.pixel(top, left[k] as usize));
                let tr = gray_f32(img.pixel(top, right[k] as usize));
                let bl = gray_f32(img.pixel(bottom, left[k] as usize));
                let br = gray_f32(img.pixel(bottom, right[k] as usize));
                let s = tlfk * tl + trfk * tr + blfk * bl + brfk * br;
                let dst = out.pixel_mut(r, (c as usize) + k);
                if out_is_integral {
                    // static_cast<T>(float_value + 0.5): the +0.5 happens
                    // in f32, then truncates.
                    assign_pixel(dst, &((s + 0.5f32) as f64));
                } else {
                    assign_pixel(dst, &f64::from(s));
                }
            }
            c += 4;
        }
        // scalar tail in f32
        let mut x = -x_scale + c as f64 * x_scale;
        while c < out_nc {
            x += x_scale;
            let left = x.floor() as i64;
            let right = (left + 1).min(in_nc - 1) as usize;
            let left = left as usize;
            let lr_frac = f32_of(x - left as f64);

            let tl = gray_f32(img.pixel(top, left));
            let tr = gray_f32(img.pixel(top, right));
            let bl = gray_f32(img.pixel(bottom, left));
            let br = gray_f32(img.pixel(bottom, right));
            let temp = (1.0f32 - tb_frac_f) * ((1.0 - lr_frac) * tl + lr_frac * tr)
                + tb_frac_f * ((1.0 - lr_frac) * bl + lr_frac * br);
            assign_pixel(out.pixel_mut(r, c as usize), &temp);
            c += 1;
        }
    }
}

/// Generic bilinear resize over pixel intensities (mixed pixel types), source
/// coordinates accumulated in f64.
pub fn resize_image_into<S, D>(img: &S, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    let src_rgb = <S::PixelType as Pixel>::is_rgb() || <S::PixelType as Pixel>::is_rgb_alpha();
    let dst_rgb = <D::PixelType as Pixel>::is_rgb() || <D::PixelType as Pixel>::is_rgb_alpha();
    let same_type =
        std::any::TypeId::of::<S::PixelType>() == std::any::TypeId::of::<D::PixelType>();
    let src_gray = <S::PixelType as Pixel>::is_gray();
    let dst_gray = <D::PixelType as Pixel>::is_gray();

    if src_rgb && dst_rgb && same_type {
        resize_rgb_simd(img, out);
        return;
    }
    if src_gray && dst_gray && same_type {
        let out_is_integral = !<D::PixelType as Pixel>::is_float();
        resize_gray_simd(img, out, out_is_integral);
        return;
    }

    let in_nr = img.num_rows() as i64;
    let in_nc = img.num_columns() as i64;
    let out_nr = out.num_rows() as i64;
    let out_nc = out.num_columns() as i64;
    if out_nr * out_nc == 0 || in_nr * in_nc == 0 {
        return;
    }

    let x_scale = (in_nc - 1) as f64 / (out_nc - 1).max(1) as f64;
    let y_scale = (in_nr - 1) as f64 / (out_nr - 1).max(1) as f64;

    let mut y = -y_scale;
    for r in 0..out_nr as usize {
        y += y_scale;
        let top = y.floor() as i64;
        let bottom = (top + 1).min(in_nr - 1) as usize;
        let top = top as usize;
        let tb_frac = y - top as f64;
        let mut x = -x_scale;
        for c in 0..out_nc as usize {
            x += x_scale;
            let left = x.floor() as i64;
            let right = (left + 1).min(in_nc - 1) as usize;
            let left = left as usize;
            let lr_frac = x - left as f64;

            let tl = get_pixel_intensity(img.pixel(top, left));
            let tr = get_pixel_intensity(img.pixel(top, right));
            let bl = get_pixel_intensity(img.pixel(bottom, left));
            let br = get_pixel_intensity(img.pixel(bottom, right));

            let temp = (1.0 - tb_frac) * ((1.0 - lr_frac) * tl + lr_frac * tr)
                + tb_frac * ((1.0 - lr_frac) * bl + lr_frac * br);
            assign_pixel(out.pixel_mut(r, c), &temp);
        }
    }
}

/// Resizes `img` into `out` at `rows x cols` using dlib's bilinear
/// `resize_image` (sizes `out` first).
pub fn resize_image<S, D>(img: &S, rows: usize, cols: usize, out: &mut D)
where
    S: GenericImage,
    S::PixelType: Pixel,
    D: GenericImage,
    D::PixelType: Pixel,
{
    out.set_image_size(rows, cols);
    resize_image_into(img, out);
}

//! Pixel types and conversions ported from `dlib/pixel.h` (dlib commit
//! 46fa4a28).
//!
//! Covers the `rgb_pixel`, `rgb_alpha_pixel`, `hsi_pixel` and `lab_pixel`
//! structs, the grayscale pixel types (`u8`, `u16`, `i32`, `f32`, `f64`),
//! [`assign_pixel`], [`get_pixel_intensity`], [`assign_pixel_intensity`] and
//! pixel serialization (`serialize(const rgb_pixel&)` etc. in `dlib/pixel.h`).
//!
//! Semantics notes (verified against the C++ source):
//! - Grayscale-to-grayscale conversion is `static_cast` truncation clamped to
//!   the destination range (there is **no** rounding); e.g.
//!   `assign_pixel(u8, 127.9f64) == 127` but `assign_pixel(u8, 300.0) == 255`
//!   and `assign_pixel(u8, -1.0) == 0`.
//! - dlib's `hsi_pixel` actually stores three `unsigned char`s whose HSI
//!   components are quantized as `component/255.0` (hue additionally scaled
//!   by `360/255`). [`HsiPixel`] keeps the shared-contract `u16` fields but
//!   preserves dlib's exact 0–255 value semantics.

use dlib_rs_core::serialize::{Deserializer, DlibSerialize, SerializeError, Serializer};

/// RGB pixel, port of `dlib::rgb_pixel` (`dlib/pixel.h`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
pub struct RgbPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// RGB pixel with alpha channel, port of `dlib::rgb_alpha_pixel`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
pub struct RgbAlphaPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

/// HSI (actually HSL, see `dlib/pixel.h` `assign_pixel_helpers::RGB2HSL`)
/// pixel, port of `dlib::hsi_pixel`.
///
/// The C++ struct stores `unsigned char h, s, i`; this port uses `u16` fields
/// (shared contract) holding dlib's exact 0–255 quantized values:
/// `h = round(hue_degrees/360*255)`, `s = round(sat*255)`, `i = round(l*255)`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
pub struct HsiPixel {
    pub h: u16,
    pub s: u16,
    pub i: u16,
}

/// CIE L*a*b* pixel, port of `dlib::lab_pixel` (`dlib/pixel.h`).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
pub struct LabPixel {
    pub l: u8,
    pub a: u8,
    pub b: u8,
}

/// Untagged-ish dynamic pixel value used to drive [`Pixel::assign_from_value`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PixelValue {
    U8(u8),
    U16(u16),
    I32(i32),
    F32(f32),
    F64(f64),
    Rgb(RgbPixel),
    Rgba(RgbAlphaPixel),
    Hsi(HsiPixel),
    Lab(LabPixel),
}

/// Trait implemented by every dlib pixel type (`pixel_traits<P>` in C++).
pub trait Pixel: Copy + Default + PartialEq + std::fmt::Debug + 'static {
    /// Dynamic view of this pixel (`pixel_traits` basic classification).
    fn to_value(&self) -> PixelValue;
    /// `dlib::assign_pixel(dest, src)` for this destination type.
    fn assign_from_value(&mut self, v: &PixelValue);
    /// Maximum pixel (`pixel_traits<P>::max()`).
    fn max_pixel() -> Self;
    /// Minimum pixel (`pixel_traits<P>::min()`).
    fn min_pixel() -> Self;
    fn is_gray() -> bool;
    fn is_rgb() -> bool;
    fn is_rgb_alpha() -> bool;
    fn is_hsi() -> bool;
    fn is_lab() -> bool;
    fn is_float() -> bool;
}

// ----------------------------------------------------------------------------------------
// grayscale <-> grayscale conversion helpers (assign_pixel_helpers::assign for
// two grayscale types in dlib/pixel.h). Semantics: static_cast truncation
// clamped to the destination range via less_or_equal_to_max /
// greater_or_equal_to_min.

fn to_u8_gray(v: &PixelValue) -> u8 {
    match *v {
        PixelValue::U8(x) => x,
        PixelValue::U16(x) => {
            if x <= u8::MAX as u16 {
                x as u8
            } else {
                u8::MAX
            }
        }
        PixelValue::I32(x) => {
            if x <= 0 {
                0
            } else if x <= u8::MAX as i32 {
                x as u8
            } else {
                u8::MAX
            }
        }
        PixelValue::F32(x) => f64_to_u8_gray(x as f64),
        PixelValue::F64(x) => f64_to_u8_gray(x),
        _ => unreachable!("color pixel in grayscale path"),
    }
}

fn f64_to_u8_gray(x: f64) -> u8 {
    if x <= 0.0 {
        0
    } else if x <= u8::MAX as f64 {
        x as u8
    } else {
        u8::MAX
    }
}

fn to_u16_gray(v: &PixelValue) -> u16 {
    match *v {
        PixelValue::U8(x) => x as u16,
        PixelValue::U16(x) => x,
        PixelValue::I32(x) => {
            if x <= 0 {
                0
            } else if x <= u16::MAX as i32 {
                x as u16
            } else {
                u16::MAX
            }
        }
        PixelValue::F32(x) => f64_to_u16_gray(x as f64),
        PixelValue::F64(x) => f64_to_u16_gray(x),
        _ => unreachable!("color pixel in grayscale path"),
    }
}

fn f64_to_u16_gray(x: f64) -> u16 {
    if x <= 0.0 {
        0
    } else if x <= u16::MAX as f64 {
        x as u16
    } else {
        u16::MAX
    }
}

fn to_i32_gray(v: &PixelValue) -> i32 {
    match *v {
        PixelValue::U8(x) => x as i32,
        PixelValue::U16(x) => x as i32,
        PixelValue::I32(x) => x,
        PixelValue::F32(x) => f64_to_i32_gray(x as f64),
        PixelValue::F64(x) => f64_to_i32_gray(x),
        _ => unreachable!("color pixel in grayscale path"),
    }
}

fn f64_to_i32_gray(x: f64) -> i32 {
    // C++: p <= max / p >= min in double, then static_cast (truncates).
    if x > i32::MAX as f64 {
        i32::MAX
    } else if x < i32::MIN as f64 {
        i32::MIN
    } else {
        x as i32
    }
}

fn to_f32_gray(v: &PixelValue) -> f32 {
    let x = match *v {
        PixelValue::U8(x) => x as f64,
        PixelValue::U16(x) => x as f64,
        PixelValue::I32(x) => x as f64,
        PixelValue::F32(x) => x as f64,
        PixelValue::F64(x) => x,
        _ => unreachable!("color pixel in grayscale path"),
    };
    // float_grayscale_pixel_traits: max() = numeric_limits<float>::max(),
    // min() = -max().
    if x > f32::MAX as f64 {
        f32::MAX
    } else if x < f32::MIN as f64 {
        f32::MIN
    } else {
        x as f32
    }
}

fn to_f64_gray(v: &PixelValue) -> f64 {
    match *v {
        PixelValue::U8(x) => x as f64,
        PixelValue::U16(x) => x as f64,
        PixelValue::I32(x) => x as f64,
        PixelValue::F32(x) => x as f64,
        PixelValue::F64(x) => x,
        _ => unreachable!("color pixel in grayscale path"),
    }
}

fn is_gray_value(v: &PixelValue) -> bool {
    matches!(
        v,
        PixelValue::U8(_)
            | PixelValue::U16(_)
            | PixelValue::I32(_)
            | PixelValue::F32(_)
            | PixelValue::F64(_)
    )
}

// ----------------------------------------------------------------------------------------
// HSL / Lab conversion internals (assign_pixel_helpers::RGB2HSL, HSL2RGB,
// RGB2Lab, Lab2RGB in dlib/pixel.h). Ported line for line.

#[derive(Clone, Copy)]
struct Hsl {
    h: f64,
    s: f64,
    l: f64,
}

#[derive(Clone, Copy)]
struct Colour {
    r: f64,
    g: f64,
    b: f64,
}

#[derive(Clone, Copy)]
struct Lab {
    l: f64,
    a: f64,
    b: f64,
}

/// `RGB2HSL` from `dlib/pixel.h` (hue in degrees, lightness/saturation 0..1).
fn rgb2hsl(c1: Colour) -> Hsl {
    let themin = c1.r.min(c1.g.min(c1.b));
    let themax = c1.r.max(c1.g.max(c1.b));
    let delta = themax - themin;
    let mut c2 = Hsl {
        h: 0.0,
        s: 0.0,
        l: (themin + themax) / 2.0,
    };
    if c2.l > 0.0 && c2.l < 1.0 {
        c2.s = delta
            / (if c2.l < 0.5 {
                2.0 * c2.l
            } else {
                2.0 - 2.0 * c2.l
            });
    }
    if delta > 0.0 {
        if themax == c1.r && themax != c1.g {
            c2.h += (c1.g - c1.b) / delta;
        }
        if themax == c1.g && themax != c1.b {
            c2.h += 2.0 + (c1.b - c1.r) / delta;
        }
        if themax == c1.b && themax != c1.r {
            c2.h += 4.0 + (c1.r - c1.g) / delta;
        }
        c2.h *= 60.0;
    }
    c2
}

/// `HSL2RGB` from `dlib/pixel.h`.
fn hsl2rgb(c1: Hsl) -> Colour {
    let mut sat = Colour {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
    if c1.h < 120.0 {
        sat.r = (120.0 - c1.h) / 60.0;
        sat.g = c1.h / 60.0;
        sat.b = 0.0;
    } else if c1.h < 240.0 {
        sat.r = 0.0;
        sat.g = (240.0 - c1.h) / 60.0;
        sat.b = (c1.h - 120.0) / 60.0;
    } else {
        sat.r = (c1.h - 240.0) / 60.0;
        sat.g = 0.0;
        sat.b = (360.0 - c1.h) / 60.0;
    }
    sat.r = sat.r.min(1.0);
    sat.g = sat.g.min(1.0);
    sat.b = sat.b.min(1.0);

    let ctmp = Colour {
        r: 2.0 * c1.s * sat.r + (1.0 - c1.s),
        g: 2.0 * c1.s * sat.g + (1.0 - c1.s),
        b: 2.0 * c1.s * sat.b + (1.0 - c1.s),
    };

    if c1.l < 0.5 {
        Colour {
            r: c1.l * ctmp.r,
            g: c1.l * ctmp.g,
            b: c1.l * ctmp.b,
        }
    } else {
        Colour {
            r: (1.0 - c1.l) * ctmp.r + 2.0 * c1.l - 1.0,
            g: (1.0 - c1.l) * ctmp.g + 2.0 * c1.l - 1.0,
            b: (1.0 - c1.l) * ctmp.b + 2.0 * c1.l - 1.0,
        }
    }
}

/// `RGB2Lab` from `dlib/pixel.h` (sRGB -> XYZ(D65) -> L*a*b*).
fn rgb2lab(c1: Colour) -> Lab {
    let mut var_r = c1.r;
    let mut var_g = c1.g;
    let mut var_b = c1.b;

    if var_r > 0.04045 {
        var_r = ((var_r + 0.055) / 1.055).powf(2.4);
    } else {
        var_r /= 12.92;
    }
    if var_g > 0.04045 {
        var_g = ((var_g + 0.055) / 1.055).powf(2.4);
    } else {
        var_g /= 12.92;
    }
    if var_b > 0.04045 {
        var_b = ((var_b + 0.055) / 1.055).powf(2.4);
    } else {
        var_b /= 12.92;
    }

    var_r *= 100.0;
    var_g *= 100.0;
    var_b *= 100.0;

    // Observer = 2 degrees, Illuminant = D65.
    let x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805;
    let y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722;
    let z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505;

    let mut var_x = x / 95.047;
    let mut var_y = y / 100.000;
    let mut var_z = z / 108.883;

    if var_x > 0.008856 {
        var_x = var_x.powf(1.0 / 3.0);
    } else {
        var_x = (7.787 * var_x) + (16.0 / 116.0);
    }
    if var_y > 0.008856 {
        var_y = var_y.powf(1.0 / 3.0);
    } else {
        var_y = (7.787 * var_y) + (16.0 / 116.0);
    }
    if var_z > 0.008856 {
        var_z = var_z.powf(1.0 / 3.0);
    } else {
        var_z = (7.787 * var_z) + (16.0 / 116.0);
    }

    Lab {
        l: ((116.0 * var_y) - 16.0).max(0.0),
        a: (500.0 * (var_x - var_y)).clamp(-128.0, 127.0),
        b: (200.0 * (var_y - var_z)).clamp(-128.0, 127.0),
    }
}

/// `Lab2RGB` from `dlib/pixel.h`.
fn lab2rgb(c1: Lab) -> Colour {
    let mut var_y = (c1.l + 16.0) / 116.0;
    let mut var_x = (c1.a / 500.0) + var_y;
    let mut var_z = var_y - (c1.b / 200.0);

    if var_y.powi(3) > 0.008856 {
        var_y = var_y.powi(3);
    } else {
        var_y = (var_y - 16.0 / 116.0) / 7.787;
    }
    if var_x.powi(3) > 0.008856 {
        var_x = var_x.powi(3);
    } else {
        var_x = (var_x - 16.0 / 116.0) / 7.787;
    }
    if var_z.powi(3) > 0.008856 {
        var_z = var_z.powi(3);
    } else {
        var_z = (var_z - 16.0 / 116.0) / 7.787;
    }

    let x = var_x * 95.047;
    let y = var_y * 100.000;
    let z = var_z * 108.883;

    let var_x = x / 100.0;
    let var_y = y / 100.0;
    let var_z = z / 100.0;

    let mut var_r = var_x * 3.2406 + var_y * -1.5372 + var_z * -0.4986;
    let mut var_g = var_x * -0.9689 + var_y * 1.8758 + var_z * 0.0415;
    let mut var_b = var_x * 0.0557 + var_y * -0.2040 + var_z * 1.0570;

    if var_r > 0.0031308 {
        var_r = 1.055 * var_r.powf(1.0 / 2.4) - 0.055;
    } else {
        var_r *= 12.92;
    }
    if var_g > 0.0031308 {
        var_g = 1.055 * var_g.powf(1.0 / 2.4) - 0.055;
    } else {
        var_g *= 12.92;
    }
    if var_b > 0.0031308 {
        var_b = 1.055 * var_b.powf(1.0 / 2.4) - 0.055;
    } else {
        var_b *= 12.92;
    }

    Colour {
        r: var_r.clamp(0.0, 1.0),
        g: var_g.clamp(0.0, 1.0),
        b: var_b.clamp(0.0, 1.0),
    }
}

// ----------------------------------------------------------------------------------------
// pixel <-> pixel free functions

/// Port of `dlib::assign_pixel(dest, src)` (`dlib/pixel.h`), including every
/// color space conversion and the alpha-blending rules.
pub fn assign_pixel<P: Pixel, Q: Pixel>(dst: &mut P, src: &Q) {
    dst.assign_from_value(&src.to_value());
}

/// Port of `dlib::get_pixel_intensity(src)` (`dlib/pixel.h`): returns the
/// pixel's intensity in `[0, 255]`-style units as an `f64` (grayscale pixels
/// return their value; RGB averages channels; alpha is ignored).
pub fn get_pixel_intensity<P: Pixel>(p: &P) -> f64 {
    match p.to_value() {
        PixelValue::U8(x) => x as f64,
        PixelValue::U16(x) => x as f64,
        PixelValue::I32(x) => x as f64,
        PixelValue::F32(x) => x as f64,
        PixelValue::F64(x) => x,
        PixelValue::Rgb(c) => (c.r as u32 + c.g as u32 + c.b as u32) as f64 / 3.0,
        // The C++ helper sets alpha to 255 first, which makes the blend a
        // plain channel average.
        PixelValue::Rgba(c) => (c.r as u32 + c.g as u32 + c.b as u32) as f64 / 3.0,
        PixelValue::Hsi(c) => c.i as f64,
        PixelValue::Lab(c) => c.l as f64,
    }
}

/// Port of `dlib::assign_pixel_intensity(dest, new_intensity)`
/// (`dlib/pixel.h`): grayscale pixels take the intensity directly (clamped,
/// truncating); color pixels are round-tripped through HSI; an alpha channel
/// is preserved.
pub fn assign_pixel_intensity<P: Pixel>(p: &mut P, v: f64) {
    if P::is_gray() {
        p.assign_from_value(&PixelValue::F64(v));
        return;
    }
    if P::is_rgb_alpha() {
        // assign_pixel_intensity_helper for has_alpha pixels: force alpha to
        // 255, go through hsi, restore the old alpha.
        let old_alpha = match p.to_value() {
            PixelValue::Rgba(c) => c.a,
            _ => 255,
        };
        // C++ sets dest.alpha = 255 before the rgb conversion, so the rgb
        // view is a plain channel copy.
        let rgb = match p.to_value() {
            PixelValue::Rgba(c) => RgbPixel {
                r: c.r,
                g: c.g,
                b: c.b,
            },
            _ => rgb_pixel_value(p),
        };
        let mut hsi = rgb_to_hsi(&rgb);
        hsi.i = f64_to_u8_gray(v) as u16;
        let mut out = *p;
        out.assign_from_value(&PixelValue::Hsi(hsi));
        if let PixelValue::Rgba(mut c) = out.to_value() {
            c.a = old_alpha;
            p.assign_from_value(&PixelValue::Rgba(c));
        }
        return;
    }
    // rgb / hsi / lab: convert to hsi, set i, convert back.
    let hsi = rgb_to_hsi(&rgb_pixel_value(p));
    let hsi = HsiPixel {
        h: hsi.h,
        s: hsi.s,
        i: f64_to_u8_gray(v) as u16,
    };
    p.assign_from_value(&PixelValue::Hsi(hsi));
}

/// Converts any pixel to an `RgbPixel` the same way `assign_pixel` would
/// (alpha blended against the pixel's own channels, per the dlib helpers).
fn rgb_pixel_value<P: Pixel>(p: &P) -> RgbPixel {
    let mut rgb = RgbPixel::default();
    rgb.assign_from_value(&p.to_value());
    rgb
}

// ----------------------------------------------------------------------------------------
// color space free functions (exact dlib/pixel.h formulas)

/// Port of `assign_pixel(hsi_pixel, rgb_pixel)` from `dlib/pixel.h`
/// (`RGB2HSL` + the `/255` quantization).
pub fn rgb_to_hsi(src: &RgbPixel) -> HsiPixel {
    let c1 = Colour {
        r: src.r as f64 / 255.0,
        g: src.g as f64 / 255.0,
        b: src.b as f64 / 255.0,
    };
    let c2 = rgb2hsl(c1);
    HsiPixel {
        h: cast_u8_wrap(c2.h / 360.0 * 255.0 + 0.5) as u16,
        s: cast_u8_wrap(c2.s * 255.0 + 0.5) as u16,
        i: cast_u8_wrap(c2.l * 255.0 + 0.5) as u16,
    }
}

/// C++ `static_cast<unsigned char>(x)` for out-of-range doubles: the value
/// is truncated toward zero and then wrapped modulo 256 (what
/// `cvttsd2si`-based codegen does in practice). dlib's `rgb_to_hsi` relies
/// on this for negative hues.
fn cast_u8_wrap(x: f64) -> u8 {
    let i = x.trunc() as i64;
    (i & 0xFF) as u8
}

/// Port of `assign_pixel(rgb_pixel, hsi_pixel)` from `dlib/pixel.h`.
pub fn hsi_to_rgb(src: &HsiPixel) -> RgbPixel {
    let h = Hsl {
        h: src.h as f64 / 255.0 * 360.0,
        s: src.s as f64 / 255.0,
        l: src.i as f64 / 255.0,
    };
    let c = hsl2rgb(h);
    RgbPixel {
        r: (c.r * 255.0 + 0.5) as u8,
        g: (c.g * 255.0 + 0.5) as u8,
        b: (c.b * 255.0 + 0.5) as u8,
    }
}

/// Port of `assign_pixel(lab_pixel, rgb_pixel)` from `dlib/pixel.h`
/// (`RGB2Lab` + the `*255`/`+128` quantization).
pub fn rgb_to_lab(src: &RgbPixel) -> LabPixel {
    let c1 = Colour {
        r: src.r as f64 / 255.0,
        g: src.g as f64 / 255.0,
        b: src.b as f64 / 255.0,
    };
    let c2 = rgb2lab(c1);
    LabPixel {
        l: ((c2.l / 100.0) * 255.0 + 0.5) as u8,
        a: (c2.a + 128.0 + 0.5) as u8,
        b: (c2.b + 128.0 + 0.5) as u8,
    }
}

/// Port of `assign_pixel(rgb_pixel, lab_pixel)` from `dlib/pixel.h`.
pub fn lab_to_rgb(src: &LabPixel) -> RgbPixel {
    let l = Lab {
        l: (src.l as f64 / 255.0) * 100.0,
        a: src.a as f64 - 128.0,
        b: src.b as f64 - 128.0,
    };
    let c = lab2rgb(l);
    RgbPixel {
        r: (c.r * 255.0 + 0.5) as u8,
        g: (c.g * 255.0 + 0.5) as u8,
        b: (c.b * 255.0 + 0.5) as u8,
    }
}

/// Alpha-blends `src` onto `dest` exactly like `assign_pixel(rgb, rgb_alpha)`
/// in `dlib/pixel.h` (fixed point arithmetic with wrapping `u32` math).
fn blend_rgba_onto_rgb(dest: &mut RgbPixel, src: &RgbAlphaPixel) {
    if src.a == 255 {
        dest.r = src.r;
        dest.g = src.g;
        dest.b = src.b;
        return;
    }
    let mut temp_r = src.r as u32;
    let mut temp_g = src.g as u32;
    let mut temp_b = src.b as u32;

    temp_r = temp_r.wrapping_sub(dest.r as u32);
    temp_g = temp_g.wrapping_sub(dest.g as u32);
    temp_b = temp_b.wrapping_sub(dest.b as u32);

    temp_r = temp_r.wrapping_mul(src.a as u32);
    temp_g = temp_g.wrapping_mul(src.a as u32);
    temp_b = temp_b.wrapping_mul(src.a as u32);

    temp_r >>= 8;
    temp_g >>= 8;
    temp_b >>= 8;

    dest.r = dest.r.wrapping_add((temp_r & 0xFF) as u8);
    dest.g = dest.g.wrapping_add((temp_g & 0xFF) as u8);
    dest.b = dest.b.wrapping_add((temp_b & 0xFF) as u8);
}

/// Alpha-blends a grayscale `src` onto `dest` exactly like
/// `assign_pixel(gray, rgb_alpha)` in `dlib/pixel.h` (fixed point arithmetic).
fn blend_rgba_onto_gray<P: Pixel>(dest: &mut P, src: &RgbAlphaPixel) {
    let avg = ((src.r as u32 + src.g as u32 + src.b as u32) / 3) as u8;
    if src.a == 255 {
        dest.assign_from_value(&PixelValue::U8(avg));
        return;
    }
    let mut temp = avg as i32;
    let dest_copy = to_i32_gray(&dest.to_value());
    temp -= dest_copy;
    temp *= src.a as i32;
    temp /= 255;
    dest.assign_from_value(&PixelValue::I32(temp + dest_copy));
}

// ----------------------------------------------------------------------------------------
// Pixel impls

impl Pixel for u8 {
    fn to_value(&self) -> PixelValue {
        PixelValue::U8(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        if is_gray_value(v) {
            *self = to_u8_gray(v);
        } else {
            match *v {
                PixelValue::Rgb(c) => {
                    *self = ((c.r as u32 + c.g as u32 + c.b as u32) / 3) as u8;
                }
                PixelValue::Rgba(c) => blend_rgba_onto_gray(self, &c),
                PixelValue::Hsi(c) => *self = to_u8_gray(&PixelValue::U16(c.i)),
                PixelValue::Lab(c) => *self = c.l,
                _ => unreachable!(),
            }
        }
    }
    fn max_pixel() -> Self {
        255
    }
    fn min_pixel() -> Self {
        0
    }
    fn is_gray() -> bool {
        true
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for u16 {
    fn to_value(&self) -> PixelValue {
        PixelValue::U16(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        if is_gray_value(v) {
            *self = to_u16_gray(v);
        } else {
            match *v {
                PixelValue::Rgb(c) => {
                    // C++ goes through an `unsigned int` average and then the
                    // generic grayscale assign.
                    let temp = (c.r as u32 + c.g as u32 + c.b as u32) / 3;
                    *self = to_u16_gray(&PixelValue::U16(temp as u16));
                }
                PixelValue::Rgba(c) => blend_rgba_onto_gray(self, &c),
                PixelValue::Hsi(c) => *self = to_u16_gray(&PixelValue::U16(c.i)),
                PixelValue::Lab(c) => *self = c.l as u16,
                _ => unreachable!(),
            }
        }
    }
    fn max_pixel() -> Self {
        u16::MAX
    }
    fn min_pixel() -> Self {
        0
    }
    fn is_gray() -> bool {
        true
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for i32 {
    fn to_value(&self) -> PixelValue {
        PixelValue::I32(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        if is_gray_value(v) {
            *self = to_i32_gray(v);
        } else {
            match *v {
                PixelValue::Rgb(c) => {
                    let temp = (c.r as u32 + c.g as u32 + c.b as u32) / 3;
                    *self = to_i32_gray(&PixelValue::U16(temp as u16));
                }
                PixelValue::Rgba(c) => blend_rgba_onto_gray(self, &c),
                PixelValue::Hsi(c) => *self = to_i32_gray(&PixelValue::U16(c.i)),
                PixelValue::Lab(c) => *self = c.l as i32,
                _ => unreachable!(),
            }
        }
    }
    fn max_pixel() -> Self {
        i32::MAX
    }
    fn min_pixel() -> Self {
        i32::MIN
    }
    fn is_gray() -> bool {
        true
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for f32 {
    fn to_value(&self) -> PixelValue {
        PixelValue::F32(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        if is_gray_value(v) {
            *self = to_f32_gray(v);
        } else {
            match *v {
                PixelValue::Rgb(c) => {
                    let temp = (c.r as u32 + c.g as u32 + c.b as u32) / 3;
                    *self = to_f32_gray(&PixelValue::U16(temp as u16));
                }
                PixelValue::Rgba(c) => blend_rgba_onto_gray(self, &c),
                PixelValue::Hsi(c) => *self = to_f32_gray(&PixelValue::U16(c.i)),
                PixelValue::Lab(c) => *self = c.l as f32,
                _ => unreachable!(),
            }
        }
    }
    fn max_pixel() -> Self {
        1.0
    }
    fn min_pixel() -> Self {
        -1.0
    }
    fn is_gray() -> bool {
        true
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
}

impl Pixel for f64 {
    fn to_value(&self) -> PixelValue {
        PixelValue::F64(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        if is_gray_value(v) {
            *self = to_f64_gray(v);
        } else {
            match *v {
                PixelValue::Rgb(c) => {
                    *self = (c.r as f64 + c.g as f64 + c.b as f64) / 3.0;
                }
                PixelValue::Rgba(c) => blend_rgba_onto_gray(self, &c),
                PixelValue::Hsi(c) => *self = c.i as f64,
                PixelValue::Lab(c) => *self = c.l as f64,
                _ => unreachable!(),
            }
        }
    }
    fn max_pixel() -> Self {
        1.0
    }
    fn min_pixel() -> Self {
        -1.0
    }
    fn is_gray() -> bool {
        true
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        true
    }
}

impl Pixel for RgbPixel {
    fn to_value(&self) -> PixelValue {
        PixelValue::Rgb(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        match v {
            PixelValue::Rgb(c) => *self = *c,
            PixelValue::Rgba(c) => blend_rgba_onto_rgb(self, c),
            PixelValue::Hsi(c) => *self = hsi_to_rgb(c),
            PixelValue::Lab(c) => *self = lab_to_rgb(c),
            gray => {
                let p = to_u8_gray(gray);
                self.r = p;
                self.g = p;
                self.b = p;
            }
        }
    }
    fn max_pixel() -> Self {
        RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }
    }
    fn min_pixel() -> Self {
        RgbPixel::default()
    }
    fn is_gray() -> bool {
        false
    }
    fn is_rgb() -> bool {
        true
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for RgbAlphaPixel {
    fn to_value(&self) -> PixelValue {
        PixelValue::Rgba(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        match v {
            PixelValue::Rgba(c) => *self = *c,
            PixelValue::Rgb(c) => {
                self.r = c.r;
                self.g = c.g;
                self.b = c.b;
                self.a = 255;
            }
            PixelValue::Hsi(c) => {
                let rgb = hsi_to_rgb(c);
                self.r = rgb.r;
                self.g = rgb.g;
                self.b = rgb.b;
                self.a = 255;
            }
            PixelValue::Lab(c) => {
                let rgb = lab_to_rgb(c);
                self.r = rgb.r;
                self.g = rgb.g;
                self.b = rgb.b;
                self.a = 255;
            }
            gray => {
                let p = to_u8_gray(gray);
                self.r = p;
                self.g = p;
                self.b = p;
                self.a = 255;
            }
        }
    }
    fn max_pixel() -> Self {
        RgbAlphaPixel {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        }
    }
    fn min_pixel() -> Self {
        RgbAlphaPixel::default()
    }
    fn is_gray() -> bool {
        false
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        true
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for HsiPixel {
    fn to_value(&self) -> PixelValue {
        PixelValue::Hsi(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        match v {
            PixelValue::Hsi(c) => *self = *c,
            PixelValue::Rgb(c) => *self = rgb_to_hsi(c),
            PixelValue::Rgba(c) => {
                // The C++ helper converts the current hsi pixel to rgb,
                // blends the rgba source onto it, then converts back.
                let mut temp = hsi_to_rgb(self);
                blend_rgba_onto_rgb(&mut temp, c);
                *self = rgb_to_hsi(&temp);
            }
            PixelValue::Lab(c) => {
                let temp = lab_to_rgb(c);
                *self = rgb_to_hsi(&temp);
            }
            gray => {
                self.h = 0;
                self.s = 0;
                self.i = to_u8_gray(gray) as u16;
            }
        }
    }
    fn max_pixel() -> Self {
        HsiPixel { h: 0, s: 0, i: 255 }
    }
    fn min_pixel() -> Self {
        HsiPixel::default()
    }
    fn is_gray() -> bool {
        false
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        true
    }
    fn is_lab() -> bool {
        false
    }
    fn is_float() -> bool {
        false
    }
}

impl Pixel for LabPixel {
    fn to_value(&self) -> PixelValue {
        PixelValue::Lab(*self)
    }
    fn assign_from_value(&mut self, v: &PixelValue) {
        match v {
            PixelValue::Lab(c) => *self = *c,
            PixelValue::Rgb(c) => *self = rgb_to_lab(c),
            PixelValue::Rgba(c) => {
                let mut temp = lab_to_rgb(self);
                blend_rgba_onto_rgb(&mut temp, c);
                *self = rgb_to_lab(&temp);
            }
            PixelValue::Hsi(c) => {
                let temp = hsi_to_rgb(c);
                *self = rgb_to_lab(&temp);
            }
            gray => {
                self.a = 128;
                self.b = 128;
                self.l = to_u8_gray(gray);
            }
        }
    }
    fn max_pixel() -> Self {
        LabPixel {
            l: 255,
            a: 128,
            b: 128,
        }
    }
    fn min_pixel() -> Self {
        LabPixel {
            l: 0,
            a: 128,
            b: 128,
        }
    }
    fn is_gray() -> bool {
        false
    }
    fn is_rgb() -> bool {
        false
    }
    fn is_rgb_alpha() -> bool {
        false
    }
    fn is_hsi() -> bool {
        false
    }
    fn is_lab() -> bool {
        true
    }
    fn is_float() -> bool {
        false
    }
}

// ----------------------------------------------------------------------------------------
// pixel serialization (dlib/pixel.h serialize/deserialize overloads; each
// channel is a raw `unsigned char` byte)

/// Serializes one pixel exactly like the matching `serialize` overload in
/// `dlib/pixel.h` (channel bytes in declaration order).
pub fn serialize_pixel<P: Pixel + DlibSerialize>(p: &P, out: &mut Serializer) {
    p.dlib_serialize(out);
}

/// Deserializes one pixel exactly like the matching `deserialize` overload in
/// `dlib/pixel.h`.
pub fn deserialize_pixel<P: Pixel + DlibSerialize>(
    inp: &mut Deserializer,
) -> Result<P, SerializeError> {
    P::dlib_deserialize(inp)
}

impl DlibSerialize for RgbPixel {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u8(self.r);
        out.write_u8(self.g);
        out.write_u8(self.b);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        Ok(RgbPixel {
            r: inp.read_u8()?,
            g: inp.read_u8()?,
            b: inp.read_u8()?,
        })
    }
}

impl DlibSerialize for RgbAlphaPixel {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u8(self.r);
        out.write_u8(self.g);
        out.write_u8(self.b);
        out.write_u8(self.a);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        Ok(RgbAlphaPixel {
            r: inp.read_u8()?,
            g: inp.read_u8()?,
            b: inp.read_u8()?,
            a: inp.read_u8()?,
        })
    }
}

impl DlibSerialize for HsiPixel {
    fn dlib_serialize(&self, out: &mut Serializer) {
        // dlib's hsi_pixel holds three unsigned chars.
        out.write_u8(self.h as u8);
        out.write_u8(self.s as u8);
        out.write_u8(self.i as u8);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        Ok(HsiPixel {
            h: inp.read_u8()? as u16,
            s: inp.read_u8()? as u16,
            i: inp.read_u8()? as u16,
        })
    }
}

impl DlibSerialize for LabPixel {
    fn dlib_serialize(&self, out: &mut Serializer) {
        out.write_u8(self.l);
        out.write_u8(self.a);
        out.write_u8(self.b);
    }
    fn dlib_deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        Ok(LabPixel {
            l: inp.read_u8()?,
            a: inp.read_u8()?,
            b: inp.read_u8()?,
        })
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_clamps() {
        // u8 destination.
        let mut d = 0u8;
        d.assign_from_value(&PixelValue::F64(-1.0));
        assert_eq!(d, 0);
        d.assign_from_value(&PixelValue::F64(300.0));
        assert_eq!(d, 255);
        d.assign_from_value(&PixelValue::I32(-5));
        assert_eq!(d, 0);
        d.assign_from_value(&PixelValue::I32(300));
        assert_eq!(d, 255);
        d.assign_from_value(&PixelValue::U16(300));
        assert_eq!(d, 255);
        // Truncation, not rounding: 127.4 -> 127, 127.9 -> 127 (C++
        // static_cast semantics; only color conversions round).
        d.assign_from_value(&PixelValue::F64(127.4));
        assert_eq!(d, 127);
        d.assign_from_value(&PixelValue::F64(127.9));
        assert_eq!(d, 127);
        // u16 destination from f64.
        let mut u = 0u16;
        u.assign_from_value(&PixelValue::F64(-1.0));
        assert_eq!(u, 0);
        u.assign_from_value(&PixelValue::F64(70000.0));
        assert_eq!(u, u16::MAX);
        u.assign_from_value(&PixelValue::U8(200));
        assert_eq!(u, 200);
        // i32 destination clamps only at the i32 range.
        let mut i = 0i32;
        i.assign_from_value(&PixelValue::F64(-0.5));
        assert_eq!(i, 0);
        i.assign_from_value(&PixelValue::F64(1e300));
        assert_eq!(i, i32::MAX);
        // f32 destination saturates at f32::MAX (dlib clamps, no inf).
        let mut f = 0f32;
        f.assign_from_value(&PixelValue::F64(1e300));
        assert_eq!(f, f32::MAX);
        f.assign_from_value(&PixelValue::F64(-1e300));
        assert_eq!(f, f32::MIN);
        // Float destinations keep the exact value otherwise.
        let mut g = 0f64;
        g.assign_from_value(&PixelValue::I32(-3));
        assert_eq!(g, -3.0);
    }

    #[test]
    fn test_assign_pixel_free_fn() {
        let src = 300.0f64;
        let mut d = 0u8;
        assign_pixel(&mut d, &src);
        assert_eq!(d, 255);
        let src = 200u16;
        let mut d = 0u8;
        assign_pixel(&mut d, &src);
        assert_eq!(d, 200);
    }

    #[test]
    fn test_rgb_gray_average() {
        let c = RgbPixel {
            r: 10,
            g: 20,
            b: 30,
        };
        let mut g = 0u8;
        g.assign_from_value(&c.to_value());
        assert_eq!(g, 20); // (10+20+30)/3 with unsigned truncating division
        let mut f = 0f64;
        f.assign_from_value(&c.to_value());
        assert_eq!(f, 20.0);
        // Gray -> rgb replicates the channel.
        let mut rgb = RgbPixel::default();
        rgb.assign_from_value(&PixelValue::U8(77));
        assert_eq!(
            rgb,
            RgbPixel {
                r: 77,
                g: 77,
                b: 77
            }
        );
        // Gray -> rgb from f64 truncates through the u8 conversion.
        rgb.assign_from_value(&PixelValue::F64(127.9));
        assert_eq!(
            rgb,
            RgbPixel {
                r: 127,
                g: 127,
                b: 127
            }
        );
        // get_pixel_intensity
        assert_eq!(get_pixel_intensity(&c), 20.0);
        assert_eq!(
            get_pixel_intensity(&RgbAlphaPixel {
                r: 10,
                g: 20,
                b: 30,
                a: 0
            }),
            20.0
        );
        assert_eq!(get_pixel_intensity(&250u8), 250.0);
    }

    #[test]
    fn test_rgb_hsi_roundtrip() {
        for r in [0u8, 1, 17, 64, 128, 200, 255] {
            for g in [0u8, 33, 90, 128, 210, 255] {
                for b in [0u8, 5, 70, 128, 190, 255] {
                    // dlib's RGB2HSL produces a negative hue when red is the
                    // maximum and blue > green; the C++ code then wraps that
                    // negative value modulo 256 into the u8 hue (see
                    // cast_u8_wrap), so those colors do not round-trip.
                    if r >= g && r >= b && b > g {
                        continue;
                    }
                    let c = RgbPixel { r, g, b };
                    let hsi = rgb_to_hsi(&c);
                    assert!(hsi.h <= 255 && hsi.s <= 255 && hsi.i <= 255);
                    let back = hsi_to_rgb(&hsi);
                    let diff = (back.r as i32 - c.r as i32).abs().max(
                        (back.g as i32 - c.g as i32)
                            .abs()
                            .max((back.b as i32 - c.b as i32).abs()),
                    );
                    // The 8-bit HSI quantization is lossy; dlib's own
                    // formulas round-trip saturated colors with a few units
                    // of error (verified against an independent evaluation).
                    assert!(diff <= 6, "roundtrip diff {diff} for {c:?}");
                }
            }
        }
    }
    #[test]
    fn test_rgb_hsi_negative_hue_wrap() {
        // dlib stores the wrapped negative hue exactly like the C++
        // static_cast<unsigned char> (oracle computed independently from the
        // formulas in dlib/pixel.h).
        let hsi = rgb_to_hsi(&RgbPixel {
            r: 128,
            g: 0,
            b: 70,
        });
        assert_eq!(
            hsi,
            HsiPixel {
                h: 234,
                s: 255,
                i: 64
            }
        );
        let hsi = rgb_to_hsi(&RgbPixel { r: 255, g: 0, b: 0 });
        assert_eq!(
            hsi,
            HsiPixel {
                h: 0,
                s: 255,
                i: 128
            }
        );
    }

    #[test]
    fn test_rgb_lab_roundtrip() {
        for c in [
            RgbPixel { r: 0, g: 0, b: 0 },
            RgbPixel {
                r: 255,
                g: 255,
                b: 255,
            },
            RgbPixel { r: 255, g: 0, b: 0 },
            RgbPixel { r: 0, g: 255, b: 0 },
            RgbPixel { r: 0, g: 0, b: 255 },
            RgbPixel {
                r: 128,
                g: 128,
                b: 128,
            },
            RgbPixel {
                r: 12,
                g: 200,
                b: 99,
            },
        ] {
            let lab = rgb_to_lab(&c);
            let _ = (lab.l, lab.a, lab.b); // all u8 by construction
            let back = lab_to_rgb(&lab);
            let diff = (back.r as i32 - c.r as i32).abs().max(
                (back.g as i32 - c.g as i32)
                    .abs()
                    .max((back.b as i32 - c.b as i32).abs()),
            );
            // 8-bit Lab quantization is lossy; pure green round-trips with
            // diff 7 under dlib's exact formulas (independent check).
            assert!(diff <= 8, "lab roundtrip diff {diff} for {c:?}");
        }
        // Grayscale into lab keeps neutral a/b.
        let mut lab = LabPixel::default();
        lab.assign_from_value(&PixelValue::U8(100));
        assert_eq!(
            lab,
            LabPixel {
                l: 100,
                a: 128,
                b: 128
            }
        );
    }

    #[test]
    fn test_rgba_blending() {
        // Fully opaque: straight copy.
        let mut rgb = RgbPixel {
            r: 10,
            g: 10,
            b: 10,
        };
        rgb.assign_from_value(&PixelValue::Rgba(RgbAlphaPixel {
            r: 200,
            g: 100,
            b: 50,
            a: 255,
        }));
        assert_eq!(
            rgb,
            RgbPixel {
                r: 200,
                g: 100,
                b: 50
            }
        );
        // Half-transparent: fixed point blend dest + (src-dest)*alpha/256.
        let mut rgb = RgbPixel {
            r: 100,
            g: 100,
            b: 100,
        };
        rgb.assign_from_value(&PixelValue::Rgba(RgbAlphaPixel {
            r: 200,
            g: 0,
            b: 100,
            a: 128,
        }));
        // (200-100)*128 >> 8 = 50 -> 150 ; (0-100)*128>>8 wraps per u32 math.
        assert_eq!(rgb.r, 150);
        assert_eq!(
            rgb.g,
            (100u8).wrapping_add(((0u32.wrapping_sub(100)).wrapping_mul(128) >> 8) as u8)
        );
        assert_eq!(rgb.b, 100);
        // Gray destination blending.
        let mut g = 100u8;
        g.assign_from_value(&PixelValue::Rgba(RgbAlphaPixel {
            r: 200,
            g: 200,
            b: 200,
            a: 128,
        }));
        // temp = 200-100=100; 100*128/255 = 50; dest = 150.
        assert_eq!(g, 150);
        // rgb -> rgba sets alpha 255.
        let mut a = RgbAlphaPixel::default();
        a.assign_from_value(&PixelValue::Rgb(RgbPixel { r: 1, g: 2, b: 3 }));
        assert_eq!(
            a,
            RgbAlphaPixel {
                r: 1,
                g: 2,
                b: 3,
                a: 255
            }
        );
        // hsi <- rgba composes through rgb.
        let mut h = HsiPixel::default();
        h.assign_from_value(&PixelValue::Rgba(RgbAlphaPixel {
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        }));
        let rgb_back = hsi_to_rgb(&h);
        assert!(rgb_back.r > 240 && rgb_back.g < 15 && rgb_back.b < 15);
    }

    #[test]
    fn test_pixel_intensity_assignment() {
        let mut g = 0u8;
        assign_pixel_intensity(&mut g, 42.9);
        assert_eq!(g, 42); // truncation
        assign_pixel_intensity(&mut g, 300.0);
        assert_eq!(g, 255);
        let mut rgb = RgbPixel { r: 255, g: 0, b: 0 };
        assign_pixel_intensity(&mut rgb, 128.0);
        // Hue/saturation preserved through the HSI round trip.
        let hsi = rgb_to_hsi(&rgb);
        assert_eq!(hsi.i, 128);
        let mut a = RgbAlphaPixel {
            r: 255,
            g: 0,
            b: 0,
            a: 77,
        };
        assign_pixel_intensity(&mut a, 128.0);
        assert_eq!(a.a, 77); // alpha preserved
        assert_eq!(
            rgb_to_hsi(&RgbPixel {
                r: a.r,
                g: a.g,
                b: a.b
            })
            .i,
            128
        );
    }

    #[test]
    fn test_pixel_serialization() {
        let mut ser = Serializer::new();
        let rgb = RgbPixel { r: 1, g: 2, b: 3 };
        let rgba = RgbAlphaPixel {
            r: 4,
            g: 5,
            b: 6,
            a: 7,
        };
        let hsi = HsiPixel { h: 8, s: 9, i: 10 };
        let lab = LabPixel {
            l: 11,
            a: 12,
            b: 13,
        };
        serialize_pixel(&rgb, &mut ser);
        serialize_pixel(&rgba, &mut ser);
        serialize_pixel(&hsi, &mut ser);
        serialize_pixel(&lab, &mut ser);
        let bytes = ser.into_inner();
        assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        let mut de = Deserializer::new(&bytes);
        assert_eq!(deserialize_pixel::<RgbPixel>(&mut de).unwrap(), rgb);
        assert_eq!(deserialize_pixel::<RgbAlphaPixel>(&mut de).unwrap(), rgba);
        assert_eq!(deserialize_pixel::<HsiPixel>(&mut de).unwrap(), hsi);
        assert_eq!(deserialize_pixel::<LabPixel>(&mut de).unwrap(), lab);
        assert_eq!(de.remaining(), 0);
        // Truncated input errors instead of panicking.
        let mut de = Deserializer::new(&bytes[..2]);
        assert!(deserialize_pixel::<RgbPixel>(&mut de).is_err());
    }

    #[test]
    fn test_scalar_dlib_serialize_roundtrip() {
        let mut ser = Serializer::new();
        5u8.dlib_serialize(&mut ser);
        6u16.dlib_serialize(&mut ser);
        (-7i32).dlib_serialize(&mut ser);
        1.5f32.dlib_serialize(&mut ser);
        2.25f64.dlib_serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        assert_eq!(u8::dlib_deserialize(&mut de).unwrap(), 5);
        assert_eq!(u16::dlib_deserialize(&mut de).unwrap(), 6);
        assert_eq!(i32::dlib_deserialize(&mut de).unwrap(), -7);
        assert_eq!(f32::dlib_deserialize(&mut de).unwrap(), 1.5);
        assert_eq!(f64::dlib_deserialize(&mut de).unwrap(), 2.25);
    }

    #[test]
    fn test_max_min_pixels() {
        assert_eq!(u8::max_pixel(), 255);
        assert_eq!(u16::min_pixel(), 0);
        assert_eq!(i32::min_pixel(), i32::MIN);
        assert_eq!(f64::max_pixel(), 1.0);
        assert_eq!(
            RgbPixel::max_pixel(),
            RgbPixel {
                r: 255,
                g: 255,
                b: 255
            }
        );
        assert_eq!(RgbAlphaPixel::max_pixel().a, 255);
        assert!(<u8 as Pixel>::is_gray());
        assert!(RgbPixel::is_rgb());
        assert!(RgbAlphaPixel::is_rgb_alpha());
        assert!(HsiPixel::is_hsi());
        assert!(LabPixel::is_lab());
        assert!(f32::is_float());
    }
}

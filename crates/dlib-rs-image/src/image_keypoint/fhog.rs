//! FHOG feature extraction, ported from `dlib/image_transforms/fhog.h`
//! (`impl_fhog::impl_extract_fhog_features` and the `cell_size == 1` variant
//! `impl_extract_fhog_features_cell_size_1`).
//!
//! The algorithm is the one from:
//!   P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan,
//!   "Object Detection with Discriminatively Trained Part Based Models",
//!   IEEE TPAMI, Vol. 32, No. 9, Sep. 2010.
//!
//! The 31 channels are:
//!   * channels 0..18  : contrast-sensitive orientation bins (18 directions),
//!   * channels 18..27 : contrast-insensitive bins (9 directions, o and o+9
//!     summed),
//!   * channels 27..31 : 4 texture/energy channels (the sums of the normalized
//!     contrast-sensitive responses, one per block normalization).
//!
//! Memory order mirrors dlib's planar output (`dlib::array<array2d<T>>`):
//! the data is plane-major, each plane row-major, i.e. element
//! `(r, c, ch)` lives at `data[ch * nr * nc + r * nc + c]`.

use crate::array2d::GenericImage;
use crate::pixel::{get_pixel_intensity, Pixel, PixelValue};

/// Number of channels in an FHOG feature map (27 + 4, as in dlib).
pub const FHOG_NUM_CHANNELS: usize = 31;

/// The 9 unit orientation vectors used to snap gradients, exactly the table in
/// `impl_fhog::impl_extract_fhog_features` (dlib/image_transforms/fhog.h).
const DIRECTIONS: [[f32; 2]; 9] = [
    [1.0000, 0.0000],
    [0.9397, 0.3420],
    [0.7660, 0.6428],
    [0.5000, 0.8660],
    [0.1736, 0.9848],
    [-0.1736, 0.9848],
    [-0.5000, 0.8660],
    [-0.7660, 0.6428],
    [-0.9397, 0.3420],
];

/// Normalization epsilon used by dlib's FHOG (`const float eps = 0.0001;`).
const EPS: f32 = 0.0001;

/// An FHOG feature map: 31 channel planes of `nr` x `nc` cells.
///
/// Port of dlib's `dlib::array<array2d<T>>` output of
/// `extract_fhog_features` (dlib/image_transforms/fhog.h). `nr`/`nc` include
/// the filter padding (`hog_nr + filter_rows_padding - 1` rows etc.), matching
/// dlib's `init_hog`.
#[derive(Clone, Debug, Default)]
pub struct HogImage {
    /// Number of cell rows (including filter padding border).
    pub nr: usize,
    /// Number of cell columns (including filter padding border).
    pub nc: usize,
    /// Plane-major data: `data[ch * nr * nc + r * nc + c]`.
    pub data: Vec<f32>,
}

impl HogImage {
    /// Always 31, matching dlib's `27+4` hog bands.
    pub fn num_channels(&self) -> usize {
        FHOG_NUM_CHANNELS
    }

    /// True if the image was too small to produce any feature cells (dlib
    /// produces an empty feature map in that case).
    pub fn is_empty(&self) -> bool {
        self.nr == 0 || self.nc == 0
    }

    /// Value of channel `ch` at cell `(r, c)` (dlib `hog[ch][r][c]`).
    pub fn at(&self, r: usize, c: usize, ch: usize) -> f64 {
        debug_assert!(ch < FHOG_NUM_CHANNELS && r < self.nr && c < self.nc);
        f64::from(self.data[ch * self.nr * self.nc + r * self.nc + c])
    }

    /// All 31 channels at cell `(r, c)` (dlib `matrix<T,31,1>` per cell).
    pub fn cell(&self, r: usize, c: usize) -> [f64; FHOG_NUM_CHANNELS] {
        let mut out = [0.0; FHOG_NUM_CHANNELS];
        for (ch, v) in out.iter_mut().enumerate() {
            *v = self.at(r, c, ch);
        }
        out
    }

    /// Iterator over all cells in row-major order, yielding
    /// `(row, column, 31 channels)`.
    pub fn iter_cells(
        &self,
    ) -> impl Iterator<Item = (usize, usize, [f64; FHOG_NUM_CHANNELS])> + '_ {
        (0..self.nr).flat_map(move |r| (0..self.nc).map(move |c| (r, c, self.cell(r, c))))
    }
}

/// Extract FHOG features with dlib's defaults for the filter padding
/// (`filter_rows_padding = filter_cols_padding = 1`).
///
/// Port of `extract_fhog_features(img, hog, cell_size = 8, 1, 1)` from
/// dlib/image_transforms/fhog.h.
pub fn extract_fhog_features<S: GenericImage>(img: &S, cell_size: i64) -> HogImage
where
    S::PixelType: Pixel,
{
    extract_fhog_features_with_padding(img, cell_size, 1, 1)
}

/// Extract FHOG features with explicit filter padding.
///
/// Port of `impl_fhog::impl_extract_fhog_features` (and its `cell_size == 1`
/// specialization) from dlib/image_transforms/fhog.h. Padding semantics follow
/// dlib's `init_hog`: each plane is `hog_nr + filter_rows_padding - 1` rows by
/// `hog_nc + filter_cols_padding - 1` columns, and the feature cells are
/// written at offset `((filter_rows_padding-1)/2, (filter_cols_padding-1)/2)`
/// with the border left at zero.
pub fn extract_fhog_features_with_padding<S: GenericImage>(
    img: &S,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> HogImage
where
    S::PixelType: Pixel,
{
    assert!(cell_size > 0 && filter_rows_padding > 0 && filter_cols_padding > 0);
    if cell_size == 1 {
        extract_cell_size_1(img, filter_rows_padding, filter_cols_padding)
    } else {
        extract_general(img, cell_size, filter_rows_padding, filter_cols_padding)
    }
}

// --------------------------------------------------------------------------------

/// Grayscale gradient (dlib's non-rgb `get_gradient`): central differences of
/// `(int)get_pixel_intensity`, returned as `(gx, gy, length_squared)`.
fn gradient_gray<S: GenericImage>(img: &S, r: i64, c: i64) -> (f32, f32, f32)
where
    S::PixelType: Pixel,
{
    let gx = (get_pixel_intensity(img.pixel(r as usize, (c + 1) as usize)) as i64
        - get_pixel_intensity(img.pixel(r as usize, (c - 1) as usize)) as i64) as i32;
    let gy = (get_pixel_intensity(img.pixel((r + 1) as usize, c as usize)) as i64
        - get_pixel_intensity(img.pixel((r - 1) as usize, c as usize)) as i64) as i32;
    let len = gx * gx + gy * gy;
    (gx as f32, gy as f32, len as f32)
}

fn rgb_channels<S: GenericImage>(img: &S, r: i64, c: i64) -> (i32, i32, i32)
where
    S::PixelType: Pixel,
{
    match img.pixel(r as usize, c as usize).to_value() {
        PixelValue::Rgb(p) => (p.r as i32, p.g as i32, p.b as i32),
        PixelValue::Rgba(p) => (p.r as i32, p.g as i32, p.b as i32),
        _ => unreachable!("rgb_channels called on non-rgb pixel"),
    }
}

/// RGB gradient (dlib's rgb `get_gradient` overloads). Per-channel central
/// differences, keeping the channel with the strongest squared gradient.
///
/// dlib processes columns in 8-wide SIMD blocks and then a scalar remainder
/// loop, and the two paths tie-break DIFFERENTLY — observably so in dlib's
/// own output:
///  - SIMD (`select(cmp, a, b)` with strict `>`): red only if strictly
///    stronger than green, then that only if strictly stronger than blue
///    (ties favor green, then blue).
///  - scalar matrix overload: red is the base, green/blue replace only when
///    strictly greater (ties favor red).
///
/// `simd_lane` selects which dlib overload to mirror. Returns
/// `(gx, gy, length_squared)`.
fn gradient_rgb<S: GenericImage>(img: &S, r: i64, c: i64, simd_lane: bool) -> (f32, f32, f32)
where
    S::PixelType: Pixel,
{
    let (r0, g0, b0) = rgb_channels(img, r, c + 1);
    let (r1, g1, b1) = rgb_channels(img, r, c - 1);
    let (r2, g2, b2) = rgb_channels(img, r + 1, c);
    let (r3, g3, b3) = rgb_channels(img, r - 1, c);
    let grad_x_red = r0 - r1;
    let grad_y_red = r2 - r3;
    let rlen = grad_x_red * grad_x_red + grad_y_red * grad_y_red;

    let gxg = g0 - g1;
    let gyg = g2 - g3;
    let glen = gxg * gxg + gyg * gyg;

    let gxb = b0 - b1;
    let gyb = b2 - b3;
    let blen = gxb * gxb + gyb * gyb;

    let (gx, gy, len) = if simd_lane {
        // cmp = rlen > glen; t = select(cmp, red, green); cmp = tlen > blen; g = select(cmp, t, blue)
        let (tgx, tgy, tlen) = if rlen > glen {
            (grad_x_red, grad_y_red, rlen)
        } else {
            (gxg, gyg, glen)
        };
        if tlen > blen {
            (tgx, tgy, tlen)
        } else {
            (gxb, gyb, blen)
        }
    } else {
        // scalar overload: red base, replace only when strictly greater
        let (mut gx, mut gy, mut len) = (grad_x_red, grad_y_red, rlen);
        if glen > len {
            gx = gxg;
            gy = gyg;
            len = glen;
        }
        if blen > len {
            gx = gxb;
            gy = gyb;
            len = blen;
        }
        (gx, gy, len)
    };
    (gx as f32, gy as f32, len as f32)
}

fn gradient_at<S: GenericImage>(img: &S, r: i64, c: i64, simd_lane: bool) -> (f32, f32, f32)
where
    S::PixelType: Pixel,
{
    if <S::PixelType as Pixel>::is_rgb() {
        gradient_rgb(img, r, c, simd_lane)
    } else {
        gradient_gray(img, r, c)
    }
}

/// Snap a gradient to one of 18 orientations (dlib's `best_o` loop).
fn best_orientation(gx: f32, gy: f32) -> usize {
    let mut best_dot = 0.0f32;
    let mut best_o = 0usize;
    for (o, dir) in DIRECTIONS.iter().enumerate() {
        let dot = gx * dir[0] + gy * dir[1];
        if dot > best_dot {
            best_dot = dot;
            best_o = o;
        } else if -dot > best_dot {
            best_dot = -dot;
            best_o = o + 9;
        }
    }
    best_o
}

// --------------------------------------------------------------------------------

/// Port of `impl_fhog::impl_extract_fhog_features` (the general
/// `cell_size > 1` path; the SIMD blocks in dlib are numerically equivalent to
/// this scalar loop).
fn extract_general<S: GenericImage>(
    img: &S,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> HogImage
where
    S::PixelType: Pixel,
{
    let img_nr = img.num_rows() as i64;
    let img_nc = img.num_columns() as i64;
    let cell = cell_size as f32;

    // cells_nr = (int)((float)img.nr()/(float)cell_size + 0.5)
    let cells_nr = (img_nr as f32 / cell + 0.5) as i64;
    let cells_nc = (img_nc as f32 / cell + 0.5) as i64;
    if cells_nr == 0 || cells_nc == 0 {
        return HogImage::default();
    }

    // hist has a 1-cell border all the way around.
    let hist_nr = (cells_nr + 2) as usize;
    let hist_nc = (cells_nc + 2) as usize;
    let mut hist = vec![[0.0f32; 18]; hist_nr * hist_nc];
    let mut norm = vec![0.0f32; (cells_nr * cells_nc) as usize];

    let hog_nr = (cells_nr - 2).max(0);
    let hog_nc = (cells_nc - 2).max(0);
    if hog_nr == 0 || hog_nc == 0 {
        return HogImage::default();
    }
    let padding_rows_offset = (filter_rows_padding - 1) / 2;
    let padding_cols_offset = (filter_cols_padding - 1) / 2;

    // init_hog: plane size includes the padding border; the border stays zero
    // and every interior cell is written below.
    let out_nr = (hog_nr + filter_rows_padding - 1) as usize;
    let out_nc = (hog_nc + filter_cols_padding - 1) as usize;
    let mut hog = HogImage {
        nr: out_nr,
        nc: out_nc,
        data: vec![0.0; FHOG_NUM_CHANNELS * out_nr * out_nc],
    };

    let visible_nr = (cells_nr * cell_size).min(img_nr) - 1;
    let visible_nc = (cells_nc * cell_size).min(img_nc) - 1;

    // First populate the gradient histograms.
    for y in 1..visible_nr {
        // const float yp = (y + 0.5)/(float)cell_size - 0.5;
        let yp = ((y as f64 + 0.5) / f64::from(cell) - 0.5) as f32;
        let iyp = yp.floor() as i64;
        let vy0 = yp - iyp as f32;
        let vy1 = 1.0 - vy0;
        // dlib: SIMD 8-wide blocks for x in [1, visible_nc-7), scalar
        // remainder for the last columns; the rgb tie-break differs between
        // the two paths (see gradient_rgb).
        let mut x = 1i64;
        while x + 7 < visible_nc {
            for lane in 0..8 {
                vote_hist(
                    img,
                    y,
                    x + lane,
                    true,
                    cell,
                    iyp,
                    vy0,
                    vy1,
                    &mut hist,
                    hist_nc as i64,
                );
            }
            x += 8;
        }
        for xx in x..visible_nc {
            vote_hist(
                img,
                y,
                xx,
                false,
                cell,
                iyp,
                vy0,
                vy1,
                &mut hist,
                hist_nc as i64,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vote_hist<S: GenericImage>(
        img: &S,
        y: i64,
        x: i64,
        simd_lane: bool,
        cell: f32,
        iyp: i64,
        vy0: f32,
        vy1: f32,
        hist: &mut [[f32; 18]],
        hist_nc: i64,
    ) where
        S::PixelType: Pixel,
    {
        {
            let (gx, gy, len) = gradient_at(img, y, x, simd_lane);
            let v = len.sqrt();
            let best_o = best_orientation(gx, gy);

            // const float xp = (x + 0.5)/(double)cell_size - 0.5;
            let xp = ((x as f64 + 0.5) / f64::from(cell) - 0.5) as f32;
            let ixp = xp.floor() as i64;
            let vx0 = xp - ixp as f32;
            let vx1 = 1.0 - vx0;

            // Add the gradient magnitude v to the 4 histograms around the
            // pixel using bilinear interpolation. dlib folds v into the x
            // fractions FIRST (vx1 *= v; vx0 *= v), then multiplies by vy —
            // keep that association for bit-exactness.
            let vx1 = vx1 * v;
            let vx0 = vx0 * v;
            let v11 = vy1 * vx1;
            let v01 = vy0 * vx1;
            let v10 = vy1 * vx0;
            let v00 = vy0 * vx0;
            let base = ((iyp + 1) * hist_nc + ixp + 1) as usize;
            hist[base][best_o] += v11;
            hist[base + hist_nc as usize][best_o] += v01;
            hist[base + 1][best_o] += v10;
            hist[base + hist_nc as usize + 1][best_o] += v00;
        }
    }

    // Compute energy in each block by summing over orientations.
    for r in 0..cells_nr {
        for c in 0..cells_nc {
            let h = &hist[((r + 1) * hist_nc as i64 + c + 1) as usize];
            let mut acc = 0.0f32;
            for o in 0..9 {
                let s = h[o] + h[o + 9];
                acc += s * s;
            }
            norm[(r * cells_nc + c) as usize] = acc;
        }
    }

    // compute features
    let n = |r: i64, c: i64| norm[(r * cells_nc + c) as usize];
    let hst = |r: i64, c: i64| &hist[((r + 1) * hist_nc as i64 + c + 1) as usize];
    for y in 0..hog_nr {
        let yy = (y + padding_rows_offset) as usize;
        for x in 0..hog_nc {
            let xx = (x + padding_cols_offset) as usize;

            // dlib's simd4f z1..z4 lane-wise sums: nn[k] normalizes over the
            // 4 blocks around reference cells in this exact lane order
            // (z1[k] + z2[k] + z3[k] + z4[k], evaluated left-to-right).
            let z = [
                ((n(y + 1, x + 1) + n(y + 1, x + 2)) + n(y + 2, x + 1)) + n(y + 2, x + 2),
                ((n(y, x + 1) + n(y, x + 2)) + n(y + 1, x + 1)) + n(y + 1, x + 2),
                ((n(y + 1, x) + n(y + 1, x + 1)) + n(y + 2, x)) + n(y + 2, x + 1),
                ((n(y, x) + n(y, x + 1)) + n(y + 1, x)) + n(y + 1, x + 1),
            ];
            let mut nn = [0.0f32; 4];
            let mut nv = [0.0f32; 4];
            for k in 0..4 {
                nn[k] = 0.2 * (z[k] + EPS).sqrt();
                nv[k] = 0.1 / nn[k];
            }

            let center = hst(y + 1, x + 1);
            let mut t = [0.0f32; 4];
            // dlib sums simd4f lanes as (l0 + l2) + (l1 + l3) (the SSE2
            // movehl/shuffle horizontal add) — replicate exactly.
            let hsum = |h: [f32; 4]| (h[0] + h[2]) + (h[1] + h[3]);

            // contrast-sensitive features: dlib processes orientations in
            let mut o = 0usize;
            while o < 18 {
                let h0 = center[o];
                let h1 = center[o + 1];
                let h2 = center[o + 2];
                let hk = |co: f32| {
                    let mut h = [0.0f32; 4];
                    for k in 0..4 {
                        h[k] = co.min(nn[k]) * nv[k];
                    }
                    h
                };
                let ha = hk(h0);
                let hb = hk(h1);
                let hc = hk(h2);
                for k in 0..4 {
                    t[k] += (ha[k] + hb[k]) + hc[k];
                }
                hog.data[o * out_nr * out_nc + yy * out_nc + xx] = hsum(ha);
                hog.data[(o + 1) * out_nr * out_nc + yy * out_nc + xx] = hsum(hb);
                hog.data[(o + 2) * out_nr * out_nc + yy * out_nc + xx] = hsum(hc);
                o += 3;
            }

            // contrast-insensitive features
            for o in 0..9 {
                let v = center[o] + center[o + 9];
                let mut h = [0.0f32; 4];
                for k in 0..4 {
                    h[k] = v.min(nn[k]) * nv[k];
                }
                hog.data[(o + 18) * out_nr * out_nc + yy * out_nc + xx] = hsum(h);
            }

            // texture features: t *= 2*0.2357
            for (k, tk) in t.iter_mut().enumerate() {
                hog.data[(27 + k) * out_nr * out_nc + yy * out_nc + xx] = *tk * 0.4714;
            }
        }
    }

    hog
}

// --------------------------------------------------------------------------------

/// Port of `impl_fhog::impl_extract_fhog_features_cell_size_1`.
fn extract_cell_size_1<S: GenericImage>(
    img: &S,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> HogImage
where
    S::PixelType: Pixel,
{
    let img_nr = img.num_rows() as i64;
    let img_nc = img.num_columns() as i64;
    if img_nr <= 2 || img_nc <= 2 {
        return HogImage::default();
    }

    let mut angle = vec![0u8; (img_nr * img_nc) as usize];
    let mut norm = vec![0.0f32; (img_nr * img_nc) as usize];

    let hog_nr = img_nr - 2;
    let hog_nc = img_nc - 2;
    let padding_rows_offset = (filter_rows_padding - 1) / 2;
    let padding_cols_offset = (filter_cols_padding - 1) / 2;
    let out_nr = (hog_nr + filter_rows_padding - 1) as usize;
    let out_nc = (hog_nc + filter_cols_padding - 1) as usize;
    let mut hog = HogImage {
        nr: out_nr,
        nc: out_nc,
        data: vec![0.0; FHOG_NUM_CHANNELS * out_nr * out_nc],
    };

    let visible_nr = img_nr - 1;
    let visible_nc = img_nc - 1;

    // First populate the (per-pixel) gradient data. dlib uses SIMD 8-wide
    // column blocks plus a scalar remainder; the rgb tie-break differs
    // between the two paths (see gradient_rgb).
    let mut set_px = |y: i64, x: i64, simd_lane: bool| {
        let (gx, gy, v) = gradient_at(img, y, x, simd_lane);
        let best_o = best_orientation(gx, gy);
        norm[(y * img_nc + x) as usize] = v;
        angle[(y * img_nc + x) as usize] = best_o as u8;
    };
    for y in 1..visible_nr {
        let mut x = 1i64;
        while x + 7 < visible_nc {
            for lane in 0..8 {
                set_px(y, x + lane, true);
            }
            x += 8;
        }
        for xx in x..visible_nc {
            set_px(y, xx, false);
        }
    }

    let n = |r: i64, c: i64| norm[(r * img_nc + c) as usize];
    let ang = |r: i64, c: i64| angle[(r * img_nc + c) as usize];

    // compute features
    for y in 0..hog_nr {
        let yy = (y + padding_rows_offset) as usize;
        for x in 0..hog_nc {
            let xx = (x + padding_cols_offset) as usize;

            // dlib's simd4f z1..z4 lane-wise sums (same lane semantics as the
            // general path; z1[k] + z2[k] + z3[k] + z4[k], left-to-right).
            let z = [
                ((n(y + 1, x + 1) + n(y + 1, x + 2)) + n(y + 2, x + 1)) + n(y + 2, x + 2),
                ((n(y, x + 1) + n(y, x + 2)) + n(y + 1, x + 1)) + n(y + 1, x + 2),
                ((n(y + 1, x) + n(y + 1, x + 1)) + n(y + 2, x)) + n(y + 2, x + 1),
                ((n(y, x) + n(y, x + 1)) + n(y + 1, x)) + n(y + 1, x + 1),
            ];
            let mut nn = [0.0f32; 4];
            let mut nv = [0.0f32; 4];
            for k in 0..4 {
                nn[k] = 0.2 * (z[k] + EPS).sqrt();
                nv[k] = 0.1 / nn[k];
            }

            let temp0 = n(y + 1, x + 1).sqrt();
            let a = ang(y + 1, x + 1) as usize;

            let mut t = [0.0f32; 4];
            let mut h = [0.0f32; 4];
            for k in 0..4 {
                h[k] = temp0.min(nn[k]) * nv[k];
                t[k] = h[k];
            }
            let vv = (h[0] + h[2]) + (h[1] + h[3]);
            hog.data[a * out_nr * out_nc + yy * out_nc + xx] = vv;
            hog.data[(a % 9 + 18) * out_nr * out_nc + yy * out_nc + xx] = vv;

            for (k, tk) in t.iter_mut().enumerate() {
                hog.data[(27 + k) * out_nr * out_nc + yy * out_nc + xx] = *tk * 0.4714;
            }
        }
    }

    hog
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    fn u8_image(nr: usize, nc: usize, f: impl Fn(usize, usize) -> u8) -> Array2D<u8> {
        let mut img = Array2D::<u8>::zeros(nr, nc);
        for r in 0..nr {
            for c in 0..nc {
                *img.get_mut(r, c) = f(r, c);
            }
        }
        img
    }

    // (a) Constant image: zero gradients -> every channel of every cell is 0.
    // 24x24 with cell_size 8 gives cells_nr = cells_nc = 3, so hog_nr =
    // hog_nc = 1 (one output cell).
    #[test]
    fn fhog_constant_image_all_zero() {
        let img = u8_image(24, 24, |_, _| 100);
        let hog = extract_fhog_features(&img, 8);
        assert_eq!(hog.nr, 1);
        assert_eq!(hog.nc, 1);
        assert_eq!(hog.num_channels(), 31);
        for ch in 0..31 {
            assert_eq!(hog.at(0, 0, ch), 0.0, "channel {ch}");
        }
    }

    // Too-small image: 16x16 with cell_size 8 -> cells 2x2 -> hog_nr = 0 ->
    // empty feature map (dlib's `hog.clear()`).
    #[test]
    fn fhog_too_small_image_is_empty() {
        let img = u8_image(16, 16, |_, _| 5);
        let hog = extract_fhog_features(&img, 8);
        assert!(hog.is_empty());
        assert_eq!(hog.num_channels(), 31);
    }

    // (b) Vertical step edge: left half 0, right half 255 in a 24x24 image,
    // cell_size 8. Every gradient is (255, 0) -> orientation bin 0 only.
    //
    // Hand-derived (see dlib loops): the center cell histogram is
    //   h = 255 * (sum of row weights 7.0) * (0.9375 + 0.9375) = 3346.875,
    // all four block norms equal h^2, so nn_k = 0.2*h (up to eps) and each
    // normalized response is 0.2h * 0.1/(0.2h) = 0.1; summed over the 4
    // blocks the sensitive channel 0 and insensitive channel 18 are 0.4, the
    // texture channels are 0.1 * 2*0.2357 = 0.04714, and every other channel
    // is exactly 0.
    #[test]
    fn fhog_vertical_edge_orientation() {
        let img = u8_image(24, 24, |_, c| if c < 12 { 0 } else { 255 });
        let hog = extract_fhog_features(&img, 8);
        assert_eq!((hog.nr, hog.nc), (1, 1));

        let ch0 = hog.at(0, 0, 0);
        assert!((ch0 - 0.4).abs() < 1e-3, "sensitive bin 0: {ch0}");
        let ch18 = hog.at(0, 0, 18);
        assert!((ch18 - 0.4).abs() < 1e-3, "insensitive bin 0: {ch18}");
        for ch in 1..18 {
            assert_eq!(hog.at(0, 0, ch), 0.0, "sensitive channel {ch}");
        }
        for ch in 19..27 {
            assert_eq!(hog.at(0, 0, ch), 0.0, "insensitive channel {ch}");
        }
        for ch in 27..31 {
            let v = hog.at(0, 0, ch);
            assert!((v - 0.1 * 2.0 * 0.2357).abs() < 1e-4, "texture {ch}: {v}");
        }
    }

    // Same edge through the cell_size == 1 specialization: one output cell per
    // pixel. At output cell (5, 11) the orientation pixel is (6, 12), which
    // has gradient (255, 0), so exactly channels 0, 18 (same clipped value
    // vv) and the 4 texture channels are nonzero, each bounded by dlib's
    // clipping limit (4 blocks * 0.1 for channels 0/18, 0.1*2*0.2357 for
    // textures). Output cell (5, 13) sits fully inside the flat region and
    // must be all zero.
    #[test]
    fn fhog_cell_size_1_edge() {
        let img = u8_image(12, 24, |_, c| if c < 12 { 0 } else { 255 });
        let hog = extract_fhog_features(&img, 1);
        assert_eq!(hog.nr, 10);
        assert_eq!(hog.nc, 22);
        let cell = hog.cell(5, 11);
        for (ch, v) in cell.iter().enumerate() {
            if ch == 0 || ch == 18 {
                assert!(*v > 0.0 && *v <= 0.4 + 1e-4, "channel {ch}: {v}");
            } else if ch >= 27 {
                assert!(*v > 0.0 && *v <= 0.04714 + 1e-4, "channel {ch}: {v}");
            } else {
                assert_eq!(*v, 0.0, "channel {ch}");
            }
        }
        for ch in 0..31 {
            assert_eq!(hog.at(5, 13, ch), 0.0, "flat cell channel {ch}");
        }
    }
}

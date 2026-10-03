//! FHOG pyramid scanner, ported from
//! `dlib/image_processing/scan_fhog_pyramid.h` for the exact instantiation
//! used by the frontal face detector:
//! `scan_fhog_pyramid<pyramid_down<6>, default_fhog_feature_extractor>`
//! (the pyramid type is hardcoded to `PyramidDown<6>`).
//!
//! Also ports the pieces of `dlib/image_transforms/spatial_filtering.h`
//! (`float_spatially_filter_image` and
//! `float_spatially_filter_image_separable`) that `apply_filters_to_fhog`
//! uses, including dlib's SIMD accumulation order (8 output columns at a
//! time with three interleaved accumulators), so the saliency values match
//! the C++ implementation bit for bit.

use dlib_rs_core::geometry::{
    center, centered_rect_point, grow_rect, move_rect, shrink_rect, Dpoint, Drectangle, Point,
    Rectangle,
};
use dlib_rs_core::matrix::max as matrix_max;
use dlib_rs_core::matrix::svd::svd3;
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, SerializeError};
use std::cmp::Ordering;

use crate::array2d::{Array2D, GenericImage};
use crate::image_keypoint::fhog::{
    extract_fhog_features_with_padding, HogImage, FHOG_NUM_CHANNELS,
};
use crate::image_transforms::interpolation::PyramidDown;
use crate::pixel::Pixel;

// --------------------------------------------------------------------------------
// fhog <-> image coordinate mapping (dlib/image_transforms/fhog.h)
// --------------------------------------------------------------------------------

/// Port of `image_to_fhog(point, cell_size, filter_rows_padding,
/// filter_cols_padding)` from dlib/image_transforms/fhog.h.
pub fn image_to_fhog(
    p: Point,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> Point {
    // There is a one pixel border around the image.
    let px = p.x() - 1;
    let py = p.y() - 1;
    // There is also a 1 "cell" border around the HOG image formation.
    Point::new(
        px / cell_size - 1 + (filter_cols_padding - 1) / 2,
        py / cell_size - 1 + (filter_rows_padding - 1) / 2,
    )
}

/// Port of `image_to_fhog(const rectangle&, ...)`.
pub fn image_to_fhog_rect(
    rect: &Rectangle,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> Rectangle {
    Rectangle::from_points(
        image_to_fhog(
            rect.tl_corner(),
            cell_size,
            filter_rows_padding,
            filter_cols_padding,
        ),
        image_to_fhog(
            rect.br_corner(),
            cell_size,
            filter_rows_padding,
            filter_cols_padding,
        ),
    )
}

/// Port of `fhog_to_image(point, cell_size, filter_rows_padding,
/// filter_cols_padding)` from dlib/image_transforms/fhog.h.
pub fn fhog_to_image(
    p: Point,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> Point {
    // Convert to image space and then set to the center of the cell.
    let px = (p.x() + 1 - (filter_cols_padding - 1) / 2) * cell_size + 1;
    let py = (p.y() + 1 - (filter_rows_padding - 1) / 2) * cell_size + 1;
    let off_x = if px >= 0 {
        cell_size / 2
    } else {
        -cell_size / 2
    };
    let off_y = if py >= 0 {
        cell_size / 2
    } else {
        -cell_size / 2
    };
    Point::new(px + off_x, py + off_y)
}

/// Port of `fhog_to_image(const rectangle&, ...)`.
pub fn fhog_to_image_rect(
    rect: &Rectangle,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
) -> Rectangle {
    Rectangle::from_points(
        fhog_to_image(
            rect.tl_corner(),
            cell_size,
            filter_rows_padding,
            filter_cols_padding,
        ),
        fhog_to_image(
            rect.br_corner(),
            cell_size,
            filter_rows_padding,
            filter_cols_padding,
        ),
    )
}

/// `std::lround` (round half away from zero).
fn lround(x: f64) -> i64 {
    x.round() as i64
}

/// `pyramid_down<N>::rect_down(rect, levels)` on the integer rectangle
/// (converted to drectangle, corners transformed, converted back).
fn rect_down_levels(rect: &Rectangle, levels: usize) -> Rectangle {
    let pyr = PyramidDown::<6>::new();
    let mut tl = Dpoint::new(rect.left() as f64, rect.top() as f64);
    let mut br = Dpoint::new(rect.right() as f64, rect.bottom() as f64);
    for _ in 0..levels {
        tl = pyr.point_down(&tl);
        br = pyr.point_down(&br);
    }
    Rectangle::new(
        lround(tl.x()),
        lround(tl.y()),
        lround(br.x()),
        lround(br.y()),
    )
}

/// `pyramid_down<N>::rect_up(rect, levels)` on the integer rectangle.
fn rect_up_levels(rect: &Rectangle, levels: usize) -> Rectangle {
    let pyr = PyramidDown::<6>::new();
    let mut tl = Dpoint::new(rect.left() as f64, rect.top() as f64);
    let mut br = Dpoint::new(rect.right() as f64, rect.bottom() as f64);
    for _ in 0..levels {
        tl = pyr.point_up(&tl);
        br = pyr.point_up(&br);
    }
    Rectangle::new(
        lround(tl.x()),
        lround(tl.y()),
        lround(br.x()),
        lround(br.y()),
    )
}

// --------------------------------------------------------------------------------
// fhog_filterbank
// --------------------------------------------------------------------------------

/// Port of `scan_fhog_pyramid::fhog_filterbank`: the 31 full 2D filters plus
/// their separable (row/column) decompositions used to speed up detection.
#[derive(Clone, Default)]
pub struct FhogFilterbank {
    /// 31 full filters, each `fhog_window_height x fhog_window_width`.
    pub filters: Vec<Matrix<f32>>,
    /// Per channel, the row filters (length = filter width each).
    pub row_filters: Vec<Vec<Vec<f32>>>,
    /// Per channel, the column filters (length = filter height each).
    pub col_filters: Vec<Vec<Vec<f32>>>,
}

impl FhogFilterbank {
    /// Port of `fhog_filterbank::num_separable_filters()`.
    pub fn num_separable_filters(&self) -> usize {
        self.row_filters.iter().map(|v| v.len()).sum::<usize>()
    }
}

// --------------------------------------------------------------------------------
// spatial filtering (dlib/image_transforms/spatial_filtering.h)
// --------------------------------------------------------------------------------

/// The `non_border` rectangle shared by the filtering routines:
/// `rectangle(first_col, first_row, last_col-1, last_row-1)`.
fn non_border_rect(filter_rows: i64, filter_cols: i64, nr: usize, nc: usize) -> Rectangle {
    let first_row = filter_rows / 2;
    let first_col = filter_cols / 2;
    let last_row = nr as i64 - (filter_rows - 1) / 2;
    let last_col = nc as i64 - (filter_cols - 1) / 2;
    Rectangle::new(first_col, first_row, last_col - 1, last_row - 1)
}

/// Port of `impl::float_spatially_filter_image`
/// (dlib/image_transforms/spatial_filtering.h), restricted to the float
/// images/filters used by `apply_filters_to_fhog`. `out` must already have
/// `nr * nc` elements. The accumulation replicates dlib's SIMD loop (8
/// columns per block, three interleaved accumulators over the filter
/// columns).
pub fn float_spatially_filter_image(
    in_img: &[f32],
    nr: usize,
    nc: usize,
    filter: &Matrix<f32>,
    add_to: bool,
    out: &mut [f32],
) {
    let fnr = filter.nr() as i64;
    let fnc = filter.nc() as i64;
    let first_row = fnr / 2;
    let first_col = fnc / 2;
    let last_row = nr as i64 - (fnr - 1) / 2;
    let last_col = nc as i64 - (fnc - 1) / 2;

    // zero_border_pixels(out, non_border); the non-border region is fully
    // overwritten below, so zeroing everything is equivalent.
    if !add_to {
        out.iter_mut().for_each(|v| *v = 0.0);
    }

    for r in first_row..last_row {
        let mut c = first_col;
        // SIMD part: 8 output columns per step.
        while c + 7 < last_col {
            let mut t1 = [0.0f32; 8];
            let mut t2 = [0.0f32; 8];
            let mut t3 = [0.0f32; 8];
            for m in 0..fnr {
                let row = &in_img[((r - first_row + m) as usize) * nc + (c - first_col) as usize..];
                let mut n = 0i64;
                while n < fnc - 2 {
                    let f0 = filter[(m as usize, n as usize)];
                    let f1 = filter[(m as usize, (n + 1) as usize)];
                    let f2 = filter[(m as usize, (n + 2) as usize)];
                    for j in 0..8 {
                        t1[j] += row[(n + j as i64) as usize] * f0;
                        t2[j] += row[(n + 1 + j as i64) as usize] * f1;
                        t3[j] += row[(n + 2 + j as i64) as usize] * f2;
                    }
                    n += 3;
                }
                while n < fnc {
                    let f0 = filter[(m as usize, n as usize)];
                    for j in 0..8 {
                        t1[j] += row[(n + j as i64) as usize] * f0;
                    }
                    n += 1;
                }
            }
            for j in 0..8 {
                let v = t1[j] + (t2[j] + t3[j]);
                let idx = r as usize * nc + (c + j as i64) as usize;
                out[idx] = if add_to { out[idx] + v } else { v };
            }
            c += 8;
        }
        // Scalar tail.
        while c < last_col {
            let mut temp = 0.0f32;
            for m in 0..fnr {
                for n in 0..fnc {
                    temp += in_img
                        [((r - first_row + m) as usize) * nc + (c - first_col + n) as usize]
                        * filter[(m as usize, n as usize)];
                }
            }
            let idx = r as usize * nc + c as usize;
            out[idx] = if add_to { out[idx] + temp } else { temp };
            c += 1;
        }
    }
}

/// Port of `float_spatially_filter_image_separable`
/// (dlib/image_transforms/spatial_filtering.h). `out` and `scratch` must
/// already have `nr * nc` elements.
// The argument list mirrors dlib's float_spatially_filter_image_separable.
#[allow(clippy::too_many_arguments)]
pub fn float_spatially_filter_image_separable(
    in_img: &[f32],
    nr: usize,
    nc: usize,
    row_filter: &[f32],
    col_filter: &[f32],
    add_to: bool,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let rs = row_filter.len() as i64;
    let cs = col_filter.len() as i64;
    let first_row = cs / 2;
    let first_col = rs / 2;
    let last_row = nr as i64 - (cs - 1) / 2;
    let last_col = nc as i64 - (rs - 1) / 2;

    if !add_to {
        out.iter_mut().for_each(|v| *v = 0.0);
    }

    // apply the row filter
    for r in 0..nr as i64 {
        let row = &in_img[r as usize * nc..];
        let mut c = first_col;
        while c + 7 < last_col {
            let mut t1 = [0.0f32; 8];
            let mut t2 = [0.0f32; 8];
            let mut t3 = [0.0f32; 8];
            let mut n = 0i64;
            while n < rs - 2 {
                let f0 = row_filter[n as usize];
                let f1 = row_filter[(n + 1) as usize];
                let f2 = row_filter[(n + 2) as usize];
                for j in 0..8 {
                    t1[j] += row[(c - first_col + n + j as i64) as usize] * f0;
                    t2[j] += row[(c - first_col + n + 1 + j as i64) as usize] * f1;
                    t3[j] += row[(c - first_col + n + 2 + j as i64) as usize] * f2;
                }
                n += 3;
            }
            while n < rs {
                let f0 = row_filter[n as usize];
                for j in 0..8 {
                    t1[j] += row[(c - first_col + n + j as i64) as usize] * f0;
                }
                n += 1;
            }
            for j in 0..8 {
                scratch[r as usize * nc + (c + j as i64) as usize] = t1[j] + (t2[j] + t3[j]);
            }
            c += 8;
        }
        while c < last_col {
            let mut temp = 0.0f32;
            for n in 0..rs {
                temp += row[(c - first_col + n) as usize] * row_filter[n as usize];
            }
            scratch[r as usize * nc + c as usize] = temp;
            c += 1;
        }
    }

    // apply the column filter
    for r in first_row..last_row {
        let mut c = first_col;
        while c + 7 < last_col {
            let mut t1 = [0.0f32; 8];
            let mut t2 = [0.0f32; 8];
            let mut t3 = [0.0f32; 8];
            let mut m = 0i64;
            while m < cs - 2 {
                let f0 = col_filter[m as usize];
                let f1 = col_filter[(m + 1) as usize];
                let f2 = col_filter[(m + 2) as usize];
                for j in 0..8 {
                    t1[j] +=
                        scratch[((r - first_row + m) as usize) * nc + (c + j as i64) as usize] * f0;
                    t2[j] += scratch
                        [((r - first_row + m + 1) as usize) * nc + (c + j as i64) as usize]
                        * f1;
                    t3[j] += scratch
                        [((r - first_row + m + 2) as usize) * nc + (c + j as i64) as usize]
                        * f2;
                }
                m += 3;
            }
            while m < cs {
                let f0 = col_filter[m as usize];
                for j in 0..8 {
                    t1[j] +=
                        scratch[((r - first_row + m) as usize) * nc + (c + j as i64) as usize] * f0;
                }
                m += 1;
            }
            for j in 0..8 {
                let v = t1[j] + (t2[j] + t3[j]);
                let idx = r as usize * nc + (c + j as i64) as usize;
                out[idx] = if add_to { out[idx] + v } else { v };
            }
            c += 8;
        }
        while c < last_col {
            let mut temp = 0.0f32;
            for m in 0..cs {
                temp += scratch[((r - first_row + m) as usize) * nc + c as usize]
                    * col_filter[m as usize];
            }
            let idx = r as usize * nc + c as usize;
            out[idx] = if add_to { out[idx] + temp } else { temp };
            c += 1;
        }
    }
}

/// Port of `impl::apply_filters_to_fhog`
/// (dlib/image_processing/scan_fhog_pyramid.h): produces the saliency image
/// for one pyramid level by summing the filtered feature planes. `out` and
/// `scratch` are resized to the plane size. Returns the valid (non-border)
/// area of the saliency image.
pub fn apply_filters_to_fhog(
    w: &FhogFilterbank,
    hog: &HogImage,
    out: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) -> Rectangle {
    let nr = hog.nr;
    let nc = hog.nc;
    out.clear();
    out.resize(nr * nc, 0.0);
    scratch.resize(nr * nc, 0.0);

    let num_separable_filters = w.num_separable_filters();
    // use the separable filters if they would be faster than running the
    // regular filters.
    let use_dense = w.filters.is_empty()
        || (num_separable_filters as f64)
            > w.filters.len() as f64 * w.filters[0].nr().min(w.filters[0].nc()) as f64 / 3.0;

    if use_dense {
        let area = non_border_rect(w.filters[0].nr() as i64, w.filters[0].nc() as i64, nr, nc);
        let plane0 = &hog.data[0..nr * nc];
        float_spatially_filter_image(plane0, nr, nc, &w.filters[0], false, out);
        for i in 1..w.filters.len() {
            let plane = &hog.data[i * nr * nc..(i + 1) * nr * nc];
            float_spatially_filter_image(plane, nr, nc, &w.filters[i], true, out);
        }
        area
    } else {
        let mut area = Rectangle::default();
        let mut have_output = false;
        for i in 0..w.row_filters.len() {
            for j in 0..w.row_filters[i].len() {
                let plane = &hog.data[i * nr * nc..(i + 1) * nr * nc];
                let a = non_border_rect(
                    w.col_filters[i][j].len() as i64,
                    w.row_filters[i][j].len() as i64,
                    nr,
                    nc,
                );
                float_spatially_filter_image_separable(
                    plane,
                    nr,
                    nc,
                    &w.row_filters[i][j],
                    &w.col_filters[i][j],
                    have_output,
                    out,
                    scratch,
                );
                if !have_output {
                    area = a;
                    have_output = true;
                }
            }
        }
        if !have_output {
            out.clear();
            out.resize(nr * nc, 0.0);
            area = Rectangle::new(0, 0, nc as i64 - 1, nr as i64 - 1);
        }
        area
    }
}

/// Port of `impl::detect_from_fhog_pyramid`.
// The argument list mirrors dlib's impl::detect_from_fhog_pyramid.
#[allow(clippy::too_many_arguments)]
fn detect_from_fhog_pyramid(
    feats: &[HogImage],
    w: &FhogFilterbank,
    thresh: f64,
    det_box_height: u64,
    det_box_width: u64,
    cell_size: i64,
    filter_rows_padding: i64,
    filter_cols_padding: i64,
    dets: &mut Vec<(f64, Rectangle)>,
) {
    dets.clear();

    let mut saliency: Vec<f32> = Vec::new();
    let mut scratch: Vec<f32> = Vec::new();

    // for all pyramid levels
    for (l, hog) in feats.iter().enumerate() {
        let area = apply_filters_to_fhog(w, hog, &mut saliency, &mut scratch);
        if area.is_empty() {
            continue;
        }
        let nc = hog.nc;

        // now search the saliency image for any detections
        for r in area.top()..=area.bottom() {
            for c in area.left()..=area.right() {
                let v = saliency[r as usize * nc + c as usize];
                // if we found a detection
                if f64::from(v) >= thresh {
                    let rect = fhog_to_image_rect(
                        &centered_rect_point(&Point::new(c, r), det_box_width, det_box_height),
                        cell_size,
                        filter_rows_padding,
                        filter_cols_padding,
                    );
                    let rect = rect_up_levels(&rect, l);
                    dets.push((f64::from(v), rect));
                }
            }
        }
    }

    // sort detections in descending order of confidence
    dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
}

// --------------------------------------------------------------------------------
// scan_fhog_pyramid
// --------------------------------------------------------------------------------

/// Port of `dlib::scan_fhog_pyramid<pyramid_down<6>>`
/// (dlib/image_processing/scan_fhog_pyramid.h).
#[derive(Clone)]
pub struct ScanFhogPyramid {
    /// The feature pyramid (one HogImage of 31 planes per level).
    feats: Vec<HogImage>,
    cell_size: i64,
    padding: u64,
    window_width: u64,
    window_height: u64,
    max_pyramid_levels: u64,
    min_pyramid_layer_width: u64,
    min_pyramid_layer_height: u64,
    nuclear_norm_regularization_strength: f64,
}

impl Default for ScanFhogPyramid {
    fn default() -> Self {
        ScanFhogPyramid::new()
    }
}

impl ScanFhogPyramid {
    /// Port of `scan_fhog_pyramid::init()` defaults.
    pub fn new() -> Self {
        ScanFhogPyramid {
            feats: Vec::new(),
            cell_size: 8,
            padding: 1,
            window_width: 64,
            window_height: 64,
            max_pyramid_levels: 1000,
            min_pyramid_layer_width: 64,
            min_pyramid_layer_height: 64,
            nuclear_norm_regularization_strength: 0.0,
        }
    }

    /// Port of `scan_fhog_pyramid::is_loaded_with_image()`.
    pub fn is_loaded_with_image(&self) -> bool {
        !self.feats.is_empty()
    }

    /// Port of `scan_fhog_pyramid::get_cell_size()`.
    pub fn get_cell_size(&self) -> i64 {
        self.cell_size
    }

    /// Port of `scan_fhog_pyramid::get_padding()`.
    pub fn get_padding(&self) -> u64 {
        self.padding
    }

    /// Port of `scan_fhog_pyramid::get_detection_window_width()`.
    pub fn get_detection_window_width(&self) -> u64 {
        self.window_width
    }

    /// Port of `scan_fhog_pyramid::get_detection_window_height()`.
    pub fn get_detection_window_height(&self) -> u64 {
        self.window_height
    }

    /// Port of `scan_fhog_pyramid::get_max_pyramid_levels()`.
    pub fn get_max_pyramid_levels(&self) -> u64 {
        self.max_pyramid_levels
    }

    /// Port of `scan_fhog_pyramid::get_min_pyramid_layer_width()`.
    pub fn get_min_pyramid_layer_width(&self) -> u64 {
        self.min_pyramid_layer_width
    }

    /// Port of `scan_fhog_pyramid::get_min_pyramid_layer_height()`.
    pub fn get_min_pyramid_layer_height(&self) -> u64 {
        self.min_pyramid_layer_height
    }

    /// The number of pyramid levels of the currently loaded image
    /// (`feats.size()`).
    pub fn num_pyramid_levels(&self) -> usize {
        self.feats.len()
    }

    /// Port of `scan_fhog_pyramid::get_fhog_window_width()`.
    pub fn get_fhog_window_width(&self) -> u64 {
        self.compute_fhog_window_size().0
    }

    /// Port of `scan_fhog_pyramid::get_fhog_window_height()`.
    pub fn get_fhog_window_height(&self) -> u64 {
        self.compute_fhog_window_size().1
    }

    /// Port of `scan_fhog_pyramid::get_num_dimensions()`:
    /// `width * height * 31`.
    pub fn get_num_dimensions(&self) -> u64 {
        let (width, height) = self.compute_fhog_window_size();
        width * height * FHOG_NUM_CHANNELS as u64
    }

    /// Debug/inspection accessor: the loaded feature pyramid levels.
    pub fn feats(&self) -> &[HogImage] {
        &self.feats
    }

    /// Port of `scan_fhog_pyramid::deserialize` (the `load` in this port):
    /// version int (must be 1), the (empty) feature extractor, the feature
    /// pyramid array, cell size, padding, detection window size, pyramid
    /// limits, the nuclear norm strength and a trailing dimension check.
    pub fn load(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        let version = inp.read_i32()?;
        if version != 1 {
            return Err(SerializeError::Malformed(
                "Unsupported version found when deserializing a scan_fhog_pyramid object.",
            ));
        }

        // default_fhog_feature_extractor serializes nothing.

        // item.feats: dlib::array<fhog_image> serializes max_size, size,
        // then the elements.
        let feats = read_fhog_image_array(inp)?;
        self.feats = feats;

        self.cell_size = i64::from(inp.read_i32()?);
        self.padding = inp.read_u64()?;
        self.window_width = inp.read_u64()?;
        self.window_height = inp.read_u64()?;
        self.max_pyramid_levels = inp.read_u64()?;
        self.min_pyramid_layer_width = inp.read_u64()?;
        self.min_pyramid_layer_height = inp.read_u64()?;
        self.nuclear_norm_regularization_strength = inp.read_f64()?;

        let dims = inp.read_i64()?;
        if self.get_num_dimensions() as i64 != dims {
            return Err(SerializeError::Malformed(
                "Number of dimensions in serialized scan_fhog_pyramid doesn't match the expected number.",
            ));
        }
        Ok(())
    }

    /// Port of `scan_fhog_pyramid::load(img)` via
    /// `impl::create_fhog_pyramid`: the number of levels is chosen by
    /// repeatedly applying `pyramid_down<6>::rect_down` until the rectangle
    /// drops below the minimum pyramid layer size; level 0 extracts FHOG
    /// features from the input image, and every subsequent level extracts
    /// them from the pyramid-downsampled *image* (not the feature planes),
    /// all with `filter_rows_padding = fhog_window_height` and
    /// `filter_cols_padding = fhog_window_width`.
    pub fn load_image<S: GenericImage>(&mut self, img: &S)
    where
        S::PixelType: Pixel,
    {
        let (width, height) = self.compute_fhog_window_size();

        // figure out how many pyramid levels we should be using based on the
        // image size
        let pyr = PyramidDown::<6>::new();
        let mut levels: u64 = 0;
        let mut rect = Drectangle::new(
            0.0,
            0.0,
            (img.num_columns() as i64 - 1) as f64,
            (img.num_rows() as i64 - 1) as f64,
        );
        loop {
            rect = pyr.rect_down(&rect);
            levels += 1;
            if !(rect.width() >= self.min_pyramid_layer_width as f64
                && rect.height() >= self.min_pyramid_layer_height as f64
                && levels < self.max_pyramid_levels)
            {
                break;
            }
        }

        // build our feature pyramid
        let mut feats = Vec::with_capacity(levels as usize);
        feats.push(extract_fhog_features_with_padding(
            img,
            self.cell_size,
            height as i64,
            width as i64,
        ));

        if levels > 1 {
            let mut temp1: Array2D<S::PixelType> = Array2D::new();
            let mut temp2: Array2D<S::PixelType>;
            pyr.apply(img, &mut temp1);
            feats.push(extract_fhog_features_with_padding(
                &temp1,
                self.cell_size,
                height as i64,
                width as i64,
            ));
            temp2 = temp1.clone();

            for _ in 2..levels {
                pyr.apply(&temp2, &mut temp1);
                feats.push(extract_fhog_features_with_padding(
                    &temp1,
                    self.cell_size,
                    height as i64,
                    width as i64,
                ));
                std::mem::swap(&mut temp1, &mut temp2);
            }
        }
        self.feats = feats;
    }

    /// Port of `scan_fhog_pyramid::build_fhog_filterbank(weights)`: reshapes
    /// each 31st slice of the weight vector into an
    /// `fhog_window_height x fhog_window_width` filter, and computes its SVD
    /// so separable filters can be used (singular values below
    /// `max(1e-4, max(w) * 1e-3)` are rounded to zero, exactly like dlib's
    /// `round_zeros`).
    pub fn build_fhog_filterbank(&self, weights: &Matrix<f64>) -> FhogFilterbank {
        let (width, height) = self.compute_fhog_window_size();
        let width = width as usize;
        let height = height as usize;
        let size = width * height;

        let mut fb = FhogFilterbank {
            filters: Vec::with_capacity(FHOG_NUM_CHANNELS),
            row_filters: vec![Vec::new(); FHOG_NUM_CHANNELS],
            col_filters: vec![Vec::new(); FHOG_NUM_CHANNELS],
        };

        for i in 0..FHOG_NUM_CHANNELS {
            // f = reshape(rowm(weights, range(i*size, (i+1)*size-1)), height, width)
            let mut f = Matrix::zeros(height, width);
            for r in 0..height {
                for c in 0..width {
                    f[(r, c)] = weights[(i * size + r * width + c, 0)];
                }
            }
            fb.filters.push(f.clone().map(|x| x as f32));

            let mut u = Matrix::new();
            let mut w = Matrix::new();
            let mut v = Matrix::new();
            svd3(&f, &mut u, &mut w, &mut v);

            // rsort_columns(u, w) and rsort_columns(v, w2): order the
            // columns of u and v by their singular value, descending.
            let mut order: Vec<usize> = (0..w.nr()).collect();
            order.sort_by(|&a, &b| w[(b, 0)].partial_cmp(&w[(a, 0)]).unwrap_or(Ordering::Equal));

            let sv_max = matrix_max(&w);
            let thresh = 1e-4_f64.max(sv_max * 0.001);

            for &j in &order {
                let val = w[(j, 0)];
                // round_zeros(w, thresh)
                let kept = if val >= thresh || val <= -thresh {
                    val
                } else {
                    0.0
                };
                if kept != 0.0 {
                    let s = kept.sqrt();
                    fb.col_filters[i].push((0..u.nr()).map(|r| (u[(r, j)] * s) as f32).collect());
                    fb.row_filters[i].push((0..v.nr()).map(|r| (v[(r, j)] * s) as f32).collect());
                }
            }
        }

        fb
    }

    /// Port of `scan_fhog_pyramid::detect(const fhog_filterbank&, dets,
    /// thresh)`.
    pub fn detect(&self, w: &FhogFilterbank, dets: &mut Vec<(f64, Rectangle)>, thresh: f64) {
        let (width, height) = self.compute_fhog_window_size();
        detect_from_fhog_pyramid(
            &self.feats,
            w,
            thresh,
            height - 2 * self.padding,
            width - 2 * self.padding,
            self.cell_size,
            height as i64,
            width as i64,
            dets,
        );
    }

    /// Port of `scan_fhog_pyramid::detect(const feature_vector_type&, ...)`
    /// which first builds the filterbank from the raw weights.
    pub fn detect_weights(
        &self,
        weights: &Matrix<f64>,
        dets: &mut Vec<(f64, Rectangle)>,
        thresh: f64,
    ) {
        let fb = self.build_fhog_filterbank(weights);
        self.detect(&fb, dets, thresh);
    }

    /// Port of `scan_fhog_pyramid::get_feature_vector(obj, psi)` where the
    /// detection `obj` is just a rectangle (the fhog scanner produces
    /// part-less detections). Returns a `num_dimensions` x 1 dense column
    /// vector.
    pub fn get_feature_vector(&self, rect: &Rectangle) -> Matrix<f64> {
        let (_mapped_rect, fhog_rect, best_level) =
            self.get_mapped_rect_and_metadata(self.feats.len() as u64, rect);

        let mut psi = Matrix::zeros(self.get_num_dimensions() as usize, 1);
        let level = &self.feats[best_level];
        let plane_nc = level.nc as i64;
        let plane_nr = level.nr as i64;

        let mut i = 0usize;
        for ii in 0..FHOG_NUM_CHANNELS {
            for r in fhog_rect.top()..=fhog_rect.bottom() {
                for c in fhog_rect.left()..=fhog_rect.right() {
                    if c >= 0 && c < plane_nc && r >= 0 && r < plane_nr {
                        psi[i] += f64::from(
                            level.data
                                [ii * level.nr * level.nc + r as usize * level.nc + c as usize],
                        );
                    }
                    i += 1;
                }
            }
        }
        psi
    }

    /// Port of `scan_fhog_pyramid::compute_fhog_window_size()`: the FHOG
    /// filter size is the detection window mapped to fhog cell coordinates
    /// (with 1x1 filter padding) grown by `padding` cells on each side.
    fn compute_fhog_window_size(&self) -> (u64, u64) {
        let rect = centered_rect_point(&Point::new(0, 0), self.window_width, self.window_height);
        let temp = grow_rect(
            &image_to_fhog_rect(&rect, self.cell_size, 1, 1),
            self.padding as i64,
        );
        (temp.width(), temp.height())
    }

    /// Port of `scan_fhog_pyramid::get_mapped_rect_and_metadata()`: finds the
    /// pyramid level whose detection window best matches `rect` and returns
    /// the image-space rectangle, the fhog-space cell rectangle and the level.
    fn get_mapped_rect_and_metadata(
        &self,
        number_pyramid_levels: u64,
        rect: &Rectangle,
    ) -> (Rectangle, Rectangle, usize) {
        let mut best_level: u64 = 0;
        let mut best_match_score = -1.0f64;
        let mut fhog_rect = Rectangle::default();

        let (width, height) = self.compute_fhog_window_size();

        for l in 0..number_pyramid_levels {
            let rect_fhog_space = image_to_fhog_rect(
                &rect_down_levels(rect, l as usize),
                self.cell_size,
                height as i64,
                width as i64,
            );

            let win_image_space = rect_up_levels(
                &fhog_to_image_rect(
                    &centered_rect_point(
                        &center(&rect_fhog_space),
                        width - 2 * self.padding,
                        height - 2 * self.padding,
                    ),
                    self.cell_size,
                    height as i64,
                    width as i64,
                ),
                l as usize,
            );

            let match_score = get_match_score(&win_image_space, rect);
            if match_score > best_match_score {
                best_match_score = match_score;
                best_level = l;
                fhog_rect = centered_rect_point(&center(&rect_fhog_space), width, height);
            }

            if rect_fhog_space.area() <= 1 {
                break;
            }
        }

        let mapped_rect = rect_up_levels(
            &fhog_to_image_rect(
                &shrink_rect(&fhog_rect, self.padding as i64),
                self.cell_size,
                height as i64,
                width as i64,
            ),
            best_level as usize,
        );
        (mapped_rect, fhog_rect, best_level as usize)
    }
}

/// Port of `scan_fhog_pyramid::get_match_score()`.
fn get_match_score(r1: &Rectangle, r2: &Rectangle) -> f64 {
    // make the rectangles overlap as much as possible before computing the
    // match score.
    let r1 = move_rect(r1, &r2.tl_corner());
    r1.intersect(r2).area() as f64 / (r1 + *r2).area() as f64
}

// --------------------------------------------------------------------------------
// deserialization of the feats arrays
// --------------------------------------------------------------------------------

/// Reads a `dlib::array<fhog_image>` (`array<array<array2d<float>>>`):
/// packed max_size, packed size, then each fhog image.
fn read_fhog_image_array(inp: &mut Deserializer) -> Result<Vec<HogImage>, SerializeError> {
    let _max_size = inp.read_u64()?;
    let size = inp.read_u64()? as usize;
    if size > inp.remaining() {
        return Err(SerializeError::Eof);
    }
    let mut v = Vec::with_capacity(size.min(1024));
    for _ in 0..size {
        v.push(read_hog_image(inp)?);
    }
    Ok(v)
}

/// Reads one fhog image: a `dlib::array<array2d<float>>` (packed max_size,
/// packed plane count, then each `array2d<float>`).
fn read_hog_image(inp: &mut Deserializer) -> Result<HogImage, SerializeError> {
    let _max_size = inp.read_u64()?;
    let planes = inp.read_u64()? as usize;
    let mut hog = HogImage::default();
    if planes == 0 {
        return Ok(hog);
    }
    if planes > 4096 {
        return Err(SerializeError::Malformed("fhog plane count"));
    }

    let mut data: Vec<f32> = Vec::new();
    for p in 0..planes {
        let mut nr = inp.read_i64()?;
        let mut nc = inp.read_i64()?;
        if nr < 0 || nc < 0 {
            nr = -nr;
            nc = -nc;
        } else {
            std::mem::swap(&mut nr, &mut nc);
        }
        if nr < 0 || nc < 0 || (nr as u64) * (nc as u64) > inp.remaining() as u64 {
            return Err(SerializeError::Malformed("fhog plane dims"));
        }
        if p == 0 {
            hog.nr = nr as usize;
            hog.nc = nc as usize;
            data = Vec::with_capacity(planes * hog.nr * hog.nc);
        } else if hog.nr != nr as usize || hog.nc != nc as usize {
            return Err(SerializeError::Malformed("inconsistent fhog plane dims"));
        }
        for _ in 0..(nr * nc) {
            data.push(inp.read_f32()?);
        }
    }
    hog.data = data;
    Ok(hog)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob() -> Vec<u8> {
        // The embedded frontal-face model blob: object_detector version int
        // followed by the scanner.
        crate::image_processing::frontal_face_detector::frontal_face_detector_bytes().to_vec()
    }

    #[test]
    fn fhog_coordinate_roundtrip() {
        // image_to_fhog of the 80x80 detection window centered at the origin
        // is the 10x10 cell rectangle (-6..3), exactly as in dlib.
        let rect = centered_rect_point(&Point::new(0, 0), 80, 80);
        let f = image_to_fhog_rect(&rect, 8, 1, 1);
        assert_eq!((f.l(), f.t(), f.r(), f.b()), (-6, -6, 3, 3));
        // Round trip on a positive interior point maps to the cell center
        // (dlib semantics).
        let p = Point::new(100, 60);
        let q = fhog_to_image(image_to_fhog(p, 8, 1, 1), 8, 1, 1);
        assert_eq!((q.x(), q.y()), (101, 61));
    }

    #[test]
    fn fhog_window_size_matches_dlib() {
        let mut scanner = ScanFhogPyramid::new();
        // defaults: cell 8, padding 1, 64x64 window -> 10x10 cells.
        assert_eq!(scanner.get_fhog_window_width(), 10);
        assert_eq!(scanner.get_fhog_window_height(), 10);
        assert_eq!(scanner.get_num_dimensions(), 3100);
        scanner.window_width = 80;
        scanner.window_height = 80;
        assert_eq!(scanner.get_fhog_window_width(), 12);
        assert_eq!(scanner.get_fhog_window_height(), 12);
        assert_eq!(scanner.get_num_dimensions(), 4464);
    }

    #[test]
    fn load_from_embedded_blob() {
        let bytes = blob();
        let mut de = Deserializer::new(&bytes);
        let version = de.read_i32().unwrap();
        assert_eq!(version, 2); // object_detector serialization version
        let mut scanner = ScanFhogPyramid::new();
        scanner.load(&mut de).unwrap();
        assert_eq!(scanner.get_cell_size(), 8);
        // dlib's frontal face model stores padding = 0 and an 80x80
        // detection window, i.e. 10x10-cell filters.
        assert_eq!(scanner.get_padding(), 0);
        assert_eq!(scanner.get_detection_window_width(), 80);
        assert_eq!(scanner.get_detection_window_height(), 80);
        assert_eq!(scanner.get_num_dimensions(), 3100);
        // The blob's saved feature pyramid is empty (config-only scanner).
        assert!(!scanner.is_loaded_with_image());
    }

    #[test]
    fn load_image_and_get_feature_vector() {
        // Deterministic gradient image.
        let mut img: crate::array2d::Array2D<u8> = crate::array2d::Array2D::zeros(64, 64);
        for r in 0..64 {
            for c in 0..64 {
                *img.pixel_mut(r, c) = ((r * 3 + c * 5) % 256) as u8;
            }
        }
        let mut scanner = ScanFhogPyramid::new();
        scanner.load_image(&img);
        assert!(scanner.is_loaded_with_image());
        // A 64x64 image with a 64x64 min pyramid layer size yields a single
        // level (rect_down gives 53x53 which is below the minimum).
        assert_eq!(scanner.num_pyramid_levels(), 1);

        let rect = Rectangle::new(0, 0, 63, 63);
        let psi = scanner.get_feature_vector(&rect);
        assert_eq!(psi.size(), scanner.get_num_dimensions() as usize);
        assert_eq!(psi.size(), 31 * 10 * 10);
        // The features are non-trivial (the gradient image has energy).
        let total: f64 = psi.iter().sum();
        assert!(total > 0.0);
    }

    #[test]
    fn detect_runs_on_synthetic_image() {
        let mut img: crate::array2d::Array2D<u8> = crate::array2d::Array2D::zeros(100, 100);
        for r in 0..100 {
            for c in 0..100 {
                *img.pixel_mut(r, c) = ((r + 2 * c) % 256) as u8;
            }
        }
        let mut scanner = ScanFhogPyramid::new();
        scanner.load_image(&img);
        assert!(scanner.num_pyramid_levels() >= 1);

        // Zero weights (plus bias) produce an all-zero filterbank; the
        // separable path is taken (no nonzero singular values) so this
        // exercises the filtering + coordinate mapping machinery.
        let w = Matrix::zeros(scanner.get_num_dimensions() as usize + 1, 1);
        let fb = scanner.build_fhog_filterbank(&w);
        assert_eq!(fb.filters.len(), 31);
        let mut dets = Vec::new();
        scanner.detect(&fb, &mut dets, 1.0);
        assert!(dets.is_empty()); // zero filters -> saliency 0 < 1
    }

    #[test]
    fn separable_filter_matches_direct_convolution() {
        // The separable SIMD path must produce (near-)identical results to a
        // straightforward reference convolution for a random-ish filter.
        let nr = 13usize;
        let nc = 17usize;
        let mut input = vec![0.0f32; nr * nc];
        let mut s: u32 = 12345;
        let mut next = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / 8388608.0 - 1.0
        };
        for v in input.iter_mut() {
            *v = next();
        }
        let row_f: Vec<f32> = (0..4).map(|_| next()).collect();
        let col_f: Vec<f32> = (0..3).map(|_| next()).collect();

        let mut out = vec![0.0f32; nr * nc];
        let mut scratch = vec![0.0f32; nr * nc];
        float_spatially_filter_image_separable(
            &input,
            nr,
            nc,
            &row_f,
            &col_f,
            false,
            &mut out,
            &mut scratch,
        );

        // Reference: separable convolution computed naively.
        let first_row = col_f.len() as i64 / 2;
        let first_col = row_f.len() as i64 / 2;
        let last_row = nr as i64 - (col_f.len() as i64 - 1) / 2;
        let last_col = nc as i64 - (row_f.len() as i64 - 1) / 2;
        for r in first_row..last_row {
            for c in first_col..last_col {
                let mut acc = 0.0f32;
                for m in 0..col_f.len() as i64 {
                    let mut racc = 0.0f32;
                    for n in 0..row_f.len() as i64 {
                        racc += input
                            [((r - first_row + m) as usize) * nc + (c - first_col + n) as usize]
                            * row_f[n as usize];
                    }
                    acc += racc * col_f[m as usize];
                }
                let got = out[r as usize * nc + c as usize];
                assert!((got - acc).abs() < 1e-3, "at ({r},{c}): {got} vs {acc}");
            }
        }
        // Everything outside the valid area is zeroed.
        for r in 0..nr as i64 {
            for c in 0..nc as i64 {
                if r < first_row || r >= last_row || c < first_col || c >= last_col {
                    assert_eq!(out[r as usize * nc + c as usize], 0.0);
                }
            }
        }
    }
}

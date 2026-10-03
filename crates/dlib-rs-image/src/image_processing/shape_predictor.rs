//! Port of `dlib/image_processing/shape_predictor.h`: the cascaded
//! regression-tree landmark regressor (`shape_predictor`), its
//! serialization format (version 1, byte-compatible with dlib's
//! `shape_predictor_68_face_landmarks.dat`), and the `impl`-namespace
//! helpers shared with `shape_predictor_trainer.h`.

use std::path::Path;

use dlib_rs_core::geometry::{
    find_affine_transform, find_similarity_transform, Dpoint, PointTransformAffine, Rectangle,
};
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, SerializeError, Serializer};

use crate::array2d::GenericImage;
use crate::image_processing::full_object_detection::FullObjectDetection;
use crate::pixel::{get_pixel_intensity, Pixel};

// ----------------------------------------------------------------------------------------
// impl::split_feature / impl::regression_tree

/// Port of `impl::split_feature` (`shape_predictor.h`): a pairwise pixel
/// difference test `features[idx1] - features[idx2] > thresh`. `idx1`/`idx2`
/// index the per-cascade-level feature pixel pool.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitFeature {
    /// First pixel index (`idx1`, an `unsigned long` in dlib).
    pub idx1: u32,
    /// Second pixel index (`idx2`, an `unsigned long` in dlib).
    pub idx2: u32,
    /// Split threshold (`thresh`, a `float` in dlib).
    pub thresh: f32,
}

impl SplitFeature {
    /// Port of `serialize(const impl::split_feature&, std::ostream&)`:
    /// `idx1`, `idx2` (packed `unsigned long`s) then `thresh` (float).
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_u64(self.idx1 as u64);
        out.write_u64(self.idx2 as u64);
        out.write_f32(self.thresh);
    }

    /// Port of `deserialize(impl::split_feature&, std::istream&)`.
    pub fn deserialize(inp: &mut Deserializer<'_>) -> Result<Self, SerializeError> {
        let idx1 = read_ulong_as_u32(inp)?;
        let idx2 = read_ulong_as_u32(inp)?;
        let thresh = inp.read_f32()?;
        Ok(SplitFeature { idx1, idx2, thresh })
    }
}

/// Port of `impl::regression_tree` (`shape_predictor.h`): a complete binary
/// tree stored as an array of splits in breadth-first order plus one leaf
/// displacement vector per leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionTree {
    /// `splits`: `2^tree_depth - 1` split nodes in breadth-first order.
    pub splits: Vec<SplitFeature>,
    /// `leaf_values`: `splits.len() + 1` leaf displacement vectors
    /// (`matrix<float,0,1>` in dlib).
    pub leaf_values: Vec<Matrix<f32>>,
}

impl RegressionTree {
    /// Number of leaves (`num_leaves()`).
    pub fn num_leaves(&self) -> usize {
        self.leaf_values.len()
    }

    /// Port of `impl::regression_tree::operator()`: walks the tree with the
    /// given feature pixel values and returns the index of the selected leaf
    /// (within `leaf_values`). Use `leaf_values[returned]` for the
    /// displacement vector.
    pub fn leaf_index(&self, feature_pixel_values: &[f32]) -> usize {
        let mut i = 0usize;
        while i < self.splits.len() {
            let s = &self.splits[i];
            if feature_pixel_values[s.idx1 as usize] - feature_pixel_values[s.idx2 as usize]
                > s.thresh
            {
                i = 2 * i + 1; // left_child(i)
            } else {
                i = 2 * i + 2; // right_child(i)
            }
        }
        i - self.splits.len()
    }

    /// Port of `serialize(const impl::regression_tree&, std::ostream&)`:
    /// `splits` then `leaf_values`, both as `std::vector`s.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_u64(self.splits.len() as u64);
        for s in &self.splits {
            s.serialize(out);
        }
        out.write_u64(self.leaf_values.len() as u64);
        for leaf in &self.leaf_values {
            leaf.serialize(out);
        }
    }

    /// Port of `deserialize(impl::regression_tree&, std::istream&)`.
    pub fn deserialize(inp: &mut Deserializer<'_>) -> Result<Self, SerializeError> {
        let n = read_count(inp)?;
        let mut splits = Vec::with_capacity(n);
        for _ in 0..n {
            splits.push(SplitFeature::deserialize(inp)?);
        }
        let n = read_count(inp)?;
        let mut leaf_values = Vec::with_capacity(n);
        for _ in 0..n {
            leaf_values.push(Matrix::<f32>::deserialize(inp)?);
        }
        Ok(RegressionTree {
            splits,
            leaf_values,
        })
    }
}

// ----------------------------------------------------------------------------------------
// impl helpers (shared with shape_predictor_trainer)

/// Port of `impl::location(shape, idx)`: returns the `idx`-th point of a
/// shape stored as a flat `[x0, y0, x1, y1, ...]` column vector.
pub(crate) fn location(shape: &Matrix<f32>, idx: usize) -> [f32; 2] {
    [*shape.get(idx * 2, 0), *shape.get(idx * 2 + 1, 0)]
}

/// Port of `impl::nearest_shape_point(shape, pt)`: index of the shape point
/// nearest to `pt`.
pub(crate) fn nearest_shape_point(shape: &Matrix<f32>, pt: &[f32; 2]) -> usize {
    // find the nearest part of the shape to this pixel
    let mut best_dist = f32::INFINITY;
    let num_shape_parts = shape.size() / 2;
    let mut best_idx = 0usize;
    for j in 0..num_shape_parts {
        let l = location(shape, j);
        let dx = l[0] - pt[0];
        let dy = l[1] - pt[1];
        // length_squared() = (double)(x*x + y*y) evaluated in float, then
        // stored into a float by the caller.
        let dist = ((dx * dx + dy * dy) as f64) as f32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = j;
        }
    }
    best_idx
}

/// Port of `impl::create_shape_relative_encoding(shape, pixel_coordinates,
/// anchor_idx, deltas)`: expresses each pixel coordinate as
/// `location(shape, anchor_idx[i]) + deltas[i]`.
pub(crate) fn create_shape_relative_encoding(
    shape: &Matrix<f32>,
    pixel_coordinates: &[[f32; 2]],
) -> (Vec<u32>, Vec<[f32; 2]>) {
    let mut anchor_idx = Vec::with_capacity(pixel_coordinates.len());
    let mut deltas = Vec::with_capacity(pixel_coordinates.len());
    for pc in pixel_coordinates {
        let a = nearest_shape_point(shape, pc);
        anchor_idx.push(a as u32);
        let l = location(shape, a);
        deltas.push([pc[0] - l[0], pc[1] - l[1]]);
    }
    (anchor_idx, deltas)
}

/// Port of `impl::find_tform_between_shapes(from_shape, to_shape)`: the
/// least-squares similarity transform mapping one shape onto the other,
/// returned as dlib stores it in `extract_feature_pixel_values` — the 2x2
/// linear part cast to `float`.
pub(crate) fn find_tform_between_shapes(
    from_shape: &Matrix<f32>,
    to_shape: &Matrix<f32>,
) -> [[f32; 2]; 2] {
    assert!(
        from_shape.size() == to_shape.size()
            && from_shape.size().is_multiple_of(2)
            && from_shape.size() > 0,
        "find_tform_between_shapes(): shapes must be equal-sized, even-length and non-empty"
    );
    let num = from_shape.size() / 2;
    if num == 1 {
        // Just use an identity transform if there is only one landmark.
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    let mut from_points = Vec::with_capacity(num);
    let mut to_points = Vec::with_capacity(num);
    for i in 0..num {
        let f = location(from_shape, i);
        let t = location(to_shape, i);
        from_points.push(Dpoint::new(f[0] as f64, f[1] as f64));
        to_points.push(Dpoint::new(t[0] as f64, t[1] as f64));
    }
    let tform = find_similarity_transform(&from_points, &to_points);
    let m = tform.get_m();
    [
        [m[0][0] as f32, m[0][1] as f32],
        [m[1][0] as f32, m[1][1] as f32],
    ]
}

/// Port of `impl::normalizing_tform(rect)`: maps `rect.tl_corner()` to (0,0)
/// and `rect.br_corner()` to (1,1). (dlib pushes the integer corners through
/// `vector<float,2>`, i.e. `long -> float -> double`; replicated here.)
pub(crate) fn normalizing_tform(rect: &Rectangle) -> PointTransformAffine {
    let tl = rect.tl_corner();
    let tr = rect.tr_corner();
    let br = rect.br_corner();
    let from_points = [
        Dpoint::new(tl.x() as f32 as f64, tl.y() as f32 as f64),
        Dpoint::new(tr.x() as f32 as f64, tr.y() as f32 as f64),
        Dpoint::new(br.x() as f32 as f64, br.y() as f32 as f64),
    ];
    let to_points = [
        Dpoint::new(0.0, 0.0),
        Dpoint::new(1.0, 0.0),
        Dpoint::new(1.0, 1.0),
    ];
    find_affine_transform(&from_points, &to_points)
}

/// Port of `impl::unnormalizing_tform(rect)`: maps (0,0) to
/// `rect.tl_corner()` and (1,1) to `rect.br_corner()`.
pub(crate) fn unnormalizing_tform(rect: &Rectangle) -> PointTransformAffine {
    let tl = rect.tl_corner();
    let tr = rect.tr_corner();
    let br = rect.br_corner();
    let from_points = [
        Dpoint::new(0.0, 0.0),
        Dpoint::new(1.0, 0.0),
        Dpoint::new(1.0, 1.0),
    ];
    let to_points = [
        Dpoint::new(tl.x() as f32 as f64, tl.y() as f32 as f64),
        Dpoint::new(tr.x() as f32 as f64, tr.y() as f32 as f64),
        Dpoint::new(br.x() as f32 as f64, br.y() as f32 as f64),
    ];
    find_affine_transform(&from_points, &to_points)
}

/// Converts a `dpoint` to dlib's integer `point` exactly like the
/// `vector<double,2> -> vector<long,2>` constructor: `floor(v + 0.5)`.
fn dpoint_round_to_point(p: &Dpoint) -> (i64, i64) {
    ((p.x() + 0.5).floor() as i64, (p.y() + 0.5).floor() as i64)
}

/// The feature value stored for one sampled pixel, mirroring
/// `feature_pixel_values[i] = get_pixel_intensity(img[p.y()][p.x()])` where
/// the storage type is the pixel's `basic_pixel_type` (unsigned char for
/// u8/rgb/hsi/lab images). Values are kept as `f32` because every comparison
/// site in dlib casts to `float` anyway.
fn pixel_feature_value<P: Pixel>(p: &P) -> f32 {
    if P::is_gray() {
        // basic_pixel_type == pixel type: stored unconverted.
        get_pixel_intensity(p) as f32
    } else {
        // assign_pixel(unsigned char, color pixel): e.g. rgb -> (r+g+b)/3
        // with integer division, clamped into [0, 255].
        get_pixel_intensity(p).floor().clamp(0.0, 255.0) as f32
    }
}

/// Port of `impl::extract_feature_pixel_values(img, rect, current_shape,
/// reference_shape, reference_pixel_anchor_idx, reference_pixel_deltas,
/// feature_pixel_values)`.
///
/// For each pooled pixel the position is `tform_to_img(tform *
/// reference_pixel_deltas[i] + location(current_shape, anchor_idx[i]))` where
/// `tform` maps the reference shape onto the current one. The resulting
/// coordinates are converted to integer pixel indices by round-to-nearest
/// (`vector<float,2> -> point` uses `floor(x + 0.5)`), i.e. **nearest-pixel
/// sampling, no interpolation**; pixels outside the image read as 0.
pub(crate) fn extract_feature_pixel_values<S>(
    img: &S,
    rect: &Rectangle,
    current_shape: &Matrix<f32>,
    reference_shape: &Matrix<f32>,
    reference_pixel_anchor_idx: &[u32],
    reference_pixel_deltas: &[[f32; 2]],
    feature_pixel_values: &mut Vec<f32>,
) where
    S: GenericImage,
    S::PixelType: Pixel,
{
    let tform = find_tform_between_shapes(reference_shape, current_shape);
    let tform_to_img = unnormalizing_tform(rect);
    // get_rect(img_): the whole image.
    let area = Rectangle::new(
        0,
        0,
        img.num_columns() as i64 - 1,
        img.num_rows() as i64 - 1,
    );

    feature_pixel_values.resize(reference_pixel_deltas.len(), 0.0);
    for i in 0..feature_pixel_values.len() {
        // Compute the point in the current shape corresponding to the i-th
        // pixel and then map it from the normalized shape space into pixel
        // space. All arithmetic through the shape-space product is float,
        // like dlib's expression templates.
        let d = reference_pixel_deltas[i];
        let loc = location(current_shape, reference_pixel_anchor_idx[i] as usize);
        let fx = tform[0][0] * d[0] + tform[0][1] * d[1] + loc[0];
        let fy = tform[1][0] * d[0] + tform[1][1] * d[1] + loc[1];
        let p = tform_to_img.apply(&Dpoint::new(fx as f64, fy as f64));
        let (px, py) = dpoint_round_to_point(&p);
        if area.contains_xy(px, py) {
            feature_pixel_values[i] = pixel_feature_value(img.pixel(py as usize, px as usize));
        } else {
            feature_pixel_values[i] = 0.0;
        }
    }
}

// ----------------------------------------------------------------------------------------
// float column-vector helpers mirroring the dlib matrix expressions used by
// the trainer (element order = storage order for these column vectors).

/// `dst += src` with dlib's empty-matrix semantics: adding to an empty
/// matrix assigns a copy of the source.
pub(crate) fn matrix_acc(dst: &mut Matrix<f32>, src: &Matrix<f32>) {
    if dst.nr() == src.nr() && dst.nc() == src.nc() {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += *s;
        }
    } else {
        *dst = src.clone();
    }
}

/// `a - b` (element-wise), like `matrix` subtraction.
pub(crate) fn matrix_sub(a: &Matrix<f32>, b: &Matrix<f32>) -> Matrix<f32> {
    let mut out = Matrix::zeros(a.nr(), a.nc());
    for k in 0..a.size() {
        *out.get_mut(k, 0) = *a.get(k, 0) - *b.get(k, 0);
    }
    out
}

/// `dot(a, a)` accumulated in float, like dlib's `dot(matrix<float,0,1>)`.
pub(crate) fn matrix_dot_self(a: &Matrix<f32>) -> f32 {
    let mut sum = 0.0f32;
    for v in a.iter() {
        sum += v * v;
    }
    sum
}

/// `pointwise_multiply(a, b)`.
pub(crate) fn matrix_pointwise_mul(a: &Matrix<f32>, b: &Matrix<f32>) -> Matrix<f32> {
    let mut out = Matrix::zeros(a.nr(), a.nc());
    for k in 0..a.size() {
        *out.get_mut(k, 0) = *a.get(k, 0) * *b.get(k, 0);
    }
    out
}

/// `dlib::reciprocal(m)`: `1/x` where `x != 0`, else `0`.
pub(crate) fn matrix_reciprocal(a: &Matrix<f32>) -> Matrix<f32> {
    let mut out = Matrix::zeros(a.nr(), a.nc());
    for k in 0..a.size() {
        let v = *a.get(k, 0);
        *out.get_mut(k, 0) = if v != 0.0 { 1.0 / v } else { 0.0 };
    }
    out
}

/// `dlib::min(m)` over all elements.
pub(crate) fn matrix_min(a: &Matrix<f32>) -> f32 {
    let mut best = f32::INFINITY;
    for v in a.iter() {
        if *v < best {
            best = *v;
        }
    }
    best
}

// ----------------------------------------------------------------------------------------
// shape_predictor

/// Port of `class dlib::shape_predictor`
/// (`dlib/image_processing/shape_predictor.h`): a cascade of random
/// regression forests operating on pixel-difference features sampled
/// relative to the current shape estimate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShapePredictor {
    /// `initial_shape`: the mean shape in the normalized box [0,1]x[0,1],
    /// stored flat as `[x0, y0, x1, y1, ...]` (`matrix<float,0,1>`).
    pub initial_shape: Matrix<f32>,
    /// `forests[cascade]`: the regression trees for each cascade level.
    pub forests: Vec<Vec<RegressionTree>>,
    /// `anchor_idx[cascade][i]`: the initial-shape point that pixel `i` is
    /// anchored to.
    pub anchor_idx: Vec<Vec<u32>>,
    /// `deltas[cascade][i]`: the offset of pixel `i` from its anchor point
    /// (`std::vector<dlib::vector<float,2>>` in dlib).
    pub deltas: Vec<Vec<[f32; 2]>>,
}

impl ShapePredictor {
    /// Port of `shape_predictor(initial_shape, forests, pixel_coordinates)`:
    /// computes the per-cascade `anchor_idx`/`deltas` encodings of the
    /// feature pixel coordinates relative to the initial shape.
    pub fn new(
        initial_shape: Matrix<f32>,
        forests: Vec<Vec<RegressionTree>>,
        pixel_coordinates: &[Vec<[f32; 2]>],
    ) -> Self {
        let mut anchor_idx = Vec::with_capacity(pixel_coordinates.len());
        let mut deltas = Vec::with_capacity(pixel_coordinates.len());
        // Each cascade uses a different set of pixels for its features. We
        // compute their representations relative to the initial shape now and
        // save it.
        for pc in pixel_coordinates {
            let (a, d) = create_shape_relative_encoding(&initial_shape, pc);
            anchor_idx.push(a);
            deltas.push(d);
        }
        ShapePredictor {
            initial_shape,
            forests,
            anchor_idx,
            deltas,
        }
    }

    /// Port of `num_parts()`.
    pub fn num_parts(&self) -> usize {
        self.initial_shape.size() / 2
    }

    /// Port of `num_features()`: total number of leaves over all cascades.
    pub fn num_features(&self) -> usize {
        let mut num = 0;
        for forest in &self.forests {
            for tree in forest {
                num += tree.num_leaves();
            }
        }
        num
    }

    /// Port of `shape_predictor::operator()(img, rect)`: runs the cascade on
    /// `img` inside `box_` and returns the predicted landmark locations.
    ///
    /// The feature pixels are sampled at integer coordinates (round to
    /// nearest); see [`extract_feature_pixel_values`].
    pub fn operator_<S>(&self, img: &S, box_: &Rectangle) -> FullObjectDetection
    where
        S: GenericImage,
        S::PixelType: Pixel,
    {
        let mut current_shape = self.initial_shape.clone();
        let mut feature_pixel_values: Vec<f32> = Vec::new();
        for iter in 0..self.forests.len() {
            extract_feature_pixel_values(
                img,
                box_,
                &current_shape,
                &self.initial_shape,
                &self.anchor_idx[iter],
                &self.deltas[iter],
                &mut feature_pixel_values,
            );
            // evaluate all the trees at this level of the cascade.
            for tree in &self.forests[iter] {
                let leaf = &tree.leaf_values[tree.leaf_index(&feature_pixel_values)];
                for (dst, src) in current_shape.iter_mut().zip(leaf.iter()) {
                    *dst += *src;
                }
            }
        }

        // convert the current_shape into a full_object_detection
        let tform_to_img = unnormalizing_tform(box_);
        let mut parts = Vec::with_capacity(current_shape.size() / 2);
        for i in 0..current_shape.size() / 2 {
            let l = location(&current_shape, i);
            let p = tform_to_img.apply(&Dpoint::new(l[0] as f64, l[1] as f64));
            let (x, y) = dpoint_round_to_point(&p);
            // dlib stores the parts as `point`s, then widens them to dpoints.
            parts.push(Dpoint::new(x as f64, y as f64));
        }
        FullObjectDetection::new(*box_, parts)
    }

    /// Port of `serialize(const shape_predictor&, std::ostream&)`. Field
    /// order: version int 1, `initial_shape`, `forests`, `anchor_idx`,
    /// `deltas`.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_i32(1);
        self.initial_shape.serialize(out);
        out.write_u64(self.forests.len() as u64);
        for forest in &self.forests {
            out.write_u64(forest.len() as u64);
            for tree in forest {
                tree.serialize(out);
            }
        }
        out.write_u64(self.anchor_idx.len() as u64);
        for level in &self.anchor_idx {
            out.write_u64(level.len() as u64);
            for &a in level {
                out.write_u64(a as u64);
            }
        }
        out.write_u64(self.deltas.len() as u64);
        for level in &self.deltas {
            out.write_u64(level.len() as u64);
            for d in level {
                out.write_f32(d[0]);
                out.write_f32(d[1]);
            }
        }
    }

    /// Port of `deserialize(shape_predictor&, std::istream&)`; only version 1
    /// is accepted, exactly like dlib.
    pub fn deserialize(&mut self, inp: &mut Deserializer<'_>) -> Result<(), SerializeError> {
        let version = inp.read_i32()?;
        if version != 1 {
            return Err(SerializeError::Malformed(
                "Unexpected version found while deserializing dlib::shape_predictor.",
            ));
        }
        self.initial_shape = Matrix::<f32>::deserialize(inp)?;

        let n = read_count(inp)?;
        let mut forests = Vec::with_capacity(n);
        for _ in 0..n {
            let n = read_count(inp)?;
            let mut forest = Vec::with_capacity(n);
            for _ in 0..n {
                forest.push(RegressionTree::deserialize(inp)?);
            }
            forests.push(forest);
        }
        self.forests = forests;

        let n = read_count(inp)?;
        let mut anchor_idx = Vec::with_capacity(n);
        for _ in 0..n {
            let n = read_count(inp)?;
            let mut level = Vec::with_capacity(n);
            for _ in 0..n {
                level.push(read_ulong_as_u32(inp)?);
            }
            anchor_idx.push(level);
        }
        self.anchor_idx = anchor_idx;

        let n = read_count(inp)?;
        let mut deltas = Vec::with_capacity(n);
        for _ in 0..n {
            let n = read_count(inp)?;
            let mut level = Vec::with_capacity(n);
            for _ in 0..n {
                let x = inp.read_f32()?;
                let y = inp.read_f32()?;
                level.push([x, y]);
            }
            deltas.push(level);
        }
        self.deltas = deltas;
        Ok(())
    }

    /// Convenience wrapper: deserializes a `shape_predictor` from a file
    /// written by dlib's `serialize()` (e.g.
    /// `shape_predictor_68_face_landmarks.dat`).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<ShapePredictor, SerializeError> {
        let data = std::fs::read(path)?;
        let mut inp = Deserializer::new(&data);
        let mut sp = ShapePredictor::default();
        sp.deserialize(&mut inp)?;
        Ok(sp)
    }
}

/// Reads one `unsigned long` (packed u64) and narrows it to `u32`; dlib's
/// index fields are `unsigned long` but only small values ever occur.
fn read_ulong_as_u32(inp: &mut Deserializer<'_>) -> Result<u32, SerializeError> {
    let v = inp.read_u64()?;
    u32::try_from(v)
        .map_err(|_| SerializeError::Malformed("shape_predictor index out of u32 range"))
}

/// Reads a `std::vector` element count, rejecting absurd lengths early so a
/// corrupt stream cannot request a huge allocation (each serialized element
/// consumes at least one input byte).
fn read_count(inp: &mut Deserializer<'_>) -> Result<usize, SerializeError> {
    let len = inp.read_u64()?;
    if len > inp.remaining() as u64 {
        return Err(SerializeError::Malformed(
            "vector length exceeds remaining input",
        ));
    }
    Ok(len as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;

    fn dat_path() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/shape_predictor_68_face_landmarks.dat"
        )
        .to_string()
    }

    #[test]
    fn test_load_68_landmark_dat() {
        let path = dat_path();
        if !std::path::Path::new(&path).exists() {
            return; // CI without the 100MB asset
        }
        let sp = ShapePredictor::load_from_file(&path).unwrap();
        assert_eq!(sp.num_parts(), 68);
        assert!(!sp.forests.is_empty());
        assert!(sp.forests.iter().all(|f| !f.is_empty()));
        assert!(sp
            .forests
            .iter()
            .all(|f| f.iter().all(|t| t.num_leaves() > 0)));
        assert_eq!(sp.anchor_idx.len(), sp.forests.len());
        assert_eq!(sp.deltas.len(), sp.forests.len());
        assert!(sp.num_features() > 0);
        // anchor indices must be valid part indices
        for level in &sp.anchor_idx {
            assert!(!level.is_empty());
            assert!(level.iter().all(|&a| (a as usize) < sp.num_parts()));
        }
    }

    #[test]
    fn test_inference_smoke_on_dat() {
        let path = dat_path();
        if !std::path::Path::new(&path).exists() {
            return; // CI without the 100MB asset
        }
        let sp = ShapePredictor::load_from_file(&path).unwrap();
        // deterministic synthetic image
        let mut img = Array2D::<u8>::zeros(120, 120);
        for r in 0..120 {
            for c in 0..120 {
                *img.get_mut(r, c) = ((r * 7 + c * 13) % 256) as u8;
            }
        }
        let rect = Rectangle::new(20, 20, 99, 99);
        let det = sp.operator_(&img, &rect);
        assert_eq!(det.num_parts(), 68);
        assert_eq!(*det.get_rect(), rect);
        for k in 0..68 {
            let p = det.part(k);
            assert!(p.x().is_finite() && p.y().is_finite());
        }
    }

    #[test]
    fn test_handmade_predictor_inference_and_roundtrip() {
        // 2 parts at (0.25,0.25) and (0.75,0.75) in the unit box; a single
        // tree with no splits and zero leaf delta, so the output is exactly
        // the unnormalizing tform applied to the initial shape.
        let mut initial_shape = Matrix::zeros(4, 1);
        *initial_shape.get_mut(0, 0) = 0.25;
        *initial_shape.get_mut(1, 0) = 0.25;
        *initial_shape.get_mut(2, 0) = 0.75;
        *initial_shape.get_mut(3, 0) = 0.75;
        let mut leaf = Matrix::zeros(4, 1);
        for k in 0..4 {
            *leaf.get_mut(k, 0) = 0.0;
        }
        let tree = RegressionTree {
            splits: vec![],
            leaf_values: vec![leaf],
        };
        let pixel_coordinates = vec![vec![[0.25f32, 0.25f32]]];
        let sp = ShapePredictor::new(initial_shape, vec![vec![tree]], &pixel_coordinates);
        assert_eq!(sp.num_parts(), 2);
        assert_eq!(sp.num_features(), 1);

        let mut img = Array2D::<u8>::zeros(20, 20);
        for r in 0..20 {
            for c in 0..20 {
                *img.get_mut(r, c) = ((r + c) % 256) as u8;
            }
        }
        let rect = Rectangle::new(0, 0, 19, 19);
        let det = sp.operator_(&img, &rect);
        // unnormalizing tform: (x,y) -> (19x, 19y), rounded to nearest by the
        // point conversion: (0.25,0.25)->(4.75,4.75)->(5,5); (0.75,0.75)->
        // (14.25,14.25)->(14,14).
        assert_eq!(*det.part(0), Dpoint::new(5.0, 5.0));
        assert_eq!(*det.part(1), Dpoint::new(14.0, 14.0));

        // serialization round-trip
        let mut ser = Serializer::new();
        sp.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let mut sp2 = ShapePredictor::default();
        sp2.deserialize(&mut de).unwrap();
        assert_eq!(sp, sp2);
        assert_eq!(de.remaining(), 0);
        // and the version byte stream starts with version=1
        assert_eq!(bytes[0], 0x01);
    }

    #[test]
    fn test_bad_version_rejected() {
        let mut ser = Serializer::new();
        ser.write_i32(3);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let mut sp = ShapePredictor::default();
        assert!(sp.deserialize(&mut de).is_err());
    }
}

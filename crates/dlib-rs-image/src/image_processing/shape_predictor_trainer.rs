//! Port of `dlib/image_processing/shape_predictor_trainer.h`: the trainer
//! that fits the cascaded regression forests of a `shape_predictor`, using
//! `dlib_rs_core::rand::Rand` (a bit-exact port of `dlib::rand`) so equal
//! seeds produce identical trees.

use std::collections::VecDeque;

use dlib_rs_core::geometry::Rectangle;
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::rand::Rand;

use crate::array2d::GenericImage;
use crate::image_processing::full_object_detection::{
    object_part_not_present, FullObjectDetection,
};
use crate::image_processing::shape_predictor::{
    create_shape_relative_encoding, extract_feature_pixel_values, matrix_acc, matrix_dot_self,
    matrix_min, matrix_pointwise_mul, matrix_reciprocal, matrix_sub, normalizing_tform,
    RegressionTree, ShapePredictor, SplitFeature,
};

use crate::pixel::Pixel;

/// Port of `shape_predictor_trainer::padding_mode_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingMode {
    /// `bounding_box_relative`.
    BoundingBoxRelative,
    /// `landmark_relative` (the default).
    LandmarkRelative,
}

/// Port of `class dlib::shape_predictor_trainer`
/// (`dlib/image_processing/shape_predictor_trainer.h`).
///
/// This trainer really only works with `unsigned char` or `rgb_pixel` images
/// (since the split thresholds are sampled in the range [-128, 128]).
#[derive(Clone, Debug)]
pub struct ShapePredictorTrainer {
    rnd: Rand,
    cascade_depth: u64,
    tree_depth: u64,
    num_trees_per_cascade_level: u64,
    nu: f64,
    oversampling_amount: u64,
    feature_pool_size: u64,
    lambda: f64,
    num_test_splits: u64,
    feature_pool_region_padding: f64,
    num_threads: u64,
    padding_mode: PaddingMode,
    oversampling_translation_jitter: f64,
    /// Kept for API parity; console progress printing is not ported.
    #[allow(dead_code)]
    verbose: bool,
}

impl Default for ShapePredictorTrainer {
    /// Port of the `shape_predictor_trainer()` default constructor.
    fn default() -> Self {
        ShapePredictorTrainer {
            rnd: Rand::new(),
            cascade_depth: 10,
            tree_depth: 4,
            num_trees_per_cascade_level: 500,
            nu: 0.1,
            oversampling_amount: 20,
            oversampling_translation_jitter: 0.0,
            feature_pool_size: 400,
            lambda: 0.1,
            num_test_splits: 20,
            feature_pool_region_padding: 0.0,
            verbose: false,
            num_threads: 0,
            padding_mode: PaddingMode::LandmarkRelative,
        }
    }
}

impl ShapePredictorTrainer {
    /// Equivalent to the default constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `get_cascade_depth()`.
    pub fn get_cascade_depth(&self) -> u64 {
        self.cascade_depth
    }

    /// Port of `set_cascade_depth(depth)` (`DLIB_CASSERT(depth > 0)`).
    pub fn set_cascade_depth(&mut self, depth: u64) -> &mut Self {
        assert!(depth > 0, "set_cascade_depth(): depth must be > 0");
        self.cascade_depth = depth;
        self
    }

    /// Port of `get_tree_depth()`.
    pub fn get_tree_depth(&self) -> u64 {
        self.tree_depth
    }

    /// Port of `set_tree_depth(depth)` (`DLIB_CASSERT(depth > 0)`).
    pub fn set_tree_depth(&mut self, depth: u64) -> &mut Self {
        assert!(depth > 0, "set_tree_depth(): depth must be > 0");
        self.tree_depth = depth;
        self
    }

    /// Port of `get_num_trees_per_cascade_level()`.
    pub fn get_num_trees_per_cascade_level(&self) -> u64 {
        self.num_trees_per_cascade_level
    }

    /// Port of `set_num_trees_per_cascade_level(num)`
    /// (`DLIB_CASSERT(num > 0)`).
    pub fn set_num_trees_per_cascade_level(&mut self, num: u64) -> &mut Self {
        assert!(
            num > 0,
            "set_num_trees_per_cascade_level(): num must be > 0"
        );
        self.num_trees_per_cascade_level = num;
        self
    }

    /// Port of `get_nu()`.
    pub fn get_nu(&self) -> f64 {
        self.nu
    }

    /// Port of `set_nu(nu)` (`DLIB_CASSERT(0 < nu && nu <= 1)`).
    pub fn set_nu(&mut self, nu: f64) -> &mut Self {
        assert!(
            0.0 < nu && nu <= 1.0,
            "set_nu(): require 0 < nu <= 1, got {nu}"
        );
        self.nu = nu;
        self
    }

    /// Port of `get_random_seed()` (the seed string of the internal
    /// `dlib::rand`).
    pub fn get_random_seed(&self) -> &str {
        self.rnd.get_seed()
    }

    /// Port of `set_random_seed(seed)`.
    pub fn set_random_seed(&mut self, seed: &str) -> &mut Self {
        self.rnd.set_seed(seed);
        self
    }

    /// Port of `get_oversampling_amount()`.
    pub fn get_oversampling_amount(&self) -> u64 {
        self.oversampling_amount
    }

    /// Port of `set_oversampling_amount(amount)`
    /// (`DLIB_CASSERT(amount > 0)`).
    pub fn set_oversampling_amount(&mut self, amount: u64) -> &mut Self {
        assert!(amount > 0, "set_oversampling_amount(): amount must be > 0");
        self.oversampling_amount = amount;
        self
    }

    /// Port of `get_oversampling_translation_jitter()`.
    pub fn get_oversampling_translation_jitter(&self) -> f64 {
        self.oversampling_translation_jitter
    }

    /// Port of `set_oversampling_translation_jitter(amount)`
    /// (`DLIB_CASSERT(amount >= 0)`).
    pub fn set_oversampling_translation_jitter(&mut self, amount: f64) -> &mut Self {
        assert!(
            amount >= 0.0,
            "set_oversampling_translation_jitter(): amount must be >= 0"
        );
        self.oversampling_translation_jitter = amount;
        self
    }

    /// Port of `get_feature_pool_size()`.
    pub fn get_feature_pool_size(&self) -> u64 {
        self.feature_pool_size
    }

    /// Port of `set_feature_pool_size(size)` (`DLIB_CASSERT(size > 1)`).
    pub fn set_feature_pool_size(&mut self, size: u64) -> &mut Self {
        assert!(size > 1, "set_feature_pool_size(): size must be > 1");
        self.feature_pool_size = size;
        self
    }

    /// Port of `get_lambda()`.
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }

    /// Port of `set_lambda(lambda)` (`DLIB_CASSERT(lambda > 0)`).
    pub fn set_lambda(&mut self, lambda: f64) -> &mut Self {
        assert!(lambda > 0.0, "set_lambda(): lambda must be > 0");
        self.lambda = lambda;
        self
    }

    /// Port of `get_num_test_splits()`.
    pub fn get_num_test_splits(&self) -> u64 {
        self.num_test_splits
    }

    /// Port of `set_num_test_splits(num)` (`DLIB_CASSERT(num > 0)`).
    pub fn set_num_test_splits(&mut self, num: u64) -> &mut Self {
        assert!(num > 0, "set_num_test_splits(): num must be > 0");
        self.num_test_splits = num;
        self
    }

    /// Port of `set_padding_mode(mode)`.
    pub fn set_padding_mode(&mut self, mode: PaddingMode) -> &mut Self {
        self.padding_mode = mode;
        self
    }

    /// Port of `get_padding_mode()`.
    pub fn get_padding_mode(&self) -> PaddingMode {
        self.padding_mode
    }

    /// Port of `get_feature_pool_region_padding()`.
    pub fn get_feature_pool_region_padding(&self) -> f64 {
        self.feature_pool_region_padding
    }

    /// Port of `set_feature_pool_region_padding(padding)`
    /// (`DLIB_CASSERT(padding > -0.5)`).
    pub fn set_feature_pool_region_padding(&mut self, padding: f64) -> &mut Self {
        assert!(
            padding > -0.5,
            "set_feature_pool_region_padding(): padding must be > -0.5"
        );
        self.feature_pool_region_padding = padding;
        self
    }

    /// Port of `be_verbose()`.
    pub fn be_verbose(&mut self) -> &mut Self {
        self.verbose = true;
        self
    }

    /// Port of `be_quiet()`.
    pub fn be_quiet(&mut self) -> &mut Self {
        self.verbose = false;
        self
    }

    /// Port of `get_num_threads()`.
    pub fn get_num_threads(&self) -> u64 {
        self.num_threads
    }

    /// Port of `set_num_threads(num)`. Values greater than 1 reproduce
    /// dlib's thread-pool block-partitioned summation order (computed
    /// deterministically here, without threads).
    pub fn set_num_threads(&mut self, num: u64) -> &mut Self {
        self.num_threads = num;
        self
    }

    /// Port of `shape_predictor_trainer::train(images, objects)`: fits the
    /// cascade of regression trees on the training objects and returns the
    /// resulting [`ShapePredictor`].
    pub fn train<I>(&self, images: &[I], objects: &[Vec<FullObjectDetection>]) -> ShapePredictor
    where
        I: GenericImage,
        I::PixelType: Pixel,
    {
        assert!(
            images.len() == objects.len() && !objects.is_empty(),
            "train(): images.size() {} must equal objects.size() {} and be > 0",
            images.len(),
            objects.len()
        );
        // make sure the objects agree on the number of parts and that there
        // is at least one full_object_detection.
        let mut num_parts = 0usize;
        let mut part_present: Vec<i32> = Vec::new();
        for objs in objects {
            for obj in objs {
                if num_parts == 0 {
                    num_parts = obj.num_parts();
                    assert!(
                        num_parts != 0,
                        "train(): you can't give objects that don't have any parts to the trainer."
                    );
                    part_present.resize(num_parts, 0);
                } else {
                    assert!(
                        obj.num_parts() == num_parts,
                        "train(): all the objects must agree on the number of parts."
                    );
                }
                for (p, flag) in part_present.iter_mut().enumerate() {
                    if *obj.part(p) != object_part_not_present() {
                        *flag = 1;
                    }
                }
            }
        }
        assert!(
            num_parts != 0,
            "train(): you must give at least one full_object_detection with parts."
        );
        assert!(
            part_present.iter().sum::<i32>() == num_parts as i32,
            "train(): each part must appear at least once in this training data."
        );

        // rnd.set_seed(get_random_seed()): every train() call restarts from
        // the stored seed string, so identical seeds give identical models.
        let mut rnd = Rand::with_seed(self.rnd.get_seed());

        let mut samples: Vec<TrainingSample> = Vec::new();
        let initial_shape = self.populate_training_sample_shapes(objects, &mut samples, &mut rnd);
        let pixel_coordinates = self.randomly_sample_pixel_coordinates(&initial_shape, &mut rnd);

        let mut forests: Vec<Vec<RegressionTree>> = vec![Vec::new(); self.cascade_depth as usize];
        // Now start doing the actual training by filling in the forests
        for cascade in 0..self.cascade_depth as usize {
            // Each cascade uses a different set of pixels for its features.
            // We compute their representations relative to the initial shape
            // first.
            let (anchor_idx, deltas) =
                create_shape_relative_encoding(&initial_shape, &pixel_coordinates[cascade]);

            // First compute the feature_pixel_values for each training sample
            // at this level of the cascade.
            for sample in samples.iter_mut() {
                let img = &images[sample.image_idx];
                extract_feature_pixel_values(
                    img,
                    &sample.rect,
                    &sample.current_shape,
                    &initial_shape,
                    &anchor_idx,
                    &deltas,
                    &mut sample.feature_pixel_values,
                );
            }

            // Now start building the trees at this cascade level.
            for _ in 0..self.num_trees_per_cascade_level {
                let tree =
                    self.make_regression_tree(&mut samples, &pixel_coordinates[cascade], &mut rnd);
                forests[cascade].push(tree);
            }
        }

        ShapePredictor::new(initial_shape, forests, &pixel_coordinates)
    }

    // --------------------------------------------------------------------------------

    /// Port of the static `object_to_shape(obj, shape, present)`: normalizes
    /// the object's parts into the unit box of its rectangle.
    fn object_to_shape(obj: &FullObjectDetection) -> (Matrix<f32>, Matrix<f32>) {
        let mut shape = Matrix::zeros(obj.num_parts() * 2, 1);
        let mut present = Matrix::zeros(obj.num_parts() * 2, 1);
        let tform_from_img = normalizing_tform(obj.get_rect());
        for i in 0..obj.num_parts() {
            if *obj.part(i) != object_part_not_present() {
                let p = tform_from_img.apply(obj.part(i));
                *shape.get_mut(2 * i, 0) = p.x() as f32;
                *shape.get_mut(2 * i + 1, 0) = p.y() as f32;
                *present.get_mut(2 * i, 0) = 1.0;
                *present.get_mut(2 * i + 1, 0) = 1.0;
            }
        }
        (shape, present)
    }

    /// Port of `populate_training_sample_shapes(objects, samples)`: builds
    /// the (oversampled) training samples and returns the mean shape.
    fn populate_training_sample_shapes(
        &self,
        objects: &[Vec<FullObjectDetection>],
        samples: &mut Vec<TrainingSample>,
        rnd: &mut Rand,
    ) -> Matrix<f32> {
        samples.clear();
        let mut mean_shape = Matrix::new();
        let mut count = Matrix::new();
        // first fill out the target shapes
        for (i, objs) in objects.iter().enumerate() {
            for obj in objs {
                let rect: Rectangle = *obj.get_rect();
                let (target_shape, present) = Self::object_to_shape(obj);
                let sample = TrainingSample {
                    image_idx: i,
                    rect,
                    target_shape: target_shape.clone(),
                    present: present.clone(),
                    current_shape: Matrix::new(),
                    diff_shape: Matrix::new(),
                    feature_pixel_values: Vec::new(),
                };
                for _ in 0..self.oversampling_amount {
                    samples.push(sample.clone());
                }
                matrix_acc(&mut mean_shape, &target_shape);
                matrix_acc(&mut count, &present);
            }
        }

        mean_shape = matrix_pointwise_mul(&mean_shape, &matrix_reciprocal(&count));

        // now go pick random initial shapes
        for i in 0..samples.len() {
            if (i as u64).is_multiple_of(self.oversampling_amount) {
                // The mean shape is what we really use as an initial shape so
                // always include it in the training set as an example
                // starting shape.
                samples[i].current_shape = mean_shape.clone();
            } else {
                let mut current = Matrix::new();
                let mut hits = Matrix::zeros(mean_shape.nr(), 1);

                let mut iter = 0;
                // Pick a few samples at random and randomly average them
                // together to make the initial shape.  Note that we make sure
                // we get at least one observation (i.e. non
                // OBJECT_PART_NOT_PRESENT) on each part location.
                while matrix_min(&hits) == 0.0 || iter < 2 {
                    iter += 1;
                    let rand_idx =
                        (rnd.get_random_32bit_number() as u64 % samples.len() as u64) as usize;
                    let alpha = rnd.get_random_double() + 0.1;
                    let alpha_f = alpha as f32;
                    // current_shape += alpha * target_shape
                    if current.size() == 0 {
                        current = samples[rand_idx].target_shape.clone();
                        for v in current.iter_mut() {
                            *v *= alpha_f;
                        }
                    } else {
                        for k in 0..current.size() {
                            *current.get_mut(k, 0) +=
                                alpha_f * *samples[rand_idx].target_shape.get(k, 0);
                        }
                    }
                    // hits += alpha * present
                    for k in 0..hits.size() {
                        *hits.get_mut(k, 0) += alpha_f * *samples[rand_idx].present.get(k, 0);
                    }
                }
                current = matrix_pointwise_mul(&current, &matrix_reciprocal(&hits));

                if self.oversampling_translation_jitter != 0.0 {
                    let off_x = rnd.get_double_in_range(
                        -self.oversampling_translation_jitter,
                        self.oversampling_translation_jitter,
                    );
                    let off_y = rnd.get_double_in_range(
                        -self.oversampling_translation_jitter,
                        self.oversampling_translation_jitter,
                    );
                    for j in 0..current.size() / 2 {
                        // float += double: add in double, round back to float.
                        *current.get_mut(2 * j, 0) =
                            ((*current.get(2 * j, 0) as f64) + off_x) as f32;
                        *current.get_mut(2 * j + 1, 0) =
                            ((*current.get(2 * j + 1, 0) as f64) + off_y) as f32;
                    }
                }

                samples[i].current_shape = current;
            }
        }
        for sample in samples.iter_mut() {
            for k in 0..sample.present.size() {
                // if this part is not present
                if *sample.present.get(k, 0) == 0.0 {
                    let v = *sample.current_shape.get(k, 0);
                    *sample.target_shape.get_mut(k, 0) = v;
                }
            }
        }

        mean_shape
    }

    /// Port of `randomly_sample_pixel_coordinates(pixel_coordinates, min_x,
    /// min_y, max_x, max_y)`.
    fn randomly_sample_pixel_coordinates_in_box(
        &self,
        pixel_coordinates: &mut Vec<[f32; 2]>,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        rnd: &mut Rand,
    ) {
        pixel_coordinates.resize(self.feature_pool_size as usize, [0.0, 0.0]);
        for p in pixel_coordinates.iter_mut() {
            p[0] = (rnd.get_random_double() * (max_x - min_x) + min_x) as f32;
            p[1] = (rnd.get_random_double() * (max_y - min_y) + min_y) as f32;
        }
    }

    /// Port of `randomly_sample_pixel_coordinates(initial_shape)`: uniform
    /// sampling inside the (padded) bounding box of the initial shape.
    fn randomly_sample_pixel_coordinates(
        &self,
        initial_shape: &Matrix<f32>,
        rnd: &mut Rand,
    ) -> Vec<Vec<[f32; 2]>> {
        let padding = self.feature_pool_region_padding;
        // Figure out the bounds on the object shapes. We will sample
        // uniformly from this box. reshape(initial_shape, size/2, 2) puts the
        // x coordinates in column 0 and the y coordinates in column 1.
        let mut min_x = *initial_shape.get(0, 0);
        let mut min_y = *initial_shape.get(1, 0);
        let mut max_x = min_x;
        let mut max_y = min_y;
        for i in 1..initial_shape.size() / 2 {
            let x = *initial_shape.get(2 * i, 0);
            let y = *initial_shape.get(2 * i + 1, 0);
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
        let mut min_x = min_x as f64;
        let mut min_y = min_y as f64;
        let mut max_x = max_x as f64;
        let mut max_y = max_y as f64;

        if self.padding_mode == PaddingMode::BoundingBoxRelative {
            min_x = 0.0f64.min(min_x);
            min_y = 0.0f64.min(min_y);
            max_x = 1.0f64.max(max_x);
            max_y = 1.0f64.max(max_y);
        }

        min_x -= padding;
        min_y -= padding;
        max_x += padding;
        max_y += padding;

        let mut pixel_coordinates: Vec<Vec<[f32; 2]>> =
            vec![Vec::new(); self.cascade_depth as usize];
        for coords in pixel_coordinates.iter_mut() {
            self.randomly_sample_pixel_coordinates_in_box(coords, min_x, min_y, max_x, max_y, rnd);
        }
        pixel_coordinates
    }

    /// Port of `make_regression_tree(samples, pixel_coordinates)`: fits one
    /// regression tree breadth-first and updates the samples' current shapes.
    fn make_regression_tree(
        &self,
        samples: &mut [TrainingSample],
        pixel_coordinates: &[[f32; 2]],
        rnd: &mut Rand,
    ) -> RegressionTree {
        let mut parts: VecDeque<(usize, usize)> = VecDeque::new();
        parts.push_back((0, samples.len()));

        let mut tree = RegressionTree {
            splits: Vec::new(),
            leaf_values: Vec::new(),
        };

        // walk the tree in breadth first order
        let num_split_nodes = ((1u64 << self.tree_depth) - 1) as usize;
        let mut sums: Vec<Matrix<f32>> = vec![Matrix::new(); num_split_nodes * 2 + 1];
        if self.num_threads > 1 {
            // Mirrors dlib's thread-pool path: each worker accumulates a
            // block sum, then the blocks are added in order. The arithmetic
            // (hence the result) is identical to the C++ multithreaded run,
            // evaluated here without threads.
            let num_workers = self.num_threads as usize;
            let num = samples.len();
            let block_size = std::cmp::max(1, num.div_ceil(num_workers));
            let mut block_sums: Vec<Matrix<f32>> = vec![Matrix::new(); num_workers];
            for (block, block_sum) in block_sums.iter_mut().enumerate() {
                let block_begin = block * block_size;
                let block_end = std::cmp::min(num, block_begin + block_size);
                for s in samples.iter_mut().take(block_end).skip(block_begin) {
                    s.diff_shape = matrix_sub(&s.target_shape, &s.current_shape);
                    matrix_acc(block_sum, &s.diff_shape);
                }
            }
            // now calculate the total result from separate blocks
            for b in &block_sums {
                matrix_acc(&mut sums[0], b);
            }
        } else {
            // synchronous implementation
            for s in samples.iter_mut() {
                s.diff_shape = matrix_sub(&s.target_shape, &s.current_shape);
                matrix_acc(&mut sums[0], &s.diff_shape);
            }
        }

        for i in 0..num_split_nodes {
            let (begin, end) = parts.pop_front().unwrap();

            let (split, left_sum, right_sum) =
                self.generate_split(samples, begin, end, pixel_coordinates, &sums[i], rnd);
            tree.splits.push(split);
            let mid = self.partition_samples(&tree.splits[i], samples, begin, end);

            parts.push_back((begin, mid));
            parts.push_back((mid, end));
            sums[2 * i + 1] = left_sum; // sums[left_child(i)]
            sums[2 * i + 2] = right_sum; // sums[right_child(i)]
        }

        // Now all the parts contain the ranges for the leaves so we can use
        // them to compute the average leaf values.
        let target_nr = samples[0].target_shape.nr();
        let mut present_counts;
        tree.leaf_values = vec![Matrix::new(); parts.len()];
        let nu_f = self.nu as f32;
        for i in 0..parts.len() {
            let (begin, end) = parts[i];
            // Get the present counts for each dimension so we can divide each
            // dimension by the number of observations we have on it to find
            // the mean displacement in each leaf.
            present_counts = Matrix::zeros(target_nr, 1);
            for s in samples.iter().take(end).skip(begin) {
                matrix_acc(&mut present_counts, &s.present);
            }
            present_counts = matrix_reciprocal(&present_counts);

            if end != begin {
                // leaf_values[i] = pointwise_multiply(present_counts,
                //                                     sums[num_split_nodes+i]*nu)
                let mut scaled = Matrix::zeros(target_nr, 1);
                for k in 0..target_nr {
                    *scaled.get_mut(k, 0) = *sums[num_split_nodes + i].get(k, 0) * nu_f;
                }
                tree.leaf_values[i] = matrix_pointwise_mul(&present_counts, &scaled);
            } else {
                tree.leaf_values[i] = Matrix::zeros(target_nr, 1);
            }

            // now adjust the current shape based on these predictions
            let leaf = tree.leaf_values[i].clone();
            for s in samples.iter_mut().take(end).skip(begin) {
                for k in 0..s.current_shape.size() {
                    *s.current_shape.get_mut(k, 0) += *leaf.get(k, 0);
                }
                // For parts that aren't present in the training data, we just
                // make sure that the target shape always matches and
                // therefore gives zero error.  So this makes the algorithm
                // simply ignore non-present landmarks.
                for k in 0..s.present.size() {
                    // if this part is not present
                    if *s.present.get(k, 0) == 0.0 {
                        let v = *s.current_shape.get(k, 0);
                        *s.target_shape.get_mut(k, 0) = v;
                    }
                }
            }
        }

        tree
    }

    /// Port of `randomly_generate_split_feature(pixel_coordinates)`: draws a
    /// pair of feature-pool pixels with probability `exp(-dist/lambda)` of
    /// being accepted per try, and a threshold uniform in `[-128, 128]/2`.
    fn randomly_generate_split_feature(
        &self,
        pixel_coordinates: &[[f32; 2]],
        rnd: &mut Rand,
    ) -> SplitFeature {
        let lambda = self.lambda;
        let mut feat = SplitFeature {
            idx1: 0,
            idx2: 0,
            thresh: 0.0,
        };
        let max_iters = (self.feature_pool_size * self.feature_pool_size) as usize;
        for _ in 0..max_iters {
            feat.idx1 = rnd.get_integer(self.feature_pool_size as i64) as u32;
            feat.idx2 = rnd.get_integer(self.feature_pool_size as i64) as u32;
            while feat.idx1 == feat.idx2 {
                feat.idx2 = rnd.get_integer(self.feature_pool_size as i64) as u32;
            }
            let d1 = pixel_coordinates[feat.idx1 as usize];
            let d2 = pixel_coordinates[feat.idx2 as usize];
            // length(vector<float,2> - vector<float,2>) = sqrt((double)(x*x +
            // y*y)) with the products in float.
            let dx = d1[0] - d2[0];
            let dy = d1[1] - d2[1];
            let dist = ((dx * dx + dy * dy) as f64).sqrt();
            let accept_prob = (-dist / lambda).exp();
            if accept_prob > rnd.get_random_double() {
                break;
            }
        }

        feat.thresh = ((rnd.get_random_double() * 256.0 - 128.0) / 2.0) as f32;

        feat
    }

    /// Port of `generate_split(samples, begin, end, pixel_coordinates, sum,
    /// left_sum, right_sum)`: tries `num_test_splits` random splits and keeps
    /// the one maximizing the between-side sum-of-deviations score. Returns
    /// `(split, left_sum, right_sum)`.
    fn generate_split(
        &self,
        samples: &[TrainingSample],
        begin: usize,
        end: usize,
        pixel_coordinates: &[[f32; 2]],
        sum: &Matrix<f32>,
        rnd: &mut Rand,
    ) -> (SplitFeature, Matrix<f32>, Matrix<f32>) {
        // generate a bunch of random splits and test them and return the best
        // one.
        let num_test_splits = self.num_test_splits as usize;

        // sample the random features we test in this function
        let mut feats: Vec<SplitFeature> = Vec::with_capacity(num_test_splits);
        for _ in 0..num_test_splits {
            feats.push(self.randomly_generate_split_feature(pixel_coordinates, rnd));
        }

        let mut left_sums: Vec<Matrix<f32>> = vec![Matrix::new(); num_test_splits];
        let mut left_cnt: Vec<u64> = vec![0; num_test_splits];

        // now compute the sums of vectors that go left for each feature
        for s in samples.iter().take(end).skip(begin) {
            for i in 0..num_test_splits {
                let f = &feats[i];
                let fv = &s.feature_pixel_values;
                if fv[f.idx1 as usize] - fv[f.idx2 as usize] > f.thresh {
                    matrix_acc(&mut left_sums[i], &s.diff_shape);
                    left_cnt[i] += 1;
                }
            }
        }

        // now figure out which feature is the best
        let mut best_score = -1.0f64;
        let mut best_feat = 0usize;
        for i in 0..num_test_splits {
            // check how well the feature splits the space.
            let right_cnt = (end - begin) as u64 - left_cnt[i];
            if left_cnt[i] != 0 && right_cnt != 0 {
                // score = dot(l,l)/left_cnt + dot(t,t)/right_cnt, evaluated
                // in float like dlib's dot()/scalar arithmetic, then stored
                // into a double.
                let temp = matrix_sub(sum, &left_sums[i]);
                let score = (matrix_dot_self(&left_sums[i]) / left_cnt[i] as f32
                    + matrix_dot_self(&temp) / right_cnt as f32) as f64;
                if score > best_score {
                    best_score = score;
                    best_feat = i;
                }
            }
        }

        let mut left_sum = left_sums.swap_remove(best_feat);
        let right_sum;
        if left_sum.size() != 0 {
            right_sum = matrix_sub(sum, &left_sum);
        } else {
            right_sum = sum.clone();
            left_sum = Matrix::zeros(sum.nr(), sum.nc());
        }
        (feats.swap_remove(best_feat), left_sum, right_sum)
    }

    /// Port of `partition_samples(split, samples, begin, end)`: quicksort-style
    /// partition of `[begin, end)` by the split test; returns the midpoint.
    fn partition_samples(
        &self,
        split: &SplitFeature,
        samples: &mut [TrainingSample],
        begin: usize,
        end: usize,
    ) -> usize {
        let mut i = begin;
        for j in begin..end {
            let fv = &samples[j].feature_pixel_values;
            let go_left = fv[split.idx1 as usize] - fv[split.idx2 as usize] > split.thresh;
            if go_left {
                samples.swap(i, j);
                i += 1;
            }
        }
        i
    }
}

/// Port of the private `training_sample<feature_type>` struct: one (possibly
/// oversampled) training example flowing through the cascade.
#[derive(Clone)]
struct TrainingSample {
    image_idx: usize,
    rect: Rectangle,
    target_shape: Matrix<f32>,
    present: Matrix<f32>,
    current_shape: Matrix<f32>,
    diff_shape: Matrix<f32>,
    feature_pixel_values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array2d::Array2D;
    use crate::image_processing::full_object_detection::FullObjectDetection;
    use dlib_rs_core::geometry::Dpoint;
    use dlib_rs_core::serialize::{Deserializer, Serializer};

    /// Builds a 100x100 u8 image with a bright 12x12 blob at the given center
    /// and returns it together with the 3 ground-truth landmark positions
    /// (blob center, left edge, bottom edge of the blob).
    fn blob_image(cx: i64, cy: i64) -> (Array2D<u8>, [Dpoint; 3]) {
        let mut img = Array2D::<u8>::zeros(100, 100);
        for r in 0..100usize {
            for c in 0..100usize {
                *img.get_mut(r, c) = ((r * 31 + c * 17) % 60) as u8;
            }
        }
        for dr in -6..=6i64 {
            for dc in -6..=6i64 {
                let r = (cy + dr) as usize;
                let c = (cx + dc) as usize;
                *img.get_mut(r, c) = 240u8;
            }
        }
        let parts = [
            Dpoint::new(cx as f64, cy as f64),
            Dpoint::new((cx - 6) as f64, cy as f64),
            Dpoint::new(cx as f64, (cy + 6) as f64),
        ];
        (img, parts)
    }

    fn tiny_training_set() -> (Vec<Array2D<u8>>, Vec<Vec<FullObjectDetection>>) {
        let centers = [(40, 40), (55, 45), (35, 55), (50, 60)];
        let mut images = Vec::new();
        let mut objects = Vec::new();
        for (cx, cy) in centers {
            let (img, parts) = blob_image(cx, cy);
            images.push(img);
            objects.push(vec![FullObjectDetection::new(
                Rectangle::new(cx - 10, cy - 10, cx + 10, cy + 10),
                parts.to_vec(),
            )]);
        }
        (images, objects)
    }

    fn tiny_trainer() -> ShapePredictorTrainer {
        let mut t = ShapePredictorTrainer::new();
        t.set_cascade_depth(2)
            .set_tree_depth(2)
            .set_num_trees_per_cascade_level(3)
            .set_feature_pool_size(40)
            .set_oversampling_amount(1)
            .set_random_seed("42");
        t
    }

    #[test]
    fn test_defaults_match_dlib() {
        let t = ShapePredictorTrainer::new();
        assert_eq!(t.get_cascade_depth(), 10);
        assert_eq!(t.get_tree_depth(), 4);
        assert_eq!(t.get_num_trees_per_cascade_level(), 500);
        assert_eq!(t.get_nu(), 0.1);
        assert_eq!(t.get_oversampling_amount(), 20);
        assert_eq!(t.get_oversampling_translation_jitter(), 0.0);
        assert_eq!(t.get_feature_pool_size(), 400);
        assert_eq!(t.get_lambda(), 0.1);
        assert_eq!(t.get_num_test_splits(), 20);
        assert_eq!(t.get_feature_pool_region_padding(), 0.0);
        assert_eq!(t.get_num_threads(), 0);
        assert_eq!(t.get_padding_mode(), PaddingMode::LandmarkRelative);
        assert_eq!(t.get_random_seed(), "");
    }

    #[test]
    fn test_train_and_dimensions() {
        let (images, objects) = tiny_training_set();
        let sp = tiny_trainer().train(&images, &objects);

        assert_eq!(sp.num_parts(), 3);
        assert_eq!(sp.forests.len(), 2);
        for forest in &sp.forests {
            assert_eq!(forest.len(), 3);
            for tree in forest {
                assert_eq!(tree.splits.len(), 3); // 2^2 - 1
                assert_eq!(tree.num_leaves(), 4); // 2^2
                for leaf in &tree.leaf_values {
                    assert_eq!((leaf.nr(), leaf.nc()), (6, 1));
                }
            }
        }
        assert_eq!(sp.anchor_idx.len(), 2);
        assert_eq!(sp.deltas.len(), 2);
        for level in &sp.anchor_idx {
            assert_eq!(level.len(), 40);
            assert!(level.iter().all(|&a| (a as usize) < sp.num_parts()));
        }
        for level in &sp.deltas {
            assert_eq!(level.len(), 40);
        }
    }

    #[test]
    fn test_serialize_deserialize_infer_identical() {
        let (images, objects) = tiny_training_set();
        let sp = tiny_trainer().train(&images, &objects);

        let rect = *objects[0][0].get_rect();
        let det1 = sp.operator_(&images[0], &rect);

        let mut ser = Serializer::new();
        sp.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut de = Deserializer::new(&bytes);
        let mut sp2 = ShapePredictor::default();
        sp2.deserialize(&mut de).unwrap();
        assert_eq!(de.remaining(), 0);

        let det2 = sp2.operator_(&images[0], &rect);
        assert_eq!(det1, det2);
    }

    #[test]
    fn test_same_seed_identical_bytes() {
        let (images, objects) = tiny_training_set();
        let sp1 = tiny_trainer().train(&images, &objects);
        let sp2 = tiny_trainer().train(&images, &objects);

        let mut ser1 = Serializer::new();
        sp1.serialize(&mut ser1);
        let mut ser2 = Serializer::new();
        sp2.serialize(&mut ser2);
        assert_eq!(ser1.into_inner(), ser2.into_inner());
    }

    #[test]
    fn test_inference_moves_with_blob() {
        // The trained predictor should place parts near the moving blob, i.e.
        // the detections must differ between images.
        let (images, objects) = tiny_training_set();
        let sp = tiny_trainer().train(&images, &objects);
        let d1 = sp.operator_(&images[0], objects[0][0].get_rect());
        let d3 = sp.operator_(&images[3], objects[3][0].get_rect());
        assert_eq!(d1.num_parts(), 3);
        let moved = (0..3).any(|k| d1.part(k) != d3.part(k));
        assert!(moved, "detections should track the moving blob");
        // and all parts finite
        for k in 0..3 {
            assert!(d1.part(k).x().is_finite() && d1.part(k).y().is_finite());
        }
    }
}

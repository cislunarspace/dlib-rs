//! Ports of `dlib::decision_function`, `dlib::distance_function` and
//! `dlib::projection_function` from `dlib/svm/function.h`, including the exact
//! `serialize`/`deserialize` byte layout of the C++ headers, plus
//! `dlib::vector_normalizer<matrix<double>>` from
//! `dlib/statistics/statistics.h`.

use crate::svm::kernels::Kernel;
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, SerializeError, Serializer};

/// Serializes a dlib `matrix<sample_type,0,1>` (a column of column vectors)
/// exactly like `serialize(matrix&, ostream&)` in `dlib/matrix/matrix.h`:
/// packed signed dims `-(len)`, `-1`, then each sample matrix.
fn serialize_sample_vector(out: &mut Serializer, v: &[Matrix<f64>]) {
    out.write_i64(-(v.len() as i64));
    out.write_i64(-1);
    for s in v {
        s.serialize(out);
    }
}

/// Mirror of the deserialize side of [`serialize_sample_vector`]; dlib's
/// `matrix<T,0,1>` deserializer rejects a column count other than 1.
fn deserialize_sample_vector(inp: &mut Deserializer) -> Result<Vec<Matrix<f64>>, SerializeError> {
    let mut nr = inp.read_i64()?;
    let mut nc = inp.read_i64()?;
    if nr < 0 || nc < 0 {
        nr = -nr;
        nc = -nc;
    }
    if nc != 1 {
        return Err(SerializeError::Malformed("basis_vectors column count"));
    }
    // Every sample matrix needs at least its two packed dimension bytes.
    if nr as u128 * 2 > inp.remaining() as u128 {
        return Err(SerializeError::Eof);
    }
    let mut v = Vec::with_capacity(nr as usize);
    for _ in 0..nr {
        v.push(Matrix::<f64>::deserialize(inp)?);
    }
    Ok(v)
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::decision_function<K>` (dlib/svm/function.h).
#[derive(Clone, Debug)]
pub struct DecisionFunction<K: Kernel> {
    /// dlib `alpha` (`scalar_vector_type`, a column vector).
    pub alpha_vector: Matrix<f64>,
    /// dlib `b`.
    pub b: f64,
    /// dlib `kernel_function`.
    pub kernel_function: K,
    /// dlib `basis_vectors` (`sample_vector_type`).
    pub basis_dictionary: Vec<Matrix<f64>>,
}

impl<K: Kernel> DecisionFunction<K> {
    /// `decision_function(alpha, b, kernel_function, basis_vectors)`.
    pub fn new(
        alpha_vector: Matrix<f64>,
        b: f64,
        kernel_function: K,
        basis_dictionary: Vec<Matrix<f64>>,
    ) -> Self {
        Self {
            alpha_vector,
            b,
            kernel_function,
            basis_dictionary,
        }
    }

    /// `operator()(x) == sum_i alpha(i)*kernel_function(x, basis_vectors(i)) - b`
    /// (dlib/svm/function.h).
    pub fn operator_(&self, x: &Matrix<f64>) -> f64
    where
        K: Kernel<SampleType = Matrix<f64>>,
    {
        let mut temp = 0.0;
        for i in 0..self.basis_dictionary.len() {
            temp +=
                self.alpha_vector[i] * self.kernel_function.operator_(x, &self.basis_dictionary[i]);
        }
        temp - self.b
    }

    /// `serialize(decision_function, out)` (dlib/svm/function.h): `alpha`
    /// (matrix), `b` (double), `kernel_function` (kernel parameters), then
    /// `basis_vectors` (a `matrix<matrix<double>,0,1>`).
    pub fn serialize(&self, out: &mut Serializer) {
        self.alpha_vector.serialize(out);
        out.write_f64(self.b);
        self.kernel_function.serialize(out);
        serialize_sample_vector(out, &self.basis_dictionary);
    }

    /// `deserialize(decision_function, in)`.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError>
    where
        K: Default,
    {
        let alpha_vector = Matrix::<f64>::deserialize(inp)?;
        let b = inp.read_f64()?;
        let mut kernel_function = K::default();
        kernel_function.deserialize(inp)?;
        let basis_dictionary = deserialize_sample_vector(inp)?;
        Ok(Self {
            alpha_vector,
            b,
            kernel_function,
            basis_dictionary,
        })
    }
}

impl<K: Kernel + Default> Default for DecisionFunction<K> {
    /// `decision_function()` — `b == 0`, default kernel, empty vectors.
    fn default() -> Self {
        Self::new(Matrix::new(), 0.0, K::default(), Vec::new())
    }
}

// ----------------------------------------------------------------------------------------

/// `trans(alpha)*kernel_matrix(kernel, basis)*alpha`, evaluated left to right
/// exactly like dlib's chained matrix multiplication
/// (`matrix_multiply_helper::eval`, dlib/matrix/matrix.h).
fn squared_kernel_expansion<K>(kernel: &K, alpha: &Matrix<f64>, basis: &[Matrix<f64>]) -> f64
where
    K: Kernel<SampleType = Matrix<f64>>,
{
    let n = basis.len();
    if n == 0 {
        return 0.0;
    }
    // row(j) = sum_i alpha(i)*K(i,j), accumulated from the first product.
    let row = |j: usize| {
        let mut t = alpha[0] * kernel.operator_(&basis[0], &basis[j]);
        for i in 1..n {
            t += alpha[i] * kernel.operator_(&basis[i], &basis[j]);
        }
        t
    };
    let mut total = row(0) * alpha[0];
    for j in 1..n {
        total += row(j) * alpha[j];
    }
    total
}

/// Port of `dlib::distance_function<K>` (dlib/svm/function.h). `b` stores the
/// squared norm of the represented point in feature space.
#[derive(Clone, Debug)]
pub struct DistanceFunction<K: Kernel> {
    alpha_vector: Matrix<f64>,
    b: f64,
    kernel_function: K,
    basis_dictionary: Vec<Matrix<f64>>,
}

impl<K: Kernel<SampleType = Matrix<f64>>> DistanceFunction<K> {
    /// `distance_function(kern)` — the origin (`b == 0`, empty vectors).
    pub fn new(kernel_function: K) -> Self {
        Self {
            alpha_vector: Matrix::new(),
            b: 0.0,
            kernel_function,
            basis_dictionary: Vec::new(),
        }
    }

    /// `distance_function(kern, samp)` — the single sample `samp`.
    pub fn with_sample(kernel_function: K, samp: &Matrix<f64>) -> Self {
        Self {
            alpha_vector: Matrix::from_row_vec(1, 1, &[1.0]),
            b: kernel_function.operator_(samp, samp),
            kernel_function,
            basis_dictionary: vec![samp.clone()],
        }
    }

    /// Converting constructor `distance_function(const decision_function<K>&)`
    /// — keeps `alpha`/basis and sets `b` to the squared feature-space norm.
    pub fn from_decision_function(f: &DecisionFunction<K>) -> Self {
        Self {
            b: squared_kernel_expansion(&f.kernel_function, &f.alpha_vector, &f.basis_dictionary),
            alpha_vector: f.alpha_vector.clone(),
            kernel_function: f.kernel_function.clone(),
            basis_dictionary: f.basis_dictionary.clone(),
        }
    }

    /// `distance_function(alpha, b, kernel_function, basis_vectors)`.
    pub fn new_full(
        alpha_vector: Matrix<f64>,
        b: f64,
        kernel_function: K,
        basis_dictionary: Vec<Matrix<f64>>,
    ) -> Self {
        Self {
            alpha_vector,
            b,
            kernel_function,
            basis_dictionary,
        }
    }

    /// `distance_function(alpha, kernel_function, basis_vectors)` — computes
    /// `b` as `trans(alpha)*kernel_matrix(...)*alpha`.
    pub fn new_computed(
        alpha_vector: Matrix<f64>,
        kernel_function: K,
        basis_dictionary: Vec<Matrix<f64>>,
    ) -> Self {
        let b = squared_kernel_expansion(&kernel_function, &alpha_vector, &basis_dictionary);
        Self::new_full(alpha_vector, b, kernel_function, basis_dictionary)
    }

    /// `get_alpha()`.
    pub fn get_alpha(&self) -> &Matrix<f64> {
        &self.alpha_vector
    }

    /// `get_squared_norm()`.
    pub fn get_squared_norm(&self) -> f64 {
        self.b
    }

    /// `get_kernel()`.
    pub fn get_kernel(&self) -> &K {
        &self.kernel_function
    }

    /// `get_basis_vectors()`.
    pub fn get_basis_vectors(&self) -> &[Matrix<f64>] {
        &self.basis_dictionary
    }

    /// `operator()(x)`: distance between `x` and this point in feature space,
    /// i.e. `sqrt(b + k(x,x) - 2*sum_i alpha(i)*k(x, basis(i)))`, clamped at 0.
    pub fn operator_(&self, x: &Matrix<f64>) -> f64 {
        let mut temp = 0.0;
        for i in 0..self.basis_dictionary.len() {
            temp +=
                self.alpha_vector[i] * self.kernel_function.operator_(x, &self.basis_dictionary[i]);
        }
        let temp = self.b + self.kernel_function.operator_(x, x) - 2.0 * temp;
        if temp > 0.0 {
            temp.sqrt()
        } else {
            0.0
        }
    }

    /// The `operator()(const distance_function&)` overload: the distance
    /// between the two points in feature space,
    /// `sqrt(b + x.b - 2*sum_ij alpha(i)*x.alpha(j)*k(basis(i), x.basis(j)))`.
    pub fn distance_to(&self, x: &DistanceFunction<K>) -> f64 {
        let mut temp = 0.0;
        for i in 0..self.basis_dictionary.len() {
            for j in 0..x.basis_dictionary.len() {
                temp += self.alpha_vector[i]
                    * x.alpha_vector[j]
                    * self
                        .kernel_function
                        .operator_(&self.basis_dictionary[i], &x.basis_dictionary[j]);
            }
        }
        let temp = self.b + x.b - 2.0 * temp;
        if temp > 0.0 {
            temp.sqrt()
        } else {
            0.0
        }
    }

    /// `serialize(distance_function, out)` (dlib/svm/function.h): same layout
    /// as `decision_function` — `alpha` matrix, `b` double, kernel, basis.
    pub fn serialize(&self, out: &mut Serializer) {
        self.alpha_vector.serialize(out);
        out.write_f64(self.b);
        self.kernel_function.serialize(out);
        serialize_sample_vector(out, &self.basis_dictionary);
    }

    /// `deserialize(distance_function, in)`.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError>
    where
        K: Default,
    {
        let alpha_vector = Matrix::<f64>::deserialize(inp)?;
        let b = inp.read_f64()?;
        let mut kernel_function = K::default();
        kernel_function.deserialize(inp)?;
        let basis_dictionary = deserialize_sample_vector(inp)?;
        Ok(Self {
            alpha_vector,
            b,
            kernel_function,
            basis_dictionary,
        })
    }
}

impl<K: Kernel<SampleType = Matrix<f64>>> std::ops::Mul<f64> for DistanceFunction<K> {
    type Output = DistanceFunction<K>;
    /// `operator*(scalar)`: `distance_function(val*alpha, val*val*b, ...)`.
    fn mul(self, val: f64) -> DistanceFunction<K> {
        DistanceFunction::new_full(
            self.alpha_vector.clone() * val,
            val * val * self.b,
            self.kernel_function,
            self.basis_dictionary,
        )
    }
}

impl<K: Kernel<SampleType = Matrix<f64>>> std::ops::Div<f64> for DistanceFunction<K> {
    type Output = DistanceFunction<K>;
    /// `operator/(scalar)`: `distance_function(alpha/val, b/val/val, ...)`.
    fn div(self, val: f64) -> DistanceFunction<K> {
        DistanceFunction::new_full(
            self.alpha_vector.clone() * (1.0 / val),
            self.b / val / val,
            self.kernel_function,
            self.basis_dictionary,
        )
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::projection_function<K>` (dlib/svm/function.h).
#[derive(Clone, Debug)]
pub struct ProjectionFunction<K: Kernel> {
    /// dlib `weights` (`scalar_matrix_type`).
    pub weights: Matrix<f64>,
    /// dlib `kernel_function`.
    pub kernel_function: K,
    /// dlib `basis_vectors` (`sample_vector_type`).
    pub basis_dictionary: Vec<Matrix<f64>>,
}

impl<K: Kernel<SampleType = Matrix<f64>>> ProjectionFunction<K> {
    /// `projection_function(weights, kernel_function, basis_vectors)`.
    pub fn new(
        weights: Matrix<f64>,
        kernel_function: K,
        basis_dictionary: Vec<Matrix<f64>>,
    ) -> Self {
        Self {
            weights,
            kernel_function,
            basis_dictionary,
        }
    }

    /// `out_vector_size() == weights.nr()`.
    pub fn out_vector_size(&self) -> usize {
        self.weights.nr()
    }

    /// `operator()(x) == weights * kernel_matrix(kernel, basis_vectors, x)`
    /// (dlib/svm/function.h); the products follow dlib's
    /// `matrix_multiply_helper` accumulation order.
    pub fn operator_(&self, x: &Matrix<f64>) -> Matrix<f64> {
        // temp1(i) = k(basis(i), x)
        let temp1: Vec<f64> = self
            .basis_dictionary
            .iter()
            .map(|b| self.kernel_function.operator_(b, x))
            .collect();
        // temp2(r) = weights(r,0)*temp1(0) + ... (matrix product)
        let mut temp2 = vec![0.0; self.weights.nr()];
        if !temp1.is_empty() {
            for (r, out) in temp2.iter_mut().enumerate() {
                let row = self.weights.rowm(r);
                let mut acc = row[0] * temp1[0];
                for (&w, &t) in row.iter().zip(temp1.iter()).skip(1) {
                    acc += w * t;
                }
                *out = acc;
            }
        }
        Matrix::from_vec(self.weights.nr(), 1, temp2).expect("column vector")
    }

    /// `serialize(projection_function, out)` (dlib/svm/function.h): `weights`
    /// matrix, kernel parameters, then the basis sample vector.
    pub fn serialize(&self, out: &mut Serializer) {
        self.weights.serialize(out);
        self.kernel_function.serialize(out);
        serialize_sample_vector(out, &self.basis_dictionary);
    }

    /// `deserialize(projection_function, in)`.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError>
    where
        K: Default,
    {
        let weights = Matrix::<f64>::deserialize(inp)?;
        let mut kernel_function = K::default();
        kernel_function.deserialize(inp)?;
        let basis_dictionary = deserialize_sample_vector(inp)?;
        Ok(Self {
            weights,
            kernel_function,
            basis_dictionary,
        })
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::vector_normalizer<matrix<double>>`
/// (dlib/statistics/statistics.h): `train` stores the sample mean and the
/// reciprocal standard deviation, `operator()(x)` returns
/// `pointwise_multiply(x - mean, 1/stddev)`.
#[derive(Clone, Debug)]
pub struct VectorNormalizer {
    m: Matrix<f64>,
    sd: Matrix<f64>,
}

impl VectorNormalizer {
    /// An untrained normalizer (empty vectors).
    pub fn new() -> Self {
        Self {
            m: Matrix::new(),
            sd: Matrix::new(),
        }
    }

    /// `vector_normalizer::train(samples)`:
    /// `m = mean(samples)`, `sd = reciprocal(sqrt(variance(samples)))` with
    /// dlib's element-wise matrix-of-matrix statistics.
    pub fn train(&mut self, samples: &[Matrix<f64>]) {
        assert!(!samples.is_empty(), "train(): samples must be nonempty");
        let d = samples[0].size();
        assert!(
            samples.iter().all(|s| s.size() == d && s.nc() == 1),
            "train(): all samples must be column vectors of equal size"
        );
        let n = samples.len();

        // mean(mat(samples)) == sum(samples) / n, accumulated in sample order.
        let mut sum = vec![0.0; d];
        for s in samples {
            for (j, &v) in s.iter().enumerate() {
                sum[j] += v;
            }
        }
        let mean: Vec<f64> = sum.iter().map(|v| v * (1.0 / n as f64)).collect();

        // variance(mat(samples)) == sum(pow(s - avg, 2)) / (n - 1)
        let mut val = vec![0.0; d];
        for s in samples {
            for j in 0..d {
                val[j] += (s[j] - mean[j]).powf(2.0);
            }
        }
        let variance: Vec<f64> = if n <= 1 {
            val
        } else {
            val.iter().map(|v| v * (1.0 / (n as f64 - 1.0))).collect()
        };
        let sd: Vec<f64> = variance
            .iter()
            .map(|v| if *v != 0.0 { 1.0 / v.sqrt() } else { 0.0 })
            .collect();

        self.m = Matrix::from_vec(d, 1, mean).expect("column vector");
        self.sd = Matrix::from_vec(d, 1, sd).expect("column vector");
    }

    /// `in_vector_size() == m.nr()`.
    pub fn in_vector_size(&self) -> usize {
        self.m.nr()
    }

    /// `out_vector_size() == m.nr()`.
    pub fn out_vector_size(&self) -> usize {
        self.m.nr()
    }

    /// `means()` — the stored mean vector.
    pub fn means(&self) -> &Matrix<f64> {
        &self.m
    }

    /// `std_devs()` — the stored reciprocal standard deviations (dlib keeps
    /// the reciprocal in this member).
    pub fn std_devs(&self) -> &Matrix<f64> {
        &self.sd
    }

    /// `operator()(x) == pointwise_multiply(x - m, sd)`.
    pub fn operator_(&self, x: &Matrix<f64>) -> Matrix<f64> {
        assert_eq!(
            (x.nr(), x.nc()),
            (self.in_vector_size(), 1),
            "operator(): x must be a column vector of the trained size"
        );
        let data: Vec<f64> = (0..x.nr())
            .map(|j| (x[j] - self.m[j]) * self.sd[j])
            .collect();
        Matrix::from_vec(x.nr(), 1, data).expect("column vector")
    }

    /// `serialize(vector_normalizer, out)` (dlib/statistics/statistics.h):
    /// `m`, `sd`, then an empty `matrix<double> pca` kept for backwards
    /// compatibility.
    pub fn serialize(&self, out: &mut Serializer) {
        self.m.serialize(out);
        self.sd.serialize(out);
        // serialize(matrix<double>()) — an empty matrix.
        out.write_i64(0);
        out.write_i64(0);
    }

    /// `deserialize(vector_normalizer, in)`; fails if the stream holds a
    /// serialized `vector_normalizer_pca` instead.
    pub fn deserialize(inp: &mut Deserializer) -> Result<Self, SerializeError> {
        let m = Matrix::<f64>::deserialize(inp)?;
        let sd = Matrix::<f64>::deserialize(inp)?;
        let pca = Matrix::<f64>::deserialize(inp)?;
        if pca.size() != 0 {
            return Err(SerializeError::Malformed(
                "serialized vector_normalizer_pca read as vector_normalizer",
            ));
        }
        Ok(Self { m, sd })
    }
}

impl Default for VectorNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::kernels::{LinearKernel, RadialBasisKernel};

    fn col(vals: &[f64]) -> Matrix<f64> {
        Matrix::from_row_vec(vals.len(), 1, vals)
    }

    #[test]
    fn decision_function_evaluation() {
        // df(x) = 2*dot(x, (1,0)) + 3*dot(x, (0,1)) - 0.5
        let df = DecisionFunction::new(
            Matrix::from_row_vec(2, 1, &[2.0, 3.0]),
            0.5,
            LinearKernel,
            vec![col(&[1.0, 0.0]), col(&[0.0, 1.0])],
        );
        assert_eq!(df.operator_(&col(&[2.0, 2.0])), 9.5);
        assert_eq!(df.operator_(&col(&[0.0, 0.0])), -0.5);
    }

    #[test]
    fn decision_function_serialization_roundtrip_is_byte_exact() {
        let df = DecisionFunction::new(
            Matrix::from_row_vec(3, 1, &[0.5, -1.25, 10.0]),
            -0.75,
            RadialBasisKernel::new(0.5),
            vec![col(&[1.0, 2.0]), col(&[-3.0, 0.25]), col(&[0.0, 0.0])],
        );

        let mut out = Serializer::new();
        df.serialize(&mut out);
        let bytes1 = out.into_inner();

        let mut inp = Deserializer::new(&bytes1);
        let df2 = DecisionFunction::<RadialBasisKernel>::deserialize(&mut inp).unwrap();
        assert_eq!(inp.remaining(), 0);

        let mut out = Serializer::new();
        df2.serialize(&mut out);
        let bytes2 = out.into_inner();
        assert_eq!(bytes1, bytes2);

        // Round-tripping preserves evaluation bit-for-bit.
        let probe = col(&[0.3, -1.0]);
        assert_eq!(df.operator_(&probe), df2.operator_(&probe));
        assert_eq!(df2.kernel_function.gamma, 0.5);
        assert_eq!(df2.b, -0.75);
        assert_eq!(df2.basis_dictionary.len(), 3);
    }

    #[test]
    fn distance_function_from_decision_function() {
        // Single support vector at (1,0) with alpha 1: the distance function
        // represents that point exactly.
        let df = DecisionFunction::new(
            Matrix::from_row_vec(1, 1, &[1.0]),
            0.0,
            RadialBasisKernel::new(0.5),
            vec![col(&[1.0, 0.0])],
        );
        let dist = DistanceFunction::from_decision_function(&df);
        let p = col(&[1.0, 0.0]);
        assert_eq!(dist.operator_(&p), 0.0);
        let far = col(&[5.0, 0.0]);
        assert!(dist.operator_(&far) > 0.0);
        assert_eq!(dist.operator_(&far), dist.operator_(&far));

        // Pairwise distance to itself is zero and matches operator_.
        let dist2 = DistanceFunction::with_sample(RadialBasisKernel::new(0.5), &far);
        let d = dist.distance_to(&dist2);
        assert!(d > 0.0);
        assert!((d - dist.operator_(&far)).abs() < 1e-12);

        // scalar * / operators scale the squared norm as dlib does
        let scaled = dist.clone() * 2.0;
        assert_eq!(scaled.get_squared_norm(), 4.0 * dist.get_squared_norm());
        let divided = dist.clone() / 2.0;
        assert_eq!(
            divided.get_squared_norm(),
            dist.get_squared_norm() / 2.0 / 2.0
        );

        // serialization roundtrip
        let mut out = Serializer::new();
        dist.serialize(&mut out);
        let bytes = out.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let dist3 = DistanceFunction::<RadialBasisKernel>::deserialize(&mut inp).unwrap();
        assert_eq!(inp.remaining(), 0);
        assert_eq!(dist3.operator_(&far), dist.operator_(&far));
    }

    #[test]
    fn projection_function_projects() {
        // identity weights over the standard basis: projection of x is x
        let proj = ProjectionFunction::new(
            Matrix::from_row_vec(2, 2, &[1.0, 0.0, 0.0, 1.0]),
            LinearKernel,
            vec![col(&[1.0, 0.0]), col(&[0.0, 1.0])],
        );
        assert_eq!(proj.out_vector_size(), 2);
        let x = col(&[7.0, -3.0]);
        let out = proj.operator_(&x);
        assert_eq!((out.nr(), out.nc()), (2, 1));
        assert_eq!(out[0], 7.0);
        assert_eq!(out[1], -3.0);

        let mut ser = Serializer::new();
        proj.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let proj2 = ProjectionFunction::<LinearKernel>::deserialize(&mut inp).unwrap();
        assert_eq!(inp.remaining(), 0);
        assert_eq!(proj2.operator_(&x)[1], -3.0);
    }

    #[test]
    fn vector_normalizer_trains_mean_and_std() {
        let mut norm = VectorNormalizer::new();
        norm.train(&[col(&[0.0, 0.0]), col(&[2.0, 2.0]), col(&[4.0, 4.0])]);
        assert_eq!(norm.in_vector_size(), 2);
        assert_eq!(norm.out_vector_size(), 2);
        assert_eq!(norm.means()[0], 2.0);
        assert_eq!(norm.means()[1], 2.0);
        // variance per dim = ((4+0+4)/2) = 4, stored reciprocal std = 1/2
        assert_eq!(norm.std_devs()[0], 0.5);
        let out = norm.operator_(&col(&[2.0, 2.0]));
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        let out = norm.operator_(&col(&[4.0, 0.0]));
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], -1.0);

        // processed samples come out (numerically) zero mean, unit std
        let samples = [
            col(&[1.0, -2.0]),
            col(&[2.0, -1.0]),
            col(&[3.0, 0.0]),
            col(&[4.0, 1.0]),
        ];
        norm.train(&samples);
        let mut mean = [0.0f64, 0.0f64];
        let mut var = [0.0f64, 0.0f64];
        let processed: Vec<Matrix<f64>> = samples.iter().map(|s| norm.operator_(s)).collect();
        for p in &processed {
            mean[0] += p[0];
            mean[1] += p[1];
        }
        mean[0] /= 4.0;
        mean[1] /= 4.0;
        for p in &processed {
            var[0] += (p[0] - mean[0]).powf(2.0);
            var[1] += (p[1] - mean[1]).powf(2.0);
        }
        var[0] /= 3.0;
        var[1] /= 3.0;
        assert!(mean[0].abs() < 1e-12 && mean[1].abs() < 1e-12);
        assert!((var[0] - 1.0).abs() < 1e-12 && (var[1] - 1.0).abs() < 1e-12);

        // serialization roundtrip keeps the trained parameters
        let mut ser = Serializer::new();
        norm.serialize(&mut ser);
        let bytes = ser.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let norm2 = VectorNormalizer::deserialize(&mut inp).unwrap();
        assert_eq!(inp.remaining(), 0);
        assert_eq!(norm2.means()[0], norm.means()[0]);
        assert_eq!(
            norm2.operator_(&samples[0])[0],
            norm.operator_(&samples[0])[0]
        );
    }
}

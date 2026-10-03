//! Kernel functions ported from `dlib/svm/kernel.h`: the dlib kernel concept
//! and its dense instantiations `radial_basis_kernel`, `polynomial_kernel`,
//! `sigmoid_kernel` and `linear_kernel` (template argument
//! `T = matrix<double>`), each with the exact `serialize`/`deserialize` byte
//! layout of the C++ headers. Sparse kernels are not ported.

use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::serialize::{Deserializer, SerializeError, Serializer};

/// The `trans(a)*b` inner product, evaluated exactly like dlib's
/// `matrix_multiply_helper::eval` (`dlib/matrix/matrix.h`): the accumulator
/// starts at the first product and adds the remaining products in index order.
fn dot(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
    assert!(!a.is_empty(), "kernel samples must be nonempty");
    assert_eq!(a.nc(), 1, "kernel samples must be column vectors");
    assert_eq!(
        (a.nr(), a.nc()),
        (b.nr(), b.nc()),
        "kernel samples must have equal dimensions"
    );
    let mut acc = a[0] * b[0];
    for k in 1..a.nr() {
        acc += a[k] * b[k];
    }
    acc
}

/// Port of the dlib kernel concept (`dlib/svm/kernel_abstract.h`): a function
/// over pairs of samples plus serialization of its parameters.
pub trait Kernel: Clone {
    /// dlib `kernel_type::sample_type`.
    type SampleType;

    /// dlib `kernel::operator()(a, b)`.
    fn operator_(&self, a: &Self::SampleType, b: &Self::SampleType) -> f64;

    /// dlib `serialize(kernel, out)` (dlib/svm/kernel.h).
    fn serialize(&self, out: &mut Serializer);

    /// dlib `deserialize(kernel, in)` (dlib/svm/kernel.h).
    fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError>;
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::radial_basis_kernel<matrix<double>>`
/// (dlib/svm/kernel.h). Default `gamma` is 0.1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialBasisKernel {
    /// Kernel parameter `gamma`.
    pub gamma: f64,
}

impl RadialBasisKernel {
    /// `radial_basis_kernel(gamma)`.
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    /// `operator()(a, b) == exp(-gamma * trans(a-b)*(a-b))`.
    pub fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        assert!(!a.is_empty(), "kernel samples must be nonempty");
        assert_eq!(a.nc(), 1, "kernel samples must be column vectors");
        assert_eq!(
            (a.nr(), a.nc()),
            (b.nr(), b.nc()),
            "kernel samples must have equal dimensions"
        );
        // const scalar_type d = trans(a-b)*(a-b);
        let d0 = a[0] - b[0];
        let mut d = d0 * d0;
        for k in 1..a.nr() {
            let diff = a[k] - b[k];
            d += diff * diff;
        }
        (-self.gamma * d).exp()
    }

    /// `serialize(radial_basis_kernel)`: one double (`gamma`).
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_f64(self.gamma);
    }

    /// `deserialize(radial_basis_kernel)`.
    pub fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.gamma = inp.read_f64()?;
        Ok(())
    }
}

impl Default for RadialBasisKernel {
    fn default() -> Self {
        Self { gamma: 0.1 }
    }
}

impl Kernel for RadialBasisKernel {
    type SampleType = Matrix<f64>;
    fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        self.operator_(a, b)
    }
    fn serialize(&self, out: &mut Serializer) {
        self.serialize(out)
    }
    fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.deserialize(inp)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::polynomial_kernel<matrix<double>>`
/// (dlib/svm/kernel.h). Defaults are `gamma = 1`, `coef = 0`, `degree = 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolynomialKernel {
    /// Kernel parameter `gamma`.
    pub gamma: f64,
    /// Kernel parameter `coef`.
    pub coef: f64,
    /// Kernel parameter `degree`.
    pub degree: f64,
}

impl PolynomialKernel {
    /// `polynomial_kernel(gamma, coef, degree)`.
    pub fn new(gamma: f64, coef: f64, degree: f64) -> Self {
        Self {
            gamma,
            coef,
            degree,
        }
    }

    /// `operator()(a, b) == pow(gamma * trans(a)*b + coef, degree)`.
    pub fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        (self.gamma * dot(a, b) + self.coef).powf(self.degree)
    }

    /// `serialize(polynomial_kernel)`: `gamma`, `coef`, `degree` as doubles.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_f64(self.gamma);
        out.write_f64(self.coef);
        out.write_f64(self.degree);
    }

    /// `deserialize(polynomial_kernel)`.
    pub fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.gamma = inp.read_f64()?;
        self.coef = inp.read_f64()?;
        self.degree = inp.read_f64()?;
        Ok(())
    }
}

impl Default for PolynomialKernel {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            coef: 0.0,
            degree: 1.0,
        }
    }
}

impl Kernel for PolynomialKernel {
    type SampleType = Matrix<f64>;
    fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        self.operator_(a, b)
    }
    fn serialize(&self, out: &mut Serializer) {
        self.serialize(out)
    }
    fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.deserialize(inp)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::sigmoid_kernel<matrix<double>>`
/// (dlib/svm/kernel.h). Defaults are `gamma = 0.1`, `coef = -1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidKernel {
    /// Kernel parameter `gamma`.
    pub gamma: f64,
    /// Kernel parameter `coef`.
    pub coef: f64,
}

impl SigmoidKernel {
    /// `sigmoid_kernel(gamma, coef)`.
    pub fn new(gamma: f64, coef: f64) -> Self {
        Self { gamma, coef }
    }

    /// `operator()(a, b) == tanh(gamma * trans(a)*b + coef)`.
    pub fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        (self.gamma * dot(a, b) + self.coef).tanh()
    }

    /// `serialize(sigmoid_kernel)`: `gamma`, `coef` as doubles.
    pub fn serialize(&self, out: &mut Serializer) {
        out.write_f64(self.gamma);
        out.write_f64(self.coef);
    }

    /// `deserialize(sigmoid_kernel)`.
    pub fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.gamma = inp.read_f64()?;
        self.coef = inp.read_f64()?;
        Ok(())
    }
}

impl Default for SigmoidKernel {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            coef: -1.0,
        }
    }
}

impl Kernel for SigmoidKernel {
    type SampleType = Matrix<f64>;
    fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        self.operator_(a, b)
    }
    fn serialize(&self, out: &mut Serializer) {
        self.serialize(out)
    }
    fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.deserialize(inp)
    }
}

// ----------------------------------------------------------------------------------------

/// Port of `dlib::linear_kernel<matrix<double>>`
/// (dlib/svm/kernel.h). It has no parameters and serializes to zero bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearKernel;

impl LinearKernel {
    /// `linear_kernel()`.
    pub fn new() -> Self {
        Self
    }

    /// `operator()(a, b) == trans(a)*b`.
    pub fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        dot(a, b)
    }

    /// `serialize(linear_kernel)`: writes nothing.
    pub fn serialize(&self, _out: &mut Serializer) {}

    /// `deserialize(linear_kernel)`: reads nothing.
    pub fn deserialize(&mut self, _inp: &mut Deserializer) -> Result<(), SerializeError> {
        Ok(())
    }
}

impl Kernel for LinearKernel {
    type SampleType = Matrix<f64>;
    fn operator_(&self, a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        self.operator_(a, b)
    }
    fn serialize(&self, out: &mut Serializer) {
        self.serialize(out)
    }
    fn deserialize(&mut self, inp: &mut Deserializer) -> Result<(), SerializeError> {
        self.deserialize(inp)
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn col(vals: &[f64]) -> Matrix<f64> {
        Matrix::from_row_vec(vals.len(), 1, vals)
    }

    #[test]
    fn radial_basis_kernel_values() {
        let k = RadialBasisKernel::new(0.5);
        let a = col(&[0.0, 0.0]);
        let b = col(&[1.0, 1.0]);
        assert_eq!(k.operator_(&a, &a), 1.0);
        assert_eq!(k.operator_(&a, &b), (-1.0f64).exp());
        // default gamma
        assert_eq!(RadialBasisKernel::default().gamma, 0.1);
        assert_eq!(
            RadialBasisKernel::default().operator_(&a, &b),
            (-0.1f64 * 2.0).exp()
        );
    }

    #[test]
    fn polynomial_kernel_values() {
        let a = col(&[2.0, 3.0]);
        let b = col(&[1.0, -1.0]);
        // dot = -1; (1*-1 + 0.5)^3
        let k = PolynomialKernel::new(1.0, 0.5, 3.0);
        assert_eq!(k.operator_(&a, &b), (-0.5f64).powf(3.0));
        // default kernel is the plain dot product
        assert_eq!(PolynomialKernel::default().operator_(&a, &b), -1.0);
    }

    #[test]
    fn sigmoid_kernel_values() {
        let a = col(&[2.0, 3.0]);
        let b = col(&[1.0, -1.0]);
        let k = SigmoidKernel::new(0.5, -0.25);
        assert_eq!(k.operator_(&a, &b), (-0.5f64 + -0.25).tanh());
        assert_eq!(SigmoidKernel::default().gamma, 0.1);
        assert_eq!(SigmoidKernel::default().coef, -1.0);
    }

    #[test]
    fn linear_kernel_is_dot_product() {
        let k = LinearKernel;
        let a = col(&[1.0, 2.0, 3.0]);
        let b = col(&[4.0, 5.0, 6.0]);
        assert_eq!(k.operator_(&a, &b), 32.0);
    }

    #[test]
    fn kernel_serialization_matches_parameter_layout() {
        // rbf: exactly one double (gamma)
        let mut out = Serializer::new();
        RadialBasisKernel::new(0.25).serialize(&mut out);
        let bytes = out.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let mut k = RadialBasisKernel::default();
        k.deserialize(&mut inp).unwrap();
        assert_eq!(k.gamma, 0.25);
        assert_eq!(inp.remaining(), 0);

        // polynomial: three doubles in order gamma, coef, degree
        let mut out = Serializer::new();
        PolynomialKernel::new(2.0, 0.5, 3.0).serialize(&mut out);
        let bytes = out.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let mut k = PolynomialKernel::default();
        k.deserialize(&mut inp).unwrap();
        assert_eq!(k, PolynomialKernel::new(2.0, 0.5, 3.0));
        assert_eq!(inp.remaining(), 0);

        // sigmoid: two doubles in order gamma, coef
        let mut out = Serializer::new();
        SigmoidKernel::new(0.7, -2.0).serialize(&mut out);
        let bytes = out.into_inner();
        let mut inp = Deserializer::new(&bytes);
        let mut k = SigmoidKernel::default();
        k.deserialize(&mut inp).unwrap();
        assert_eq!(k, SigmoidKernel::new(0.7, -2.0));
        assert_eq!(inp.remaining(), 0);

        // linear: zero bytes
        let mut out = Serializer::new();
        LinearKernel.serialize(&mut out);
        assert!(out.into_inner().is_empty());
    }

    #[test]
    fn trait_versions_agree_with_inherent_versions() {
        fn eval(
            k: &impl Kernel<SampleType = Matrix<f64>>,
            a: &Matrix<f64>,
            b: &Matrix<f64>,
        ) -> f64 {
            k.operator_(a, b)
        }
        let a = col(&[1.0, 0.0]);
        let b = col(&[0.0, 2.0]);
        assert_eq!(
            eval(&RadialBasisKernel::new(0.5), &a, &b),
            (-0.5f64 * 5.0).exp()
        );
        assert_eq!(eval(&LinearKernel, &a, &b), 0.0);
    }
}

//! Matrix port of dlib's `dlib/matrix/*.h`.
//!
//! Storage is row-major (`data[r*nc + c]`), matching dlib's default
//! `row_major_layout`; the serialization order matches
//! `dlib::serialize(const matrix&)` (negated dims, then row-major elements).
//!
//! Decompositions are accessed as `matrix::lu::LuDecomposition`, etc.

pub mod cholesky;
pub mod eigenvalue;
pub mod la;
pub mod lu;
pub mod qr;
pub mod svd;

// ---------------------------------------------------------------------------
// Matrix<T> type and operations (ported from dlib/matrix/matrix.h) follow.
// ---------------------------------------------------------------------------

use num_traits::{Float, NumAssign, One, Zero};
use std::fmt;
use std::iter::{FromIterator, Sum};
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A dense row-major matrix, port of `dlib::matrix<T>` from `dlib/matrix/matrix.h`.
///
/// Elements are stored row-major: `data[r*nc + c]` (dlib's default
/// `row_major_layout`). dlib's column-major layout variant is not ported.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Matrix<T> {
    pub(crate) data: Vec<T>,
    pub(crate) nr: usize,
    pub(crate) nc: usize,
}

impl<T> Matrix<T> {
    /// An empty 0x0 matrix (dlib default constructor).
    pub fn new() -> Self {
        Matrix {
            data: Vec::new(),
            nr: 0,
            nc: 0,
        }
    }

    /// Number of rows (`nr()` in dlib).
    pub fn nr(&self) -> usize {
        self.nr
    }

    /// Number of columns (`nc()` in dlib).
    pub fn nc(&self) -> usize {
        self.nc
    }

    /// Total number of elements (`size()` in dlib).
    pub fn size(&self) -> usize {
        self.nr * self.nc
    }

    /// True if the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// True if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.nr == self.nc
    }

    /// Resizes to `nr` x `nc`.
    ///
    /// dlib's `set_size` leaves the values uninitialized; this port
    /// default-fills new slots (and keeps old values where the buffer is
    /// reused) so no uninitialized memory can be observed.
    pub fn set_size(&mut self, nr: usize, nc: usize)
    where
        T: Default + Clone,
    {
        self.data.resize(nr * nc, T::default());
        self.nr = nr;
        self.nc = nc;
    }

    /// Immutable element access, dlib's `m(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> &T {
        assert!(
            r < self.nr && c < self.nc,
            "Matrix::get: index ({r},{c}) out of bounds for {}x{} matrix",
            self.nr,
            self.nc
        );
        &self.data[r * self.nc + c]
    }

    /// Mutable element access, dlib's `m(r, c) = x`.
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        assert!(
            r < self.nr && c < self.nc,
            "Matrix::get_mut: index ({r},{c}) out of bounds for {}x{} matrix",
            self.nr,
            self.nc
        );
        &mut self.data[r * self.nc + c]
    }

    pub fn swap(&mut self, r1: usize, c1: usize, r2: usize, c2: usize) {
        let i1 = r1 * self.nc + c1;
        let i2 = r2 * self.nc + c2;
        assert!(
            i1 < self.size() && i2 < self.size(),
            "swap: index out of bounds"
        );
        self.data.swap(i1, i2);
    }

    /// Iterate over elements in row-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Mutably iterate over elements in row-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Builds a new matrix by applying `f` to every element (dlib matrix
    /// expressions like `sqrt(m)`, `exp(m)` evaluate eagerly here).
    pub fn map<U, F: FnMut(T) -> U>(&self, mut f: F) -> Matrix<U>
    where
        T: Clone,
    {
        Matrix {
            data: self.data.iter().cloned().map(&mut f).collect(),
            nr: self.nr,
            nc: self.nc,
        }
    }

    /// Take ownership of the row-major element buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone> Matrix<T> {
    /// Builds a matrix from a row-major buffer; errors if `data.len() != nr*nc`.
    pub fn from_vec(nr: usize, nc: usize, data: Vec<T>) -> Result<Self, &'static str> {
        if data.len() != nr.saturating_mul(nc) {
            return Err("from_vec: data.len() must equal nr*nc");
        }
        Ok(Matrix { data, nr, nc })
    }

    /// Infallible constructor from a row-major slice of known length
    /// (convenience for tests and literals; dlib has no direct analogue).
    pub fn from_row_vec(nr: usize, nc: usize, data: &[T]) -> Self {
        assert_eq!(
            data.len(),
            nr * nc,
            "from_row_vec: data.len() must equal nr*nc"
        );
        Matrix {
            data: data.to_vec(),
            nr,
            nc,
        }
    }

    /// Sub-matrix, port of `subm(m, row, col, nr, nc)` from
    /// `dlib/matrix/matrix_subexp.h`: `(row, col)` is the top-left corner and
    /// `nr`/`nc` are the dimensions of the block (NOT end indices), with
    /// `row + nr <= m.nr()` and `col + nc <= m.nc()`.
    pub fn subm(&self, row: usize, col: usize, nr: usize, nc: usize) -> Matrix<T> {
        assert!(
            row + nr <= self.nr && col + nc <= self.nc,
            "subm: block ({row}..{}, {col}..{}) outside {}x{} matrix",
            row + nr,
            col + nc,
            self.nr,
            self.nc
        );
        let mut data = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                data.push(self.get(row + r, col + c).clone());
            }
        }
        Matrix { data, nr, nc }
    }

    /// Row `r` as a 1 x nc matrix (dlib `rowm(m, r)`).
    pub fn rowm(&self, r: usize) -> Matrix<T> {
        self.row(r)
    }

    /// Column `c` as an nr x 1 matrix (dlib `colm(m, c)`).
    pub fn colm(&self, c: usize) -> Matrix<T> {
        self.col(c)
    }

    /// Row `r` as a 1 x nc matrix.
    pub fn row(&self, r: usize) -> Matrix<T> {
        assert!(
            r < self.nr,
            "row: index {r} out of bounds for {} rows",
            self.nr
        );
        Matrix {
            data: self.data[r * self.nc..(r + 1) * self.nc].to_vec(),
            nr: 1,
            nc: self.nc,
        }
    }

    /// Column `c` as an nr x 1 matrix.
    pub fn col(&self, c: usize) -> Matrix<T> {
        assert!(
            c < self.nc,
            "col: index {c} out of bounds for {} cols",
            self.nc
        );
        Matrix {
            data: (0..self.nr)
                .map(|r| self.data[r * self.nc + c].clone())
                .collect(),
            nr: self.nr,
            nc: 1,
        }
    }

    /// Overwrites the block at `(row, col)` (dimensions `src.nr() x src.nc()`)
    /// with `src`; port of `set_subm(m, row, col, nr, nc) = src`.
    pub fn set_subm(&mut self, row: usize, col: usize, src: &Matrix<T>) {
        assert!(
            row + src.nr <= self.nr && col + src.nc <= self.nc,
            "set_subm: block ({row}..{}, {col}..{}) outside {}x{} matrix",
            row + src.nr,
            col + src.nc,
            self.nr,
            self.nc
        );
        for r in 0..src.nr {
            for c in 0..src.nc {
                *self.get_mut(row + r, col + c) = src.get(r, c).clone();
            }
        }
    }

    /// Overwrites row `r` (dlib `set_rowm(m, r) = src`, src is 1 x nc).
    pub fn set_rowm(&mut self, r: usize, src: &Matrix<T>) {
        assert!(r < self.nr, "set_rowm: row {r} out of bounds");
        assert_eq!(
            (src.nr, src.nc),
            (1, self.nc),
            "set_rowm: source must be 1x{}",
            self.nc
        );
        for c in 0..self.nc {
            *self.get_mut(r, c) = src.data[c].clone();
        }
    }

    /// Overwrites column `c` (dlib `set_colm(m, c) = src`, src is nr x 1).
    pub fn set_colm(&mut self, c: usize, src: &Matrix<T>) {
        assert!(c < self.nc, "set_colm: col {c} out of bounds");
        assert_eq!(
            (src.nr, src.nc),
            (self.nr, 1),
            "set_colm: source must be {}x1",
            self.nr
        );
        for r in 0..self.nr {
            *self.get_mut(r, c) = src.data[r].clone();
        }
    }

    /// Reinterprets the element buffer as an `nr` x `nc` matrix in the same
    /// row-major order (dlib `reshape(m, nr, nc)`).
    pub fn reshape(&self, nr: usize, nc: usize) -> Matrix<T> {
        assert_eq!(
            self.size(),
            nr * nc,
            "reshape: source has {} elements, target is {nr}x{nc}",
            self.size()
        );
        Matrix {
            data: self.data.clone(),
            nr,
            nc,
        }
    }

    /// Transpose (dlib `trans(m)`).
    pub fn transpose(&self) -> Matrix<T> {
        // Flat map: out[c*nr + r] = in[r*nc + c]; i % nr == r, i / nr == c.
        let data = (0..self.size())
            .map(|i| self.data[(i % self.nr) * self.nc + i / self.nr].clone())
            .collect();
        Matrix {
            data,
            nr: self.nc,
            nc: self.nr,
        }
    }
}

impl<T: Zero + Clone> Matrix<T> {
    /// An `nr` x `nc` matrix of zeros (dlib `matrix<T> m(nr, nc)` is
    /// uninitialized; `zeros_matrix<T>(nr, nc)` is the zero-filled analogue).
    pub fn zeros(nr: usize, nc: usize) -> Self {
        Matrix {
            data: vec![T::zero(); nr * nc],
            nr,
            nc,
        }
    }
}

impl<T: One + Clone> Matrix<T> {
    /// An `nr` x `nc` matrix of ones (dlib `ones_matrix`).
    pub fn ones(nr: usize, nc: usize) -> Self {
        Matrix {
            data: vec![T::one(); nr * nc],
            nr,
            nc,
        }
    }

    /// The `n` x `n` identity matrix (dlib `identity_matrix<T>(n)`).
    pub fn identity(n: usize) -> Self
    where
        T: Zero + Clone,
    {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * (n + 1)] = T::one();
        }
        m
    }

    /// Alias for [`Matrix::identity`] (dlib `eye(n)`).
    pub fn eye(n: usize) -> Self
    where
        T: Zero + Clone,
    {
        Matrix::identity(n)
    }
}

// -- indexing ---------------------------------------------------------------

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &T {
        self.get(idx.0, idx.1)
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        self.get_mut(idx.0, idx.1)
    }
}

/// Flat linear index in row-major order (`data[r*nc + c]`), matching dlib's
/// linear indexing for `row_major_layout`.
impl<T> Index<usize> for Matrix<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        assert!(
            i < self.size(),
            "index {i} out of bounds for {} elements",
            self.size()
        );
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(
            i < self.size(),
            "index {i} out of bounds for {} elements",
            self.size()
        );
        &mut self.data[i]
    }
}

// -- iteration ---------------------------------------------------------------

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<T> FromIterator<T> for Matrix<T> {
    /// Collects a flat row-major buffer into a column vector (nr = len, nc = 1).
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        let nr = data.len();
        Matrix { data, nr, nc: 1 }
    }
}

// -- operators (dlib operator overloads) --------------------------------------

impl<T> Add for Matrix<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Matrix<T>;
    fn add(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(
            (self.nr, self.nc),
            (rhs.nr, rhs.nc),
            "Matrix addition: shapes {}x{} and {}x{} do not match",
            self.nr,
            self.nc,
            rhs.nr,
            rhs.nc
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Matrix {
            data,
            nr: self.nr,
            nc: self.nc,
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Matrix<T>;
    fn sub(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(
            (self.nr, self.nc),
            (rhs.nr, rhs.nc),
            "Matrix subtraction: shapes {}x{} and {}x{} do not match",
            self.nr,
            self.nc,
            rhs.nr,
            rhs.nc
        );
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Matrix {
            data,
            nr: self.nr,
            nc: self.nc,
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        let (m, k, n) = (self.nr, self.nc, rhs.nc);
        assert_eq!(
            k, rhs.nr,
            "Matrix multiply: inner dimensions do not match ({}x{} * {}x{})",
            self.nr, self.nc, rhs.nr, rhs.nc
        );
        // Cache-friendly i-k-j loop: streams over both operands' rows.
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a.is_zero() {
                    continue;
                }
                let out_row = &mut out.data[i * n..(i + 1) * n];
                let rhs_row = &rhs.data[p * n..(p + 1) * n];
                for (o, b) in out_row.iter_mut().zip(rhs_row.iter()) {
                    *o = *o + a * *b;
                }
            }
        }
        out
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Matrix<T>;
    fn mul(self, s: T) -> Matrix<T> {
        let data = self.data.iter().map(|a| *a * s).collect();
        Matrix {
            data,
            nr: self.nr,
            nc: self.nc,
        }
    }
}

impl<T> Div<T> for Matrix<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Matrix<T>;
    fn div(self, s: T) -> Matrix<T> {
        let data = self.data.iter().map(|a| *a / s).collect();
        Matrix {
            data,
            nr: self.nr,
            nc: self.nc,
        }
    }
}

impl<T> Neg for Matrix<T>
where
    T: Copy + Neg<Output = T>,
{
    type Output = Matrix<T>;
    fn neg(self) -> Matrix<T> {
        let data = self.data.iter().map(|a| -*a).collect();
        Matrix {
            data,
            nr: self.nr,
            nc: self.nc,
        }
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Copy + Add<Output = T>,
{
    fn add_assign(&mut self, rhs: Matrix<T>) {
        assert_eq!(
            (self.nr, self.nc),
            (rhs.nr, rhs.nc),
            "Matrix +=: shapes {}x{} and {}x{} do not match",
            self.nr,
            self.nc,
            rhs.nr,
            rhs.nc
        );
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a + *b;
        }
    }
}

impl<T> SubAssign for Matrix<T>
where
    T: Copy + Sub<Output = T>,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        assert_eq!(
            (self.nr, self.nc),
            (rhs.nr, rhs.nc),
            "Matrix -=: shapes {}x{} and {}x{} do not match",
            self.nr,
            self.nc,
            rhs.nr,
            rhs.nc
        );
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a = *a - *b;
        }
    }
}

impl<T> MulAssign<T> for Matrix<T>
where
    T: Copy + Mul<Output = T>,
{
    fn mul_assign(&mut self, s: T) {
        for a in self.data.iter_mut() {
            *a = *a * s;
        }
    }
}

// -- free functions (dlib/matrix/matrix_utilities.h) ---------------------------

/// Transpose, port of `dlib::trans(m)` from `dlib/matrix/matrix_utilities.h`.
pub fn trans<T: Clone>(m: &Matrix<T>) -> Matrix<T> {
    m.transpose()
}

/// Sub-matrix with top-left corner `(row, col)` and size `nr x nc`, port of
/// `dlib::subm(m, row, col, nr, nc)` from `dlib/matrix/matrix_subexp.h`.
pub fn subm<T: Clone>(m: &Matrix<T>, row: usize, col: usize, nr: usize, nc: usize) -> Matrix<T> {
    m.subm(row, col, nr, nc)
}

/// Row `r` as a 1 x nc matrix, port of `dlib::rowm(m, r)`.
pub fn rowm<T: Clone>(m: &Matrix<T>, r: usize) -> Matrix<T> {
    m.rowm(r)
}

/// Column `c` as an nr x 1 matrix, port of `dlib::colm(m, c)`.
pub fn colm<T: Clone>(m: &Matrix<T>, c: usize) -> Matrix<T> {
    m.colm(c)
}

/// Sum of all elements, port of `dlib::sum(m)`.
pub fn sum<T>(m: &Matrix<T>) -> T
where
    T: Float + NumAssign + Sum,
{
    m.data.iter().copied().sum()
}

/// Mean of all elements, port of `dlib::mean(m)`.
pub fn mean<T>(m: &Matrix<T>) -> T
where
    T: Float + NumAssign + Sum,
{
    let n = num_traits::NumCast::from(m.size()).expect("size fits in float");
    sum(m) / n
}

/// Maximum element, port of `dlib::max(m)`. Panics on an empty matrix.
pub fn max<T: Float + NumAssign>(m: &Matrix<T>) -> T {
    m.data
        .iter()
        .copied()
        .fold(None::<T>, |acc, x| {
            Some(match acc {
                Some(a) if a > x => a,
                _ => x,
            })
        })
        .expect("max: matrix is empty")
}

/// Minimum element, port of `dlib::min(m)`. Panics on an empty matrix.
pub fn min<T: Float + NumAssign>(m: &Matrix<T>) -> T {
    m.data
        .iter()
        .copied()
        .fold(None::<T>, |acc, x| {
            Some(match acc {
                Some(a) if a < x => a,
                _ => x,
            })
        })
        .expect("min: matrix is empty")
}

/// Element-wise absolute value, port of `dlib::abs(m)`.
pub fn abs<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::abs)
}

/// Element-wise square root, port of `dlib::sqrt(m)`.
pub fn sqrt<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::sqrt)
}

/// Element-wise exponential, port of `dlib::exp(m)`.
pub fn exp<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::exp)
}

/// Element-wise natural logarithm, port of `dlib::log(m)`.
pub fn log<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::ln)
}

/// Element-wise power, port of `dlib::pow(m, e)`.
pub fn pow<T: Float + NumAssign>(m: &Matrix<T>, e: T) -> Matrix<T> {
    m.map(|x| x.powf(e))
}

/// Element-wise sine, port of `dlib::sin(m)`.
pub fn sin<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::sin)
}

/// Element-wise cosine, port of `dlib::cos(m)`.
pub fn cos<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::cos)
}

/// Element-wise tangent, port of `dlib::tan(m)`.
pub fn tan<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::tan)
}

/// Element-wise arc tangent, port of `dlib::atan(m)`.
pub fn atan<T: Float + NumAssign>(m: &Matrix<T>) -> Matrix<T> {
    m.map(Float::atan)
}

/// Dot product treating both matrices as flat vectors, port of `dlib::dot(a, b)`
/// (requires matching sizes).
pub fn dot<T>(a: &Matrix<T>, b: &Matrix<T>) -> T
where
    T: Float + NumAssign + Sum,
{
    assert_eq!(
        a.size(),
        b.size(),
        "dot: sizes {} and {} do not match",
        a.size(),
        b.size()
    );
    a.data.iter().zip(b.data.iter()).map(|(x, y)| *x * *y).sum()
}

/// Trace (sum of diagonal), port of `dlib::trace(m)`; requires a square matrix.
pub fn trace<T>(m: &Matrix<T>) -> T
where
    T: Float + NumAssign + Sum,
{
    assert!(
        m.is_square(),
        "trace: matrix must be square, got {}x{}",
        m.nr,
        m.nc
    );
    (0..m.nr).map(|i| m.data[i * (m.nc + 1)]).sum()
}

/// Euclidean length (Frobenius norm for matrices), port of `dlib::length(m)`
/// (`sqrt(sum(squared(m)))`).
pub fn length<T>(m: &Matrix<T>) -> T
where
    T: Float + NumAssign + Sum,
{
    sum(&hadamard(m, m)).sqrt()
}

/// Element-wise product, port of `dlib::hadamard` / `pointwise_multiply`.
pub fn hadamard<T>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T>
where
    T: Copy + Mul<Output = T>,
{
    assert_eq!(
        (a.nr, a.nc),
        (b.nr, b.nc),
        "hadamard: shapes {}x{} and {}x{} do not match",
        a.nr,
        a.nc,
        b.nr,
        b.nc
    );
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| *x * *y)
        .collect();
    Matrix {
        data,
        nr: a.nr,
        nc: a.nc,
    }
}

// -- Display -------------------------------------------------------------------

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.nr {
            for c in 0..self.nc {
                if c > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", self.data[r * self.nc + c])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// -- serialization (dlib/matrix/matrix.h serialize/deserialize) -----------------

/// Upper bound on either matrix dimension accepted by [`Matrix::deserialize`].
const MAX_DIM: i64 = 1_000_000_000;

impl<T: crate::serialize::DlibSerialize> Matrix<T> {
    /// Serializes exactly like `dlib::serialize(const matrix&)`
    /// (dlib/matrix/matrix.h): packed signed dims `-(nr as i64)`,
    /// `-(nc as i64)`, then the elements in row-major order.
    pub fn serialize(&self, out: &mut crate::serialize::Serializer) {
        out.write_i64(-(self.nr as i64));
        out.write_i64(-(self.nc as i64));
        for r in 0..self.nr {
            for c in 0..self.nc {
                self.data[r * self.nc + c].dlib_serialize(out);
            }
        }
    }

    /// Deserializes exactly like `dlib::deserialize(matrix&)`
    /// (dlib/matrix/matrix.h): reads two packed i64 dims; if either is
    /// negative both are negated (current format, written by `serialize`),
    /// otherwise they are accepted as-is (legacy positive-dims format);
    /// then `nr*nc` elements in row-major order.
    pub fn deserialize(
        inp: &mut crate::serialize::Deserializer,
    ) -> Result<Matrix<T>, crate::serialize::SerializeError> {
        let mut nr = inp.read_i64()?;
        let mut nc = inp.read_i64()?;
        if nr < 0 || nc < 0 {
            nr = -nr;
            nc = -nc;
        }
        if nr > MAX_DIM || nc > MAX_DIM {
            return Err(crate::serialize::SerializeError::Malformed("matrix dims"));
        }
        let count = nr
            .checked_mul(nc)
            .ok_or(crate::serialize::SerializeError::Malformed("matrix dims"))?;
        if count as u128 > inp.remaining() as u128 {
            // Each element takes at least one byte; more bytes than remain
            // means the buffer cannot possibly hold the claimed matrix.
            return Err(crate::serialize::SerializeError::Eof);
        }
        let mut data = Vec::with_capacity(count.min(1 << 20) as usize);
        for _ in 0..count {
            data.push(T::dlib_deserialize(inp)?);
        }
        Ok(Matrix {
            data,
            nr: nr as usize,
            nc: nc as usize,
        })
    }
}

impl<T: crate::serialize::DlibSerialize> crate::serialize::DlibSerialize for Matrix<T> {
    fn dlib_serialize(&self, out: &mut crate::serialize::Serializer) {
        self.serialize(out);
    }
    fn dlib_deserialize(
        inp: &mut crate::serialize::Deserializer,
    ) -> Result<Self, crate::serialize::SerializeError> {
        Matrix::deserialize(inp)
    }
}

// -- tests ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn m23() -> Matrix<f64> {
        Matrix::from_row_vec(2, 3, &[1., 2., 3., 4., 5., 6.])
    }

    #[test]
    fn constructors_and_access() {
        let m = Matrix::<f64>::new();
        assert_eq!((m.nr(), m.nc(), m.size()), (0, 0, 0));
        assert!(m.is_empty());
        assert!(!Matrix::from_row_vec(1, 2, &[0., 0.]).is_square());

        let z = Matrix::<i32>::zeros(2, 3);
        assert_eq!((z.nr(), z.nc()), (2, 3));
        assert!(z.iter().all(|&x| x == 0));

        assert!(Matrix::<f64>::ones(3, 2).iter().all(|&x| x == 1.0));

        let i = Matrix::<f64>::identity(3);
        assert_eq!(
            i,
            Matrix::from_row_vec(3, 3, &[1., 0., 0., 0., 1., 0., 0., 0., 1.])
        );
        assert_eq!(Matrix::<f64>::eye(2), Matrix::identity(2));
        assert!(i.is_square());

        assert_eq!(
            Matrix::from_vec(2, 2, vec![1., 2., 3., 4.]),
            Ok(Matrix::from_row_vec(2, 2, &[1., 2., 3., 4.]))
        );
        assert!(Matrix::from_vec(2, 2, vec![1.]).is_err());

        let mut m = m23();
        assert_eq!(*m.get(1, 2), 6.0);
        assert_eq!(m[(0, 1)], 2.0);
        m[(0, 1)] = 9.0;
        assert_eq!(*m.get_mut(0, 1), 9.0);
        assert_eq!(m[0], 1.0); // flat row-major
        assert_eq!(m[2], 3.0);
        m.swap(0, 1, 1, 1);
        assert_eq!(m[(0, 1)], 5.0);
        assert_eq!(m[(1, 1)], 9.0);
    }

    #[test]
    fn set_size_default_fills() {
        let mut m = Matrix::<f64>::new();
        m.set_size(2, 2);
        assert_eq!(m, Matrix::zeros(2, 2));
    }

    #[test]
    fn matmul_hand_computed() {
        // 2x3 * 3x2
        let a = m23();
        let b = Matrix::from_row_vec(3, 2, &[7., 8., 9., 10., 11., 12.]);
        let c = a.clone() * b;
        assert_eq!((c.nr(), c.nc()), (2, 2));
        // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(c, Matrix::from_row_vec(2, 2, &[58., 64., 139., 154.]));

        let i = Matrix::<f64>::identity(3);
        assert_eq!(a.clone() * i.clone(), a);
        assert_eq!(i.clone() * i.clone(), i);
    }

    #[test]
    fn transpose_subm_rowm_colm() {
        let a = m23();
        let t = trans(&a);
        assert_eq!((t.nr(), t.nc()), (3, 2));
        assert_eq!(t, Matrix::from_row_vec(3, 2, &[1., 4., 2., 5., 3., 6.]));
        assert_eq!(t.transpose(), a);

        // dlib subm(m, row, col, nr, nc): corner + size
        let s = subm(&a, 1, 1, 1, 2);
        assert_eq!((s.nr(), s.nc()), (1, 2));
        assert_eq!(s, Matrix::from_row_vec(1, 2, &[5., 6.]));
        let s2 = a.subm(0, 1, 2, 1);
        assert_eq!(s2, Matrix::from_row_vec(2, 1, &[2., 5.]));
        assert_eq!(a.subm(0, 0, a.nr(), a.nc()), a);

        let r = rowm(&a, 1);
        assert_eq!((r.nr(), r.nc()), (1, 3));
        assert_eq!(r, Matrix::from_row_vec(1, 3, &[4., 5., 6.]));
        assert_eq!(a.row(0), Matrix::from_row_vec(1, 3, &[1., 2., 3.]));

        let c = colm(&a, 2);
        assert_eq!((c.nr(), c.nc()), (2, 1));
        assert_eq!(c, Matrix::from_row_vec(2, 1, &[3., 6.]));
        assert_eq!(a.col(1), Matrix::from_row_vec(2, 1, &[2., 5.]));
    }

    #[test]
    fn set_subm_rowm_colm_reshape() {
        let mut m = Matrix::<i32>::zeros(3, 3);
        let blk = Matrix::from_row_vec(2, 2, &[1, 2, 3, 4]);
        m.set_subm(1, 1, &blk);
        assert_eq!(m, Matrix::from_row_vec(3, 3, &[0, 0, 0, 0, 1, 2, 0, 3, 4]));

        m.set_rowm(0, &Matrix::from_row_vec(1, 3, &[7, 8, 9]));
        assert_eq!(m, Matrix::from_row_vec(3, 3, &[7, 8, 9, 0, 1, 2, 0, 3, 4]));

        m.set_colm(0, &Matrix::from_row_vec(3, 1, &[5, 6, 7]));
        assert_eq!(m, Matrix::from_row_vec(3, 3, &[5, 8, 9, 6, 1, 2, 7, 3, 4]));

        let flat = Matrix::from_row_vec(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let r = flat.reshape(3, 2);
        assert_eq!((r.nr(), r.nc()), (3, 2));
        assert_eq!(r, Matrix::from_row_vec(3, 2, &[1., 2., 3., 4., 5., 6.]));
    }

    #[test]
    fn arithmetic_ops() {
        let a = m23();
        let b = Matrix::from_row_vec(2, 3, &[1., 1., 1., 2., 2., 2.]);

        assert_eq!(
            a.clone() + b.clone(),
            Matrix::from_row_vec(2, 3, &[2., 3., 4., 6., 7., 8.])
        );
        assert_eq!(
            a.clone() - b.clone(),
            Matrix::from_row_vec(2, 3, &[0., 1., 2., 2., 3., 4.])
        );
        assert_eq!(
            a.clone() * 2.0,
            Matrix::from_row_vec(2, 3, &[2., 4., 6., 8., 10., 12.])
        );
        assert_eq!(
            a.clone() / 2.0,
            Matrix::from_row_vec(2, 3, &[0.5, 1., 1.5, 2., 2.5, 3.])
        );
        assert_eq!(
            -a.clone(),
            Matrix::from_row_vec(2, 3, &[-1., -2., -3., -4., -5., -6.])
        );

        let mut c = a.clone();
        c += b.clone();
        assert_eq!(c, a.clone() + b.clone());
        c -= b.clone();
        assert_eq!(c, a);
        let mut d = a.clone();
        d *= 3.0;
        assert_eq!(d, a.clone() * 3.0);
    }

    #[test]
    fn numeric_free_functions() {
        let a = m23();
        assert_eq!(sum(&a), 21.0);
        assert_eq!(mean(&a), 3.5);
        assert_eq!(max(&a), 6.0);
        assert_eq!(min(&a), 1.0);

        let sq = Matrix::from_row_vec(1, 3, &[-4., 9., 16.]);
        assert_eq!(abs(&sq), Matrix::from_row_vec(1, 3, &[4., 9., 16.]));
        assert_eq!(sqrt(&abs(&sq)), Matrix::from_row_vec(1, 3, &[2., 3., 4.]));
        assert!((exp(&Matrix::from_row_vec(1, 1, &[0.0])))[(0, 0)] - std::f64::consts::E < 1e-3);
        assert!((log(&Matrix::from_row_vec(1, 1, &[std::f64::consts::E])))[(0, 0)] - 1.0 < 1e-12);
        assert_eq!(
            pow(&Matrix::from_row_vec(1, 2, &[2., 3.]), 2.0),
            Matrix::from_row_vec(1, 2, &[4., 9.])
        );
        assert!((sin(&Matrix::from_row_vec(1, 1, &[0.0])))[(0, 0)].abs() < 1e-12);
        assert!((cos(&Matrix::from_row_vec(1, 1, &[0.0])))[(0, 0)] - 1.0 < 1e-12);
        assert!((tan(&Matrix::from_row_vec(1, 1, &[0.0])))[(0, 0)].abs() < 1e-12);
        assert!(
            (atan(&Matrix::from_row_vec(1, 1, &[1.0])))[(0, 0)] - std::f64::consts::FRAC_PI_4
                < 1e-12
        );

        let v = Matrix::from_row_vec(1, 3, &[1., 2., 3.]);
        assert_eq!(dot(&v, &v), 14.0);
        assert_eq!(trace(&Matrix::<f64>::identity(3)), 3.0);
        assert_eq!(length(&v), 14f64.sqrt());

        let p = Matrix::from_row_vec(1, 3, &[1., 2., 3.]);
        let q = Matrix::from_row_vec(1, 3, &[4., 5., 6.]);
        assert_eq!(
            hadamard(&p, &q),
            Matrix::from_row_vec(1, 3, &[4., 10., 18.])
        );
    }

    #[test]
    fn iteration_and_display() {
        let a = m23();
        let collected: Vec<f64> = a.iter().copied().collect();
        assert_eq!(collected, vec![1., 2., 3., 4., 5., 6.]);
        let mut b = a.clone();
        for x in b.iter_mut() {
            *x += 1.0;
        }
        assert_eq!(b, Matrix::from_row_vec(2, 3, &[2., 3., 4., 5., 6., 7.]));
        let owned: Vec<f64> = a.clone().into_iter().collect();
        assert_eq!(owned.len(), 6);
        let s = format!("{}", a);
        assert_eq!(s, "1 2 3\n4 5 6\n");
    }

    #[test]
    fn serialize_roundtrip_and_layout() {
        use crate::serialize::{Deserializer, DlibSerialize, Serializer};

        let a = m23();
        let mut ser = Serializer::new();
        a.serialize(&mut ser);

        // Byte layout: packed i64 of -nr and -nc prefix the element stream.
        let mut expect = Serializer::new();
        expect.write_i64(-2);
        expect.write_i64(-3);
        let bytes = ser.as_bytes();
        assert_eq!(&bytes[..expect.as_bytes().len()], expect.as_bytes());

        let mut de = Deserializer::new(bytes);
        let b = Matrix::<f64>::deserialize(&mut de).unwrap();
        assert_eq!(b, a);
        assert_eq!(de.remaining(), 0);

        // Via the DlibSerialize trait.
        let mut ser2 = Serializer::new();
        a.dlib_serialize(&mut ser2);
        let mut de2 = Deserializer::new(ser2.as_bytes());
        assert_eq!(Matrix::<f64>::dlib_deserialize(&mut de2).unwrap(), a);
    }

    #[test]
    fn deserialize_legacy_positive_dims() {
        use crate::serialize::{Deserializer, DlibSerialize, Serializer};

        // Legacy format: plain positive dims, then row-major elements.
        let mut ser = Serializer::new();
        ser.write_i64(2);
        ser.write_i64(2);
        for x in [1.5f64, 2.5, 3.5, 4.5] {
            x.dlib_serialize(&mut ser);
        }
        let mut de = Deserializer::new(ser.as_bytes());
        let m = Matrix::<f64>::deserialize(&mut de).unwrap();
        assert_eq!(m, Matrix::from_row_vec(2, 2, &[1.5, 2.5, 3.5, 4.5]));
        assert_eq!(de.remaining(), 0);
    }

    #[test]
    fn deserialize_rejects_bad_dims() {
        use crate::serialize::{Deserializer, Serializer};

        let mut ser = Serializer::new();
        ser.write_i64(-(2_000_000_000i64));
        ser.write_i64(-1);
        let mut de = Deserializer::new(ser.as_bytes());
        assert!(Matrix::<f64>::deserialize(&mut de).is_err());

        // Claims 1000x1000 elements but the buffer is empty -> Eof error.
        let mut ser2 = Serializer::new();
        ser2.write_i64(-1000);
        ser2.write_i64(-1000);
        let mut de2 = Deserializer::new(ser2.as_bytes());
        assert!(Matrix::<f64>::deserialize(&mut de2).is_err());
    }
}

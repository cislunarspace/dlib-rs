//! LU decomposition, ported line-by-line from `dlib/matrix/matrix_lu.h`
//! (which was adapted from the JAMA part of NIST's TNT library).
//!
//! The non-LAPACK branch of the dlib implementation is ported: a
//! "left-looking", dot-product, Crout/Doolittle algorithm with partial
//! pivoting. All pivot selection and loop structure match the C++ source.

use crate::matrix::Matrix;

/// Port of `dlib::lu_decomposition<double>` (class `lu_decomposition` in
/// `dlib/matrix/matrix_lu.h`).
///
/// Factors a square matrix `A` (any `m x n` with `m >= n` is accepted by the
/// C++ code, but the dlib requires-clauses for `det`/`solve` demand square
/// input, so this port asserts `nr == nc`).
pub struct LuDecomposition {
    /// Internal storage of the decomposition (L below the diagonal, U on and
    /// above it), port of the `LU` member.
    lu: Matrix<f64>,
    m: usize,
    n: usize,
    pivsign: f64,
    /// Pivot permutation, port of the `piv` member (`rowm(A, piv) == L*U`).
    piv: Vec<usize>,
}

impl LuDecomposition {
    /// Port of the `lu_decomposition(const matrix_exp&)` constructor.
    ///
    /// # Panics
    /// Panics if `a` is empty or not square (dlib `DLIB_ASSERT` on the
    /// `det`/`solve` requires clauses).
    pub fn new(a: &Matrix<f64>) -> Self {
        assert!(
            a.nr() == a.nc() && a.size() > 0,
            "lu_decomposition::new(A): A must be a non-empty square matrix"
        );

        let m = a.nr();
        let n = a.nc();
        let mut lu = a.clone();
        let mut piv: Vec<usize> = (0..m).collect();
        let mut pivsign = 1.0f64;

        // Use a "left-looking", dot-product, Crout/Doolittle algorithm.
        let mut lucolj = vec![0.0f64; m];

        for j in 0..n {
            // Make a copy of the j-th column to localize references.
            for i in 0..m {
                lucolj[i] = lu[(i, j)];
            }

            // Apply previous transformations.
            for i in 0..m {
                let kmax = i.min(j);
                let mut s = 0.0f64;
                if kmax > 0 {
                    for k in 0..kmax {
                        s += lu[(i, k)] * lucolj[k];
                    }
                }
                lucolj[i] -= s;
                lu[(i, j)] = lucolj[i];
            }

            // Find pivot and exchange if necessary.
            let mut p = j;
            for i in (j + 1)..m {
                if lucolj[i].abs() > lucolj[p].abs() {
                    p = i;
                }
            }
            if p != j {
                for k in 0..n {
                    let t = lu[(p, k)];
                    lu[(p, k)] = lu[(j, k)];
                    lu[(j, k)] = t;
                }
                piv.swap(p, j);
                pivsign = -pivsign;
            }

            // Compute multipliers.
            if j < m && lu[(j, j)] != 0.0 {
                for i in (j + 1)..m {
                    let v = lu[(i, j)] / lu[(j, j)];
                    lu[(i, j)] = v;
                }
            }
        }

        LuDecomposition {
            lu,
            m,
            n,
            pivsign,
            piv,
        }
    }

    /// Port of `is_square()`.
    pub fn is_square(&self) -> bool {
        self.m == self.n
    }

    /// Port of `nr()`.
    pub fn nr(&self) -> usize {
        self.m
    }

    /// Port of `nc()`.
    pub fn nc(&self) -> usize {
        self.n
    }

    /// The raw combined LU factors (read-only access to the `LU` member).
    pub fn lu(&self) -> &Matrix<f64> {
        &self.lu
    }

    /// Port of `get_pivot()`: the permutation `piv` with
    /// `rowm(A, piv) == L*U`.
    pub fn pivots(&self) -> &[usize] {
        &self.piv
    }

    /// Port of the `pivsign` member (exposed via `det()` in dlib): `+1.0` or
    /// `-1.0` depending on the number of row interchanges.
    pub fn pivsign(&self) -> f64 {
        self.pivsign
    }

    /// Port of `get_l()`: unit lower triangular factor.
    pub fn get_l(&self) -> Matrix<f64> {
        let mm = if self.lu.nr() >= self.lu.nc() {
            self.lu.nr()
        } else {
            self.m
        };
        let mut l = Matrix::zeros(mm, self.m);
        for i in 0..mm {
            for j in 0..self.m {
                l[(i, j)] = if i > j {
                    self.lu[(i, j)]
                } else if i == j {
                    1.0
                } else {
                    0.0
                };
            }
        }
        l
    }

    /// Port of `get_u()`: upper triangular factor.
    pub fn get_u(&self) -> Matrix<f64> {
        let nn = if self.lu.nr() >= self.lu.nc() {
            self.n
        } else {
            self.lu.nc()
        };
        let mut u = Matrix::zeros(self.n, nn);
        for i in 0..self.n {
            for j in 0..nn {
                u[(i, j)] = if i <= j { self.lu[(i, j)] } else { 0.0 };
            }
        }
        u
    }

    /// Port of `is_singular()`: true if the upper triangular factor `U` (and
    /// hence `A`) is singular.
    pub fn is_singular(&self) -> bool {
        assert!(
            self.is_square(),
            "lu_decomposition::is_singular(): only valid for square matrices"
        );
        let mut min_val = f64::INFINITY;
        let mut max_val = 0.0f64;
        for i in 0..self.n {
            let v = self.lu[(i, i)].abs();
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let mut eps = max_val;
        if eps != 0.0 {
            eps *= f64::EPSILON.sqrt() / 10.0;
        } else {
            eps = 1.0; // there is no max so just use 1
        }
        min_val < eps
    }

    /// Port of `det()`. Returns 0 for singular matrices (exactly as dlib does
    /// to avoid a `prod()` swamping an effectively zero diagonal element).
    pub fn det(&self) -> f64 {
        assert!(
            self.is_square(),
            "lu_decomposition::det(): only valid for square matrices"
        );
        if self.is_singular() {
            return 0.0;
        }
        let mut prod = 1.0f64;
        for i in 0..self.n {
            prod *= self.lu[(i, i)];
        }
        prod * self.pivsign
    }

    /// Port of `solve(B)`: solves `A*X = B` for `X` (an `n x B.nc()` matrix).
    ///
    /// Returns `Err` when the matrix is singular (dlib's triangular solvers
    /// would silently produce inf/NaN there) or when `B.nr() != nr()`.
    pub fn solve(&self, b: &Matrix<f64>) -> Result<Matrix<f64>, &'static str> {
        if !self.is_square() || b.nr() != self.nr() {
            return Err("lu_decomposition::solve(): invalid argument dimensions");
        }
        if self.is_singular() {
            return Err("lu_decomposition::solve(): matrix is singular");
        }

        let n = self.n;
        let nx = b.nc();

        // Copy right hand side with pivoting: X = rowm(B, piv)
        let mut x = Matrix::zeros(n, nx);
        for r in 0..n {
            for c in 0..nx {
                x[(r, c)] = b[(self.piv[r], c)];
            }
        }

        // Solve L*Y = B(piv,:) (unit lower triangular).
        for j in 0..nx {
            for i in 1..n {
                let mut s = 0.0;
                for k in 0..i {
                    s += self.lu[(i, k)] * x[(k, j)];
                }
                let v = x[(i, j)] - s;
                x[(i, j)] = v;
            }
        }
        // Solve U*X = Y (non-unit upper triangular).
        for j in 0..nx {
            for i in (0..n).rev() {
                let mut s = 0.0;
                for k in (i + 1)..n {
                    s += self.lu[(i, k)] * x[(k, j)];
                }
                let v = (x[(i, j)] - s) / self.lu[(i, i)];
                x[(i, j)] = v;
            }
        }
        Ok(x)
    }
}

// ----------------------------------------------------------------------------
// tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_matrix(nr: usize, nc: usize, seed: u64) -> Matrix<f64> {
        let mut x = seed;
        let mut m = Matrix::zeros(nr, nc);
        for r in 0..nr {
            for c in 0..nc {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                m[(r, c)] = ((x >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
            }
        }
        m
    }

    fn mat_mul(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
        let mut c = Matrix::zeros(a.nr(), b.nc());
        for i in 0..a.nr() {
            for j in 0..b.nc() {
                let mut s = 0.0;
                for k in 0..a.nc() {
                    s += a[(i, k)] * b[(k, j)];
                }
                c[(i, j)] = s;
            }
        }
        c
    }

    fn max_abs(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        (0..a.nr())
            .flat_map(move |r| (0..a.nc()).map(move |c| (r, c)))
            .map(|(r, c)| (a[(r, c)] - b[(r, c)]).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_lu_reconstruction() {
        let a = lcg_matrix(4, 4, 42);
        let lu = LuDecomposition::new(&a);
        let l = lu.get_l();
        let u = lu.get_u();

        // P*A == L*U where P permutes rows by `piv`.
        let mut pa = Matrix::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                pa[(r, c)] = a[(lu.pivots()[r], c)];
            }
        }
        assert!(max_abs(&pa, &mat_mul(&l, &u)) < 1e-12);
        assert_eq!(lu.pivsign(), -1.0); // permutation for this seed is odd
    }

    #[test]
    fn test_lu_det() {
        // det of [[2,0,0],[0,3,0],[0,0,4]] is 24
        let mut a = Matrix::zeros(3, 3);
        a[(0, 0)] = 2.0;
        a[(1, 1)] = 3.0;
        a[(2, 2)] = 4.0;
        let lu = LuDecomposition::new(&a);
        assert!(!lu.is_singular());
        assert!((lu.det() - 24.0).abs() < 1e-12);

        // 3x3 hand value: det = 1*(2*5-0*6) - 3*(1*5-0*4) + 5*(1*6-2*4) = 10-15-10 = -15
        let mut b = Matrix::zeros(3, 3);
        let vals = [[1.0, 3.0, 5.0], [1.0, 2.0, 0.0], [4.0, 6.0, 5.0]];
        for r in 0..3 {
            for c in 0..3 {
                b[(r, c)] = vals[r][c];
            }
        }
        let lub = LuDecomposition::new(&b);
        assert!((lub.det() + 15.0).abs() < 1e-10);

        // singular matrix
        let mut s = Matrix::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                s[(r, c)] = (r as f64) + 1.0; // rank 1
            }
        }
        let lus = LuDecomposition::new(&s);
        assert!(lus.is_singular());
        assert_eq!(lus.det(), 0.0);
        assert!(lus.solve(&Matrix::zeros(3, 1)).is_err());
    }

    #[test]
    fn test_lu_solve_residual() {
        let a = lcg_matrix(6, 6, 7);
        let b = lcg_matrix(6, 3, 99);
        let lu = LuDecomposition::new(&a);
        let x = lu.solve(&b).unwrap();
        let resid = max_abs(&mat_mul(&a, &x), &b);
        assert!(resid < 1e-10, "residual {}", resid);
    }
}

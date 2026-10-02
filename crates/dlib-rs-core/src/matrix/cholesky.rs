//! Cholesky decomposition, ported line-by-line from
//! `dlib/matrix/matrix_cholesky.h` (adapted from the JAMA part of NIST's
//! TNT library).
//!
//! The non-LAPACK branch of the dlib implementation is ported, including the
//! exact `isspd` detection heuristics and the fallback behaviour of `solve`
//! on non-SPD input.

use crate::matrix::Matrix;

/// Port of `dlib::cholesky_decomposition<double>` from
/// `dlib/matrix/matrix_cholesky.h`.
pub struct CholeskyDecomposition {
    /// Lower triangular factor (port of `L_`).
    l_: Matrix<f64>,
    /// True if the factored matrix was symmetric positive definite (port of
    /// `isspd`).
    isspd: bool,
}

impl CholeskyDecomposition {
    /// Port of the `cholesky_decomposition(const matrix_exp&)` constructor.
    ///
    /// # Panics
    /// Panics if `a` is not a non-empty square matrix (dlib `DLIB_ASSERT`).
    pub fn new(a: &Matrix<f64>) -> Self {
        assert!(
            a.nr() == a.nc() && a.size() > 0,
            "cholesky_decomposition::new(A): A must be a non-empty square matrix"
        );

        let mut isspd = true;
        let n = a.nc();
        let mut l_ = Matrix::zeros(n, n);

        // do nothing if the matrix is empty (unreachable due to the assert,
        // kept to mirror the source structure)
        if a.size() == 0 {
            return CholeskyDecomposition { l_, isspd };
        }

        let eps = f64::EPSILON;
        let mut max_diag = 0.0f64;
        for i in 0..n {
            max_diag = max_diag.max(a[(i, i)].abs());
        }
        let eps2 = max_diag * f64::EPSILON.sqrt() / 100.0;

        // compute the upper left corner
        if a[(0, 0)] > 0.0 {
            l_[(0, 0)] = a[(0, 0)].sqrt();
            if a[(0, 0)] <= eps2 {
                isspd = false;
            }
        } else {
            isspd = false;
            l_[(0, 0)] = 0.0;
        }

        // compute the first column
        for r in 1..a.nr() {
            if l_[(0, 0)] > eps * a[(r, 0)].abs() {
                l_[(r, 0)] = a[(r, 0)] / l_[(0, 0)];
            } else {
                isspd = false;
                l_[(r, 0)] = 0.0;
            }

            isspd = isspd && (a[(r, 0)] - a[(0, r)]).abs() <= eps * a[(r, 0)].abs();
        }

        // now compute all the other columns
        for c in 1..a.nc() {
            // compute the diagonal element
            let mut temp = a[(c, c)];
            for i in 0..c {
                temp -= l_[(c, i)] * l_[(c, i)];
            }

            if temp > 0.0 {
                l_[(c, c)] = temp.sqrt();
                if temp <= eps2 {
                    isspd = false;
                }
            } else {
                l_[(c, c)] = 0.0;
                isspd = false;
            }

            for r in 0..c {
                l_[(r, c)] = 0.0;
            }

            // compute the non diagonal elements
            for r in (c + 1)..a.nr() {
                temp = a[(r, c)];
                for i in 0..c {
                    temp -= l_[(r, i)] * l_[(c, i)];
                }

                if l_[(c, c)] > eps * temp.abs() {
                    l_[(r, c)] = temp / l_[(c, c)];
                } else {
                    isspd = false;
                    l_[(r, c)] = 0.0;
                }

                isspd = isspd && (a[(r, c)] - a[(c, r)]).abs() <= eps * a[(r, c)].abs();
            }
        }

        CholeskyDecomposition { l_, isspd }
    }

    /// Port of `is_spd()`.
    pub fn is_spd(&self) -> bool {
        self.isspd
    }

    /// Port of `get_l()`: the lower triangular factor.
    pub fn l(&self) -> &Matrix<f64> {
        &self.l_
    }

    /// Convenience accessor mirroring dlib usage `L*trans(L)`: returns the
    /// reconstructed matrix `L * L'`.
    pub fn matrix(&self) -> Matrix<f64> {
        let n = self.l_.nr();
        let mut out = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.l_[(i, k)] * self.l_[(j, k)];
                }
                out[(i, j)] = s;
            }
        }
        out
    }

    /// Port of `solve(B)`. Solves `A*X = B` for `X` given `A = L*L'`, via the
    /// two triangular solves `L*y = b` and `L'*X = y`.
    ///
    /// Exactly as in dlib, this uses the computed `L_` factor even when the
    /// input matrix was not SPD.
    pub fn solve(&self, b: &Matrix<f64>) -> Matrix<f64> {
        assert!(
            self.l_.nr() == b.nr(),
            "cholesky_decomposition::solve(B): B.nr() must equal L.nr()"
        );
        let n = self.l_.nr();
        let nx = b.nc();
        let mut x = b.clone();

        // Solve L*y = b (non-unit lower triangular).
        for j in 0..nx {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..i {
                    s += self.l_[(i, k)] * x[(k, j)];
                }
                let v = (x[(i, j)] - s) / self.l_[(i, i)];
                x[(i, j)] = v;
            }
        }
        // Solve L'*X = y (transposed lower = upper, non-unit).
        for j in 0..nx {
            for i in (0..n).rev() {
                let mut s = 0.0;
                for k in (i + 1)..n {
                    s += self.l_[(k, i)] * x[(k, j)];
                }
                let v = (x[(i, j)] - s) / self.l_[(i, i)];
                x[(i, j)] = v;
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        (0..a.nr())
            .flat_map(move |r| (0..a.nc()).map(move |c| (r, c)))
            .map(|(r, c)| (a[(r, c)] - b[(r, c)]).abs())
            .fold(0.0, f64::max)
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

    fn lcg(seed: u64) -> u64 {
        seed.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    #[test]
    fn test_spd_reconstruction() {
        // SPD: A = M*M' + n*I for a random M (deterministic LCG).
        let n = 5usize;
        let mut m = Matrix::zeros(n, n);
        let mut x = 123u64;
        for r in 0..n {
            for c in 0..n {
                x = lcg(x);
                m[(r, c)] = ((x >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
            }
        }
        let mt = {
            let mut t = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    t[(r, c)] = m[(c, r)];
                }
            }
            t
        };
        let a = {
            let mut a = mat_mul(&m, &mt);
            for i in 0..n {
                a[(i, i)] += n as f64;
            }
            a
        };

        let chol = CholeskyDecomposition::new(&a);
        assert!(chol.is_spd());
        assert!(max_abs(&chol.matrix(), &a) < 1e-12, "L*L' must equal A");

        // solve residual
        let mut b = Matrix::zeros(n, 2);
        let mut xb = 5u64;
        for r in 0..n {
            for c in 0..2 {
                xb = lcg(xb);
                b[(r, c)] = ((xb >> 33) as f64) / (1u64 << 31) as f64 - 1.0;
            }
        }
        let sol = chol.solve(&b);
        assert!(max_abs(&mat_mul(&a, &sol), &b) < 1e-10);
    }

    #[test]
    fn test_non_spd_detection() {
        // indefinite symmetric matrix (eigenvalues 1, -1)
        let mut a = Matrix::zeros(2, 2);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = -1.0;
        assert!(!CholeskyDecomposition::new(&a).is_spd());

        // asymmetric matrix
        let mut b = Matrix::zeros(2, 2);
        b[(0, 0)] = 4.0;
        b[(1, 1)] = 4.0;
        b[(0, 1)] = 3.0;
        b[(1, 0)] = 1.0;
        assert!(!CholeskyDecomposition::new(&b).is_spd());
    }
}

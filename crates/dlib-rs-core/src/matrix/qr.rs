//! QR decomposition, ported line-by-line from `dlib/matrix/matrix_qr.h`
//! (adapted from the JAMA part of NIST's TNT library).
//!
//! The non-LAPACK branch of the dlib implementation is ported: Householder
//! reflections with the exact norm accumulation via `hypot`.

use crate::matrix::Matrix;

/// Port of `dlib::qr_decomposition<double>` from
/// `dlib/matrix/matrix_qr.h`.
pub struct QrDecomposition {
    /// Internal storage (port of `QR_`): the compact Householder form.
    qr_: Matrix<f64>,
    m: usize,
    n: usize,
    /// Diagonal of R (port of `Rdiag`).
    rdiag: Vec<f64>,
}

impl QrDecomposition {
    /// Port of the `qr_decomposition(const matrix_exp&)` constructor.
    ///
    /// # Panics
    /// Panics if `a.nr() < a.nc()` or `a` is empty (dlib `DLIB_ASSERT`).
    pub fn new(a: &Matrix<f64>) -> Self {
        assert!(
            a.nr() >= a.nc() && a.size() > 0,
            "qr_decomposition::new(A): requires A.nr() >= A.nc() and non-empty A"
        );

        let mut qr_ = a.clone();
        let m = a.nr();
        let n = a.nc();
        let mut rdiag = vec![0.0f64; n];

        // Main loop.
        for k in 0..n {
            // Compute 2-norm of k-th column without under/overflow.
            let mut nrm = 0.0f64;
            for i in k..m {
                nrm = nrm.hypot(qr_[(i, k)]);
            }

            if nrm != 0.0 {
                // Form k-th Householder vector.
                if qr_[(k, k)] < 0.0 {
                    nrm = -nrm;
                }
                for i in k..m {
                    let v = qr_[(i, k)] / nrm;
                    qr_[(i, k)] = v;
                }
                qr_[(k, k)] += 1.0;

                // Apply transformation to remaining columns.
                for j in (k + 1)..n {
                    let mut s = 0.0f64;
                    for i in k..m {
                        s += qr_[(i, k)] * qr_[(i, j)];
                    }
                    s = -s / qr_[(k, k)];
                    for i in k..m {
                        let v = qr_[(i, j)] + s * qr_[(i, k)];
                        qr_[(i, j)] = v;
                    }
                }
            }
            rdiag[k] = -nrm;
        }

        QrDecomposition { qr_, m, n, rdiag }
    }

    /// Port of `nr()`.
    pub fn nr(&self) -> usize {
        self.m
    }

    /// Port of `nc()`.
    pub fn nc(&self) -> usize {
        self.n
    }

    /// The raw compact Householder form (read-only access to `QR_`).
    pub fn qr(&self) -> &Matrix<f64> {
        &self.qr_
    }

    /// Port of `is_full_rank()`: true if `R` (and hence `A`) has full rank.
    pub fn is_full_rank(&self) -> bool {
        let mut eps = 0.0f64;
        for v in &self.rdiag {
            eps = eps.max(v.abs());
        }
        if eps != 0.0 {
            eps *= f64::EPSILON.sqrt() / 100.0;
        } else {
            eps = 1.0; // there is no max so just use 1
        }

        // check if any of the elements of Rdiag are effectively 0
        let mut min_val = f64::INFINITY;
        for v in &self.rdiag {
            min_val = min_val.min(v.abs());
        }
        min_val > eps
    }

    /// Port of `get_r()`: the `n x n` upper triangular factor.
    pub fn r(&self) -> Matrix<f64> {
        let mut r = Matrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                if i < j {
                    r[(i, j)] = self.qr_[(i, j)];
                } else if i == j {
                    r[(i, j)] = self.rdiag[i];
                } else {
                    r[(i, j)] = 0.0;
                }
            }
        }
        r
    }

    /// Port of `get_q()`: the `m x n` economy-size orthogonal factor
    /// (non-LAPACK branch of the dlib `get_q` template).
    pub fn q(&self) -> Matrix<f64> {
        let m = self.m;
        let n = self.n;
        let mut x = Matrix::zeros(m, n);
        for k in (0..n).rev() {
            for i in 0..m {
                x[(i, k)] = 0.0;
            }
            x[(k, k)] = 1.0;
            for j in k..n {
                if self.qr_[(k, k)] != 0.0 {
                    let mut s = 0.0f64;
                    for i in k..m {
                        s += self.qr_[(i, k)] * x[(i, j)];
                    }
                    s = -s / self.qr_[(k, k)];
                    for i in k..m {
                        let v = x[(i, j)] + s * self.qr_[(i, k)];
                        x[(i, j)] = v;
                    }
                }
            }
        }
        x
    }

    /// Port of `solve(B)`: least-squares solution of `A*X = B`, returning the
    /// `n x B.nc()` result. Dispatches between the vector and matrix variants
    /// exactly like the dlib non-LAPACK `solve`.
    ///
    /// # Panics
    /// Panics if `b.nr() != nr()` (dlib `DLIB_ASSERT`).
    pub fn solve(&self, b: &Matrix<f64>) -> Matrix<f64> {
        assert!(
            b.nr() == self.nr(),
            "qr_decomposition::solve(B): B.nr() must equal nr()"
        );
        if b.nc() == 1 {
            self.solve_vect(b)
        } else {
            self.solve_mat(b)
        }
    }

    /// Port of the private `solve_vect` (single right-hand side).
    fn solve_vect(&self, b: &Matrix<f64>) -> Matrix<f64> {
        let m = self.m;
        let n = self.n;
        let mut x = vec![0.0f64; m];
        for i in 0..m {
            x[i] = b[(i, 0)];
        }

        // Compute Y = transpose(Q)*B
        for k in 0..n {
            let mut s = 0.0f64;
            for (i, xi) in x.iter().enumerate().take(m).skip(k) {
                s += self.qr_[(i, k)] * xi;
            }
            s = -s / self.qr_[(k, k)];
            for (i, xi) in x.iter_mut().enumerate().take(m).skip(k) {
                *xi += s * self.qr_[(i, k)];
            }
        }
        // Solve R*X = Y;
        for k in (0..n).rev() {
            x[k] /= self.rdiag[k];
            for i in 0..k {
                x[i] -= x[k] * self.qr_[(i, k)];
            }
        }

        // return n x 1 portion of x
        let mut out = Matrix::zeros(n, 1);
        for i in 0..n {
            out[(i, 0)] = x[i];
        }
        out
    }

    /// Port of the private `solve_mat` (multiple right-hand sides).
    fn solve_mat(&self, b: &Matrix<f64>) -> Matrix<f64> {
        let m = self.m;
        let n = self.n;
        let nx = b.nc();
        let mut x = b.clone();

        // Compute Y = transpose(Q)*B
        for k in 0..n {
            for j in 0..nx {
                let mut s = 0.0f64;
                for i in k..m {
                    s += self.qr_[(i, k)] * x[(i, j)];
                }
                s = -s / self.qr_[(k, k)];
                for i in k..m {
                    let v = x[(i, j)] + s * self.qr_[(i, k)];
                    x[(i, j)] = v;
                }
            }
        }
        // Solve R*X = Y;
        for k in (0..n).rev() {
            for j in 0..nx {
                let v = x[(k, j)] / self.rdiag[k];
                x[(k, j)] = v;
            }
            for i in 0..k {
                for j in 0..nx {
                    let v = x[(i, j)] - x[(k, j)] * self.qr_[(i, k)];
                    x[(i, j)] = v;
                }
            }
        }

        // return n x nx portion of X
        let mut out = Matrix::zeros(n, nx);
        for i in 0..n {
            for j in 0..nx {
                out[(i, j)] = x[(i, j)];
            }
        }
        out
    }
}

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
    fn test_qr_reconstruction() {
        let a = lcg_matrix(6, 4, 3);
        let qr = QrDecomposition::new(&a);
        assert!(qr.is_full_rank());
        let q = qr.q();
        let r = qr.r();

        // Q' * Q == I (n x n, economy size)
        let qt = {
            let mut t = Matrix::zeros(4, 6);
            for i in 0..6 {
                for j in 0..4 {
                    t[(j, i)] = q[(i, j)];
                }
            }
            t
        };
        let mut ident = Matrix::zeros(4, 4);
        for i in 0..4 {
            ident[(i, i)] = 1.0;
        }
        assert!(max_abs(&mat_mul(&qt, &q), &ident) < 1e-12);

        // Q * R == A
        assert!(max_abs(&mat_mul(&q, &r), &a) < 1e-12);
    }

    #[test]
    fn test_qr_least_squares() {
        // Overdetermined 4x2 system; compare against normal equations.
        let a = lcg_matrix(4, 2, 11);
        let b = lcg_matrix(4, 1, 12);
        let qr = QrDecomposition::new(&a);
        let x = qr.solve(&b);

        // normal equations: (A'*A) x = A'*b
        let at = {
            let mut t = Matrix::zeros(2, 4);
            for i in 0..4 {
                for j in 0..2 {
                    t[(j, i)] = a[(i, j)];
                }
            }
            t
        };
        let ata = mat_mul(&at, &a);
        let atb = mat_mul(&at, &b);
        // solve 2x2 by hand
        let det = ata[(0, 0)] * ata[(1, 1)] - ata[(0, 1)] * ata[(1, 0)];
        let mut xn = Matrix::zeros(2, 1);
        xn[(0, 0)] = (ata[(1, 1)] * atb[(0, 0)] - ata[(0, 1)] * atb[(1, 0)]) / det;
        xn[(1, 0)] = (ata[(0, 0)] * atb[(1, 0)] - ata[(1, 0)] * atb[(0, 0)]) / det;
        assert!(max_abs(&x, &xn) < 1e-9, "QR least squares mismatch");
    }

    #[test]
    fn test_qr_solve_matrix_rhs() {
        let a = lcg_matrix(5, 3, 21);
        let b = lcg_matrix(5, 2, 22);
        let qr = QrDecomposition::new(&a);
        let x = qr.solve(&b);
        assert_eq!(x.nr(), 3);
        assert_eq!(x.nc(), 2);
        // residual A*x - b must be orthogonal-ish small; just check ||A x - b||
        let axb = mat_mul(&a, &x);
        let mut resid = 0.0f64;
        for r in 0..5 {
            for c in 0..2 {
                resid = resid.max((axb[(r, c)] - b[(r, c)]).abs());
            }
        }
        assert!(resid < 1.0); // overdetermined, only small relative to data
    }
}

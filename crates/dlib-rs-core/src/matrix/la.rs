//! Linear algebra free functions, ported from `dlib/matrix/matrix_la.h`:
//! `inv`, `det`, and `pinv`.
//!
//! In dlib, `inv`/`det` dispatch on the compile-time dimensions; for
//! dynamically sized matrices (`NR == NC == 0`, which is what this crate's
//! `Matrix<f64>` corresponds to) the generic implementations are used:
//! `inv` via `lu_decomposition(...).solve(identity_matrix(...))` and `det`
//! via `lu_decomposition(...).det()`. Those are the paths ported here.

use crate::matrix::lu::LuDecomposition;
use crate::matrix::svd::svd3;
use crate::matrix::Matrix;

/// Port of `dlib::inv(const matrix_exp&)` for dynamically sized matrices
/// (`dlib/matrix/matrix_la.h`, `inv_helper<EXP,0>`): solves
/// `A * X = I` with the LU decomposition.
///
/// Unlike dlib (whose triangular solvers silently produce garbage), this
/// returns `Err` when the matrix is singular or non-square.
pub fn inv(m: &Matrix<f64>) -> Result<Matrix<f64>, &'static str> {
    if m.nr() != m.nc() {
        return Err("inv(): you can only apply inv() to a square matrix");
    }
    if m.size() == 0 {
        return Err("inv(): matrix is empty");
    }
    let lu = LuDecomposition::new(m);
    lu.solve(&identity(m.nr()))
}

/// Port of `dlib::det(const matrix_exp&)` for dynamically sized matrices
/// (`dlib/matrix/matrix_la.h`, `det_helper<EXP,0>`):
/// `lu_decomposition(m).det()`.
///
/// # Panics
/// Panics on a non-square matrix (dlib `DLIB_ASSERT`).
pub fn det(m: &Matrix<f64>) -> f64 {
    assert!(
        m.nr() == m.nc(),
        "det(): you can only apply det() to a square matrix"
    );
    LuDecomposition::new(m).det()
}

/// Port of `dlib::pinv_helper` from `dlib/matrix/matrix_la.h`: computes the
/// pseudoinverse of `m` using `svd3`, fastest when `m.nc() <= m.nr()`.
fn pinv_helper(m: &Matrix<f64>, tol: f64) -> Matrix<f64> {
    let mut u = Matrix::new();
    let mut w = Matrix::new();
    let mut v = Matrix::new();
    svd3(m, &mut u, &mut w, &mut v);

    let machine_eps = f64::EPSILON;
    let max_w = (0..w.nr()).map(|i| w[(i, 0)]).fold(0.0f64, f64::max);
    // compute a reasonable epsilon below which we round to zero before doing
    // the reciprocal.  Unless a non-zero tol is given then we just use
    // tol*max(w).
    let eps = if tol != 0.0 {
        tol * max_w
    } else {
        machine_eps * (m.nr().max(m.nc()) as f64) * max_w
    };

    // now compute the pseudoinverse:
    // scale_columns(v, reciprocal(round_zeros(w,eps))) * trans(u)
    let n = v.nr();
    let mut scaled = Matrix::zeros(n, n);
    for j in 0..n {
        // reciprocal(round_zeros(w(j), eps)): dlib's reciprocal maps 0 to 0
        let r = if w[(j, 0)].abs() <= eps {
            0.0
        } else {
            1.0 / w[(j, 0)]
        };
        for i in 0..n {
            scaled[(i, j)] = v[(i, j)] * r;
        }
    }
    mat_mul(&scaled, &transpose(&u))
}

/// Port of `dlib::pinv(const matrix_exp&, double tol = 0)` from
/// `dlib/matrix/matrix_la.h`: the Moore-Penrose pseudoinverse via the SVD.
///
/// # Panics
/// Panics if `tol < 0` (dlib `DLIB_ASSERT`).
pub fn pinv(m: &Matrix<f64>, tol: f64) -> Matrix<f64> {
    assert!(tol >= 0.0, "pinv(): tol can't be negative");
    // if m has more columns then rows then it is more efficient to
    // compute the pseudo-inverse of its transpose.
    if m.nc() > m.nr() {
        transpose(&pinv_helper(&transpose(m), tol))
    } else {
        pinv_helper(m, tol)
    }
}

// ----------------------------------------------------------------------------
// private helpers
// ----------------------------------------------------------------------------

fn identity(n: usize) -> Matrix<f64> {
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = 1.0;
    }
    m
}

fn transpose(m: &Matrix<f64>) -> Matrix<f64> {
    let mut t = Matrix::zeros(m.nc(), m.nr());
    for r in 0..m.nr() {
        for c in 0..m.nc() {
            t[(c, r)] = m[(r, c)];
        }
    }
    t
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

    fn max_abs(a: &Matrix<f64>, b: &Matrix<f64>) -> f64 {
        (0..a.nr())
            .flat_map(move |r| (0..a.nc()).map(move |c| (r, c)))
            .map(|(r, c)| (a[(r, c)] - b[(r, c)]).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_inv() {
        let a = lcg_matrix(5, 5, 77);
        let ainv = inv(&a).unwrap();
        let ident = identity(5);
        assert!(
            max_abs(&mat_mul(&a, &ainv), &ident) < 1e-10,
            "A*inv(A) != I"
        );
        assert!(max_abs(&mat_mul(&ainv, &a), &ident) < 1e-10);

        // singular matrix -> Err
        let mut s = Matrix::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                s[(r, c)] = (r as f64) + 1.0;
            }
        }
        assert!(inv(&s).is_err());
    }

    #[test]
    fn test_det_vs_cofactor_3x3() {
        let mut a = Matrix::zeros(3, 3);
        let vals = [[1.2, -3.0, 0.5], [4.0, 2.2, -1.0], [0.3, 5.0, 1.7]];
        for r in 0..3 {
            for c in 0..3 {
                a[(r, c)] = vals[r][c];
            }
        }
        let cofactor = vals[0][0] * (vals[1][1] * vals[2][2] - vals[1][2] * vals[2][1])
            - vals[0][1] * (vals[1][0] * vals[2][2] - vals[1][2] * vals[2][0])
            + vals[0][2] * (vals[1][0] * vals[2][1] - vals[1][1] * vals[2][0]);
        assert!((det(&a) - cofactor).abs() < 1e-9);

        // singular: det == 0
        let mut s = Matrix::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                s[(r, c)] = (r as f64) + 1.0;
            }
        }
        assert_eq!(det(&s), 0.0);
    }

    #[test]
    fn test_pinv() {
        // square invertible: pinv == inv
        let a = lcg_matrix(4, 4, 88);
        let p = pinv(&a, 0.0);
        let ainv = inv(&a).unwrap();
        assert!(max_abs(&p, &ainv) < 1e-8);

        // rectangular wide matrix: A * pinv(A) * A == A
        let b = lcg_matrix(3, 5, 99);
        let bp = pinv(&b, 0.0);
        assert_eq!((bp.nr(), bp.nc()), (5, 3));
        assert!(max_abs(&mat_mul(&mat_mul(&b, &bp), &b), &b) < 1e-9);

        // rectangular tall matrix
        let c = lcg_matrix(5, 2, 123);
        let cp = pinv(&c, 0.0);
        assert_eq!((cp.nr(), cp.nc()), (2, 5));
        assert!(max_abs(&mat_mul(&mat_mul(&c, &cp), &c), &c) < 1e-9);

        // rank-deficient: pinv still gives A*pinv(A)*A == A
        let mut r1 = Matrix::zeros(4, 3);
        for r in 0..4 {
            for c in 0..3 {
                r1[(r, c)] = ((r + 1) % 4) as f64 * (c as f64 + 1.0);
            }
        }
        let r1p = pinv(&r1, 0.0);
        assert!(max_abs(&mat_mul(&mat_mul(&r1, &r1p), &r1), &r1) < 1e-9);
    }
}

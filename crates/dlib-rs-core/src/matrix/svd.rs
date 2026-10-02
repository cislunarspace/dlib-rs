//! Singular value decomposition, ported line-by-line from
//! `dlib/matrix/matrix_la.h` (`svd4`, `svd3`, `svd2`, `svd`).
//!
//! The non-LAPACK branch of the dlib implementation is ported: Householder
//! bidiagonalization followed by the implicit-shift QR diagonalization from
//! the Algol code in "Handbook for Automatic Computation, vol. II, Linear
//! Algebra", with dlib's added iteration cap of 300 per singular value.

use crate::matrix::Matrix;

/// Which columns of `u` to compute; port of `dlib::svd_u_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SvdUMode {
    /// Port of `SVD_NO_U`.
    NoU,
    /// Port of `SVD_SKINNY_U`: `u` is `m x n`.
    SkinnyU,
    /// Port of `SVD_FULL_U`: `u` is `m x m`.
    FullU,
}

/// Port of `dlib::svd4` (non-LAPACK branch).
///
/// Given the singular value decomposition `a = u * diagm(q) * trans(v)` for
/// an `m x n` matrix `a` with `m >= n`: after the call `u` is columnwise
/// orthogonal (`m x m` for `SVD_FULL_U`, `m x n` for `SVD_SKINNY_U`), `q` is
/// an `n x 1` matrix of singular values and `v` is `n x n` orthogonal.
///
/// Returns the error code: `0` if no errors, `k` on failure to converge at
/// the `k`-th singular value.
fn svd4(
    u_mode: SvdUMode,
    withv: bool,
    a: &Matrix<f64>,
    u: &mut Matrix<f64>,
    q: &mut Matrix<f64>,
    v: &mut Matrix<f64>,
) -> usize {
    assert!(
        a.nr() >= a.nc(),
        "svd4(): you have given an invalidly sized matrix (a.nr() < a.nc())"
    );

    let mut eps = f64::EPSILON;
    let tol = f64::MIN_POSITIVE / eps;

    let m = a.nr();
    let n = a.nc();
    let mut retval = 0usize;

    let mut e = vec![0.0f64; n];
    q.set_size(n, 1);
    match u_mode {
        SvdUMode::FullU => u.set_size(m, m),
        _ => u.set_size(m, n),
    }
    if withv {
        v.set_size(n, n);
    }

    let mut g;
    let mut x;
    let mut f;
    let mut h;
    let mut s;

    /* Copy 'a' to 'u' */
    for i in 0..m {
        for j in 0..n {
            u[(i, j)] = a[(i, j)];
        }
    }

    /* Householder's reduction to bidiagonal form. */
    g = 0.0;
    x = 0.0;
    for i in 0..n {
        e[i] = g;
        s = 0.0;
        let l = i + 1;

        for j in i..m {
            s += u[(j, i)] * u[(j, i)];
        }

        if s < tol {
            g = 0.0;
        } else {
            f = u[(i, i)];
            g = if f < 0.0 { s.sqrt() } else { -s.sqrt() };
            h = f * g - s;
            u[(i, i)] = f - g;

            for j in l..n {
                s = 0.0;

                for k in i..m {
                    s += u[(k, i)] * u[(k, j)];
                }

                f = s / h;

                for k in i..m {
                    let val = u[(k, j)] + f * u[(k, i)];
                    u[(k, j)] = val;
                }
            }
        }

        q[(i, 0)] = g;
        s = 0.0;

        for j in l..n {
            s += u[(i, j)] * u[(i, j)];
        }

        if s < tol {
            g = 0.0;
        } else {
            f = u[(i, l)];
            g = if f < 0.0 { s.sqrt() } else { -s.sqrt() };
            h = f * g - s;
            u[(i, l)] = f - g;

            for j in l..n {
                e[j] = u[(i, j)] / h;
            }

            for j in l..m {
                s = 0.0;

                for k in l..n {
                    s += u[(j, k)] * u[(i, k)];
                }

                for k in l..n {
                    let val = u[(j, k)] + s * e[k];
                    u[(j, k)] = val;
                }
            }
        }

        let y = q[(i, 0)].abs() + e[i].abs();
        if y > x {
            x = y;
        }
    }

    /* accumulation of right-hand transformations */
    if withv {
        // l and g carry over from the bidiagonal loop: after the final
        // iteration (i == n-1) both s-loops are empty so g == 0 and l == n.
        let mut l = n;
        g = 0.0;
        for i in (0..n).rev() {
            if g != 0.0 {
                h = u[(i, i + 1)] * g;

                for j in l..n {
                    v[(j, i)] = u[(i, j)] / h;
                }

                for j in l..n {
                    s = 0.0;

                    for k in l..n {
                        s += u[(i, k)] * v[(k, j)];
                    }

                    for k in l..n {
                        let val = v[(k, j)] + s * v[(k, i)];
                        v[(k, j)] = val;
                    }
                }
            }

            for j in l..n {
                v[(i, j)] = 0.0;
                v[(j, i)] = 0.0;
            }

            v[(i, i)] = 1.0;
            g = e[i];
            l = i;
        }
    }

    /* accumulation of left-hand transformations */
    if u_mode != SvdUMode::NoU {
        for i in n..u.nr() {
            for j in n..u.nc() {
                u[(i, j)] = 0.0;
            }

            if i < u.nc() {
                u[(i, i)] = 1.0;
            }
        }
    }

    if u_mode != SvdUMode::NoU {
        for i in (0..n).rev() {
            let l = i + 1;
            g = q[(i, 0)];

            for j in l..u.nc() {
                u[(i, j)] = 0.0;
            }

            if g != 0.0 {
                h = u[(i, i)] * g;

                for j in l..u.nc() {
                    s = 0.0;

                    for k in l..m {
                        s += u[(k, i)] * u[(k, j)];
                    }

                    f = s / h;

                    for k in i..m {
                        let val = u[(k, j)] + f * u[(k, i)];
                        u[(k, j)] = val;
                    }
                }

                for j in i..m {
                    let val = u[(j, i)] / g;
                    u[(j, i)] = val;
                }
            } else {
                for j in i..m {
                    u[(j, i)] = 0.0;
                }
            }

            u[(i, i)] += 1.0;
        }
    }

    /* diagonalization of the bidiagonal form */
    eps *= x;

    'k_loop: for k in (0..n).rev() {
        let mut iter = 0;

        'test_f_splitting: loop {
            // test_f_splitting: look for a small e(l)
            let mut l = k;
            let mut cancellation = false;
            loop {
                if e[l].abs() <= eps {
                    break; // goto test_f_convergence
                }
                // At l == 0 the C++ code reads q(-1), which dlib indexes
                // from the end; since e(0) == 0 in every reachable state
                // this branch is dead, so treat it as convergence.
                if l == 0 {
                    break;
                }
                if q[(l - 1, 0)].abs() <= eps {
                    cancellation = true;
                    break; // goto cancellation
                }
                l -= 1;
            }

            if cancellation {
                /* cancellation of e(l) if l > 0 */
                let mut c = 0.0f64;
                let mut s2 = 1.0f64;
                let l1 = l - 1;

                let mut goto_convergence = false;
                for i in l..=k {
                    f = s2 * e[i];
                    e[i] *= c;

                    if f.abs() <= eps {
                        goto_convergence = true;
                        break;
                    }

                    g = q[(i, 0)];
                    h = (f * f + g * g).sqrt();
                    q[(i, 0)] = h;
                    c = g / h;
                    s2 = -f / h;

                    if u_mode != SvdUMode::NoU {
                        for j in 0..m {
                            let y = u[(j, l1)];
                            let z = u[(j, i)];
                            u[(j, l1)] = y * c + z * s2;
                            u[(j, i)] = -y * s2 + z * c;
                        }
                    }
                }
                let _ = goto_convergence; // falls through to test_f_convergence
            }

            // test_f_convergence:
            let z = q[(k, 0)];
            if l == k {
                // convergence:
                if z < 0.0 {
                    /* q(k) is made non-negative */
                    q[(k, 0)] = -z;
                    if withv {
                        for j in 0..n {
                            let val = -v[(j, k)];
                            v[(j, k)] = val;
                        }
                    }
                }
                continue 'k_loop;
            }

            /* shift from bottom 2x2 minor */
            iter += 1;
            if iter > 300 {
                retval = k;
                break 'k_loop;
            }
            x = q[(l, 0)];
            let y = q[(k - 1, 0)];
            g = e[k - 1];
            h = e[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = (f * f + 1.0).sqrt();
            f = ((x - z) * (x + z) + h * (y / (if f < 0.0 { f - g } else { f + g }) - h)) / x;

            /* next QR transformation */
            let mut c = 1.0f64;
            let mut s2 = 1.0f64;

            for i in l + 1..=k {
                g = e[i];
                let mut y = q[(i, 0)];
                h = s2 * g;
                g *= c;
                let mut z = (f * f + h * h).sqrt();
                e[i - 1] = z;
                c = f / z;
                s2 = h / z;
                f = x * c + g * s2;
                g = -x * s2 + g * c;
                h = y * s2;
                y *= c;

                if withv {
                    for j in 0..n {
                        x = v[(j, i - 1)];
                        z = v[(j, i)];
                        v[(j, i - 1)] = x * c + z * s2;
                        v[(j, i)] = -x * s2 + z * c;
                    }
                }

                z = (f * f + h * h).sqrt();
                q[(i - 1, 0)] = z;
                if z != 0.0 {
                    c = f / z;
                    s2 = h / z;
                }
                f = c * g + s2 * y;
                x = -s2 * g + c * y;
                if u_mode != SvdUMode::NoU {
                    for j in 0..m {
                        let y2 = u[(j, i - 1)];
                        let z2 = u[(j, i)];
                        u[(j, i - 1)] = y2 * c + z2 * s2;
                        u[(j, i)] = -y2 * s2 + z2 * c;
                    }
                }
            }

            e[l] = 0.0;
            e[k] = f;
            q[(k, 0)] = x;

            continue 'test_f_splitting;
        }
    }

    retval
}

/// Port of `dlib::svd3` from `dlib/matrix/matrix_la.h`.
///
/// Computes `m = u * w * trans(v)` where `u` is `m.nr() x m.nc()` with
/// orthonormal columns, `w` is `m.nc() x 1` singular values, and `v` is
/// `m.nc() x m.nc()` orthogonal. Returns true on convergence (dlib ignores
/// the `svd4` error code here; this port surfaces it).
pub fn svd3(
    m: &Matrix<f64>,
    u: &mut Matrix<f64>,
    w: &mut Matrix<f64>,
    v: &mut Matrix<f64>,
) -> bool {
    if m.nr() >= m.nc() {
        svd4(SvdUMode::SkinnyU, true, m, u, w, v) == 0
    } else {
        let mt = transpose(m);
        let mut q = Matrix::new();
        let ok = svd4(SvdUMode::FullU, true, &mt, v, &mut q, u) == 0;

        // if u isn't the size we want then pad it (and w) with zeros
        if u.nc() < m.nc() {
            // w = join_cols(q, zeros(m.nc()-u.nc(),1))
            w.set_size(m.nc(), 1);
            for i in 0..q.nr() {
                w[(i, 0)] = q[(i, 0)];
            }
            for i in q.nr()..m.nc() {
                w[(i, 0)] = 0.0;
            }
            // u = join_rows(u, zeros(u.nr(), extra))
            let old = u.clone();
            let rows = u.nr();
            u.set_size(rows, m.nc());
            for i in 0..rows {
                for j in 0..old.nc() {
                    u[(i, j)] = old[(i, j)];
                }
                for j in old.nc()..m.nc() {
                    u[(i, j)] = 0.0;
                }
            }
        } else {
            copy_matrix(w, &q);
        }
        ok
    }
}

/// Port of `dlib::svd2` from `dlib/matrix/matrix_la.h`.
///
/// `svd2(withu, withv, a, u, q, v)`: if `withu` is true `u` is computed as
/// `a.nr() x a.nr()` (`SVD_FULL_U`), otherwise not computed at all. `q` is
/// the `a.nc() x 1` singular values, `v` the `a.nc() x a.nc()` right
/// singular vectors (only if `withv`). Returns the error code (0 = success,
/// `k` = failure to converge at the `k`-th singular value), like dlib.
pub fn svd2(
    withu: bool,
    withv: bool,
    a: &Matrix<f64>,
    u: &mut Matrix<f64>,
    q: &mut Matrix<f64>,
    v: &mut Matrix<f64>,
) -> usize {
    let u_mode = if withu {
        SvdUMode::FullU
    } else {
        SvdUMode::NoU
    };
    svd4(u_mode, withv, a, u, q, v)
}

/// Port of `dlib::svd` from `dlib/matrix/matrix_la.h`, same argument order
/// `svd(m, u, w, v)` as dlib.
///
/// Computes the singular value decomposition `m == u * w * trans(v)` where
/// `w` is the `m.nc() x m.nc()` diagonal matrix of singular values (`diagm`
/// of the singular value vector, exactly like the dlib `svd` wrapper),
/// `u` is `m.nr() x m.nc()` with orthonormal columns and `v` is
/// `m.nc() x m.nc()` orthogonal (both from `svd3`).
///
/// Returns true on convergence.
pub fn svd(m: &Matrix<f64>, u: &mut Matrix<f64>, w: &mut Matrix<f64>, v: &mut Matrix<f64>) -> bool {
    let mut big_w = Matrix::new();
    let ok = svd3(m, u, &mut big_w, v);
    w.set_size(big_w.nr(), big_w.nr());
    for i in 0..w.nr() {
        for j in 0..w.nc() {
            w[(i, j)] = 0.0;
        }
        w[(i, i)] = big_w[(i, 0)];
    }
    ok
}

// ----------------------------------------------------------------------------
// private helpers
// ----------------------------------------------------------------------------

fn transpose(m: &Matrix<f64>) -> Matrix<f64> {
    let mut t = Matrix::zeros(m.nc(), m.nr());
    for r in 0..m.nr() {
        for c in 0..m.nc() {
            t[(c, r)] = m[(r, c)];
        }
    }
    t
}

fn copy_matrix(dst: &mut Matrix<f64>, src: &Matrix<f64>) {
    dst.set_size(src.nr(), src.nc());
    for r in 0..src.nr() {
        for c in 0..src.nc() {
            dst[(r, c)] = src[(r, c)];
        }
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
    fn test_svd_square() {
        let a = lcg_matrix(4, 4, 5);
        let mut w = Matrix::new();
        let mut u = Matrix::new();
        let mut v = Matrix::new();
        assert!(svd(&a, &mut u, &mut w, &mut v));
        assert_eq!((u.nr(), u.nc()), (4, 4));
        assert_eq!((w.nr(), w.nc()), (4, 4));
        assert_eq!((v.nr(), v.nc()), (4, 4));
        assert!(max_abs(&mat_mul(&mat_mul(&u, &w), &transpose(&v)), &a) < 1e-10);
    }

    #[test]
    fn test_svd_rectangular_5x3() {
        let a = lcg_matrix(5, 3, 17);
        let mut w = Matrix::new();
        let mut u = Matrix::new();
        let mut v = Matrix::new();
        assert!(svd(&a, &mut u, &mut w, &mut v));
        assert_eq!((u.nr(), u.nc()), (5, 3));
        assert_eq!((w.nr(), w.nc()), (3, 3));
        assert_eq!((v.nr(), v.nc()), (3, 3));
        assert!(max_abs(&mat_mul(&mat_mul(&u, &w), &transpose(&v)), &a) < 1e-10);

        // wide matrix 3x5 via svd3 transposed path
        let at = transpose(&a);
        let mut u2 = Matrix::new();
        let mut w2 = Matrix::new();
        let mut v2 = Matrix::new();
        assert!(svd3(&at, &mut u2, &mut w2, &mut v2));
        assert_eq!((u2.nr(), u2.nc()), (3, 5));
        assert_eq!((w2.nr(), w2.nc()), (5, 1));
        assert_eq!((v2.nr(), v2.nc()), (5, 5));
        // u2 * diagm(w2) * v2' should equal at (w2 padded with zeros)
        let mut w2d = Matrix::zeros(5, 5);
        for i in 0..5 {
            w2d[(i, i)] = w2[(i, 0)];
        }
        assert!(max_abs(&mat_mul(&mat_mul(&u2, &w2d), &transpose(&v2)), &at) < 1e-10);
    }

    #[test]
    fn test_svd2_and_singular_values() {
        let a = lcg_matrix(6, 3, 31);
        let mut u = Matrix::new();
        let mut q = Matrix::new();
        let mut v = Matrix::new();
        assert_eq!(svd2(true, true, &a, &mut u, &mut q, &mut v), 0);
        assert_eq!((u.nr(), u.nc()), (6, 6));
        assert_eq!((q.nr(), q.nc()), (3, 1));
        assert_eq!((v.nr(), v.nc()), (3, 3));
        let mut w = Matrix::zeros(3, 3);
        for i in 0..3 {
            w[(i, i)] = q[(i, 0)];
        }
        // reconstruct with only the first 3 columns of u (thin part)
        let mut ut = Matrix::zeros(6, 3);
        for r in 0..6 {
            for c in 0..3 {
                ut[(r, c)] = u[(r, c)];
            }
        }
        assert!(max_abs(&mat_mul(&mat_mul(&ut, &w), &transpose(&v)), &a) < 1e-10);
        // all singular values non-negative (dlib makes q(k) non-negative)
        for i in 0..3 {
            assert!(q[(i, 0)] >= 0.0);
        }
    }

    #[test]
    fn test_svd_rank_deficient() {
        // rank-1 matrix: outer product
        let mut a = Matrix::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                a[(r, c)] = (r as f64 + 1.0) * (c as f64 + 2.0);
            }
        }
        let mut w = Matrix::new();
        let mut u = Matrix::new();
        let mut v = Matrix::new();
        assert!(svd(&a, &mut u, &mut w, &mut v));
        assert!(max_abs(&mat_mul(&mat_mul(&u, &w), &transpose(&v)), &a) < 1e-10);
        // exactly one nonzero singular value
        let nonzeros = (0..4).filter(|i| w[(*i, *i)].abs() > 1e-9).count();
        assert_eq!(nonzeros, 1);
    }
}

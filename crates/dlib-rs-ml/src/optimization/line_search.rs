//! Port of `dlib/optimization/optimization_line_search.h` (Davis E. King,
//! Boost Software License). The bracketing phase follows block 2.6.2 and the
//! sectioning phase block 2.6.4 from "Practical Methods of Optimization" by
//! R. Fletcher, exactly as in the C++ header.

use dlib_rs_core::matrix::Matrix;

/// dlib `put_in_range(a, b, val)` from `dlib/algs.h`: clamps `val` into the
/// inclusive range spanned by `a` and `b` (order independent).
fn put_in_range(a: f64, b: f64, val: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

/// Port of the 4-argument `poly_min_extrap(f0, d0, f1, d1, limit)` from
/// `dlib/optimization/optimization_line_search.h`. Given the value and
/// derivative of a function at 0 and 1, fits a cubic and returns the
/// (clamped) location of its minimum as a fraction in `[0, limit]`.
fn poly_min_extrap(f0: f64, d0: f64, f1: f64, d1: f64, limit: f64) -> f64 {
    let n = 3.0 * (f1 - f0) - 2.0 * d0 - d1;
    let e = d0 + d1 - 2.0 * (f1 - f0);

    // find the minimum of the derivative of the polynomial

    let mut temp = f64::max(n * n - 3.0 * e * d0, 0.0);

    if temp < 0.0 {
        return 0.5;
    }

    temp = temp.sqrt();

    if e.abs() <= f64::EPSILON {
        return 0.5;
    }

    // figure out the two possible min values
    let x1 = (temp - n) / (3.0 * e);
    let x2 = -(temp + n) / (3.0 * e);

    // compute the value of the interpolating polynomial at these two points
    let y1 = f0 + d0 * x1 + n * x1 * x1 + e * x1 * x1 * x1;
    let y2 = f0 + d0 * x2 + n * x2 * x2 + e * x2 * x2 * x2;

    // pick the best point
    let x = if y1 < y2 { x1 } else { x2 };

    // now make sure the minimum is within the allowed range of [0,limit]
    put_in_range(0.0, limit, x)
}

/// Port of `dlib::line_search(f, f0, der, d0, rho, sigma, min_f, max_iter)`
/// from `dlib/optimization/optimization_line_search.h`. `f` and `der`
/// evaluate the objective and its directional derivative at points along the
/// line `x + alpha * dir` (dlib's `make_line_search_function`).
///
/// Returns `(alpha, f(alpha), der(alpha), num_iterations)`. The public
/// wrapper [`line_search`] drops `alpha` per the crate contract; `find_min`
/// uses this full form to take the step.
#[allow(clippy::too_many_arguments)] // mirrors the dlib line_search() signature
pub(crate) fn line_search_full<F, D>(
    f: F,
    f0: f64,
    der: D,
    d0: f64,
    x: &Matrix<f64>,
    dir: &Matrix<f64>,
    rho: f64,
    sigma: f64,
    min_f: f64,
    max_iter: usize,
) -> (f64, f64, f64, usize)
where
    F: Fn(&Matrix<f64>) -> f64,
    D: Fn(&Matrix<f64>) -> f64,
{
    debug_assert!(
        0.0 < rho && rho < sigma && sigma < 1.0 && max_iter > 0,
        "line_search(): invalid arguments (rho={rho}, sigma={sigma}, max_iter={max_iter})"
    );

    // 1 <= tau1a < tau1b. Controls the alpha jump size during the bracketing
    // phase of the search.
    let tau1a = 1.4;
    let tau1b = 9.0;

    // it must be the case that 0 < tau2 < tau3 <= 1/2 for the algorithm to
    // function correctly but the specific values of tau2 and tau3 aren't
    // super important.
    let tau2 = 1.0 / 10.0;
    let tau3 = 1.0 / 2.0;

    // Stop right away and return a step size of 0 if the gradient is 0 at the
    // starting point
    if d0.abs() <= f0.abs() * f64::EPSILON {
        return (0.0, f0, d0, 0);
    }

    // Stop right away if the current value is good enough according to min_f
    if f0 <= min_f {
        return (0.0, f0, d0, 0);
    }

    // Figure out a reasonable upper bound on how large alpha can get.
    let mu = (min_f - f0) / (rho * d0);

    let mut alpha = 1.0;
    if mu < 0.0 {
        alpha = -alpha;
    }
    alpha = put_in_range(0.0, 0.65 * mu, alpha);

    let mut last_alpha = 0.0;
    let mut last_val = f0;
    let mut last_val_der = d0;

    // The bracketing stage will find a range of points [a,b]
    // that contains a reasonable solution to the line search
    let mut a;
    let mut b;

    // These variables will hold the values and derivatives of f(a) and f(b)
    let mut a_val;
    let mut b_val;
    let mut a_val_der;
    let mut b_val_der;

    // This thresh value represents the Wolfe curvature condition
    let thresh = (sigma * d0).abs();

    let eval = |alpha: f64, f: &F, der: &D| -> (f64, f64) {
        let point = x.clone() + dir.clone() * alpha;
        (f(&point), der(&point))
    };

    let mut itr: usize = 0;
    // do the bracketing stage to find the bracket range [a,b]
    loop {
        itr += 1;
        let (val, val_der) = eval(alpha, &f, &der);

        // we are done with the line search since we found a value smaller
        // than the minimum f value
        if val <= min_f {
            return (alpha, val, val_der, itr);
        }

        if val > f0 + rho * alpha * d0 || val >= last_val {
            a_val = last_val;
            a_val_der = last_val_der;
            b_val = val;
            b_val_der = val_der;

            a = last_alpha;
            b = alpha;
            break;
        }

        if val_der.abs() <= thresh {
            return (alpha, val, val_der, itr);
        }

        // if we are stuck not making progress then quit with the current alpha
        if last_alpha == alpha || itr >= max_iter {
            return (alpha, val, val_der, itr);
        }

        if val_der >= 0.0 {
            a_val = val;
            a_val_der = val_der;
            b_val = last_val;
            b_val_der = last_val_der;

            a = alpha;
            b = last_alpha;
            break;
        }

        let temp = alpha;
        // Pick a larger range [first, last].  We will pick the next alpha in
        // that range.
        let (first, last);
        if mu > 0.0 {
            first = f64::min(mu, alpha + tau1a * (alpha - last_alpha));
            last = f64::min(mu, alpha + tau1b * (alpha - last_alpha));
        } else {
            first = f64::max(mu, alpha + tau1a * (alpha - last_alpha));
            last = f64::max(mu, alpha + tau1b * (alpha - last_alpha));
        }

        // pick a point between first and last by doing some kind of
        // interpolation
        if last_alpha < alpha {
            alpha = last_alpha
                + (alpha - last_alpha)
                    * poly_min_extrap(last_val, last_val_der, val, val_der, 1e10);
        } else {
            alpha = alpha
                + (last_alpha - alpha)
                    * poly_min_extrap(val, val_der, last_val, last_val_der, 1e10);
        }

        alpha = put_in_range(first, last, alpha);

        last_alpha = temp;

        last_val = val;
        last_val_der = val_der;
    }

    // Now do the sectioning phase from 2.6.4
    loop {
        itr += 1;
        let first = a + tau2 * (b - a);
        let last = b - tau3 * (b - a);

        // use interpolation to pick alpha between first and last
        alpha = a + (b - a) * poly_min_extrap(a_val, a_val_der, b_val, b_val_der, 1.0);
        alpha = put_in_range(first, last, alpha);

        let (val, val_der) = eval(alpha, &f, &der);

        // we are done with the line search since we found a value smaller
        // than the minimum f value or we ran out of iterations.
        if val <= min_f || itr >= max_iter {
            return (alpha, val, val_der, itr);
        }

        // stop if the interval gets so small that it isn't shrinking any more
        // due to rounding error
        if a == first || b == last {
            return (b, b_val, b_val_der, itr);
        }

        // If alpha has basically become zero then just stop.  Think of it
        // like this, if we take the largest possible alpha step will the
        // objective function change at all?  If not then there isn't any
        // point looking for a better alpha.
        let max_possible_alpha = f64::max(a.abs(), b.abs());
        if (max_possible_alpha * d0).abs() <= f0.abs() * f64::EPSILON {
            return (alpha, val, val_der, itr);
        }

        if val > f0 + rho * alpha * d0 || val >= a_val {
            b = alpha;
            b_val = val;
            b_val_der = val_der;
        } else {
            if val_der.abs() <= thresh {
                return (alpha, val, val_der, itr);
            }

            if (b - a) * val_der >= 0.0 {
                b = a;
                b_val = a_val;
                b_val_der = a_val_der;
            }

            a = alpha;
            a_val = val;
            a_val_der = val_der;
        }
    }
}

/// Port of `dlib::line_search(f, f0, der, d0, rho, sigma, min_f, max_iter)`
/// from `dlib/optimization/optimization_line_search.h` (strong Wolfe
/// conditions via `rho`/`sigma`; Fletcher blocks 2.6.2/2.6.4).
///
/// `f`/`der` evaluate the objective and its directional derivative at points
/// `x + alpha * dir`. Returns `(f_alpha, d_alpha, num_iterations)`.
#[allow(clippy::too_many_arguments)] // mirrors the dlib line_search() signature
pub fn line_search<F, D>(
    f: F,
    der: D,
    f0: f64,
    d0: f64,
    x: &Matrix<f64>,
    dir: &Matrix<f64>,
    rho: f64,
    sigma: f64,
    min_f: f64,
    max_iter: usize,
) -> (f64, f64, usize)
where
    F: Fn(&Matrix<f64>) -> f64,
    D: Fn(&Matrix<f64>) -> f64,
{
    let (_alpha, f_alpha, d_alpha, itr) =
        line_search_full(f, f0, der, d0, x, dir, rho, sigma, min_f, max_iter);
    (f_alpha, d_alpha, itr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlib_rs_core::matrix::dot;

    // f(x) = ||x||^2 along descent direction d: minimizer at
    // alpha* = -<x,d>/||d||^2.
    fn check_quadratic(x0: &[f64], d: &[f64]) {
        let n = x0.len();
        let x = Matrix::from_row_vec(n, 1, x0);
        let dir = Matrix::from_row_vec(n, 1, d);
        let f = |p: &Matrix<f64>| dot(p, p);
        let der = |p: &Matrix<f64>| 2.0 * dot(p, &dir);
        let f0 = f(&x);
        let d0 = der(&x);
        // With loose sigma the search correctly stops at the first
        // Wolfe-sufficient point, so use a tight curvature tolerance to
        // pin the minimizer of this quadratic.
        let (alpha, f_alpha, d_alpha, _itr) =
            line_search_full(f, f0, der, d0, &x, &dir, 1e-12, 1e-10, -1e100, 100);
        let expected_alpha = -dot(&x, &dir) / dot(&dir, &dir);
        // The sectioning phase stops when the bracket stops shrinking due to
        // rounding (dlib's `a == first || b == last` early-out), which pins
        // alpha to ~1e-8 relative accuracy here.
        assert!(
            (alpha - expected_alpha).abs() < 1e-6,
            "alpha={alpha} expected={expected_alpha}"
        );
        assert!(d_alpha.abs() < 1e-6, "d_alpha={d_alpha}");
        let p = x.clone() + dir.clone() * expected_alpha;
        assert!((f_alpha - dot(&p, &p)).abs() < 1e-12);
    }

    #[test]
    fn quadratic_line_search() {
        check_quadratic(&[3.0, 4.0], &[-1.0, 0.0]);
        check_quadratic(&[3.0, 4.0], &[-1.0, -1.0]);
        check_quadratic(&[-2.0, 5.0, 1.0], &[0.5, -1.5, 2.0]);
        check_quadratic(&[10.0], &[-3.0]);
    }

    #[test]
    fn public_wrapper_matches_full() {
        let x = Matrix::from_row_vec(2, 1, &[3.0, 4.0]);
        let dir = Matrix::from_row_vec(2, 1, &[1.0, -1.0]);
        let f = |p: &Matrix<f64>| dot(p, p);
        let der = |p: &Matrix<f64>| 2.0 * dot(p, &dir);
        let r = line_search(f, der, f(&x), der(&x), &x, &dir, 0.01, 0.9, -1e100, 100);
        assert!(r.1.abs() < 1e-9);
        assert!(r.0 > 0.0);
    }
}

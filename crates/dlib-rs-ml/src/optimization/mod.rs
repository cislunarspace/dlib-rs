//! Optimization port of `dlib/optimization/*.h`: find_min/find_max, line
//! search, search strategies (CG/BFGS/L-BFGS/Newton) and stop strategies.

pub mod bobyqa;
pub mod line_search;
pub mod search_strategies;
pub mod stop_strategies;

pub use search_strategies::{
    BfgsSearchStrategy, CgSearchStrategy, LbfgsSearchStrategy, NewtonSearchStrategy, SearchStrategy,
};
pub use stop_strategies::{
    GradientNormStopStrategy, ObjectiveDeltaStopStrategy, StopStrategies, StopStrategy,
};

use dlib_rs_core::matrix::{dot, Matrix};
use line_search::line_search_full;
use std::cell::RefCell;

fn all_finite(m: &Matrix<f64>) -> bool {
    (0..m.size()).all(|i| m[i].is_finite())
}

/// Port of `dlib::find_min(search_strategy, stop_strategy, f, der, x, min_f)`
/// from `dlib/optimization/optimization.h`. Minimizes `f` starting from `x0`
/// using the analytic gradient `grad`. dlib modifies `x` in place and returns
/// the objective; this port returns `(f_min, x)`.
///
/// The line search parameters come from the search strategy
/// (`get_wolfe_rho`, `get_wolfe_sigma`, `get_max_line_search_iterations`).
/// Panics if the objective or gradient produce non-finite values (dlib
/// throws `error` there).
#[allow(clippy::too_many_arguments)] // mirrors the dlib find_min() template signature
pub fn find_min<S, SS, F, G>(
    search_strategy: &mut S,
    stop_strategy: &mut SS,
    f: F,
    grad: G,
    mut x: Matrix<f64>,
    min_f: f64,
) -> (f64, Matrix<f64>)
where
    S: SearchStrategy,
    SS: StopStrategies,
    F: Fn(&Matrix<f64>) -> f64,
    G: Fn(&Matrix<f64>) -> Matrix<f64>,
{
    let mut f_value = f(&x);
    let mut g = grad(&x);

    if !f_value.is_finite() || !all_finite(&g) {
        panic!("The objective function generated non-finite outputs");
    }

    while stop_strategy.should_continue_search(f_value, &g) && f_value > min_f {
        let s = search_strategy.get_next_direction(&x, f_value, &g);

        // dlib's make_line_search_function(f/der, x, s, out) stores the last
        // evaluated value/gradient through a reference; mirror that with cells.
        let f_cell = RefCell::new(f_value);
        let g_cell = RefCell::new(g.clone());
        let lf = |p: &Matrix<f64>| -> f64 {
            let v = f(p);
            *f_cell.borrow_mut() = v;
            v
        };
        let ld = |p: &Matrix<f64>| -> f64 {
            let gr = grad(p);
            let d = dot(&gr, &s);
            *g_cell.borrow_mut() = gr;
            d
        };

        let (alpha, _f_alpha, _d_alpha, _itr) = line_search_full(
            lf,
            f_value,
            ld,
            dot(&g, &s), // compute initial gradient for the line search
            &x,
            &s,
            search_strategy.get_wolfe_rho(),
            search_strategy.get_wolfe_sigma(),
            min_f,
            search_strategy.get_max_line_search_iterations(),
        );

        // Take the search step indicated by the above line search
        x += s * alpha;

        f_value = *f_cell.borrow();
        g = g_cell.borrow().clone();

        if !f_value.is_finite() || !all_finite(&g) {
            panic!("The objective function generated non-finite outputs");
        }
    }

    (f_value, x)
}

/// Port of `dlib::find_max(search_strategy, stop_strategy, f, der, x, max_f)`
/// from `dlib/optimization/optimization.h`: a copy of `find_min` with the
/// signs flipped to look for a maximum. Returns `(f_max, x)`.
#[allow(clippy::too_many_arguments)] // mirrors the dlib find_max() template signature
pub fn find_max<S, SS, F, G>(
    search_strategy: &mut S,
    stop_strategy: &mut SS,
    f: F,
    grad: G,
    mut x: Matrix<f64>,
    max_f: f64,
) -> (f64, Matrix<f64>)
where
    S: SearchStrategy,
    SS: StopStrategies,
    F: Fn(&Matrix<f64>) -> f64,
    G: Fn(&Matrix<f64>) -> Matrix<f64>,
{
    // This function is basically just a copy of find_min() but with - put in
    // the right places to flip things around so that it ends up looking for
    // the max rather than the min.
    let mut f_value = -f(&x);
    let mut g = -grad(&x);

    if !f_value.is_finite() || !all_finite(&g) {
        panic!("The objective function generated non-finite outputs");
    }

    while stop_strategy.should_continue_search(f_value, &g) && f_value > -max_f {
        let s = search_strategy.get_next_direction(&x, f_value, &g);

        let f_cell = RefCell::new(-f_value);
        let g_cell = RefCell::new(-g.clone());
        let lf = |p: &Matrix<f64>| -> f64 {
            let v = f(p);
            *f_cell.borrow_mut() = v;
            -v
        };
        let ld = |p: &Matrix<f64>| -> f64 {
            let gr = grad(p);
            let d = dot(&gr, &s);
            *g_cell.borrow_mut() = gr;
            -d
        };

        let (alpha, _f_alpha, _d_alpha, _itr) = line_search_full(
            lf,
            f_value,
            ld,
            dot(&g, &s), // compute initial gradient for the line search
            &x,
            &s,
            search_strategy.get_wolfe_rho(),
            search_strategy.get_wolfe_sigma(),
            -max_f,
            search_strategy.get_max_line_search_iterations(),
        );

        // Take the search step indicated by the above line search
        x += s * alpha;

        // Don't forget to negate these outputs from the line search since
        // they are from the unnegated versions of f() and der()
        f_value = -*f_cell.borrow();
        g = -g_cell.borrow().clone();

        if !f_value.is_finite() || !all_finite(&g) {
            panic!("The objective function generated non-finite outputs");
        }

        // Gradient is zero, no more progress is possible.  So stop.
        if alpha == 0.0 {
            break;
        }
    }

    (-f_value, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(p: &Matrix<f64>) -> f64 {
        let (x, y) = (p[(0, 0)], p[(1, 0)]);
        (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
    }

    fn rosenbrock_grad(p: &Matrix<f64>) -> Matrix<f64> {
        let (x, y) = (p[(0, 0)], p[(1, 0)]);
        Matrix::from_row_vec(
            2,
            1,
            &[
                -2.0 * (1.0 - x) - 400.0 * x * (y - x * x),
                200.0 * (y - x * x),
            ],
        )
    }

    fn assert_rosenbrock_converged(f_final: f64, x: &Matrix<f64>) {
        let err = ((x[(0, 0)] - 1.0).powi(2) + (x[(1, 0)] - 1.0).powi(2)).sqrt();
        assert!(
            err < 1e-8,
            "x error too large: {err} x={:?}",
            (x[(0, 0)], x[(1, 0)])
        );
        assert!(f_final < 1e-14, "f too large: {f_final}");
    }

    #[test]
    fn find_min_bfgs_rosenbrock() {
        let mut strat = BfgsSearchStrategy::new();
        let mut stop = ObjectiveDeltaStopStrategy::new(1e-13).with_max_iterations(100);
        let x0 = Matrix::from_row_vec(2, 1, &[-1.2, 1.0]);
        let (fv, x) = find_min(
            &mut strat,
            &mut stop,
            rosenbrock,
            rosenbrock_grad,
            x0,
            -1e100,
        );
        assert_rosenbrock_converged(fv, &x);
    }

    #[test]
    fn find_min_lbfgs_rosenbrock() {
        let mut strat = LbfgsSearchStrategy::default();
        let mut stop = ObjectiveDeltaStopStrategy::new(1e-13).with_max_iterations(100);
        let x0 = Matrix::from_row_vec(2, 1, &[-1.2, 1.0]);
        let (fv, x) = find_min(
            &mut strat,
            &mut stop,
            rosenbrock,
            rosenbrock_grad,
            x0,
            -1e100,
        );
        assert_rosenbrock_converged(fv, &x);
    }

    #[test]
    fn find_min_cg_rosenbrock() {
        let mut strat = CgSearchStrategy::new();
        let mut stop = ObjectiveDeltaStopStrategy::new(1e-13).with_max_iterations(100);
        let x0 = Matrix::from_row_vec(2, 1, &[-1.2, 1.0]);
        let (fv, x) = find_min(
            &mut strat,
            &mut stop,
            rosenbrock,
            rosenbrock_grad,
            x0,
            -1e100,
        );
        assert_rosenbrock_converged(fv, &x);
    }

    #[test]
    fn find_max_quadratic() {
        let f = |p: &Matrix<f64>| -(p[(0, 0)] - 3.0) * (p[(0, 0)] - 3.0);
        let g = |p: &Matrix<f64>| Matrix::from_row_vec(1, 1, &[-2.0 * (p[(0, 0)] - 3.0)]);
        let mut strat = BfgsSearchStrategy::new();
        let mut stop = ObjectiveDeltaStopStrategy::new(1e-12).with_max_iterations(50);
        let x0 = Matrix::from_row_vec(1, 1, &[0.0]);
        let (fv, x) = find_max(&mut strat, &mut stop, f, g, x0, 0.0);
        assert!((x[(0, 0)] - 3.0).abs() < 1e-9, "x={}", x[(0, 0)]);
        assert!(fv.abs() < 1e-9, "f={fv}");
    }
}

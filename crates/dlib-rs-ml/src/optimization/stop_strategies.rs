//! Port of `dlib/optimization/optimization_stop_strategies.h` (Davis E.
//! King, Boost Software License): objective-delta and gradient-norm stop
//! strategies.

use dlib_rs_core::matrix::{length, Matrix};

/// Common interface for dlib stop strategies
/// (`dlib/optimization/optimization_stop_strategies_abstract.h`): dlib's
/// `should_continue_search(x, funct_value, funct_derivative)`.
pub trait StopStrategies {
    /// Returns `true` while the optimizer should keep going.
    fn should_continue_search(&mut self, funct_value: f64, funct_derivative: &Matrix<f64>) -> bool;

    /// Resets the internal iteration counters / first-call state.
    fn reset(&mut self);
}

/// Port of dlib `objective_delta_stop_strategy(min_delta [, max_iter])`:
/// stop when the objective changes by less than `min_delta` between
/// iterations (the first call always continues) or after `max_iter`
/// iterations (when enabled).
#[derive(Default)]
pub struct ObjectiveDeltaStopStrategy {
    verbose: bool,
    been_used: bool,
    min_delta: f64,
    max_iter: usize,
    cur_iter: usize,
    prev_funct_value: f64,
}

impl ObjectiveDeltaStopStrategy {
    /// dlib `objective_delta_stop_strategy(min_delta)` (`min_delta` defaults
    /// to 1e-7 in C++).
    pub fn new(min_delta: f64) -> Self {
        assert!(
            min_delta >= 0.0,
            "objective_delta_stop_strategy(min_delta): min_delta can't be negative"
        );
        Self {
            verbose: false,
            been_used: false,
            min_delta,
            max_iter: 0,
            cur_iter: 0,
            prev_funct_value: 0.0,
        }
    }

    /// dlib `objective_delta_stop_strategy(min_delta, max_iter)`.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        assert!(
            max_iter > 0,
            "objective_delta_stop_strategy(min_delta, max_iter): max_iter can't be 0"
        );
        self.max_iter = max_iter;
        self
    }

    /// dlib `be_verbose()`.
    pub fn be_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Convenience inverse of [`StopStrategies::should_continue_search`].
    pub fn should_stop(&mut self, f_value: f64) -> bool {
        !self.should_continue_search(f_value, &Matrix::zeros(0, 0))
    }
}

impl StopStrategies for ObjectiveDeltaStopStrategy {
    fn should_continue_search(
        &mut self,
        funct_value: f64,
        _funct_derivative: &Matrix<f64>,
    ) -> bool {
        if self.verbose {
            println!("iteration: {}   objective: {}", self.cur_iter, funct_value);
        }

        self.cur_iter += 1;
        if self.been_used {
            // Check if we have hit the max allowable number of iterations.
            // (but only check if max_iter is enabled (i.e. not 0)).
            if self.max_iter != 0 && self.cur_iter > self.max_iter {
                return false;
            }

            // check if the function change was too small
            if (funct_value - self.prev_funct_value).abs() < self.min_delta {
                return false;
            }
        }

        self.been_used = true;
        self.prev_funct_value = funct_value;
        true
    }

    fn reset(&mut self) {
        self.been_used = false;
        self.cur_iter = 0;
        self.prev_funct_value = 0.0;
    }
}

/// Port of dlib `gradient_norm_stop_strategy(min_norm [, max_iter])`: stop
/// when the gradient norm drops below `min_norm` or after `max_iter`
/// iterations (when enabled).
#[derive(Default)]
pub struct GradientNormStopStrategy {
    verbose: bool,
    min_norm: f64,
    max_iter: usize,
    cur_iter: usize,
}

impl GradientNormStopStrategy {
    /// dlib `gradient_norm_stop_strategy(min_norm)` (`min_norm` defaults to
    /// 1e-7 in C++).
    pub fn new(min_norm: f64) -> Self {
        assert!(
            min_norm >= 0.0,
            "gradient_norm_stop_strategy(min_norm): min_norm can't be negative"
        );
        Self {
            verbose: false,
            min_norm,
            max_iter: 0,
            cur_iter: 0,
        }
    }

    /// dlib `gradient_norm_stop_strategy(min_norm, max_iter)`.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        assert!(
            max_iter > 0,
            "gradient_norm_stop_strategy(min_norm, max_iter): max_iter can't be 0"
        );
        self.max_iter = max_iter;
        self
    }

    /// dlib `be_verbose()`.
    pub fn be_verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl StopStrategies for GradientNormStopStrategy {
    fn should_continue_search(&mut self, funct_value: f64, funct_derivative: &Matrix<f64>) -> bool {
        if self.verbose {
            println!(
                "iteration: {}   objective: {}   gradient norm: {}",
                self.cur_iter,
                funct_value,
                length(funct_derivative)
            );
        }

        self.cur_iter += 1;

        // Check if we have hit the max allowable number of iterations.
        // (but only check if max_iter is enabled (i.e. not 0)).
        if self.max_iter != 0 && self.cur_iter > self.max_iter {
            return false;
        }

        // check if the gradient norm is too small
        if length(funct_derivative) < self.min_norm {
            return false;
        }

        true
    }

    fn reset(&mut self) {
        self.cur_iter = 0;
    }
}

/// dlib only defines the two concrete stop strategies above; this alias
/// mirrors the default-usage name in the shared crate contract.
pub type StopStrategy = ObjectiveDeltaStopStrategy;

#[cfg(test)]
mod tests {
    use super::*;

    fn g0() -> Matrix<f64> {
        Matrix::zeros(0, 0)
    }

    #[test]
    fn objective_delta_first_call_continues() {
        let mut s = ObjectiveDeltaStopStrategy::new(1e-7);
        // First call: been_used == false, so always continue.
        assert!(s.should_continue_search(100.0, &g0()));
        // Huge drop continues, tiny drop stops.
        assert!(s.should_continue_search(50.0, &g0()));
        assert!(!s.should_continue_search(50.0 + 1e-9, &g0()));
    }

    #[test]
    fn objective_delta_max_iter() {
        let mut s = ObjectiveDeltaStopStrategy::new(0.0).with_max_iterations(3);
        assert!(s.should_continue_search(3.0, &g0()));
        assert!(s.should_continue_search(2.0, &g0()));
        assert!(s.should_continue_search(1.0, &g0()));
        // cur_iter == 4 > 3 now.
        assert!(!s.should_continue_search(0.5, &g0()));
        s.reset();
        assert!(s.should_continue_search(3.0, &g0()));
    }

    #[test]
    fn objective_delta_should_stop_helper() {
        let mut s = ObjectiveDeltaStopStrategy::new(0.1);
        assert!(!s.should_stop(10.0));
        assert!(!s.should_stop(5.0));
        assert!(s.should_stop(5.05));
    }

    #[test]
    fn gradient_norm_behavior() {
        let mut s = GradientNormStopStrategy::new(1e-6);
        let big = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        let small = Matrix::from_row_vec(2, 1, &[1e-9, 1e-9]);
        assert!(s.should_continue_search(5.0, &big));
        assert!(!s.should_continue_search(5.0, &small));
        s.reset();
        assert!(s.should_continue_search(5.0, &big));

        let mut t = GradientNormStopStrategy::new(0.0).with_max_iterations(1);
        assert!(t.should_continue_search(1.0, &big));
        assert!(!t.should_continue_search(1.0, &big));
    }
}

//! Port of `dlib/optimization/optimization_search_strategies.h` (Davis E.
//! King, Boost Software License): cg / bfgs / lbfgs / newton search
//! strategies for use with [`crate::optimization::find_min`].

use dlib_rs_core::matrix::{dot, la, trans, Matrix};

/// Common interface mirroring dlib's `search_strategy_concept`
/// (`dlib/optimization/optimization_search_strategies_abstract.h`).
pub trait SearchStrategy {
    /// dlib `get_next_direction(x, function_value, gradient)`.
    fn get_next_direction(
        &mut self,
        x: &Matrix<f64>,
        function_value: f64,
        gradient: &Matrix<f64>,
    ) -> Matrix<f64>;

    /// dlib `get_wolfe_rho()` (Armijo condition parameter for line search).
    fn get_wolfe_rho(&self) -> f64 {
        0.01
    }

    /// dlib `get_wolfe_sigma()` (curvature condition parameter).
    fn get_wolfe_sigma(&self) -> f64 {
        0.9
    }

    /// dlib `get_max_line_search_iterations()`.
    fn get_max_line_search_iterations(&self) -> usize {
        100
    }

    /// Name of the strategy ("bfgs" | "lbfgs" | "cg" | "newton").
    fn name(&self) -> &'static str;
}

// ----------------------------------------------------------------------------------------

/// Port of dlib `cg_search_strategy`: Polak-Ribiere conjugate gradient
/// (Fletcher eq. 4.1.12, page 83), restarting with steepest descent when the
/// previous gradient norm degenerates.
#[derive(Default)]
pub struct CgSearchStrategy {
    been_used: bool,
    prev_derivative: Option<Matrix<f64>>,
    prev_direction: Option<Matrix<f64>>,
}

impl CgSearchStrategy {
    /// dlib `cg_search_strategy()`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SearchStrategy for CgSearchStrategy {
    fn get_next_direction(
        &mut self,
        _x: &Matrix<f64>,
        _function_value: f64,
        funct_derivative: &Matrix<f64>,
    ) -> Matrix<f64> {
        let prev_direction = if !self.been_used {
            self.been_used = true;
            -funct_derivative.clone()
        } else {
            let prev_derivative = self.prev_derivative.as_ref().unwrap();
            // Use the Polak-Ribiere (4.1.12) conjugate gradient described by
            // Fletcher on page 83
            let temp = dot(prev_derivative, prev_derivative);
            // If this value hits zero then just use the direction of steepest descent.
            if temp.abs() < f64::EPSILON {
                self.prev_derivative = Some(funct_derivative.clone());
                self.prev_direction = Some(-funct_derivative.clone());
                return -funct_derivative.clone();
            }

            let b = dot(
                &(funct_derivative.clone() - prev_derivative.clone()),
                funct_derivative,
            ) / temp;
            -funct_derivative.clone() + self.prev_direction.as_ref().unwrap().clone() * b
        };

        self.prev_derivative = Some(funct_derivative.clone());
        // store for next call (dlib stores prev_direction; mirror that state)
        self.prev_direction = Some(prev_direction.clone());
        prev_direction
    }

    fn get_wolfe_rho(&self) -> f64 {
        0.001
    }

    fn get_wolfe_sigma(&self) -> f64 {
        0.01
    }

    fn name(&self) -> &'static str {
        "cg"
    }
}

// ----------------------------------------------------------------------------------------

/// Port of dlib `bfgs_search_strategy`: full BFGS quasi-Newton update
/// (Fletcher eq. 3.2.12, page 55) with the Nocedal-Wright scaled initial H.
#[derive(Default)]
pub struct BfgsSearchStrategy {
    been_used: bool,
    been_used_twice: bool,
    prev_x: Option<Matrix<f64>>,
    prev_derivative: Option<Matrix<f64>>,
    h: Option<Matrix<f64>>,
}

impl BfgsSearchStrategy {
    /// dlib `bfgs_search_strategy()`.
    pub fn new() -> Self {
        Self::default()
    }
}

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

impl SearchStrategy for BfgsSearchStrategy {
    fn get_next_direction(
        &mut self,
        x: &Matrix<f64>,
        _function_value: f64,
        funct_derivative: &Matrix<f64>,
    ) -> Matrix<f64> {
        if !self.been_used {
            self.been_used = true;
            self.h = Some(Matrix::identity(x.size()));
        } else {
            // update H with the BFGS formula from (3.2.12) on page 55 of Fletcher
            let prev_x = self.prev_x.as_ref().unwrap();
            let prev_derivative = self.prev_derivative.as_ref().unwrap();
            let delta = x.clone() - prev_x.clone();
            let gamma = funct_derivative.clone() - prev_derivative.clone();

            let dg = dot(&delta, &gamma);

            // Try to set the initial value of the H matrix to something
            // reasonable if we are still in the early stages of figuring out
            // what it is (Nocedal & Wright, quasi-Newton chapter).
            if !self.been_used_twice {
                let gg = dot(&gamma, &gamma);
                if gg.abs() > f64::EPSILON {
                    let temp = put_in_range(0.01, 100.0, dg / gg);
                    let n = self.h.as_ref().unwrap().nr();
                    let mut h = Matrix::identity(n);
                    h *= temp;
                    self.h = Some(h);
                    self.been_used_twice = true;
                }
            }

            let h = self.h.clone().unwrap();
            let h_nr = h.nr();
            let hg = h.clone() * gamma.clone(); // H*gamma
            let gh = (trans(&gamma) * h.clone()).transpose(); // trans(trans(gamma)*H)
            let ghg = dot(&gamma, &hg);
            if ghg < f64::INFINITY && dg < f64::INFINITY && dg != 0.0 {
                let term1 = (delta.clone() * trans(&delta)) * (1.0 + ghg / dg) / dg;
                let term2 = (delta.clone() * trans(&gh) + hg.clone() * trans(&delta)) / dg;
                self.h = Some(h + term1 - term2);
            } else {
                self.h = Some(Matrix::identity(h_nr));
                self.been_used_twice = false;
            }
        }

        self.prev_x = Some(x.clone());
        let dir = -(self.h.as_ref().unwrap().clone() * funct_derivative.clone());
        self.prev_derivative = Some(funct_derivative.clone());
        dir
    }

    fn name(&self) -> &'static str {
        "bfgs"
    }
}

// ----------------------------------------------------------------------------------------

/// One L-BFGS curvature pair (dlib `lbfgs_search_strategy::data_helper`).
struct DataHelper {
    s: Matrix<f64>,
    y: Matrix<f64>,
    rho: f64,
}

/// Port of dlib `lbfgs_search_strategy(max_size)`: two-loop recursion from
/// algorithm 7.4 in Nocedal & Wright over a fixed-size history queue.
pub struct LbfgsSearchStrategy {
    max_size: usize,
    been_used: bool,
    prev_x: Option<Matrix<f64>>,
    prev_derivative: Option<Matrix<f64>>,
    data: Vec<DataHelper>,
    alpha: Vec<f64>,
}

impl Default for LbfgsSearchStrategy {
    fn default() -> Self {
        Self {
            max_size: 10,
            been_used: false,
            prev_x: None,
            prev_derivative: None,
            data: Vec::new(),
            alpha: Vec::new(),
        }
    }
}

impl LbfgsSearchStrategy {
    /// dlib `lbfgs_search_strategy(max_size)`.
    pub fn with_size(max_size: usize) -> Self {
        assert!(
            max_size > 0,
            "lbfgs_search_strategy(max_size): max_size can't be zero"
        );
        Self {
            max_size,
            ..Self::default()
        }
    }
}

impl SearchStrategy for LbfgsSearchStrategy {
    fn get_next_direction(
        &mut self,
        x: &Matrix<f64>,
        _function_value: f64,
        funct_derivative: &Matrix<f64>,
    ) -> Matrix<f64> {
        let mut prev_direction = -funct_derivative.clone();

        if !self.been_used {
            self.been_used = true;
        } else {
            // add an element into the stored data sequence
            let s = x.clone() - self.prev_x.as_ref().unwrap().clone();
            let y = funct_derivative.clone() - self.prev_derivative.as_ref().unwrap().clone();
            let temp = dot(&s, &y);
            // only accept this bit of data if temp isn't zero
            if temp.abs() > f64::EPSILON {
                let rho = 1.0 / temp;
                self.data.push(DataHelper { s, y, rho });
            } else {
                self.data.clear();
            }

            if !self.data.is_empty() {
                // This block of code is from algorithm 7.4 in the Nocedal book.

                self.alpha.resize(self.data.len(), 0.0);
                for i in (0..self.data.len()).rev() {
                    self.alpha[i] = self.data[i].rho * dot(&self.data[i].s, &prev_direction);
                    prev_direction -= self.data[i].y.clone() * self.alpha[i];
                }

                // Take a guess at what the first H matrix should be (Nocedal &
                // Wright, large scale unconstrained optimization chapter).
                let last = &self.data[self.data.len() - 1];
                let mut h0 = 1.0 / last.rho / dot(&last.y, &last.y);
                h0 = put_in_range(0.001, 1000.0, h0);
                prev_direction *= h0;

                for i in 0..self.data.len() {
                    let beta = self.data[i].rho * dot(&self.data[i].y, &prev_direction);
                    prev_direction += self.data[i].s.clone() * (self.alpha[i] - beta);
                }
            }
        }

        if self.data.len() > self.max_size {
            // remove the oldest element in the data sequence
            self.data.remove(0);
        }

        self.prev_x = Some(x.clone());
        self.prev_derivative = Some(funct_derivative.clone());
        prev_direction
    }

    fn name(&self) -> &'static str {
        "lbfgs"
    }
}

// ----------------------------------------------------------------------------------------

/// Port of dlib `newton_search_strategy(hessian)`: direction
/// `-inv(hessian(x)) * gradient` (dlib
/// `newton_search_strategy_obj::get_next_direction`).
pub struct NewtonSearchStrategy<H>
where
    H: Fn(&Matrix<f64>) -> Matrix<f64>,
{
    hessian: H,
}

impl<H> NewtonSearchStrategy<H>
where
    H: Fn(&Matrix<f64>) -> Matrix<f64>,
{
    /// dlib `newton_search_strategy(hessian)`.
    pub fn new(hessian: H) -> Self {
        Self { hessian }
    }
}

impl<H> SearchStrategy for NewtonSearchStrategy<H>
where
    H: Fn(&Matrix<f64>) -> Matrix<f64>,
{
    fn get_next_direction(
        &mut self,
        x: &Matrix<f64>,
        _function_value: f64,
        funct_derivative: &Matrix<f64>,
    ) -> Matrix<f64> {
        let h = (self.hessian)(x);
        let h_inv = la::inv(&h)
            .unwrap_or_else(|e| panic!("newton_search_strategy: hessian inversion failed: {e}"));
        -(h_inv * funct_derivative.clone())
    }

    fn name(&self) -> &'static str {
        "newton"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(v: &[f64]) -> Matrix<f64> {
        Matrix::from_row_vec(v.len(), 1, v)
    }

    #[test]
    fn cg_first_direction_is_steepest_descent() {
        let mut cg = CgSearchStrategy::new();
        let g = col(&[1.0, 2.0]);
        let d = cg.get_next_direction(&col(&[0.0, 0.0]), 0.0, &g);
        assert_eq!((d[(0, 0)], d[(1, 0)]), (-1.0, -2.0));
        assert_eq!(cg.get_wolfe_rho(), 0.001);
        assert_eq!(cg.get_wolfe_sigma(), 0.01);
        assert_eq!(cg.name(), "cg");
    }

    #[test]
    fn lbfgs_defaults_and_max_size() {
        let mut lb = LbfgsSearchStrategy::default();
        assert_eq!(lb.name(), "lbfgs");
        let x0 = col(&[0.0, 0.0]);
        let g0 = col(&[1.0, 1.0]);
        let _ = lb.get_next_direction(&x0, 0.0, &g0);
        assert!(lb.data.is_empty());
        // second call adds a pair
        let x1 = col(&[0.5, 0.5]);
        let g1 = col(&[0.5, 0.5]);
        let _ = lb.get_next_direction(&x1, 0.0, &g1);
        assert_eq!(lb.data.len(), 1);
    }

    #[test]
    fn newton_direction_solves_newton_system() {
        // f(x) = x^T A x / 2 with A = diag(2, 4): newton dir = -A^{-1} g.
        let mut n = NewtonSearchStrategy::new(|_x: &Matrix<f64>| {
            Matrix::from_row_vec(2, 2, &[2.0, 0.0, 0.0, 4.0])
        });
        let g = col(&[2.0, 8.0]);
        let d = n.get_next_direction(&col(&[1.0, 1.0]), 0.0, &g);
        assert!((d[(0, 0)] + 1.0).abs() < 1e-12);
        assert!((d[(1, 0)] + 2.0).abs() < 1e-12);
    }
}

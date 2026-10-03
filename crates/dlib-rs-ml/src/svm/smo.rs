//! SMO-based SVM trainers ported from `dlib/svm/svm_c_trainer.h`
//! (`svm_c_trainer`), `dlib/svm/svm_nu_trainer.h` (`svm_nu_trainer`),
//! `dlib/svm/svr_trainer.h` (`svr_trainer`), the QP solvers they drive
//! (`dlib/optimization/optimization_solve_qp2_using_smo.h` and
//! `..._solve_qp3_using_smo.h`), the `symmetric_matrix_cache<float>` kernel
//! caching strategy (`dlib/matrix/symmetric_matrix_cache.h`), `maximum_nu` and
//! `randomize_samples` (`dlib/svm/svm.h`).
//!
//! Solver paths (mirroring dlib exactly):
//! * `svm_c_trainer` (C-SVC)  -> `solve_qp3_using_smo`
//! * `svm_nu_trainer` (nu-SVC) -> `solve_qp2_using_smo`
//! * `svr_trainer` (epsilon-SVR) -> `solve_qp3_using_smo` on the doubled
//!   `+K -K / -K +K` problem of `svr_trainer::op_quad`.
//!
//! The Q matrix is cached in `f32` exactly like dlib's
//! `symmetric_matrix_cache<float>`: kernel entries are computed in `f64`,
//! truncated to `f32` when stored, and the SMO working-set arithmetic mixes
//! `f32`/`f64` exactly where the C++ expressions do.

use crate::svm::function::DecisionFunction;
use crate::svm::kernels::Kernel;
use dlib_rs_core::matrix::Matrix;
use dlib_rs_core::rand::Rand;

/// dlib's `tau` constant used by both SMO solvers.
const TAU: f64 = 1e-12;

/// Errors thrown by the dlib SMO solvers on infeasible parameters.
#[derive(Debug, thiserror::Error)]
pub enum SvmError {
    /// dlib `invalid_nu_error` (`solve_qp2_using_smo.h`).
    #[error("Invalid nu of {nu}. It is required that: 0 < nu < {max_nu}")]
    InvalidNu {
        /// The offending `nu`.
        nu: f64,
        /// The largest feasible `nu` for the given labels.
        max_nu: f64,
    },
    /// dlib `invalid_qp3_error` (`solve_qp3_using_smo.h`).
    #[error("Invalid QP3 constraint parameters of B: {b}, Cp: {cp}, Cn: {cn}")]
    InvalidQp3 {
        /// The offending `B`.
        b: f64,
        /// The offending `Cp`.
        cp: f64,
        /// The offending `Cn`.
        cn: f64,
    },
}

// ----------------------------------------------------------------------------------------
// The lazily evaluated Q matrices and dlib's symmetric_matrix_cache<float>

/// The lazily evaluated square matrix handed to the SMO solvers.
trait QMatrix {
    /// Order of the square matrix.
    fn dim(&self) -> usize;
    /// Element `(r, c)` in `f64` (truncated to `f32` by the cache).
    fn eval(&self, r: usize, c: usize) -> f64;
}

/// `diagm(y)*kernel_matrix(kernel, x)*diagm(y)` — the Q matrix of the
/// C-SVC and nu-SVC trainers (dlib/svm/svm_c_trainer.h:226,
/// dlib/svm/svm_nu_trainer.h:185).
struct LabeledKernelMatrix<'a, K: Kernel<SampleType = Matrix<f64>>> {
    x: &'a [Matrix<f64>],
    y: &'a [f64],
    kernel: &'a K,
}

impl<K: Kernel<SampleType = Matrix<f64>>> QMatrix for LabeledKernelMatrix<'_, K> {
    fn dim(&self) -> usize {
        self.y.len()
    }
    fn eval(&self, r: usize, c: usize) -> f64 {
        (self.y[r] * self.kernel.operator_(&self.x[r], &self.x[c])) * self.y[c]
    }
}

/// `make_quad(kernel_matrix(kernel, x))` — the doubled `+K -K / -K +K` Q
/// matrix of the epsilon-SVR trainer (dlib/svm/svr_trainer.h `op_quad`).
struct QuadKernelMatrix<'a, K: Kernel<SampleType = Matrix<f64>>> {
    x: &'a [Matrix<f64>],
    kernel: &'a K,
}

impl<K: Kernel<SampleType = Matrix<f64>>> QMatrix for QuadKernelMatrix<'_, K> {
    fn dim(&self) -> usize {
        2 * self.x.len()
    }
    fn eval(&self, r: usize, c: usize) -> f64 {
        let n = self.x.len();
        let (r0, sign_r) = if r < n { (r, 1.0) } else { (r - n, -1.0) };
        let (c0, sign_c) = if c < n { (c, 1.0) } else { (c - n, -1.0) };
        sign_r * sign_c * self.kernel.operator_(&self.x[r0], &self.x[c0])
    }
}

/// Port of `dlib::op_symm_cache<EXP, float>`
/// (dlib/matrix/symmetric_matrix_cache.h): caches whole columns of a
/// symmetric matrix as `f32`, evicting round-robin. dlib additionally
/// reference-counts aliased columns to pick a victim and grow the cache; this
/// port copies columns out instead of aliasing them, so the round-robin
/// victim is always free and the cache never grows — numerically identical
/// (a recomputed column holds the same `f32` values).
struct SymmetricMatrixCache<M: QMatrix> {
    m: M,
    diag_cache: Vec<f32>,
    lookup: Vec<i64>,
    rlookup: Vec<i64>,
    cache: Vec<Vec<f32>>,
    next: usize,
}

impl<M: QMatrix> SymmetricMatrixCache<M> {
    /// `op_symm_cache(m, max_size_megabytes)`.
    fn new(m: M, max_size_megabytes: i64) -> Self {
        let n = m.dim();
        assert!(n > 0, "symmetric_matrix_cache of an empty matrix");
        let diag_cache: Vec<f32> = (0..n).map(|i| m.eval(i, i) as f32).collect();
        // init(): how many columns fit in the budget; never fewer than 2.
        let mut max_size = (max_size_megabytes * 1024 * 1024) / (n as i64 * 4);
        if max_size <= 1 {
            max_size = 2;
        }
        let size = max_size.min(n as i64) as usize;
        SymmetricMatrixCache {
            m,
            diag_cache,
            lookup: vec![-1; n],
            rlookup: vec![-1; size],
            cache: vec![Vec::new(); size],
            next: 0,
        }
    }

    /// `op_symm_cache::apply(r, c)`.
    fn get(&mut self, r: usize, c: usize) -> f32 {
        if self.lookup[c] != -1 {
            self.cache[self.lookup[c] as usize][r]
        } else if r == c {
            self.diag_cache[r]
        } else if self.lookup[r] != -1 {
            // the matrix is symmetric so this is legit
            self.cache[self.lookup[r] as usize][c]
        } else {
            self.add_col_to_cache(c);
            self.cache[self.lookup[c] as usize][r]
        }
    }

    /// Order of the wrapped matrix (`m.nr()`).
    fn dim(&self) -> usize {
        self.m.dim()
    }

    /// `op_symm_cache::col(i)` — makes sure column `i` is cached and returns
    /// it (dlib bumps the round-robin pointer when it would evict the column
    /// that was just fetched).
    fn col(&mut self, i: usize) -> &Vec<f32> {
        if self.lookup[i] == -1 {
            self.add_col_to_cache(i);
        }
        if self.lookup[i] as usize == self.next {
            // if this column was the next to be replaced then make sure that
            // doesn't happen
            self.next = (self.next + 1) % self.cache.len();
        }
        &self.cache[self.lookup[i] as usize]
    }

    /// `op_symm_cache::diag()` — the precomputed `f32` diagonal.
    fn diag(&self) -> &[f32] {
        &self.diag_cache
    }

    /// `op_symm_cache::add_col_to_cache(c)`.
    fn add_col_to_cache(&mut self, c: usize) {
        let next = self.next;
        // make_sure_next_is_unreferenced(): all reference counts are zero in
        // this port, so the round-robin victim is always free.
        if self.rlookup[next] != -1 {
            self.lookup[self.rlookup[next] as usize] = -1;
        }
        self.lookup[c] = next as i64;
        self.rlookup[next] = c as i64;
        let n = self.m.dim();
        let col: Vec<f32> = (0..n).map(|r| self.m.eval(r, c) as f32).collect();
        self.cache[next] = col;
        self.next = (next + 1) % self.cache.len();
    }
}

// ----------------------------------------------------------------------------------------
// solve_qp3_using_smo (dlib/optimization/optimization_solve_qp3_using_smo.h)

/// `Cp`/`Cn` box limits plus `tau`, the parameters shared by the qp3
/// working-set routines.
struct QpLimits {
    cp: f64,
    cn: f64,
    tau: f64,
}

/// The `operator()` parameters of `solve_qp3_using_smo`.
struct Qp3Params {
    b: f64,
    cp: f64,
    cn: f64,
    eps: f64,
}

/// Port of `dlib::solve_qp3_using_smo`.
#[derive(Default)]
struct SolveQp3UsingSmo {
    /// Gradient of `f(alpha)` at the solution (`get_gradient()`).
    df: Vec<f64>,
}

impl SolveQp3UsingSmo {
    /// `solve_qp3_using_smo::operator()(Q, p, y, B, Cp, Cn, alpha, eps)`;
    /// returns `alpha` (dlib also returns an iteration count the trainers
    /// ignore).
    fn solve<M: QMatrix>(
        &mut self,
        q: &mut SymmetricMatrixCache<M>,
        p: &[f64],
        y: &[f64],
        params: Qp3Params,
    ) -> Result<Vec<f64>, SvmError> {
        let mut alpha = set_initial_alpha_qp3(y, params.b, params.cp, params.cn)?;

        // initialize df.  Compute df = Q*alpha + p
        self.df = p.to_vec();
        for (r, &alpha_r) in alpha.iter().enumerate() {
            if alpha_r != 0.0 {
                let col = q.col(r).clone();
                for (df_k, &q_k) in self.df.iter_mut().zip(col.iter()) {
                    *df_k += alpha_r * q_k as f64;
                }
            }
        }

        let limits = QpLimits {
            cp: params.cp,
            cn: params.cn,
            tau: TAU,
        };

        // now perform the actual optimization of alpha
        while let Some((i, j)) = self.find_working_group(q, y, &alpha, &limits, params.eps) {
            let old_alpha_i = alpha[i];
            let old_alpha_j = alpha[j];

            self.optimize_working_pair(q, y, &mut alpha, &limits, i, j);

            // update the df vector now that we have modified alpha(i)/alpha(j)
            let delta_alpha_i = alpha[i] - old_alpha_i;
            let delta_alpha_j = alpha[j] - old_alpha_j;

            let q_i = q.col(i).clone();
            let q_j = q.col(j).clone();
            for (df_k, (&q_ik, &q_jk)) in self.df.iter_mut().zip(q_i.iter().zip(q_j.iter())) {
                *df_k += q_ik as f64 * delta_alpha_i + q_jk as f64 * delta_alpha_j;
            }
        }

        Ok(alpha)
    }

    /// `solve_qp3_using_smo::get_gradient()`.
    fn get_gradient(&self) -> &[f64] {
        &self.df
    }

    /// `solve_qp3_using_smo::find_working_group`.
    fn find_working_group<M: QMatrix>(
        &self,
        q: &mut SymmetricMatrixCache<M>,
        y: &[f64],
        alpha: &[f64],
        limits: &QpLimits,
        eps: f64,
    ) -> Option<(usize, usize)> {
        let n = alpha.len();
        let mut ip = 0usize;
        let mut jp = 0usize;

        let mut ip_val = f64::NEG_INFINITY;
        let mut jp_val = f64::INFINITY;

        // loop over the alphas and find the maximum ip and in indices.
        for i in 0..n {
            if y[i] == 1.0 {
                if alpha[i] < limits.cp && -self.df[i] > ip_val {
                    ip_val = -self.df[i];
                    ip = i;
                }
            } else if alpha[i] > 0.0 && self.df[i] > ip_val {
                ip_val = self.df[i];
                ip = i;
            }
        }

        let mut mp = f64::NEG_INFINITY;

        let q_ip = q.col(ip).clone();
        let q_diag = q.diag();

        // now we need to find the minimum jp indices
        for j in 0..n {
            if y[j] == 1.0 {
                if alpha[j] > 0.0 {
                    let b = ip_val + self.df[j];
                    if self.df[j] > mp {
                        mp = self.df[j];
                    }

                    if b > 0.0 {
                        // scalar_type a = Q_ip(ip) + Q_diag(j) - 2*y(ip)*Q_ip(j);
                        // (the Q terms are float, so their sum is computed in f32)
                        let a = (q_ip[ip] + q_diag[j]) as f64 - (2.0 * y[ip]) * q_ip[j] as f64;
                        let a = if a <= 0.0 { limits.tau } else { a };
                        let temp = -b * b / a;
                        if temp < jp_val {
                            jp_val = temp;
                            jp = j;
                        }
                    }
                }
            } else if alpha[j] < limits.cn {
                let b = ip_val - self.df[j];
                if -self.df[j] > mp {
                    mp = -self.df[j];
                }

                if b > 0.0 {
                    // a = Q_ip(ip) + Q_diag(j) + 2*y(ip)*Q_ip(j);
                    let a = (q_ip[ip] + q_diag[j]) as f64 + (2.0 * y[ip]) * q_ip[j] as f64;
                    let a = if a <= 0.0 { limits.tau } else { a };
                    let temp = -b * b / a;
                    if temp < jp_val {
                        jp_val = temp;
                        jp = j;
                    }
                }
            }
        }

        // if we are at the optimal point then return false so the caller knows
        // to stop optimizing
        if mp + ip_val < eps {
            return None;
        }

        Some((ip, jp))
    }

    /// `solve_qp3_using_smo::optimize_working_pair`.
    fn optimize_working_pair<M: QMatrix>(
        &self,
        q: &mut SymmetricMatrixCache<M>,
        y: &[f64],
        alpha: &mut [f64],
        limits: &QpLimits,
        i: usize,
        j: usize,
    ) {
        let ci = if y[i] > 0.0 { limits.cp } else { limits.cn };
        let cj = if y[j] > 0.0 { limits.cp } else { limits.cn };

        if y[i] != y[j] {
            // quad_coef = Q(i,i)+Q(j,j)+2*Q(j,i);  (f32 arithmetic)
            let quad_coef = (q.get(i, i) + q.get(j, j) + 2.0 * q.get(j, i)) as f64;
            let quad_coef = if quad_coef <= 0.0 {
                limits.tau
            } else {
                quad_coef
            };
            let delta = (-self.df[i] - self.df[j]) / quad_coef;
            let diff = alpha[i] - alpha[j];
            alpha[i] += delta;
            alpha[j] += delta;

            if diff > 0.0 {
                if alpha[j] < 0.0 {
                    alpha[j] = 0.0;
                    alpha[i] = diff;
                }
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = -diff;
            }

            if diff > ci - cj {
                if alpha[i] > ci {
                    alpha[i] = ci;
                    alpha[j] = ci - diff;
                }
            } else if alpha[j] > cj {
                alpha[j] = cj;
                alpha[i] = cj + diff;
            }
        } else {
            // quad_coef = Q(i,i)+Q(j,j)-2*Q(j,i);  (f32 arithmetic)
            let quad_coef = (q.get(i, i) + q.get(j, j) - 2.0 * q.get(j, i)) as f64;
            let quad_coef = if quad_coef <= 0.0 {
                limits.tau
            } else {
                quad_coef
            };
            let delta = (self.df[i] - self.df[j]) / quad_coef;
            let sum = alpha[i] + alpha[j];
            alpha[i] -= delta;
            alpha[j] += delta;

            if sum > ci {
                if alpha[i] > ci {
                    alpha[i] = ci;
                    alpha[j] = sum - ci;
                }
            } else if alpha[j] < 0.0 {
                alpha[j] = 0.0;
                alpha[i] = sum;
            }

            if sum > cj {
                if alpha[j] > cj {
                    alpha[j] = cj;
                    alpha[i] = sum - cj;
                }
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = sum;
            }
        }
    }
}

/// `solve_qp3_using_smo::set_initial_alpha`.
fn set_initial_alpha_qp3(y: &[f64], b: f64, cp: f64, cn: f64) -> Result<Vec<f64>, SvmError> {
    let mut alpha = vec![0.0; y.len()];

    // It's easy in the B == 0 case
    if b == 0.0 {
        return Ok(alpha);
    }

    let c = if b > 0.0 { cp } else { cn };

    let temp = b.abs() / c;
    let num = temp.floor() as i64;
    let num_total = temp.ceil() as i64;

    let b_sign: f64 = if b > 0.0 { 1.0 } else { -1.0 };

    let mut count = 0i64;
    for i in 0..y.len() {
        if y[i] == b_sign {
            if count < num {
                count += 1;
                alpha[i] = c;
            } else {
                if count < num_total {
                    count += 1;
                    alpha[i] = c * (temp - temp.floor());
                }
                break;
            }
        }
    }

    if count != num_total {
        return Err(SvmError::InvalidQp3 { b, cp, cn });
    }
    Ok(alpha)
}

// ----------------------------------------------------------------------------------------
// solve_qp2_using_smo (dlib/optimization/optimization_solve_qp2_using_smo.h)

/// Port of `dlib::solve_qp2_using_smo`.
#[derive(Default)]
struct SolveQp2UsingSmo {
    /// Gradient of `f(alpha)` at the solution (`get_gradient()`).
    df: Vec<f64>,
}

impl SolveQp2UsingSmo {
    /// `solve_qp2_using_smo::operator()(Q, y, nu, alpha, eps)`.
    fn solve<M: QMatrix>(
        &mut self,
        q: &mut SymmetricMatrixCache<M>,
        y: &[f64],
        nu: f64,
        eps: f64,
    ) -> Result<Vec<f64>, SvmError> {
        let n = q.dim();
        self.df = vec![0.0; n];
        let mut alpha = set_initial_alpha_qp2(y, nu)?;

        // initialize df.  Compute df = Q*alpha
        for (r, &alpha_r) in alpha.iter().enumerate() {
            if alpha_r != 0.0 {
                let col = q.col(r).clone();
                for (df_k, &q_k) in self.df.iter_mut().zip(col.iter()) {
                    *df_k += alpha_r * q_k as f64;
                }
            }
        }

        // now perform the actual optimization of alpha
        while let Some((i, j)) = self.find_working_group(q, y, &alpha, TAU, eps) {
            let old_alpha_i = alpha[i];
            let old_alpha_j = alpha[j];

            self.optimize_working_pair(q, &mut alpha, TAU, i, j);

            // update the df vector now that we have modified alpha(i)/alpha(j)
            let delta_alpha_i = alpha[i] - old_alpha_i;
            let delta_alpha_j = alpha[j] - old_alpha_j;

            let q_i = q.col(i).clone();
            let q_j = q.col(j).clone();
            for (df_k, (&q_ik, &q_jk)) in self.df.iter_mut().zip(q_i.iter().zip(q_j.iter())) {
                *df_k += q_ik as f64 * delta_alpha_i + q_jk as f64 * delta_alpha_j;
            }
        }

        Ok(alpha)
    }

    /// `solve_qp2_using_smo::get_gradient()`.
    fn get_gradient(&self) -> &[f64] {
        &self.df
    }

    /// `solve_qp2_using_smo::find_working_group`.
    fn find_working_group<M: QMatrix>(
        &self,
        q: &mut SymmetricMatrixCache<M>,
        y: &[f64],
        alpha: &[f64],
        tau: f64,
        eps: f64,
    ) -> Option<(usize, usize)> {
        let n = alpha.len();
        let mut ip = 0usize;
        let mut jp = 0usize;
        let mut in_ = 0usize;
        let mut jn = 0usize;

        let mut ip_val = f64::NEG_INFINITY;
        let mut jp_val = f64::INFINITY;
        let mut in_val = f64::NEG_INFINITY;
        let mut jn_val = f64::INFINITY;

        // loop over the alphas and find the maximum ip and in indices.
        for i in 0..n {
            if y[i] == 1.0 {
                if alpha[i] < 1.0 && -self.df[i] > ip_val {
                    ip_val = -self.df[i];
                    ip = i;
                }
            } else if alpha[i] > 0.0 && self.df[i] > in_val {
                in_val = self.df[i];
                in_ = i;
            }
        }

        let mut mp = f64::INFINITY;
        let mut mn = f64::INFINITY;

        let q_ip = q.col(ip).clone();
        let q_in = q.col(in_).clone();
        let q_diag = q.diag();

        // now we need to find the minimum jp and jn indices
        for j in 0..n {
            if y[j] == 1.0 {
                if alpha[j] > 0.0 {
                    let b = ip_val + self.df[j];
                    if -self.df[j] < mp {
                        mp = -self.df[j];
                    }

                    if b > 0.0 {
                        // a = Q_ip(ip) + Q_diag(j) - 2*Q_ip(j);  (f32 arithmetic)
                        let a = (q_ip[ip] + q_diag[j] - 2.0 * q_ip[j]) as f64;
                        let a = if a <= 0.0 { tau } else { a };
                        let temp = -b * b / a;
                        if temp < jp_val {
                            jp_val = temp;
                            jp = j;
                        }
                    }
                }
            } else if alpha[j] < 1.0 {
                let b = in_val - self.df[j];
                if self.df[j] < mn {
                    mn = self.df[j];
                }

                if b > 0.0 {
                    // a = Q_in(in) + Q_diag(j) - 2*Q_in(j);  (f32 arithmetic)
                    let a = (q_in[in_] + q_diag[j] - 2.0 * q_in[j]) as f64;
                    let a = if a <= 0.0 { tau } else { a };
                    let temp = -b * b / a;
                    if temp < jn_val {
                        jn_val = temp;
                        jn = j;
                    }
                }
            }
        }

        // if we are at the optimal point then return false so the caller knows
        // to stop optimizing
        let m1 = ip_val - mp;
        let m2 = in_val - mn;
        let mx = if m1 < m2 { m2 } else { m1 };
        if mx < eps {
            return None;
        }

        if jp_val < jn_val {
            Some((ip, jp))
        } else {
            Some((in_, jn))
        }
    }

    /// `solve_qp2_using_smo::optimize_working_pair`.
    fn optimize_working_pair<M: QMatrix>(
        &self,
        q: &mut SymmetricMatrixCache<M>,
        alpha: &mut [f64],
        tau: f64,
        i: usize,
        j: usize,
    ) {
        // quad_coef = Q(i,i)+Q(j,j)-2*Q(j,i);  (f32 arithmetic)
        let quad_coef = (q.get(i, i) + q.get(j, j) - 2.0 * q.get(j, i)) as f64;
        let quad_coef = if quad_coef <= 0.0 { tau } else { quad_coef };
        let delta = (self.df[i] - self.df[j]) / quad_coef;
        let sum = alpha[i] + alpha[j];
        alpha[i] -= delta;
        alpha[j] += delta;

        if sum > 1.0 {
            if alpha[i] > 1.0 {
                alpha[i] = 1.0;
                alpha[j] = sum - 1.0;
            } else if alpha[j] > 1.0 {
                alpha[j] = 1.0;
                alpha[i] = sum - 1.0;
            }
        } else {
            if alpha[j] < 0.0 {
                alpha[j] = 0.0;
                alpha[i] = sum;
            } else if alpha[i] < 0.0 {
                alpha[i] = 0.0;
                alpha[j] = sum;
            }
        }
    }
}

/// `solve_qp2_using_smo::set_initial_alpha`.
fn set_initial_alpha_qp2(y: &[f64], nu: f64) -> Result<Vec<f64>, SvmError> {
    let mut alpha = vec![0.0; y.len()];
    let l = y.len() as f64;
    let temp = nu * l / 2.0;
    let num = temp.floor() as i64;
    let num_total = temp.ceil() as i64;

    let (count, has_slack) = fill_initial_alpha_side(y, &mut alpha, 1.0, num, num_total, temp);
    if count != num_total || !has_slack {
        return Err(SvmError::InvalidNu {
            nu,
            max_nu: 2.0 * count as f64 / y.len() as f64,
        });
    }

    let (count, has_slack) = fill_initial_alpha_side(y, &mut alpha, -1.0, num, num_total, temp);
    if count != num_total || !has_slack {
        return Err(SvmError::InvalidNu {
            nu,
            max_nu: 2.0 * count as f64 / y.len() as f64,
        });
    }
    Ok(alpha)
}

/// One pass of `set_initial_alpha` over the samples with `label`.
fn fill_initial_alpha_side(
    y: &[f64],
    alpha: &mut [f64],
    label: f64,
    num: i64,
    num_total: i64,
    temp: f64,
) -> (i64, bool) {
    let mut has_slack = false;
    let mut count = 0i64;
    for i in 0..y.len() {
        if y[i] == label {
            if count < num {
                count += 1;
                alpha[i] = 1.0;
            } else {
                has_slack = true;
                if num_total > num {
                    count += 1;
                    alpha[i] = temp - temp.floor();
                }
                break;
            }
        }
    }
    (count, has_slack)
}

// ----------------------------------------------------------------------------------------
// maximum_nu (dlib/svm/svm.h)

/// Port of `dlib::maximum_nu(y)`: `2*min(pos_count, neg_count)/y.size()`.
pub fn maximum_nu(y: &[f64]) -> f64 {
    let mut pos_count = 0i64;
    let mut neg_count = 0i64;
    for &v in y {
        if v == 1.0 {
            pos_count += 1;
        } else if v == -1.0 {
            neg_count += 1;
        }
    }
    2.0 * pos_count.min(neg_count) as f64 / y.len() as f64
}

// ----------------------------------------------------------------------------------------
// shared trainer helpers

/// `DLIB_ASSERT(is_binary_classification_problem(x, y))`.
fn assert_binary_classification_problem(x: &[Matrix<f64>], y: &[f64]) {
    assert!(
        !x.is_empty() && x.len() == y.len(),
        "train(x, y): x and y must be nonempty and the same length"
    );
    assert!(
        y.iter().all(|v| *v == 1.0 || *v == -1.0),
        "train(x, y): labels must all be +1 or -1"
    );
    assert!(
        y.contains(&1.0) && y.contains(&-1.0),
        "train(x, y): both classes must be present"
    );
}

/// Keeps the nonzero alphas with their samples, mirroring the tail of the
/// trainers' `do_train` (sv_count / sv_alpha / support_vectors).
fn extract_support_vectors(alpha: &[f64], x: &[Matrix<f64>]) -> (Matrix<f64>, Vec<Matrix<f64>>) {
    let sv_count = alpha.iter().filter(|a| **a != 0.0).count();
    let mut sv_alpha = Vec::with_capacity(sv_count);
    let mut support_vectors = Vec::with_capacity(sv_count);
    for (a, xi) in alpha.iter().zip(x) {
        if *a != 0.0 {
            sv_alpha.push(*a);
            support_vectors.push(xi.clone());
        }
    }
    (
        Matrix::from_vec(sv_count, 1, sv_alpha).expect("column vector"),
        support_vectors,
    )
}

/// `svm_c_trainer::calculate_b` (dlib/svm/svm_c_trainer.h).
fn calculate_b_c(y: &[f64], alpha: &[f64], df: &[f64], cpos: f64, cneg: f64) -> f64 {
    let mut num_free = 0i64;
    let mut sum_free = 0.0;

    let mut upper_bound = f64::NEG_INFINITY;
    let mut lower_bound = f64::INFINITY;

    for i in 0..alpha.len() {
        if y[i] == 1.0 {
            if alpha[i] == cpos {
                if df[i] > upper_bound {
                    upper_bound = df[i];
                }
            } else if alpha[i] == 0.0 {
                if df[i] < lower_bound {
                    lower_bound = df[i];
                }
            } else {
                num_free += 1;
                sum_free += df[i];
            }
        } else if alpha[i] == cneg {
            if -df[i] < lower_bound {
                lower_bound = -df[i];
            }
        } else if alpha[i] == 0.0 {
            if -df[i] > upper_bound {
                upper_bound = -df[i];
            }
        } else {
            num_free += 1;
            sum_free -= df[i];
        }
    }

    if num_free > 0 {
        sum_free / num_free as f64
    } else {
        (upper_bound + lower_bound) / 2.0
    }
}

/// `svm_nu_trainer::calculate_rho_and_b` (dlib/svm/svm_nu_trainer.h); returns
/// `(rho, b)`.
fn calculate_rho_and_b(y: &[f64], alpha: &[f64], df: &[f64]) -> (f64, f64) {
    let mut num_p_free = 0i64;
    let mut num_n_free = 0i64;
    let mut sum_p_free = 0.0;
    let mut sum_n_free = 0.0;

    let mut upper_bound_p = f64::NEG_INFINITY;
    let mut upper_bound_n = f64::NEG_INFINITY;
    let mut lower_bound_p = f64::INFINITY;
    let mut lower_bound_n = f64::INFINITY;

    for i in 0..alpha.len() {
        if y[i] == 1.0 {
            if alpha[i] == 1.0 {
                if df[i] > upper_bound_p {
                    upper_bound_p = df[i];
                }
            } else if alpha[i] == 0.0 {
                if df[i] < lower_bound_p {
                    lower_bound_p = df[i];
                }
            } else {
                num_p_free += 1;
                sum_p_free += df[i];
            }
        } else if alpha[i] == 1.0 {
            if df[i] > upper_bound_n {
                upper_bound_n = df[i];
            }
        } else if alpha[i] == 0.0 {
            if df[i] < lower_bound_n {
                lower_bound_n = df[i];
            }
        } else {
            num_n_free += 1;
            sum_n_free += df[i];
        }
    }

    let r1 = if num_p_free > 0 {
        sum_p_free / num_p_free as f64
    } else {
        (upper_bound_p + lower_bound_p) / 2.0
    };

    let r2 = if num_n_free > 0 {
        sum_n_free / num_n_free as f64
    } else {
        (upper_bound_n + lower_bound_n) / 2.0
    };

    let rho = (r1 + r2) / 2.0;
    let b = (r1 - r2) / 2.0 / rho;
    (rho, b)
}

/// `svr_trainer::calculate_b` (dlib/svm/svr_trainer.h), including dlib's
/// swapped-argument call to `find_min_and_max` (the "upper" bound is seeded
/// with `min(df)` and the "lower" bound with `max(df)`).
fn calculate_b_svr(alpha: &[f64], df: &[f64], c: f64) -> f64 {
    let mut num_free = 0i64;
    let mut sum_free = 0.0;

    // find_min_and_max(df, upper_bound, lower_bound) — min lands in
    // upper_bound, max lands in lower_bound, exactly as dlib calls it.
    let mut upper_bound = df[0];
    let mut lower_bound = df[0];
    for &v in df {
        if v > lower_bound {
            lower_bound = v;
        }
        if v < upper_bound {
            upper_bound = v;
        }
    }

    let half = alpha.len() / 2;
    for i in 0..alpha.len() {
        if i < half {
            if alpha[i] == c {
                if df[i] > upper_bound {
                    upper_bound = df[i];
                }
            } else if alpha[i] == 0.0 {
                if df[i] < lower_bound {
                    lower_bound = df[i];
                }
            } else {
                num_free += 1;
                sum_free += df[i];
            }
        } else if alpha[i] == c {
            if -df[i] < lower_bound {
                lower_bound = -df[i];
            }
        } else if alpha[i] == 0.0 {
            if -df[i] > upper_bound {
                upper_bound = -df[i];
            }
        } else {
            num_free += 1;
            sum_free -= df[i];
        }
    }

    if num_free > 0 {
        sum_free / num_free as f64
    } else {
        (upper_bound + lower_bound) / 2.0
    }
}

// ----------------------------------------------------------------------------------------
// svm_c_trainer (dlib/svm/svm_c_trainer.h)

/// Port of `dlib::svm_c_trainer<K>` (C-SVC). Defaults: `C = 1`,
/// `cache_size = 200` (MiB), `eps = 0.001`.
#[derive(Clone, Debug)]
pub struct SvmCTrainer<K: Kernel<SampleType = Matrix<f64>>> {
    kernel_function: K,
    cpos: f64,
    cneg: f64,
    cache_size: i64,
    eps: f64,
}

impl<K: Kernel<SampleType = Matrix<f64>>> SvmCTrainer<K> {
    /// `svm_c_trainer(kernel, C)` (requires `C > 0`).
    pub fn new(kernel: K, c: f64) -> Self {
        assert!(0.0 < c, "svm_c_trainer: C must be greater than 0");
        Self {
            kernel_function: kernel,
            cpos: c,
            cneg: c,
            cache_size: 200,
            eps: 0.001,
        }
    }

    /// `set_cache_size(cache_size)` (MiB).
    pub fn set_cache_size(&mut self, cache_size: i64) {
        assert!(cache_size > 0, "set_cache_size: must be positive");
        self.cache_size = cache_size;
    }

    /// `get_cache_size()`.
    pub fn get_cache_size(&self) -> i64 {
        self.cache_size
    }

    /// `set_epsilon(eps)`.
    pub fn set_epsilon(&mut self, eps: f64) {
        assert!(eps > 0.0, "set_epsilon: must be positive");
        self.eps = eps;
    }

    /// `get_epsilon()`.
    pub fn get_epsilon(&self) -> f64 {
        self.eps
    }

    /// `set_kernel(k)`.
    pub fn set_kernel(&mut self, k: K) {
        self.kernel_function = k;
    }

    /// `get_kernel()`.
    pub fn get_kernel(&self) -> &K {
        &self.kernel_function
    }

    /// `set_c(C)`.
    pub fn set_c(&mut self, c: f64) {
        assert!(c > 0.0, "set_c: C must be greater than 0");
        self.cpos = c;
        self.cneg = c;
    }

    /// `get_c_class1()`.
    pub fn get_c_class1(&self) -> f64 {
        self.cpos
    }

    /// `get_c_class2()`.
    pub fn get_c_class2(&self) -> f64 {
        self.cneg
    }

    /// `set_c_class1(C)`.
    pub fn set_c_class1(&mut self, c: f64) {
        assert!(c > 0.0, "set_c_class1: C must be greater than 0");
        self.cpos = c;
    }

    /// `set_c_class2(C)`.
    pub fn set_c_class2(&mut self, c: f64) {
        assert!(c > 0.0, "set_c_class2: C must be greater than 0");
        self.cneg = c;
    }

    /// `train(x, y) -> decision_function` — solves the C-SVC QP with
    /// `solve_qp3_using_smo` over `diagm(y)*K*diagm(y)` cached as `f32`.
    pub fn train(&self, x: &[Matrix<f64>], y: &[f64]) -> Result<DecisionFunction<K>, SvmError> {
        assert_binary_classification_problem(x, y);

        let q = LabeledKernelMatrix {
            x,
            y,
            kernel: &self.kernel_function,
        };
        let mut cache = SymmetricMatrixCache::new(q, self.cache_size);

        let mut solver = SolveQp3UsingSmo::default();
        let p = vec![-1.0; y.len()];
        let alpha = solver.solve(
            &mut cache,
            &p,
            y,
            Qp3Params {
                b: 0.0,
                cp: self.cpos,
                cn: self.cneg,
                eps: self.eps,
            },
        )?;

        let b = calculate_b_c(y, &alpha, solver.get_gradient(), self.cpos, self.cneg);
        let alpha: Vec<f64> = alpha.iter().zip(y).map(|(a, label)| a * label).collect();

        let (sv_alpha, support_vectors) = extract_support_vectors(&alpha, x);
        Ok(DecisionFunction::new(
            sv_alpha,
            b,
            self.kernel_function.clone(),
            support_vectors,
        ))
    }
}

impl<K: Kernel<SampleType = Matrix<f64>> + Default> Default for SvmCTrainer<K> {
    fn default() -> Self {
        Self {
            kernel_function: K::default(),
            cpos: 1.0,
            cneg: 1.0,
            cache_size: 200,
            eps: 0.001,
        }
    }
}

// ----------------------------------------------------------------------------------------
// svm_nu_trainer (dlib/svm/svm_nu_trainer.h)

/// Port of `dlib::svm_nu_trainer<K>` (nu-SVC). Defaults: `nu = 0.1`,
/// `cache_size = 200` (MiB), `eps = 0.001`.
#[derive(Clone, Debug)]
pub struct SvmNuTrainer<K: Kernel<SampleType = Matrix<f64>>> {
    kernel_function: K,
    nu: f64,
    cache_size: i64,
    eps: f64,
}

impl<K: Kernel<SampleType = Matrix<f64>>> SvmNuTrainer<K> {
    /// `svm_nu_trainer(kernel, nu)` (requires `0 < nu <= 1`).
    pub fn new(kernel: K, nu: f64) -> Self {
        assert!(
            0.0 < nu && nu <= 1.0,
            "svm_nu_trainer: nu must be in (0, 1]"
        );
        Self {
            kernel_function: kernel,
            nu,
            cache_size: 200,
            eps: 0.001,
        }
    }

    /// `set_cache_size(cache_size)` (MiB).
    pub fn set_cache_size(&mut self, cache_size: i64) {
        assert!(cache_size > 0, "set_cache_size: must be positive");
        self.cache_size = cache_size;
    }

    /// `get_cache_size()`.
    pub fn get_cache_size(&self) -> i64 {
        self.cache_size
    }

    /// `set_epsilon(eps)`.
    pub fn set_epsilon(&mut self, eps: f64) {
        assert!(eps > 0.0, "set_epsilon: must be positive");
        self.eps = eps;
    }

    /// `get_epsilon()`.
    pub fn get_epsilon(&self) -> f64 {
        self.eps
    }

    /// `set_kernel(k)`.
    pub fn set_kernel(&mut self, k: K) {
        self.kernel_function = k;
    }

    /// `get_kernel()`.
    pub fn get_kernel(&self) -> &K {
        &self.kernel_function
    }

    /// `set_nu(nu)`.
    pub fn set_nu(&mut self, nu: f64) {
        assert!(0.0 < nu && nu <= 1.0, "set_nu: nu must be in (0, 1]");
        self.nu = nu;
    }

    /// `get_nu()`.
    pub fn get_nu(&self) -> f64 {
        self.nu
    }

    /// `train(x, y) -> decision_function` — solves the nu-SVC QP with
    /// `solve_qp2_using_smo` over `diagm(y)*K*diagm(y)` cached as `f32`.
    pub fn train(&self, x: &[Matrix<f64>], y: &[f64]) -> Result<DecisionFunction<K>, SvmError> {
        assert_binary_classification_problem(x, y);

        let q = LabeledKernelMatrix {
            x,
            y,
            kernel: &self.kernel_function,
        };
        let mut cache = SymmetricMatrixCache::new(q, self.cache_size);

        let mut solver = SolveQp2UsingSmo::default();
        let alpha = solver.solve(&mut cache, y, self.nu, self.eps)?;

        let (rho, b) = calculate_rho_and_b(y, &alpha, solver.get_gradient());
        let alpha: Vec<f64> = alpha
            .iter()
            .zip(y)
            .map(|(a, label)| (a * label) * (1.0 / rho))
            .collect();

        let (sv_alpha, support_vectors) = extract_support_vectors(&alpha, x);
        Ok(DecisionFunction::new(
            sv_alpha,
            b,
            self.kernel_function.clone(),
            support_vectors,
        ))
    }
}

impl<K: Kernel<SampleType = Matrix<f64>> + Default> Default for SvmNuTrainer<K> {
    fn default() -> Self {
        Self {
            kernel_function: K::default(),
            nu: 0.1,
            cache_size: 200,
            eps: 0.001,
        }
    }
}

// ----------------------------------------------------------------------------------------
// svr_trainer (dlib/svm/svr_trainer.h)

/// Port of `dlib::svr_trainer<K>` (epsilon-insensitive support vector
/// regression). Defaults: `C = 1`, `eps_insensitivity = 0.1`,
/// `cache_size = 200` (MiB), `eps = 0.001`.
#[derive(Clone, Debug)]
pub struct SvrTrainer<K: Kernel<SampleType = Matrix<f64>>> {
    kernel_function: K,
    c: f64,
    eps_insensitivity: f64,
    cache_size: i64,
    eps: f64,
}

impl<K: Kernel<SampleType = Matrix<f64>>> SvrTrainer<K> {
    /// `set_cache_size(cache_size)` (MiB).
    pub fn set_cache_size(&mut self, cache_size: i64) {
        assert!(cache_size > 0, "set_cache_size: must be positive");
        self.cache_size = cache_size;
    }

    /// `get_cache_size()`.
    pub fn get_cache_size(&self) -> i64 {
        self.cache_size
    }

    /// `set_epsilon(eps)` (solver tolerance).
    pub fn set_epsilon(&mut self, eps: f64) {
        assert!(eps > 0.0, "set_epsilon: must be positive");
        self.eps = eps;
    }

    /// `get_epsilon()`.
    pub fn get_epsilon(&self) -> f64 {
        self.eps
    }

    /// `set_epsilon_insensitivity(eps)`.
    pub fn set_epsilon_insensitivity(&mut self, eps: f64) {
        assert!(eps > 0.0, "set_epsilon_insensitivity: must be positive");
        self.eps_insensitivity = eps;
    }

    /// `get_epsilon_insensitivity()`.
    pub fn get_epsilon_insensitivity(&self) -> f64 {
        self.eps_insensitivity
    }

    /// `set_kernel(k)`.
    pub fn set_kernel(&mut self, k: K) {
        self.kernel_function = k;
    }

    /// `get_kernel()`.
    pub fn get_kernel(&self) -> &K {
        &self.kernel_function
    }

    /// `set_c(C)`.
    pub fn set_c(&mut self, c: f64) {
        assert!(c > 0.0, "set_c: C must be greater than 0");
        self.c = c;
    }

    /// `get_c()`.
    pub fn get_c(&self) -> f64 {
        self.c
    }

    /// `train(x, y) -> decision_function` — solves the doubled epsilon-SVR QP
    /// with `solve_qp3_using_smo` over `make_quad(kernel_matrix(x))`.
    pub fn train(&self, x: &[Matrix<f64>], y: &[f64]) -> Result<DecisionFunction<K>, SvmError> {
        assert!(
            !x.is_empty() && x.len() == y.len(),
            "train(x, y): x and y must be nonempty and the same length"
        );

        let n = x.len();
        let q = QuadKernelMatrix {
            x,
            kernel: &self.kernel_function,
        };
        let mut cache = SymmetricMatrixCache::new(q, self.cache_size);

        // p = uniform_matrix(2n, 1, eps_insensitivity) + join_cols(y, -y)
        let p: Vec<f64> = (0..2 * n)
            .map(|i| {
                if i < n {
                    self.eps_insensitivity + y[i]
                } else {
                    self.eps_insensitivity - y[i - n]
                }
            })
            .collect();
        // y = join_cols(uniform_matrix(n, 1, 1), uniform_matrix(n, 1, -1))
        let labels: Vec<f64> = (0..2 * n).map(|i| if i < n { 1.0 } else { -1.0 }).collect();

        let mut solver = SolveQp3UsingSmo::default();
        let alpha = solver.solve(
            &mut cache,
            &p,
            &labels,
            Qp3Params {
                b: 0.0,
                cp: self.c,
                cn: self.c,
                eps: self.eps,
            },
        )?;

        let b = calculate_b_svr(&alpha, solver.get_gradient(), self.c);
        // alpha = -rowm(alpha, range(0, n-1)) + rowm(alpha, range(n, 2n-1))
        let alpha: Vec<f64> = (0..n).map(|i| -alpha[i] + alpha[i + n]).collect();

        let (sv_alpha, support_vectors) = extract_support_vectors(&alpha, x);
        Ok(DecisionFunction::new(
            sv_alpha,
            -b,
            self.kernel_function.clone(),
            support_vectors,
        ))
    }
}

impl<K: Kernel<SampleType = Matrix<f64>> + Default> Default for SvrTrainer<K> {
    fn default() -> Self {
        Self {
            kernel_function: K::default(),
            c: 1.0,
            eps_insensitivity: 0.1,
            cache_size: 200,
            eps: 0.001,
        }
    }
}

// ----------------------------------------------------------------------------------------
// randomize_samples (dlib/svm/svm.h)

/// Port of `dlib::randomize_samples(t, r)`
/// (dlib/svm/svm.h): Fisher-Yates style shuffle driven by
/// `r.get_random_32bit_number() % (n+1)` swaps.
pub fn randomize_samples<T>(t: &mut [T], rnd: &mut Rand) {
    let mut n = t.len() as i64 - 1;
    while n > 0 {
        // pick a random index to swap into t[n]
        let idx = (rnd.get_random_32bit_number() as u64 % (n as u64 + 1)) as usize;
        t.swap(idx, n as usize);
        n -= 1;
    }
}

/// Port of `dlib::randomize_samples(t, u, r)` — shuffles `t` and `u` in
/// lockstep (samples and targets).
pub fn randomize_samples_with_targets<T, U>(t: &mut [T], u: &mut [U], rnd: &mut Rand) {
    assert_eq!(t.len(), u.len(), "randomize_samples(t, u): size mismatch");
    let mut n = t.len() as i64 - 1;
    while n > 0 {
        // pick a random index to swap into t[n]
        let idx = (rnd.get_random_32bit_number() as u64 % (n as u64 + 1)) as usize;
        t.swap(idx, n as usize);
        u.swap(idx, n as usize);
        n -= 1;
    }
}

/// Port of `dlib::randomize_samples(t, u, v, r)` — shuffles three vectors in
/// lockstep.
pub fn randomize_samples_with_triples<T, U, V>(
    t: &mut [T],
    u: &mut [U],
    v: &mut [V],
    rnd: &mut Rand,
) {
    assert_eq!(
        t.len(),
        u.len(),
        "randomize_samples(t, u, v): t/u size mismatch"
    );
    assert_eq!(
        t.len(),
        v.len(),
        "randomize_samples(t, u, v): t/v size mismatch"
    );
    let mut n = t.len() as i64 - 1;
    while n > 0 {
        // pick a random index to swap into t[n]
        let idx = (rnd.get_random_32bit_number() as u64 % (n as u64 + 1)) as usize;
        t.swap(idx, n as usize);
        u.swap(idx, n as usize);
        v.swap(idx, n as usize);
        n -= 1;
    }
}

// ----------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svm::kernels::{LinearKernel, RadialBasisKernel};

    fn col(vals: &[f64]) -> Matrix<f64> {
        Matrix::from_row_vec(vals.len(), 1, vals)
    }

    fn blobs20() -> (Vec<Matrix<f64>>, Vec<f64>) {
        let pts = [
            [0.5, 0.4],
            [1.2, 0.1],
            [0.3, 1.1],
            [1.5, 1.3],
            [0.9, 0.2],
            [0.1, 0.7],
            [1.1, 0.9],
            [0.6, 1.5],
            [1.4, 0.5],
            [0.2, 0.2],
            [2.8, 2.9],
            [3.1, 2.7],
            [2.6, 3.2],
            [3.3, 3.4],
            [2.9, 3.1],
            [3.2, 2.5],
            [2.7, 2.8],
            [3.5, 3.0],
            [2.5, 3.5],
            [3.0, 3.3],
        ];
        let x: Vec<Matrix<f64>> = pts.iter().map(|p| col(p)).collect();
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 1.0 } else { -1.0 }).collect();
        (x, y)
    }

    #[test]
    fn svm_c_linear_separable() {
        let x = vec![
            col(&[2.0, 2.0]),
            col(&[-2.0, -2.0]),
            col(&[2.0, -2.0]),
            col(&[-3.0, -1.0]),
        ];
        let y = vec![1.0, -1.0, 1.0, -1.0];

        let trainer = SvmCTrainer::new(LinearKernel, 10.0);
        let df = trainer.train(&x, &y).unwrap();

        // every training point is classified correctly
        for (xi, &yi) in x.iter().zip(&y) {
            assert!(df.operator_(xi) * yi > 0.0, "missed {}", yi);
        }
        // at most one support vector per sample
        let sv = df.basis_dictionary.len();
        assert!((1..=4).contains(&sv), "unexpected sv count {sv}");

        // deterministic: identical alphas across two runs
        let df2 = trainer.train(&x, &y).unwrap();
        assert_eq!(df.alpha_vector.nr(), df2.alpha_vector.nr());
        for i in 0..df.alpha_vector.nr() {
            assert_eq!(df.alpha_vector[i], df2.alpha_vector[i]);
        }
        assert_eq!(df.b, df2.b);
        // bit-exact match with C++ dlib 20.x on this machine (golden run)
        assert_eq!(df.b.to_bits(), 0xbf35_5555_5555_5555);
    }

    #[test]
    fn svm_c_rbf_two_blobs() {
        let (x, y) = blobs20();
        let trainer = SvmCTrainer::new(RadialBasisKernel::new(0.5), 5.0);
        let df = trainer.train(&x, &y).unwrap();

        for (xi, &yi) in x.iter().zip(&y) {
            assert!(
                df.operator_(xi) * yi > 0.0,
                "point misclassified: df={} y={}",
                df.operator_(xi),
                yi
            );
        }

        let df2 = trainer.train(&x, &y).unwrap();
        assert_eq!(df.alpha_vector.nr(), df2.alpha_vector.nr());
        for i in 0..df.alpha_vector.nr() {
            assert_eq!(df.alpha_vector[i], df2.alpha_vector[i]);
        }
        assert_eq!(df.b, df2.b);
        assert_eq!(df.kernel_function.gamma, 0.5);
        // bit-exact match with C++ dlib 20.x on this machine (golden run)
        assert_eq!(df.b.to_bits(), 0xbfbd_56d5_6536_d43c);
        assert_eq!(df.alpha_vector[0].to_bits(), 0x3fc9_bb79_9cf2_8c4d);
    }

    #[test]
    fn svm_nu_classifies_and_is_deterministic() {
        let (x, y) = blobs20();
        let nu = 0.1;
        assert!(nu < maximum_nu(&y));
        let trainer = SvmNuTrainer::new(RadialBasisKernel::new(0.5), nu);
        let df = trainer.train(&x, &y).unwrap();
        for (xi, &yi) in x.iter().zip(&y) {
            assert!(df.operator_(xi) * yi > 0.0);
        }
        let df2 = trainer.train(&x, &y).unwrap();
        assert_eq!(df.alpha_vector.nr(), df2.alpha_vector.nr());
        for i in 0..df.alpha_vector.nr() {
            assert_eq!(df.alpha_vector[i], df2.alpha_vector[i]);
        }
        assert_eq!(df.b, df2.b);
        // bit-exact match with C++ dlib 20.x on this machine (golden run);
        // alpha(4) also guards the multiply-by-reciprocal semantics of dlib's
        // matrix/scalar operator/
        assert_eq!(df.b.to_bits(), 0xbfbd_529e_91cd_ee9b);
        assert_eq!(df.alpha_vector[4].to_bits(), 0xbf8d_f048_d908_b8cc);
    }

    #[test]
    fn svm_nu_rejects_infeasible_nu() {
        // y has 1 of 4 negatives => maximum_nu = 0.5; nu = 0.9 must fail
        let x = vec![col(&[0.0]), col(&[1.0]), col(&[2.0]), col(&[9.0])];
        let y = vec![1.0, 1.0, 1.0, -1.0];
        assert!((maximum_nu(&y) - 0.5).abs() < 1e-15);
        let trainer = SvmNuTrainer::new(LinearKernel, 0.9);
        let err = trainer.train(&x, &y).unwrap_err();
        assert!(matches!(err, SvmError::InvalidNu { nu, .. } if nu == 0.9));
    }

    #[test]
    fn svr_fits_linear_function() {
        // y = 2*x0 + 1 exactly; a linear-kernel SVR should reproduce it
        let inputs = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let x: Vec<Matrix<f64>> = inputs.iter().map(|p| col(p)).collect();
        let y: Vec<f64> = inputs.iter().map(|p| 2.0 * p[0] + 1.0).collect();

        let mut trainer = SvrTrainer::default();
        trainer.set_kernel(LinearKernel);
        trainer.set_c(10.0);
        trainer.set_epsilon_insensitivity(0.01);
        let df = trainer.train(&x, &y).unwrap();

        for (xi, &yi) in x.iter().zip(&y) {
            assert!((df.operator_(xi) - yi).abs() < 0.2);
        }

        let df2 = trainer.train(&x, &y).unwrap();
        assert_eq!(df.b, df2.b);
        // bit-exact match with C++ dlib 20.x on this machine (golden run)
        assert_eq!(df.b.to_bits(), 0xbff0_28f5_c28f_5c24);
        assert_eq!(df.alpha_vector.nr(), df2.alpha_vector.nr());
        for i in 0..df.alpha_vector.nr() {
            assert_eq!(df.alpha_vector[i], df2.alpha_vector[i]);
        }
    }

    #[test]
    fn maximum_nu_value() {
        assert_eq!(maximum_nu(&[1.0, 1.0, 1.0, -1.0]), 0.5);
        assert_eq!(maximum_nu(&[1.0, -1.0]), 1.0);
        assert_eq!(maximum_nu(&[1.0, -1.0, -1.0, -1.0, -1.0]), 0.4);
    }

    #[test]
    fn randomize_samples_is_deterministic_per_seed() {
        let base: Vec<i32> = (0..10).collect();
        let labels: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let mut a = base.clone();
        let mut la = labels.clone();
        randomize_samples_with_targets(&mut a, &mut la, &mut Rand::with_seed("svm-shuffle"));

        let mut b = base.clone();
        let mut lb = labels.clone();
        randomize_samples_with_targets(&mut b, &mut lb, &mut Rand::with_seed("svm-shuffle"));

        assert_eq!(a, b, "same seed must give the same permutation");
        assert_eq!(la, lb);
        assert_ne!(a, base, "the shuffle must actually shuffle");

        // it is a permutation of the input
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, base);

        // pairs stay aligned
        for (v, l) in a.iter().zip(&la) {
            let orig = base.iter().position(|p| p == v).unwrap();
            assert_eq!(labels[orig], *l);
        }

        // single-vector overload uses the same rand stream
        let mut c = base.clone();
        randomize_samples(&mut c, &mut Rand::with_seed("svm-shuffle"));
        assert_eq!(c, a);

        // triple overload
        let mut t1 = base.clone();
        let mut t2 = base.clone();
        let mut t3 = base.clone();
        randomize_samples_with_triples(&mut t1, &mut t2, &mut t3, &mut Rand::with_seed("svm"));
        assert_eq!(t1, t2);
        assert_eq!(t1, t3);
        let mut sorted = t1.clone();
        sorted.sort();
        assert_eq!(sorted, base);
    }
}

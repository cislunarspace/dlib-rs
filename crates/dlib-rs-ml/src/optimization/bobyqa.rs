//! Port of `dlib/optimization/optimization_bobyqa.h`.
//!
//! The code below is derived from M.J.D. Powell's BOBYQA Fortran code (as
//! translated to C++ in dlib by Davis E. King) and is ported line-by-line to
//! preserve the exact floating point evaluation order of the original.  The
//! Fortran 1-based indexing is kept by padding the 1-D work vectors with one
//! unused leading element; the column-major 2-D matrices (`xpt`, `bmat`,
//! `zmat`) are accessed through the [`a2!`] macro with 1-based indices.

#![allow(clippy::needless_range_loop)] // faithful port of f2c loop structure
#![allow(clippy::too_many_arguments)] // faithful port of Fortran signatures
#![allow(unused_assignments)] // Fortran-style initializations kept for port fidelity
#![allow(clippy::neg_cmp_op_on_partial_ord)] // faithful to the dlib assertion semantics
#![allow(clippy::assign_op_pattern)] // keep the f2c statement forms
#![allow(clippy::manual_memcpy)] // faithful port of f2c copy loops
#![allow(clippy::manual_swap)] // faithful port of f2c swap sequences
#![allow(clippy::collapsible_if)] // faithful port of f2c nesting

use dlib_rs_core::matrix::Matrix;

/// 1-based access into a Fortran column-major matrix with first dimension
/// `$d1` (number of rows), stored flat: `v[(i-1) + (j-1)*d1]`.
macro_rules! a2 {
    ($v:expr, $i:expr, $j:expr, $d1:expr) => {
        $v[($i - 1) + ($j - 1) * $d1]
    };
}

/// 1-based access into the (2, N) array `ptsaux` of `rescue_`.
macro_rules! p2 {
    ($v:expr, $i:expr, $j:expr) => {
        $v[$i + 2 * $j]
    };
}

/// Default final trust region radius (`rho_end`).  dlib has no default (every
/// caller passes one; its own tests use `1e-6`/`1e-8`); we default to `1e-9`.
const DEFAULT_RHO_END: f64 = 1e-9;
/// Default upper bound on the number of objective evaluations, as used by the
/// dlib test suite (`10000`).
const DEFAULT_MAX_F_EVALS: i64 = 10000;
/// Bounds used by dlib callers (e.g. `dlib/test/optimization.cpp`) to express
/// "unconstrained" problems: `[-1e100, 1e100]`.
const HUGE_BOUND: f64 = 1e100;

#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

type CalFun<'a> = &'a dyn Fn(&[f64]) -> f64;

/// Seeks the minimum of `f` starting from `x0` (a column vector, modified in
/// place) using Powell's BOBYQA algorithm without explicit bounds.
///
/// Port of `dlib::find_min_bobyqa` (`dlib/optimization/optimization_bobyqa.h`)
/// as used by dlib itself for unconstrained problems: bounds are
/// `[-1e100, 1e100]` (cf. `dlib/test/optimization.cpp`), `npt = 2*n+1` (the
/// value recommended in `optimization_bobyqa_abstract.h`), `rho_end = 1e-9`
/// and `max_f_evals = 10000`.  `radius` is `rho_begin` and should be about one
/// tenth of the greatest expected change to a variable.
///
/// Returns `(f(x_min), number_of_evaluations)`; `x0` is set to the minimizer.
pub fn find_min_bobyqa<F>(
    f: F,
    x0: &mut Matrix<f64>,
    radius: f64,
) -> Result<(f64, usize), &'static str>
where
    F: Fn(&Matrix<f64>) -> f64,
{
    let n = x0.nr();
    let lower = Matrix::from_row_vec(n, 1, &vec![-HUGE_BOUND; n]);
    let upper = Matrix::from_row_vec(n, 1, &vec![HUGE_BOUND; n]);
    find_min_bobyqa_with_bounds(f, x0, radius, &lower, &upper)
}

/// Bounded variant of [`find_min_bobyqa`]; port of `dlib::find_min_bobyqa(f,
/// x, npt, x_lower, x_upper, rho_begin, rho_end, max_f_evals)` with the
/// dlib-recommended defaults `npt = 2*n+1`, `rho_end = 1e-9` and
/// `max_f_evals = 10000`.  `radius` is `rho_begin`.
///
/// Requires (as dlib asserts): column vectors of equal size with `n > 1`,
/// `x0` within `[lower, upper]`, and `min(upper - lower) > 2 * radius`.
pub fn find_min_bobyqa_with_bounds<F>(
    f: F,
    x0: &mut Matrix<f64>,
    radius: f64,
    lower: &Matrix<f64>,
    upper: &Matrix<f64>,
) -> Result<(f64, usize), &'static str>
where
    F: Fn(&Matrix<f64>) -> f64,
{
    let n = x0.nr();
    if x0.nc() != 1 || lower.nc() != 1 || upper.nc() != 1 {
        return Err("find_min_bobyqa(): x, x_lower and x_upper must be column vectors");
    }
    if lower.nr() != n || upper.nr() != n {
        return Err("find_min_bobyqa(): x, x_lower and x_upper must all have the same length");
    }
    if n <= 1 {
        return Err("find_min_bobyqa(): x.size() must be greater than 1");
    }
    if !(0.0 < DEFAULT_RHO_END && DEFAULT_RHO_END < radius) {
        return Err("find_min_bobyqa(): require 0 < rho_end < rho_begin");
    }
    let npt = 2 * n + 1;
    if !(n + 2 <= npt && npt <= (n + 1) * (n + 2) / 2) {
        return Err("find_min_bobyqa(): npt is not in the required interval");
    }
    let mut min_width = f64::INFINITY;
    let mut min_low = f64::INFINITY;
    let mut min_upp = f64::INFINITY;
    for i in 0..n {
        min_width = min_width.min(upper[(i, 0)] - lower[(i, 0)]);
        min_low = min_low.min(x0[(i, 0)] - lower[(i, 0)]);
        min_upp = min_upp.min(upper[(i, 0)] - x0[(i, 0)]);
    }
    if !(min_width > 2.0 * radius) {
        return Err("find_min_bobyqa(): min(x_upper - x_lower) must be > 2*rho_begin");
    }
    if !(min_low >= 0.0 && min_upp >= 0.0) {
        return Err("find_min_bobyqa(): x must be within [x_lower, x_upper]");
    }

    let nfun = n;
    let calfun = move |xs: &[f64]| -> f64 { f(&Matrix::from_row_vec(nfun, 1, xs)) };

    let mut st = Bobyqa::new(n, npt);
    for i in 1..=n {
        st.x[i] = x0[(i - 1, 0)];
        st.xl[i] = lower[(i - 1, 0)];
        st.xu[i] = upper[(i - 1, 0)];
    }
    let (fmin, nfev) = st.bobyqa_top(&calfun, radius, DEFAULT_RHO_END, DEFAULT_MAX_F_EVALS)?;
    for i in 1..=n {
        x0[(i - 1, 0)] = st.x[i];
    }
    Ok((fmin, nfev as usize))
}

/// All working state of the BOBYQA implementation (`bobyqa_implementation` in
/// `dlib/optimization/optimization_bobyqa.h`).  Every 1-D array keeps the
/// Fortran 1-based indexing by leaving element 0 unused.
struct Bobyqa {
    n: usize,
    npt: usize,
    np: usize,
    nptm: usize,
    nh: usize,
    ndim: usize,
    x: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    xbase: Vec<f64>,
    xopt: Vec<f64>,
    gopt: Vec<f64>,
    sl: Vec<f64>,
    su: Vec<f64>,
    xnew: Vec<f64>,
    xalt: Vec<f64>,
    d: Vec<f64>,
    vlag: Vec<f64>,
    fval: Vec<f64>,
    hq: Vec<f64>,
    pq: Vec<f64>,
    /// npt x n, column-major (Fortran layout).
    xpt: Vec<f64>,
    /// ndim x n, column-major (Fortran layout).
    bmat: Vec<f64>,
    /// npt x nptm, column-major (Fortran layout).
    zmat: Vec<f64>,
    /// working space of `bobyqb_`, 1-based, length 2*npt.
    w: Vec<f64>,
}

impl Bobyqa {
    fn new(n: usize, npt: usize) -> Self {
        let np = n + 1;
        Bobyqa {
            n,
            npt,
            np,
            nptm: npt - np,
            nh: n * np / 2,
            ndim: npt + n,
            x: vec![0.0; n + 1],
            xl: vec![0.0; n + 1],
            xu: vec![0.0; n + 1],
            xbase: vec![0.0; n + 1],
            xopt: vec![0.0; n + 1],
            gopt: vec![0.0; n + 1],
            sl: vec![0.0; n + 1],
            su: vec![0.0; n + 1],
            xnew: vec![0.0; n + 1],
            xalt: vec![0.0; n + 1],
            d: vec![0.0; n + 1],
            vlag: vec![0.0; npt + n + 1],
            fval: vec![0.0; npt + 1],
            hq: vec![0.0; n * np / 2 + 1],
            pq: vec![0.0; npt + 1],
            xpt: vec![0.0; npt * n],
            bmat: vec![0.0; (npt + n) * n],
            zmat: vec![0.0; npt * (npt - np)],
            w: vec![0.0; 2 * npt + 1],
        }
    }

    /// Port of `bobyqa_`: NPT sanity check, bounds adjustment of the initial
    /// X, then the call of BOBYQB.
    fn bobyqa_top(
        &mut self,
        calfun: CalFun,
        rhobeg: f64,
        rhoend: f64,
        maxfun: i64,
    ) -> Result<(f64, i64), &'static str> {
        let n = self.n;
        let npt = self.npt;
        let zero = 0.0;

        if npt < n + 2 || npt > (n + 2) * self.np / 2 {
            return Err("Return from BOBYQA because NPT is not in the required interval");
        }

        // Return if there is insufficient space between the bounds.  Modify
        // the initial X if necessary in order to avoid conflicts between the
        // bounds and the construction of the first quadratic model.  The
        // lower and upper bounds on moves from the updated X are set in SL
        // and SU.
        for j in 1..=n {
            let temp = self.xu[j] - self.xl[j];
            if temp < rhobeg + rhobeg {
                return Err("Return from BOBYQA because one of the differences in x_lower and x_upper is less than 2*rho_begin");
            }
            self.sl[j] = self.xl[j] - self.x[j];
            self.su[j] = self.xu[j] - self.x[j];
            if self.sl[j] >= -rhobeg {
                if self.sl[j] >= zero {
                    self.x[j] = self.xl[j];
                    self.sl[j] = zero;
                    self.su[j] = temp;
                } else {
                    self.x[j] = self.xl[j] + rhobeg;
                    self.sl[j] = -rhobeg;
                    self.su[j] = (self.xu[j] - self.x[j]).max(rhobeg);
                }
            } else if self.su[j] <= rhobeg {
                if self.su[j] <= zero {
                    self.x[j] = self.xu[j];
                    self.sl[j] = -temp;
                    self.su[j] = zero;
                } else {
                    self.x[j] = self.xu[j] - rhobeg;
                    self.sl[j] = (self.xl[j] - self.x[j]).min(-rhobeg);
                    self.su[j] = rhobeg;
                }
            }
        }

        // Make the call of BOBYQB.
        self.bobyqb(calfun, rhobeg, rhoend, maxfun)
    }

    /// Port of `bobyqb_`, the main BOBYQA iteration.  The Fortran `goto`
    /// labels (L20, L60, L90, L190, L210, L230, L360, L650, L680, L720) are
    /// expressed as a state machine over `Phase`.
    fn bobyqb(
        &mut self,
        calfun: CalFun,
        rhobeg: f64,
        rhoend: f64,
        maxfun: i64,
    ) -> Result<(f64, i64), &'static str> {
        let n = self.n;
        let npt = self.npt;
        let ndim = self.ndim;
        let nptm = self.nptm;

        let half = 0.5;
        let one = 1.0;
        let ten = 10.0;
        let tenth = 0.1;
        let two = 2.0;
        let zero = 0.0;

        let (mut nf, mut kopt) = self.prelim(calfun, rhobeg, maxfun)?;
        let mut f = 0.0;
        let mut xoptsq = zero;
        for i in 1..=n {
            self.xopt[i] = a2!(self.xpt, kopt, i, npt);
            xoptsq += self.xopt[i] * self.xopt[i];
        }
        let mut fsave = self.fval[1];
        if nf < npt as i64 {
            return Err("Return from BOBYQA because the objective function has been called max_f_evals times.");
        }
        let mut kbase = 1usize;

        let mut rho = rhobeg;
        let mut delta = rho;
        let mut nresc = nf;
        let mut ntrits: i64 = 0;
        let mut diffa = zero;
        let mut diffb = zero;
        let mut diffc = zero;
        let mut itest: i64 = 0;
        let mut nfsav = nf;

        let mut knew = 0usize;
        let mut alpha = 0.0;
        let mut cauchy = 0.0;
        let mut adelt = 0.0;
        let mut denom = 0.0;
        let mut beta = 0.0;
        let mut dsq = 0.0;
        let mut dnorm = 0.0;
        let mut distsq = 0.0;
        let mut ratio = 0.0;
        let mut vquad = 0.0;

        #[derive(Clone, Copy, PartialEq)]
        enum Phase {
            P20,
            P60,
            P90,
            P190,
            P210,
            P230,
            P360,
            P650,
            P680,
            P720,
        }
        let mut phase = Phase::P20;

        loop {
            match phase {
                Phase::P20 => {
                    // Update GOPT if necessary before the first iteration and
                    // after each call of RESCUE that makes a call of CALFUN.
                    if kopt != kbase {
                        let mut ih = 0usize;
                        for j in 1..=n {
                            for i in 1..=j {
                                ih += 1;
                                if i < j {
                                    self.gopt[j] += self.hq[ih] * self.xopt[i];
                                }
                                self.gopt[i] += self.hq[ih] * self.xopt[j];
                            }
                        }
                        if nf > npt as i64 {
                            for k in 1..=npt {
                                let mut temp = zero;
                                for j in 1..=n {
                                    temp += a2!(self.xpt, k, j, npt) * self.xopt[j];
                                }
                                temp = self.pq[k] * temp;
                                for i in 1..=n {
                                    self.gopt[i] += temp * a2!(self.xpt, k, i, npt);
                                }
                            }
                        }
                    }
                    phase = Phase::P60;
                }

                Phase::P60 => {
                    let (dsq_out, crvmin) = self.trsbox(delta);
                    dsq = dsq_out;
                    dnorm = delta.min(dsq.sqrt());
                    if dnorm < half * rho {
                        ntrits = -1;
                        distsq = sq(ten * rho);
                        if nf <= nfsav + 2 {
                            phase = Phase::P650;
                            continue;
                        }

                        // The following choice between labels 650 and 680
                        // depends on whether or not our work with the current
                        // RHO seems to be complete.  Either RHO is decreased
                        // or termination occurs if the errors in the quadratic
                        // model at the last three interpolation points compare
                        // favourably with predictions of likely improvements
                        // to the model within distance HALF*RHO of XOPT.
                        let errbig = diffa.max(diffb).max(diffc);
                        let frhosq = rho * 0.125 * rho;
                        if crvmin > zero && errbig > frhosq * crvmin {
                            phase = Phase::P650;
                            continue;
                        }
                        let bdtol = errbig / rho;
                        let mut go_650 = false;
                        for j in 1..=n {
                            let mut bdtest = bdtol;
                            if self.xnew[j] == self.sl[j] {
                                bdtest = self.w[j];
                            }
                            if self.xnew[j] == self.su[j] {
                                bdtest = -self.w[j];
                            }
                            if bdtest < bdtol {
                                let mut curv = self.hq[(j + j * j) / 2];
                                for k in 1..=npt {
                                    curv += self.pq[k] * sq(a2!(self.xpt, k, j, npt));
                                }
                                bdtest += half * curv * rho;
                                if bdtest < bdtol {
                                    go_650 = true;
                                    break;
                                }
                            }
                        }
                        if go_650 {
                            phase = Phase::P650;
                        } else {
                            phase = Phase::P680;
                        }
                    } else {
                        ntrits += 1;
                        phase = Phase::P90;
                    }
                }

                Phase::P90 => {
                    // Severe cancellation is likely to occur if XOPT is too
                    // far from XBASE.  If the following test holds, then XBASE
                    // is shifted so that XOPT becomes zero.
                    if dsq <= xoptsq * 0.001 {
                        let fracsq = xoptsq * 0.25;
                        let mut sumpq = zero;
                        for k in 1..=npt {
                            sumpq += self.pq[k];
                            let mut sum = -half * xoptsq;
                            for i in 1..=n {
                                sum += a2!(self.xpt, k, i, npt) * self.xopt[i];
                            }
                            self.w[npt + k] = sum;
                            let temp = fracsq - half * sum;
                            for i in 1..=n {
                                self.w[i] = a2!(self.bmat, k, i, ndim);
                                self.vlag[i] = sum * a2!(self.xpt, k, i, npt) + temp * self.xopt[i];
                                let ip = npt + i;
                                for j in 1..=i {
                                    a2!(self.bmat, ip, j, ndim) = a2!(self.bmat, ip, j, ndim)
                                        + self.w[i] * self.vlag[j]
                                        + self.vlag[i] * self.w[j];
                                }
                            }
                        }

                        // Then the revisions of BMAT that depend on ZMAT are
                        // calculated.
                        for jj in 1..=nptm {
                            let mut sumz = zero;
                            let mut sumw = zero;
                            for k in 1..=npt {
                                sumz += a2!(self.zmat, k, jj, npt);
                                self.vlag[k] = self.w[npt + k] * a2!(self.zmat, k, jj, npt);
                                sumw += self.vlag[k];
                            }
                            for j in 1..=n {
                                let mut sum = (fracsq * sumz - half * sumw) * self.xopt[j];
                                for k in 1..=npt {
                                    sum += self.vlag[k] * a2!(self.xpt, k, j, npt);
                                }
                                self.w[j] = sum;
                                for k in 1..=npt {
                                    a2!(self.bmat, k, j, ndim) += sum * a2!(self.zmat, k, jj, npt);
                                }
                            }
                            for i in 1..=n {
                                let ip = i + npt;
                                let temp = self.w[i];
                                for j in 1..=i {
                                    a2!(self.bmat, ip, j, ndim) += temp * self.w[j];
                                }
                            }
                        }

                        // The following instructions complete the shift,
                        // including the changes to the second derivative
                        // parameters of the quadratic model.
                        let mut ih = 0usize;
                        for j in 1..=n {
                            self.w[j] = -half * sumpq * self.xopt[j];
                            for k in 1..=npt {
                                self.w[j] += self.pq[k] * a2!(self.xpt, k, j, npt);
                                a2!(self.xpt, k, j, npt) -= self.xopt[j];
                            }
                            for i in 1..=j {
                                ih += 1;
                                self.hq[ih] = self.hq[ih]
                                    + self.w[i] * self.xopt[j]
                                    + self.xopt[i] * self.w[j];
                                a2!(self.bmat, npt + i, j, ndim) = a2!(self.bmat, npt + j, i, ndim);
                            }
                        }
                        for i in 1..=n {
                            self.xbase[i] += self.xopt[i];
                            self.xnew[i] -= self.xopt[i];
                            self.sl[i] -= self.xopt[i];
                            self.su[i] -= self.xopt[i];
                            self.xopt[i] = zero;
                        }
                        xoptsq = zero;
                    }
                    if ntrits == 0 {
                        phase = Phase::P210;
                    } else {
                        phase = Phase::P230;
                    }
                }

                Phase::P190 => {
                    // XBASE is also moved to XOPT by a call of RESCUE.  This
                    // calculation is more expensive than the previous shift,
                    // and is called only if rounding errors have reduced by at
                    // least a factor of two the denominator of the formula
                    // for updating the H matrix.
                    nfsav = nf;
                    kbase = kopt;
                    self.rescue(calfun, maxfun, delta, &mut nf, &mut kopt);

                    // XOPT is updated now in case the branch below to label
                    // 720 is taken.
                    xoptsq = zero;
                    if kopt != kbase {
                        for i in 1..=n {
                            self.xopt[i] = a2!(self.xpt, kopt, i, npt);
                            xoptsq += self.xopt[i] * self.xopt[i];
                        }
                    }
                    if nf < 0 {
                        nf = maxfun;
                        return Err("Return from BOBYQA because the objective function has been called max_f_evals times.");
                    }
                    nresc = nf;
                    if nfsav < nf {
                        nfsav = nf;
                        phase = Phase::P20;
                    } else if ntrits > 0 {
                        phase = Phase::P60;
                    } else {
                        phase = Phase::P210;
                    }
                }

                Phase::P210 => {
                    // Pick two alternative vectors of variables, relative to
                    // XBASE, that are suitable as new positions of the KNEW-th
                    // interpolation point.
                    let (a, c) = self.altmov(kopt, knew, adelt);
                    alpha = a;
                    cauchy = c;
                    for i in 1..=n {
                        self.d[i] = self.xnew[i] - self.xopt[i];
                    }
                    phase = Phase::P230;
                }

                Phase::P230 => {
                    // Calculate VLAG and BETA for the current choice of D.
                    // The Fortran `goto L230` after swapping in the Cauchy
                    // step is expressed as a loop here.
                    let mut next = Phase::P360;
                    loop {
                        for k in 1..=npt {
                            let mut suma = zero;
                            let mut sumb = zero;
                            let mut sum = zero;
                            for j in 1..=n {
                                suma += a2!(self.xpt, k, j, npt) * self.d[j];
                                sumb += a2!(self.xpt, k, j, npt) * self.xopt[j];
                                sum += a2!(self.bmat, k, j, ndim) * self.d[j];
                            }
                            self.w[k] = suma * (half * suma + sumb);
                            self.vlag[k] = sum;
                            self.w[npt + k] = suma;
                        }
                        beta = zero;
                        for jj in 1..=nptm {
                            let mut sum = zero;
                            for k in 1..=npt {
                                sum += a2!(self.zmat, k, jj, npt) * self.w[k];
                            }
                            beta -= sum * sum;
                            for k in 1..=npt {
                                self.vlag[k] += sum * a2!(self.zmat, k, jj, npt);
                            }
                        }
                        dsq = zero;
                        let mut bsum = zero;
                        let mut dx = zero;
                        for j in 1..=n {
                            dsq += sq(self.d[j]);
                            let mut sum = zero;
                            for k in 1..=npt {
                                sum += self.w[k] * a2!(self.bmat, k, j, ndim);
                            }
                            bsum += sum * self.d[j];
                            let jp = npt + j;
                            for i in 1..=n {
                                sum += a2!(self.bmat, jp, i, ndim) * self.d[i];
                            }
                            self.vlag[jp] = sum;
                            bsum += sum * self.d[j];
                            dx += self.d[j] * self.xopt[j];
                        }
                        beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta - bsum;
                        self.vlag[kopt] += one;

                        // If NTRITS is zero, the denominator may be increased
                        // by replacing the step D of ALTMOV by a Cauchy step.
                        // Then RESCUE may be called if rounding errors have
                        // damaged the chosen denominator.
                        if ntrits == 0 {
                            denom = sq(self.vlag[knew]) + alpha * beta;
                            if denom < cauchy && cauchy > zero {
                                for i in 1..=n {
                                    self.xnew[i] = self.xalt[i];
                                    self.d[i] = self.xnew[i] - self.xopt[i];
                                }
                                cauchy = zero;
                                continue; // goto L230
                            }
                            if denom <= half * sq(self.vlag[knew]) {
                                if nf > nresc {
                                    next = Phase::P190;
                                    break;
                                }
                                return Err(
                                    "Return from BOBYQA because of much cancellation in a denominator."
                                );
                            }
                        } else {
                            // Alternatively, if NTRITS is positive, then set
                            // KNEW to the index of the next interpolation
                            // point to be deleted to make room for a trust
                            // region step.
                            let delsq = delta * delta;
                            let mut scaden = zero;
                            let mut biglsq = zero;
                            knew = 0;
                            for k in 1..=npt {
                                if k == kopt {
                                    continue;
                                }
                                let mut hdiag = zero;
                                for jj in 1..=nptm {
                                    hdiag += sq(a2!(self.zmat, k, jj, npt));
                                }
                                let den = beta * hdiag + sq(self.vlag[k]);
                                let mut distsq2 = zero;
                                for j in 1..=n {
                                    distsq2 += sq(a2!(self.xpt, k, j, npt) - self.xopt[j]);
                                }
                                let temp = one.max(sq(distsq2 / delsq));
                                if temp * den > scaden {
                                    scaden = temp * den;
                                    knew = k;
                                    denom = den;
                                }
                                biglsq = biglsq.max(temp * sq(self.vlag[k]));
                            }
                            if scaden <= half * biglsq {
                                if nf > nresc {
                                    next = Phase::P190;
                                    break;
                                }
                                return Err(
                                    "Return from BOBYQA because of much cancellation in a denominator."
                                );
                            }
                        }
                        break;
                    }
                    phase = next;
                }

                Phase::P360 => {
                    // Put the variables for the next calculation of the
                    // objective function in X, with any adjustments for the
                    // bounds, and calculate the value of the objective
                    // function at XBASE+XNEW, unless the limit on the number
                    // of calculations of F has been reached.
                    for i in 1..=n {
                        self.x[i] = self.xl[i].max(self.xbase[i] + self.xnew[i]).min(self.xu[i]);
                        if self.xnew[i] == self.sl[i] {
                            self.x[i] = self.xl[i];
                        }
                        if self.xnew[i] == self.su[i] {
                            self.x[i] = self.xu[i];
                        }
                    }
                    if nf >= maxfun {
                        return Err("Return from BOBYQA because the objective function has been called max_f_evals times.");
                    }
                    nf += 1;
                    f = calfun(&self.x[1..=n]);
                    if ntrits == -1 {
                        fsave = f;
                        phase = Phase::P720;
                        continue;
                    }

                    // Use the quadratic model to predict the change in F due
                    // to the step D, and set DIFF to the error of this
                    // prediction.
                    let fopt = self.fval[kopt];
                    vquad = zero;
                    let mut ih = 0usize;
                    for j in 1..=n {
                        vquad += self.d[j] * self.gopt[j];
                        for i in 1..=j {
                            ih += 1;
                            let mut temp = self.d[i] * self.d[j];
                            if i == j {
                                temp = half * temp;
                            }
                            vquad += self.hq[ih] * temp;
                        }
                    }
                    for k in 1..=npt {
                        vquad += half * self.pq[k] * sq(self.w[npt + k]);
                    }
                    let diff = f - fopt - vquad;
                    diffc = diffb;
                    diffb = diffa;
                    diffa = diff.abs();
                    if dnorm > rho {
                        nfsav = nf;
                    }

                    // Pick the next value of DELTA after a trust region step.
                    if ntrits > 0 {
                        if vquad >= zero {
                            return Err(
                                "Return from BOBYQA because a trust region step has failed to reduce Q."
                            );
                        }
                        ratio = (f - fopt) / vquad;
                        if ratio <= tenth {
                            delta = (half * delta).min(dnorm);
                        } else if ratio <= 0.7 {
                            delta = (half * delta).max(dnorm);
                        } else {
                            delta = (half * delta).max(dnorm + dnorm);
                        }
                        if delta <= rho * 1.5 {
                            delta = rho;
                        }

                        // Recalculate KNEW and DENOM if the new F is less
                        // than FOPT.
                        if f < fopt {
                            let ksav = knew;
                            let densav = denom;
                            let delsq = delta * delta;
                            let mut scaden = zero;
                            let mut biglsq = zero;
                            knew = 0;
                            for k in 1..=npt {
                                let mut hdiag = zero;
                                for jj in 1..=nptm {
                                    hdiag += sq(a2!(self.zmat, k, jj, npt));
                                }
                                let den = beta * hdiag + sq(self.vlag[k]);
                                let mut distsq2 = zero;
                                for j in 1..=n {
                                    distsq2 += sq(a2!(self.xpt, k, j, npt) - self.xnew[j]);
                                }
                                let temp = one.max(sq(distsq2 / delsq));
                                if temp * den > scaden {
                                    scaden = temp * den;
                                    knew = k;
                                    denom = den;
                                }
                                biglsq = biglsq.max(temp * sq(self.vlag[k]));
                            }
                            if scaden <= half * biglsq {
                                knew = ksav;
                                denom = densav;
                            }
                        }
                    }

                    // Update BMAT and ZMAT, so that the KNEW-th interpolation
                    // point can be moved.  Also update the second derivative
                    // terms of the model.
                    self.update(beta, denom, knew);
                    let mut ih = 0usize;
                    let pqold = self.pq[knew];
                    self.pq[knew] = zero;
                    for i in 1..=n {
                        let temp = pqold * a2!(self.xpt, knew, i, npt);
                        for j in 1..=i {
                            ih += 1;
                            self.hq[ih] += temp * a2!(self.xpt, knew, j, npt);
                        }
                    }
                    for jj in 1..=nptm {
                        let temp = diff * a2!(self.zmat, knew, jj, npt);
                        for k in 1..=npt {
                            self.pq[k] += temp * a2!(self.zmat, k, jj, npt);
                        }
                    }

                    // Include the new interpolation point, and make the
                    // changes to GOPT at the old XOPT that are caused by the
                    // updating of the quadratic model.
                    self.fval[knew] = f;
                    for i in 1..=n {
                        a2!(self.xpt, knew, i, npt) = self.xnew[i];
                        self.w[i] = a2!(self.bmat, knew, i, ndim);
                    }
                    for k in 1..=npt {
                        let mut suma = zero;
                        for jj in 1..=nptm {
                            suma += a2!(self.zmat, knew, jj, npt) * a2!(self.zmat, k, jj, npt);
                        }
                        let mut sumb = zero;
                        for j in 1..=n {
                            sumb += a2!(self.xpt, k, j, npt) * self.xopt[j];
                        }
                        let temp = suma * sumb;
                        for i in 1..=n {
                            self.w[i] += temp * a2!(self.xpt, k, i, npt);
                        }
                    }
                    for i in 1..=n {
                        self.gopt[i] += diff * self.w[i];
                    }

                    // Update XOPT, GOPT and KOPT if the new calculated F is
                    // less than FOPT.
                    if f < fopt {
                        kopt = knew;
                        xoptsq = zero;
                        let mut ih = 0usize;
                        for j in 1..=n {
                            self.xopt[j] = self.xnew[j];
                            xoptsq += self.xopt[j] * self.xopt[j];
                            for i in 1..=j {
                                ih += 1;
                                if i < j {
                                    self.gopt[j] += self.hq[ih] * self.d[i];
                                }
                                self.gopt[i] += self.hq[ih] * self.d[j];
                            }
                        }
                        for k in 1..=npt {
                            let mut temp = zero;
                            for j in 1..=n {
                                temp += a2!(self.xpt, k, j, npt) * self.d[j];
                            }
                            temp = self.pq[k] * temp;
                            for i in 1..=n {
                                self.gopt[i] += temp * a2!(self.xpt, k, i, npt);
                            }
                        }
                    }

                    // Calculate the parameters of the least Frobenius norm
                    // interpolant to the current data, the gradient of this
                    // interpolant at XOPT being put into VLAG(NPT+I).
                    if ntrits > 0 {
                        for k in 1..=npt {
                            self.vlag[k] = self.fval[k] - self.fval[kopt];
                            self.w[k] = zero;
                        }
                        for j in 1..=nptm {
                            let mut sum = zero;
                            for k in 1..=npt {
                                sum += a2!(self.zmat, k, j, npt) * self.vlag[k];
                            }
                            for k in 1..=npt {
                                self.w[k] += sum * a2!(self.zmat, k, j, npt);
                            }
                        }
                        for k in 1..=npt {
                            let mut sum = zero;
                            for j in 1..=n {
                                sum += a2!(self.xpt, k, j, npt) * self.xopt[j];
                            }
                            self.w[k + npt] = self.w[k];
                            self.w[k] = sum * self.w[k];
                        }
                        let mut gqsq = zero;
                        let mut gisq = zero;
                        for i in 1..=n {
                            let mut sum = zero;
                            for k in 1..=npt {
                                sum = sum
                                    + a2!(self.bmat, k, i, ndim) * self.vlag[k]
                                    + a2!(self.xpt, k, i, npt) * self.w[k];
                            }
                            if self.xopt[i] == self.sl[i] {
                                gqsq += sq(zero.min(self.gopt[i]));
                                gisq += sq(zero.min(sum));
                            } else if self.xopt[i] == self.su[i] {
                                gqsq += sq(zero.max(self.gopt[i]));
                                gisq += sq(zero.max(sum));
                            } else {
                                gqsq += sq(self.gopt[i]);
                                gisq += sum * sum;
                            }
                            self.vlag[npt + i] = sum;
                        }

                        // Test whether to replace the new quadratic model by
                        // the least Frobenius norm interpolant, making the
                        // replacement if the test is satisfied.
                        itest += 1;
                        if gqsq < ten * gisq {
                            itest = 0;
                        }
                        if itest >= 3 {
                            for i in 1..=npt.max(self.nh) {
                                if i <= n {
                                    self.gopt[i] = self.vlag[npt + i];
                                }
                                if i <= npt {
                                    self.pq[i] = self.w[npt + i];
                                }
                                if i <= self.nh {
                                    self.hq[i] = zero;
                                }
                                itest = 0;
                            }
                        }
                    }

                    // If a trust region step has provided a sufficient
                    // decrease in F, then branch for another trust region
                    // calculation.
                    if ntrits == 0 {
                        phase = Phase::P60;
                        continue;
                    }
                    if f <= fopt + tenth * vquad {
                        phase = Phase::P60;
                        continue;
                    }

                    // Alternatively, find out if the interpolation points are
                    // close enough to the best point so far.
                    distsq = sq(two * delta).max(sq(ten * rho));
                    phase = Phase::P650;
                }

                Phase::P650 => {
                    knew = 0;
                    for k in 1..=npt {
                        let mut sum = zero;
                        for j in 1..=n {
                            sum += sq(a2!(self.xpt, k, j, npt) - self.xopt[j]);
                        }
                        if sum > distsq {
                            knew = k;
                            distsq = sum;
                        }
                    }

                    // If KNEW is positive, then ALTMOV finds alternative new
                    // positions for the KNEW-th interpolation point within
                    // distance ADELT of XOPT.  Otherwise, there is a branch
                    // to label 60 for another trust region iteration, unless
                    // the calculations with the current RHO are complete.
                    if knew > 0 {
                        let dist = distsq.sqrt();
                        if ntrits == -1 {
                            delta = (tenth * delta).min(half * dist);
                            if delta <= rho * 1.5 {
                                delta = rho;
                            }
                        }
                        ntrits = 0;
                        adelt = (tenth * dist).min(delta).max(rho);
                        dsq = adelt * adelt;
                        phase = Phase::P90;
                        continue;
                    }
                    if ntrits == -1 {
                        phase = Phase::P680;
                        continue;
                    }
                    if ratio > zero {
                        phase = Phase::P60;
                        continue;
                    }
                    if delta.max(dnorm) > rho {
                        phase = Phase::P60;
                        continue;
                    }
                    phase = Phase::P680;
                }

                Phase::P680 => {
                    // The calculations with the current value of RHO are
                    // complete.  Pick the next values of RHO and DELTA.
                    if rho > rhoend {
                        delta = half * rho;
                        ratio = rho / rhoend;
                        if ratio <= 16.0 {
                            rho = rhoend;
                        } else if ratio <= 250.0 {
                            rho = ratio.sqrt() * rhoend;
                        } else {
                            rho = tenth * rho;
                        }
                        delta = delta.max(rho);
                        ntrits = 0;
                        nfsav = nf;
                        phase = Phase::P60;
                        continue;
                    }

                    // Return from the calculation, after another
                    // Newton-Raphson step, if it is too short to have been
                    // tried before.
                    if ntrits == -1 {
                        phase = Phase::P360;
                        continue;
                    }
                    phase = Phase::P720;
                }

                Phase::P720 => {
                    if self.fval[kopt] <= fsave {
                        for i in 1..=n {
                            self.x[i] =
                                self.xl[i].max(self.xbase[i] + self.xopt[i]).min(self.xu[i]);
                            if self.xopt[i] == self.sl[i] {
                                self.x[i] = self.xl[i];
                            }
                            if self.xopt[i] == self.su[i] {
                                self.x[i] = self.xu[i];
                            }
                        }
                        f = self.fval[kopt];
                    }
                    return Ok((f, nf));
                }
            }
        }
    }

    /// Port of `altmov_`: picks two alternative vectors of variables,
    /// relative to XBASE, suitable as new positions of the KNEW-th
    /// interpolation point.  Returns `(alpha, cauchy)`.
    fn altmov(&mut self, kopt: usize, knew: usize, adelt: f64) -> (f64, f64) {
        let n = self.n;
        let npt = self.npt;
        let ndim = self.ndim;
        let half = 0.5;
        let one = 1.0;
        let zero = 0.0;
        let const_ = one + 2.0_f64.sqrt();

        let mut glag = vec![0.0; n + 1];
        let mut hcol = vec![0.0; npt + 1];
        let mut w = vec![0.0; 2 * n + 2];

        // Set the first NPT components of W to the leading elements of the
        // KNEW-th column of the H matrix.
        for k in 1..=npt {
            hcol[k] = zero;
        }
        for j in 1..=npt - n - 1 {
            let temp = a2!(self.zmat, knew, j, npt);
            for k in 1..=npt {
                hcol[k] += temp * a2!(self.zmat, k, j, npt);
            }
        }
        let alpha = hcol[knew];
        let ha = half * alpha;

        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.
        for i in 1..=n {
            glag[i] = a2!(self.bmat, knew, i, ndim);
        }
        for k in 1..=npt {
            let mut temp = zero;
            for j in 1..=n {
                temp += a2!(self.xpt, k, j, npt) * self.xopt[j];
            }
            temp = hcol[k] * temp;
            for i in 1..=n {
                glag[i] += temp * a2!(self.xpt, k, i, npt);
            }
        }

        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point.
        let mut presav = zero;
        let mut ksav = 0usize;
        let mut stpsav = 0.0;
        let mut ibdsav = 0i64;
        for k in 1..=npt {
            if k == kopt {
                continue;
            }
            let mut dderiv = zero;
            let mut distsq = zero;
            for i in 1..=n {
                let temp = a2!(self.xpt, k, i, npt) - self.xopt[i];
                dderiv += glag[i] * temp;
                distsq += temp * temp;
            }
            let mut subd = adelt / distsq.sqrt();
            let mut slbd = -subd;
            let mut ilbd: i64 = 0;
            let mut iubd: i64 = 0;
            let sumin = one.min(subd);

            // Revise SLBD and SUBD if necessary because of the bounds in SL
            // and SU.
            for i in 1..=n {
                let temp = a2!(self.xpt, k, i, npt) - self.xopt[i];
                if temp > zero {
                    if slbd * temp < self.sl[i] - self.xopt[i] {
                        slbd = (self.sl[i] - self.xopt[i]) / temp;
                        ilbd = -(i as i64);
                    }
                    if subd * temp > self.su[i] - self.xopt[i] {
                        subd = sumin.max((self.su[i] - self.xopt[i]) / temp);
                        iubd = i as i64;
                    }
                } else if temp < zero {
                    if slbd * temp > self.su[i] - self.xopt[i] {
                        slbd = (self.su[i] - self.xopt[i]) / temp;
                        ilbd = i as i64;
                    }
                    if subd * temp < self.sl[i] - self.xopt[i] {
                        subd = sumin.max((self.sl[i] - self.xopt[i]) / temp);
                        iubd = -(i as i64);
                    }
                }
            }

            // Seek a large modulus of the KNEW-th Lagrange function when the
            // index of the other interpolation point on the line through XOPT
            // is KNEW.
            let mut step;
            let mut vlag;
            let mut isbd: i64;
            if k == knew {
                let diff = dderiv - one;
                step = slbd;
                vlag = slbd * (dderiv - slbd * diff);
                isbd = ilbd;
                let temp = subd * (dderiv - subd * diff);
                if temp.abs() > vlag.abs() {
                    step = subd;
                    vlag = temp;
                    isbd = iubd;
                }
                let tempd = half * dderiv;
                let tempa = tempd - diff * slbd;
                let tempb = tempd - diff * subd;
                if tempa * tempb < zero {
                    let temp = tempd * tempd / diff;
                    if temp.abs() > vlag.abs() {
                        step = tempd / diff;
                        vlag = temp;
                        isbd = 0;
                    }
                }
            } else {
                // Search along each of the other lines through XOPT and
                // another point.
                step = slbd;
                vlag = slbd * (one - slbd);
                isbd = ilbd;
                let temp = subd * (one - subd);
                if temp.abs() > vlag.abs() {
                    step = subd;
                    vlag = temp;
                    isbd = iubd;
                }
                if subd > half && vlag.abs() < 0.25 {
                    step = half;
                    vlag = 0.25;
                    isbd = 0;
                }
                vlag *= dderiv;
            }

            // Calculate PREDSQ for the current line search and maintain
            // PRESAV.
            let temp = step * (one - step) * distsq;
            let predsq = vlag * vlag * (vlag * vlag + ha * temp * temp);
            if predsq > presav {
                presav = predsq;
                ksav = k;
                stpsav = step;
                ibdsav = isbd;
            }
        }

        // Construct XNEW in a way that satisfies the bound constraints
        // exactly.
        for i in 1..=n {
            // (ksav == 0 cannot occur when altmov is invoked with knew > 0;
            // guard the read to keep the port memory-safe anyway.)
            let xks = if ksav > 0 {
                a2!(self.xpt, ksav, i, npt)
            } else {
                0.0
            };
            let temp = self.xopt[i] + stpsav * (xks - self.xopt[i]);
            self.xnew[i] = self.sl[i].max(self.su[i].min(temp));
        }
        if ibdsav < 0 {
            let j = (-ibdsav) as usize;
            self.xnew[j] = self.sl[j];
        }
        if ibdsav > 0 {
            let j = ibdsav as usize;
            self.xnew[j] = self.su[j];
        }

        // Prepare for the iterative method that assembles the constrained
        // Cauchy step in W.  The sum of squares of the fixed components of W
        // is formed in WFIXSQ, and the free components of W are set to
        // BIGSTP.
        let bigstp = adelt + adelt;
        let mut iflag = 0;
        let mut cauchy = 0.0;
        let mut csave = 0.0;
        'l100: loop {
            let mut wfixsq = zero;
            let mut ggfree = zero;
            for i in 1..=n {
                w[i] = zero;
                let tempa = (self.xopt[i] - self.sl[i]).min(glag[i]);
                let tempb = (self.xopt[i] - self.su[i]).max(glag[i]);
                if tempa > zero || tempb < zero {
                    w[i] = bigstp;
                    ggfree += sq(glag[i]);
                }
            }
            if ggfree == zero {
                cauchy = zero;
                break 'l100; // goto L200
            }

            // Investigate whether more components of W can be fixed.
            let mut step = 0.0;
            loop {
                // L120
                let temp = adelt * adelt - wfixsq;
                if temp > zero {
                    let wsqsav = wfixsq;
                    step = (temp / ggfree).sqrt();
                    ggfree = zero;
                    for i in 1..=n {
                        if w[i] == bigstp {
                            let temp = self.xopt[i] - step * glag[i];
                            if temp <= self.sl[i] {
                                w[i] = self.sl[i] - self.xopt[i];
                                wfixsq += sq(w[i]);
                            } else if temp >= self.su[i] {
                                w[i] = self.su[i] - self.xopt[i];
                                wfixsq += sq(w[i]);
                            } else {
                                ggfree += sq(glag[i]);
                            }
                        }
                    }
                    if wfixsq > wsqsav && ggfree > zero {
                        continue;
                    }
                }
                break;
            }

            // Set the remaining free components of W and all components of
            // XALT, except that W may be scaled later.
            let mut gw = zero;
            for i in 1..=n {
                if w[i] == bigstp {
                    w[i] = -step * glag[i];
                    self.xalt[i] = self.sl[i].max(self.su[i].min(self.xopt[i] + w[i]));
                } else if w[i] == zero {
                    self.xalt[i] = self.xopt[i];
                } else if glag[i] > zero {
                    self.xalt[i] = self.sl[i];
                } else {
                    self.xalt[i] = self.su[i];
                }
                gw += glag[i] * w[i];
            }

            // Set CURV to the curvature of the KNEW-th Lagrange function
            // along W.  Scale W by a factor less than one if that can reduce
            // the modulus of the Lagrange function at XOPT+W.  Set CAUCHY to
            // the final value of the square of this function.
            let mut curv = zero;
            for k in 1..=npt {
                let mut temp = zero;
                for j in 1..=n {
                    temp += a2!(self.xpt, k, j, npt) * w[j];
                }
                curv += hcol[k] * temp * temp;
            }
            if iflag == 1 {
                curv = -curv;
            }
            if curv > -gw && curv < -const_ * gw {
                let scale = -gw / curv;
                for i in 1..=n {
                    let temp = self.xopt[i] + scale * w[i];
                    self.xalt[i] = self.sl[i].max(self.su[i].min(temp));
                }
                cauchy = sq(half * gw * scale);
            } else {
                cauchy = sq(gw + half * curv);
            }

            // If IFLAG is zero, then XALT is calculated as before after
            // reversing the sign of GLAG.  Thus two XALT vectors become
            // available.  The one that is chosen is the one that gives the
            // larger value of CAUCHY.
            if iflag == 0 {
                for i in 1..=n {
                    glag[i] = -glag[i];
                    w[n + i] = self.xalt[i];
                }
                csave = cauchy;
                iflag = 1;
                continue 'l100;
            }
            if csave > cauchy {
                for i in 1..=n {
                    self.xalt[i] = w[n + i];
                }
                cauchy = csave;
            }
            break 'l100;
        }
        (alpha, cauchy)
    }

    /// Port of `prelim_`: sets the elements of XBASE, XPT, FVAL, GOPT, HQ,
    /// PQ, BMAT and ZMAT for the first iteration.  Returns `(nf, kopt)`.
    fn prelim(
        &mut self,
        calfun: CalFun,
        rhobeg: f64,
        maxfun: i64,
    ) -> Result<(i64, usize), &'static str> {
        let n = self.n;
        let npt = self.npt;
        let np = self.np;
        let ndim = self.ndim;
        let half = 0.5;
        let one = 1.0;
        let two = 2.0;
        let zero = 0.0;
        let rhosq = rhobeg * rhobeg;
        let recip = one / rhosq;

        // Set XBASE to the initial vector of variables, and set the initial
        // elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
        for j in 1..=n {
            self.xbase[j] = self.x[j];
            for k in 1..=npt {
                a2!(self.xpt, k, j, npt) = zero;
            }
            for i in 1..=ndim {
                a2!(self.bmat, i, j, ndim) = zero;
            }
        }
        for ih in 1..=n * np / 2 {
            self.hq[ih] = zero;
        }
        for k in 1..=npt {
            self.pq[k] = zero;
            for j in 1..=npt - np {
                a2!(self.zmat, k, j, npt) = zero;
            }
        }

        // Begin the initialization procedure.  NF becomes one more than the
        // number of function values so far.
        let mut nf: i64 = 0;
        let mut kopt = 1usize;
        let mut fbeg = 0.0;
        let mut ipt = 0usize;
        let mut jpt = 0usize;
        let mut stepa = 0.0;
        let mut stepb = 0.0;
        loop {
            // L50
            let nfm = nf as usize;
            let nfx = nfm.saturating_sub(n);
            nf += 1;
            let nfu = nf as usize;
            if nfm <= n << 1 {
                if nfm >= 1 && nfm <= n {
                    stepa = rhobeg;
                    if self.su[nfm] == zero {
                        stepa = -stepa;
                    }
                    a2!(self.xpt, nfu, nfm, npt) = stepa;
                } else if nfm > n {
                    stepa = a2!(self.xpt, nfu - n, nfx, npt);
                    stepb = -rhobeg;
                    if self.sl[nfx] == zero {
                        stepb = (two * rhobeg).min(self.su[nfx]);
                    }
                    if self.su[nfx] == zero {
                        stepb = (-two * rhobeg).max(self.sl[nfx]);
                    }
                    a2!(self.xpt, nfu, nfx, npt) = stepb;
                }
            } else {
                let mut itemp = (nfm - np) / n;
                jpt = nfm - itemp * n - n;
                ipt = jpt + itemp;
                if ipt > n {
                    itemp = jpt;
                    jpt = ipt - n;
                    ipt = itemp;
                }
                a2!(self.xpt, nfu, ipt, npt) = a2!(self.xpt, ipt + 1, ipt, npt);
                a2!(self.xpt, nfu, jpt, npt) = a2!(self.xpt, jpt + 1, jpt, npt);
            }

            // Calculate the next value of F.  The least function value so far
            // and its index are required.
            for j in 1..=n {
                self.x[j] = self.xl[j]
                    .max(self.xbase[j] + a2!(self.xpt, nfu, j, npt))
                    .min(self.xu[j]);
                if a2!(self.xpt, nfu, j, npt) == self.sl[j] {
                    self.x[j] = self.xl[j];
                }
                if a2!(self.xpt, nfu, j, npt) == self.su[j] {
                    self.x[j] = self.xu[j];
                }
            }
            let f = calfun(&self.x[1..=n]);
            self.fval[nfu] = f;
            if nfu == 1 {
                fbeg = f;
                kopt = 1;
            } else if f < self.fval[kopt] {
                kopt = nfu;
            }

            // Set the nonzero initial elements of BMAT and the quadratic
            // model in the cases when NF is at most 2*N+1.
            if nfu <= (n << 1) + 1 {
                if nfu >= 2 && nfu <= n + 1 {
                    self.gopt[nfm] = (f - fbeg) / stepa;
                    if (npt as i64) < nf + n as i64 {
                        a2!(self.bmat, 1, nfm, ndim) = -one / stepa;
                        a2!(self.bmat, nfu, nfm, ndim) = one / stepa;
                        a2!(self.bmat, npt + nfm, nfm, ndim) = -half * rhosq;
                    }
                } else if nfu >= n + 2 {
                    let ih = nfx * (nfx + 1) / 2;
                    let temp = (f - fbeg) / stepb;
                    let diff = stepb - stepa;
                    self.hq[ih] = two * (temp - self.gopt[nfx]) / diff;
                    self.gopt[nfx] = (self.gopt[nfx] * stepb - temp * stepa) / diff;
                    if stepa * stepb < zero && f < self.fval[nfu - n] {
                        self.fval[nfu] = self.fval[nfu - n];
                        self.fval[nfu - n] = f;
                        if kopt == nfu {
                            kopt = nfu - n;
                        }
                        a2!(self.xpt, nfu - n, nfx, npt) = stepb;
                        a2!(self.xpt, nfu, nfx, npt) = stepa;
                    }
                    a2!(self.bmat, 1, nfx, ndim) = -(stepa + stepb) / (stepa * stepb);
                    a2!(self.bmat, nfu, nfx, ndim) = -half / a2!(self.xpt, nfu - n, nfx, npt);
                    a2!(self.bmat, nfu - n, nfx, ndim) =
                        -a2!(self.bmat, 1, nfx, ndim) - a2!(self.bmat, nfu, nfx, ndim);
                    a2!(self.zmat, 1, nfx, npt) = two.sqrt() / (stepa * stepb);
                    a2!(self.zmat, nfu, nfx, npt) = half.sqrt() / rhosq;
                    a2!(self.zmat, nfu - n, nfx, npt) =
                        -a2!(self.zmat, 1, nfx, npt) - a2!(self.zmat, nfu, nfx, npt);
                }
            } else {
                // Set the off-diagonal second derivatives of the Lagrange
                // functions and the initial quadratic model.
                let ih = ipt * (ipt - 1) / 2 + jpt;
                a2!(self.zmat, 1, nfx, npt) = recip;
                a2!(self.zmat, nfu, nfx, npt) = recip;
                a2!(self.zmat, ipt + 1, nfx, npt) = -recip;
                a2!(self.zmat, jpt + 1, nfx, npt) = -recip;
                let temp = a2!(self.xpt, nfu, ipt, npt) * a2!(self.xpt, nfu, jpt, npt);
                self.hq[ih] = (fbeg - self.fval[ipt + 1] - self.fval[jpt + 1] + f) / temp;
            }
            if nf < npt as i64 && nf < maxfun {
                continue;
            }
            break;
        }
        Ok((nf, kopt))
    }

    /// Port of `rescue_`: regenerates BMAT and ZMAT from scratch when
    /// rounding errors have damaged the denominator of the H update.  Sets
    /// `nf = -1` (like the Fortran) if MAXFUN prevents further progress.
    fn rescue(&mut self, calfun: CalFun, maxfun: i64, delta: f64, nf: &mut i64, kopt: &mut usize) {
        let n = self.n;
        let npt = self.npt;
        let np = self.np;
        let ndim = self.ndim;
        let nptm = self.nptm;
        let half = 0.5;
        let one = 1.0;
        let zero = 0.0;
        let sfrac = half / np as f64;

        let mut ptsaux = vec![0.0; 2 * n + 2];
        let mut ptsid = vec![0.0; npt + 1];
        let mut w = vec![0.0; ndim + npt + 1];

        // Shift the interpolation points so that XOPT becomes the origin, and
        // set the elements of ZMAT to zero.  The value of SUMPQ is required in
        // the updating of HQ below.  The squares of the distances from XOPT to
        // the other interpolation points are set at the end of W.  Increments
        // of WINC may be added later to these squares to balance the
        // consideration of the choice of point that is going to become
        // current.
        let mut sumpq = zero;
        let mut winc = zero;
        for k in 1..=npt {
            let mut distsq = zero;
            for j in 1..=n {
                a2!(self.xpt, k, j, npt) -= self.xopt[j];
                distsq += sq(a2!(self.xpt, k, j, npt));
            }
            sumpq += self.pq[k];
            w[ndim + k] = distsq;
            winc = winc.max(distsq);
            for j in 1..=nptm {
                a2!(self.zmat, k, j, npt) = zero;
            }
        }

        // Update HQ so that HQ and PQ define the second derivatives of the
        // model after XBASE has been shifted to the trust region centre.
        let mut ih = 0usize;
        for j in 1..=n {
            w[j] = half * sumpq * self.xopt[j];
            for k in 1..=npt {
                w[j] += self.pq[k] * a2!(self.xpt, k, j, npt);
            }
            for i in 1..=j {
                ih += 1;
                self.hq[ih] = self.hq[ih] + w[i] * self.xopt[j] + w[j] * self.xopt[i];
            }
        }

        // Shift XBASE, SL, SU and XOPT.  Set the elements of BMAT to zero,
        // and also set the elements of PTSAUX.
        for j in 1..=n {
            self.xbase[j] += self.xopt[j];
            self.sl[j] -= self.xopt[j];
            self.su[j] -= self.xopt[j];
            self.xopt[j] = zero;
            p2!(ptsaux, 1, j) = delta.min(self.su[j]);
            p2!(ptsaux, 2, j) = (-delta).max(self.sl[j]);
            if p2!(ptsaux, 1, j) + p2!(ptsaux, 2, j) < zero {
                let temp = p2!(ptsaux, 1, j);
                p2!(ptsaux, 1, j) = p2!(ptsaux, 2, j);
                p2!(ptsaux, 2, j) = temp;
            }
            if p2!(ptsaux, 2, j).abs() < half * p2!(ptsaux, 1, j).abs() {
                p2!(ptsaux, 2, j) = half * p2!(ptsaux, 1, j);
            }
            for i in 1..=ndim {
                a2!(self.bmat, i, j, ndim) = zero;
            }
        }
        let fbase = self.fval[*kopt];

        // Set the identifiers of the artificial interpolation points that are
        // along a coordinate direction from XOPT, and set the corresponding
        // nonzero elements of BMAT and ZMAT.
        ptsid[1] = sfrac;
        for j in 1..=n {
            let jp = j + 1;
            let jpn = jp + n;
            ptsid[jp] = j as f64 + sfrac;
            if jpn <= npt {
                ptsid[jpn] = j as f64 / np as f64 + sfrac;
                let temp = one / (p2!(ptsaux, 1, j) - p2!(ptsaux, 2, j));
                a2!(self.bmat, jp, j, ndim) = -temp + one / p2!(ptsaux, 1, j);
                a2!(self.bmat, jpn, j, ndim) = temp + one / p2!(ptsaux, 2, j);
                a2!(self.bmat, 1, j, ndim) =
                    -a2!(self.bmat, jp, j, ndim) - a2!(self.bmat, jpn, j, ndim);
                a2!(self.zmat, 1, j, npt) =
                    2.0_f64.sqrt() / (p2!(ptsaux, 1, j) * p2!(ptsaux, 2, j)).abs();
                a2!(self.zmat, jp, j, npt) = a2!(self.zmat, 1, j, npt) * p2!(ptsaux, 2, j) * temp;
                a2!(self.zmat, jpn, j, npt) = -a2!(self.zmat, 1, j, npt) * p2!(ptsaux, 1, j) * temp;
            } else {
                a2!(self.bmat, 1, j, ndim) = -one / p2!(ptsaux, 1, j);
                a2!(self.bmat, jp, j, ndim) = one / p2!(ptsaux, 1, j);
                a2!(self.bmat, j + npt, j, ndim) = -half * sq(p2!(ptsaux, 1, j));
            }
        }

        // Set any remaining identifiers with their nonzero elements of ZMAT.
        if npt >= n + np {
            for k in (np << 1)..=npt {
                let iw = (((k - np) as f64 - half) / n as f64) as usize;
                let ip = k - np - iw * n;
                let mut iq = ip + iw;
                if iq > n {
                    iq -= n;
                }
                ptsid[k] = ip as f64 + iq as f64 / np as f64 + sfrac;
                let temp = one / (p2!(ptsaux, 1, ip) * p2!(ptsaux, 1, iq));
                a2!(self.zmat, 1, k - np, npt) = temp;
                a2!(self.zmat, ip + 1, k - np, npt) = -temp;
                a2!(self.zmat, iq + 1, k - np, npt) = -temp;
                a2!(self.zmat, k, k - np, npt) = temp;
            }
        }
        let mut nrem = npt;
        let mut kold = 1usize;
        let mut knew = *kopt;
        let mut beta = 0.0;
        let mut denom = 0.0;

        // Reorder the provisional points in the way that exchanges
        // PTSID(KOLD) with PTSID(KNEW).
        'l80: loop {
            for j in 1..=n {
                let temp = a2!(self.bmat, kold, j, ndim);
                a2!(self.bmat, kold, j, ndim) = a2!(self.bmat, knew, j, ndim);
                a2!(self.bmat, knew, j, ndim) = temp;
            }
            for j in 1..=nptm {
                let temp = a2!(self.zmat, kold, j, npt);
                a2!(self.zmat, kold, j, npt) = a2!(self.zmat, knew, j, npt);
                a2!(self.zmat, knew, j, npt) = temp;
            }
            ptsid[kold] = ptsid[knew];
            ptsid[knew] = zero;
            w[ndim + knew] = zero;
            nrem -= 1;
            if knew != *kopt {
                let temp = self.vlag[kold];
                self.vlag[kold] = self.vlag[knew];
                self.vlag[knew] = temp;

                // Update the BMAT and ZMAT matrices so that the status of the
                // KNEW-th interpolation point can be changed from provisional
                // to original.  The nonnegative values of W(NDIM+K) are
                // required in the search below.
                self.update(beta, denom, knew);
                if nrem == 0 {
                    break 'l80; // goto L350
                }
                for k in 1..=npt {
                    w[ndim + k] = w[ndim + k].abs();
                }
            }

            // Pick the index KNEW of an original interpolation point that has
            // not yet replaced one of the provisional interpolation points.
            'l120: loop {
                let mut dsqmin = zero;
                for k in 1..=npt {
                    if w[ndim + k] > zero && (dsqmin == zero || w[ndim + k] < dsqmin) {
                        knew = k;
                        dsqmin = w[ndim + k];
                    }
                }
                if dsqmin == zero {
                    break 'l80; // goto L260
                }

                // Form the W-vector of the chosen original interpolation
                // point.
                for j in 1..=n {
                    w[npt + j] = a2!(self.xpt, knew, j, npt);
                }
                for k in 1..=npt {
                    let mut sum = zero;
                    if k == *kopt {
                        // nothing
                    } else if ptsid[k] == zero {
                        for j in 1..=n {
                            sum += w[npt + j] * a2!(self.xpt, k, j, npt);
                        }
                    } else {
                        let ip = ptsid[k] as i64 as usize;
                        if ip > 0 {
                            sum = w[npt + ip] * p2!(ptsaux, 1, ip);
                        }
                        let iq = ((np as f64) * ptsid[k] - (ip * np) as f64) as i64 as usize;
                        if iq > 0 {
                            let mut iw = 1;
                            if ip == 0 {
                                iw = 2;
                            }
                            sum += w[npt + iq] * p2!(ptsaux, iw, iq);
                        }
                    }
                    w[k] = half * sum * sum;
                }

                // Calculate VLAG and BETA for the required updating of the H
                // matrix if XPT(KNEW,.) is reinstated in the set of
                // interpolation points.
                for k in 1..=npt {
                    let mut sum = zero;
                    for j in 1..=n {
                        sum += a2!(self.bmat, k, j, ndim) * w[npt + j];
                    }
                    self.vlag[k] = sum;
                }
                beta = zero;
                for j in 1..=nptm {
                    let mut sum = zero;
                    for k in 1..=npt {
                        sum += a2!(self.zmat, k, j, npt) * w[k];
                    }
                    beta -= sum * sum;
                    for k in 1..=npt {
                        self.vlag[k] += sum * a2!(self.zmat, k, j, npt);
                    }
                }
                let mut bsum = zero;
                let mut distsq = zero;
                for j in 1..=n {
                    let mut sum = zero;
                    for k in 1..=npt {
                        sum += a2!(self.bmat, k, j, ndim) * w[k];
                    }
                    let jp = j + npt;
                    bsum += sum * w[jp];
                    for ip in npt + 1..=ndim {
                        sum += a2!(self.bmat, ip, j, ndim) * w[ip];
                    }
                    bsum += sum * w[jp];
                    self.vlag[jp] = sum;
                    distsq += sq(a2!(self.xpt, knew, j, npt));
                }
                beta = half * distsq * distsq + beta - bsum;
                self.vlag[*kopt] += one;

                // KOLD is set to the index of the provisional interpolation
                // point that is going to be deleted to make way for the
                // KNEW-th original interpolation point.
                denom = zero;
                let mut vlmxsq = zero;
                for k in 1..=npt {
                    if ptsid[k] != zero {
                        let mut hdiag = zero;
                        for j in 1..=nptm {
                            hdiag += sq(a2!(self.zmat, k, j, npt));
                        }
                        let den = beta * hdiag + sq(self.vlag[k]);
                        if den > denom {
                            kold = k;
                            denom = den;
                        }
                    }
                    vlmxsq = vlmxsq.max(sq(self.vlag[k]));
                }
                if denom <= vlmxsq * 0.01 {
                    w[ndim + knew] = -w[ndim + knew] - winc;
                    continue 'l120;
                }
                continue 'l80; // goto L80
            }
        }

        // L260: when this label is reached, all the final positions of the
        // interpolation points have been chosen although any changes have not
        // been included yet in XPT.  The following cycle through the new
        // interpolation points begins by putting the new point in XPT(KPT,.)
        // and by setting PQ(KPT) to zero, except that a RETURN occurs if
        // MAXFUN prohibits another value of F.
        for kpt in 1..=npt {
            if ptsid[kpt] == zero {
                continue;
            }
            if *nf >= maxfun {
                *nf = -1;
                break;
            }
            let mut ih = 0usize;
            for j in 1..=n {
                w[j] = a2!(self.xpt, kpt, j, npt);
                a2!(self.xpt, kpt, j, npt) = zero;
                let temp = self.pq[kpt] * w[j];
                for i in 1..=j {
                    ih += 1;
                    self.hq[ih] += temp * w[i];
                }
            }
            self.pq[kpt] = zero;
            let ip = ptsid[kpt] as i64 as usize;
            let iq = ((np as f64) * ptsid[kpt] - (ip * np) as f64) as i64 as usize;
            let mut xp = 0.0;
            let mut xq = 0.0;
            if ip > 0 {
                xp = p2!(ptsaux, 1, ip);
                a2!(self.xpt, kpt, ip, npt) = xp;
            }
            if iq > 0 {
                xq = p2!(ptsaux, 1, iq);
                if ip == 0 {
                    xq = p2!(ptsaux, 2, iq);
                }
                a2!(self.xpt, kpt, iq, npt) = xq;
            }

            // Set VQUAD to the value of the current model at the new point.
            let mut vquad = fbase;
            let mut ihp = 0usize;
            if ip > 0 {
                ihp = (ip + ip * ip) / 2;
                vquad += xp * (self.gopt[ip] + half * xp * self.hq[ihp]);
            }
            if iq > 0 {
                let ihq = (iq + iq * iq) / 2;
                vquad += xq * (self.gopt[iq] + half * xq * self.hq[ihq]);
                if ip > 0 {
                    let iw = ihp.max(ihq) - (ip as i64 - iq as i64).unsigned_abs() as usize;
                    vquad += xp * xq * self.hq[iw];
                }
            }
            for k in 1..=npt {
                let mut temp = zero;
                if ip > 0 {
                    temp += xp * a2!(self.xpt, k, ip, npt);
                }
                if iq > 0 {
                    temp += xq * a2!(self.xpt, k, iq, npt);
                }
                vquad += half * self.pq[k] * temp * temp;
            }

            // Calculate F at the new interpolation point, and set DIFF to the
            // factor that is going to multiply the KPT-th Lagrange function
            // when the model is updated to provide interpolation to the new
            // function value.
            for i in 1..=n {
                w[i] = self.xl[i]
                    .max(self.xbase[i] + a2!(self.xpt, kpt, i, npt))
                    .min(self.xu[i]);
                if a2!(self.xpt, kpt, i, npt) == self.sl[i] {
                    w[i] = self.xl[i];
                }
                if a2!(self.xpt, kpt, i, npt) == self.su[i] {
                    w[i] = self.xu[i];
                }
            }
            *nf += 1;
            let f = calfun(&w[1..=n]);
            self.fval[kpt] = f;
            if f < self.fval[*kopt] {
                *kopt = kpt;
            }
            let diff = f - vquad;

            // Update the quadratic model.
            for i in 1..=n {
                self.gopt[i] += diff * a2!(self.bmat, kpt, i, ndim);
            }
            for k in 1..=npt {
                let mut sum = zero;
                for j in 1..=nptm {
                    sum += a2!(self.zmat, k, j, npt) * a2!(self.zmat, kpt, j, npt);
                }
                let temp = diff * sum;
                if ptsid[k] == zero {
                    self.pq[k] += temp;
                } else {
                    let ip = ptsid[k] as i64 as usize;
                    let iq = ((np as f64) * ptsid[k] - (ip * np) as f64) as i64 as usize;
                    let ihq = (iq * iq + iq) / 2;
                    if ip == 0 {
                        self.hq[ihq] += temp * sq(p2!(ptsaux, 2, iq));
                    } else {
                        let ihp = (ip * ip + ip) / 2;
                        self.hq[ihp] += temp * sq(p2!(ptsaux, 1, ip));
                        if iq > 0 {
                            self.hq[ihq] += temp * sq(p2!(ptsaux, 1, iq));
                            let iw = ihp.max(ihq) - (iq as i64 - ip as i64).unsigned_abs() as usize;
                            self.hq[iw] += temp * p2!(ptsaux, 1, ip) * p2!(ptsaux, 1, iq);
                        }
                    }
                }
            }
            ptsid[kpt] = zero;
        }
        // L350
    }

    /// Port of `trsbox_`: finds the vector XNEW inside the trust region
    /// (radius `delta`) and the bounds that minimizes the quadratic model.
    /// Returns `(dsq, crvmin)`.
    fn trsbox(&mut self, delta: f64) -> (f64, f64) {
        let n = self.n;
        let half = 0.5;
        let one = 1.0;
        let onemin = -1.0;
        let zero = 0.0;

        let mut gnew = vec![0.0; n + 1];
        let mut xbdi = vec![0.0; n + 1];
        let mut s = vec![0.0; n + 1];
        let mut hs = vec![0.0; n + 1];
        let mut hred = vec![0.0; n + 1];

        // The sign of GOPT(I) gives the sign of the change to the I-th
        // variable that will reduce Q from its value at XOPT.  Thus XBDI(I)
        // shows whether or not to fix the I-th variable at one of its bounds
        // initially, with NACT being set to the number of fixed variables.
        let mut iterc = 0usize;
        let mut nact = 0usize;
        for i in 1..=n {
            xbdi[i] = zero;
            if self.xopt[i] <= self.sl[i] {
                if self.gopt[i] >= zero {
                    xbdi[i] = onemin;
                }
            } else if self.xopt[i] >= self.su[i] {
                if self.gopt[i] <= zero {
                    xbdi[i] = one;
                }
            }
            if xbdi[i] != zero {
                nact += 1;
            }
            self.d[i] = zero;
            gnew[i] = self.gopt[i];
        }
        let mut delsq = delta * delta;
        let mut qred = zero;
        let mut crvmin = onemin;

        let mut itermax = 0usize;
        let mut ggsav = 0.0;
        let mut gredsq = 0.0;
        let mut dredsq = 0.0;
        let mut dredg = 0.0;
        let mut sredg = 0.0;
        let mut _itcsav = 0usize;

        'l20: loop {
            // Set the next search direction of the conjugate gradient method.
            let mut beta = zero;
            'l30: loop {
                let mut stepsq = zero;
                for i in 1..=n {
                    if xbdi[i] != zero {
                        s[i] = zero;
                    } else if beta == zero {
                        s[i] = -gnew[i];
                    } else {
                        s[i] = beta * s[i] - gnew[i];
                    }
                    stepsq += sq(s[i]);
                }
                if stepsq == zero {
                    break 'l20;
                }
                if beta == zero {
                    gredsq = stepsq;
                    itermax = iterc + n - nact;
                }
                if gredsq * delsq <= qred * 1e-4 * qred {
                    break 'l20;
                }

                // Multiply the search direction by the second derivative
                // matrix of Q (L210) and calculate some scalars for the choice
                // of steplength.
                self.hsmul(&s, &mut hs);
                // L50
                let mut resid = delsq;
                let mut ds = zero;
                let mut shs = zero;
                for i in 1..=n {
                    if xbdi[i] == zero {
                        resid -= sq(self.d[i]);
                        ds += s[i] * self.d[i];
                        shs += s[i] * hs[i];
                    }
                }
                if resid > zero {
                    let temp = (stepsq * resid + ds * ds).sqrt();
                    let blen = if ds < zero {
                        (temp - ds) / stepsq
                    } else {
                        resid / (temp + ds)
                    };
                    let mut stplen = blen;
                    if shs > zero {
                        stplen = blen.min(gredsq / shs);
                    }

                    // Reduce STPLEN if necessary in order to preserve the
                    // simple bounds, letting IACT be the index of the new
                    // constrained variable.
                    let mut iact = 0usize;
                    for i in 1..=n {
                        if s[i] != zero {
                            let xsum = self.xopt[i] + self.d[i];
                            let temp = if s[i] > zero {
                                (self.su[i] - xsum) / s[i]
                            } else {
                                (self.sl[i] - xsum) / s[i]
                            };
                            if temp < stplen {
                                stplen = temp;
                                iact = i;
                            }
                        }
                    }

                    // Update CRVMIN, GNEW and D.  Set SDEC to the decrease
                    // that occurs in Q.
                    let mut sdec = zero;
                    if stplen > zero {
                        iterc += 1;
                        let temp = shs / stepsq;
                        if iact == 0 && temp > zero {
                            crvmin = crvmin.min(temp);
                            if crvmin == onemin {
                                crvmin = temp;
                            }
                        }
                        ggsav = gredsq;
                        gredsq = zero;
                        for i in 1..=n {
                            gnew[i] += stplen * hs[i];
                            if xbdi[i] == zero {
                                gredsq += sq(gnew[i]);
                            }
                            self.d[i] += stplen * s[i];
                        }
                        sdec = (stplen * (ggsav - half * stplen * shs)).max(zero);
                        qred += sdec;
                    }

                    // Restart the conjugate gradient method if it has hit a
                    // new bound.
                    if iact > 0 {
                        nact += 1;
                        xbdi[iact] = one;
                        if s[iact] < zero {
                            xbdi[iact] = onemin;
                        }
                        delsq -= sq(self.d[iact]);
                        if delsq > zero {
                            continue 'l20;
                        }
                    } else if stplen < blen {
                        // If STPLEN is less than BLEN, then either apply
                        // another conjugate gradient iteration or RETURN.
                        if iterc == itermax {
                            break 'l20;
                        }
                        if sdec <= qred * 0.01 {
                            break 'l20;
                        }
                        beta = gredsq / ggsav;
                        continue 'l30;
                    }
                }

                // L90: prepare for the alternative iteration by calculating
                // some scalars and by multiplying the reduced D by the second
                // derivative matrix of Q.
                crvmin = zero;
                'l100: loop {
                    if nact >= n - 1 {
                        break 'l20;
                    }
                    dredsq = zero;
                    dredg = zero;
                    gredsq = zero;
                    for i in 1..=n {
                        if xbdi[i] == zero {
                            dredsq += sq(self.d[i]);
                            dredg += self.d[i] * gnew[i];
                            gredsq += sq(gnew[i]);
                            s[i] = self.d[i];
                        } else {
                            s[i] = zero;
                        }
                    }
                    _itcsav = iterc;
                    self.hsmul(&s, &mut hs);
                    // (crvmin is zero here and iterc == itcsav, so the
                    // Fortran falls through to: HRED = HS, goto L120.)
                    for i in 1..=n {
                        hred[i] = hs[i];
                    }
                    'l120: loop {
                        // Let the search direction S be a linear combination
                        // of the reduced D and the reduced G that is
                        // orthogonal to the reduced D.
                        iterc += 1;
                        let mut temp = gredsq * dredsq - dredg * dredg;
                        if temp <= qred * 1e-4 * qred {
                            break 'l20;
                        }
                        temp = temp.sqrt();
                        for i in 1..=n {
                            if xbdi[i] == zero {
                                s[i] = (dredg * self.d[i] - dredsq * gnew[i]) / temp;
                            } else {
                                s[i] = zero;
                            }
                        }
                        sredg = -temp;

                        // By considering the simple bounds on the variables,
                        // calculate an upper bound on the tangent of half the
                        // angle of the alternative iteration, namely ANGBD.
                        let mut angbd = one;
                        let mut iact = 0usize;
                        let mut xsav = 0.0;
                        for i in 1..=n {
                            if xbdi[i] == zero {
                                let tempa = self.xopt[i] + self.d[i] - self.sl[i];
                                let tempb = self.su[i] - self.xopt[i] - self.d[i];
                                if tempa <= zero {
                                    nact += 1;
                                    xbdi[i] = onemin;
                                    continue 'l100;
                                } else if tempb <= zero {
                                    nact += 1;
                                    xbdi[i] = one;
                                    continue 'l100;
                                }
                                let ssq = sq(self.d[i]) + sq(s[i]);
                                temp = ssq - sq(self.xopt[i] - self.sl[i]);
                                if temp > zero {
                                    temp = temp.sqrt() - s[i];
                                    if angbd * temp > tempa {
                                        angbd = tempa / temp;
                                        iact = i;
                                        xsav = onemin;
                                    }
                                }
                                temp = ssq - sq(self.su[i] - self.xopt[i]);
                                if temp > zero {
                                    temp = temp.sqrt() + s[i];
                                    if angbd * temp > tempb {
                                        angbd = tempb / temp;
                                        iact = i;
                                        xsav = one;
                                    }
                                }
                            }
                        }

                        // Calculate HHD and some curvatures for the
                        // alternative iteration (L210 then L150).
                        self.hsmul(&s, &mut hs);
                        let mut shs = zero;
                        let mut dhs = zero;
                        let mut dhd = zero;
                        for i in 1..=n {
                            if xbdi[i] == zero {
                                shs += s[i] * hs[i];
                                dhs += self.d[i] * hs[i];
                                dhd += self.d[i] * hred[i];
                            }
                        }

                        // Seek the greatest reduction in Q for a range of
                        // equally spaced values of ANGT in [0,ANGBD].
                        let mut redmax = zero;
                        let mut isav = 0usize;
                        let mut redsav = zero;
                        let mut rdprev = 0.0;
                        let mut rdnext = 0.0;
                        let iu = (angbd * 17.0 + 3.1) as usize;
                        let mut angt = 0.0;
                        for i in 1..=iu {
                            angt = angbd * i as f64 / iu as f64;
                            let sth = (angt + angt) / (one + angt * angt);
                            let temp = shs + angt * (angt * dhd - dhs - dhs);
                            let rednew = sth * (angt * dredg - sredg - half * sth * temp);
                            if rednew > redmax {
                                redmax = rednew;
                                isav = i;
                                rdprev = redsav;
                            } else if i == isav + 1 {
                                rdnext = rednew;
                            }
                            redsav = rednew;
                        }

                        // Return if the reduction is zero.  Otherwise, set
                        // the sine and cosine of the angle of the alternative
                        // iteration, and calculate SDEC.
                        if isav == 0 {
                            break 'l20;
                        }
                        if isav < iu {
                            let temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
                            angt = angbd * (isav as f64 + half * temp) / iu as f64;
                        }
                        let cth = (one - angt * angt) / (one + angt * angt);
                        let sth = (angt + angt) / (one + angt * angt);
                        let temp = shs + angt * (angt * dhd - dhs - dhs);
                        let sdec = sth * (angt * dredg - sredg - half * sth * temp);
                        if sdec <= zero {
                            break 'l20;
                        }

                        // Update GNEW, D and HRED.  If the angle of the
                        // alternative iteration is restricted by a bound on a
                        // free variable, that variable is fixed at the bound.
                        dredg = zero;
                        gredsq = zero;
                        for i in 1..=n {
                            gnew[i] = gnew[i] + (cth - one) * hred[i] + sth * hs[i];
                            if xbdi[i] == zero {
                                self.d[i] = cth * self.d[i] + sth * s[i];
                                dredg += self.d[i] * gnew[i];
                                gredsq += sq(gnew[i]);
                            }
                            hred[i] = cth * hred[i] + sth * hs[i];
                        }
                        qred += sdec;
                        if iact > 0 && isav == iu {
                            nact += 1;
                            xbdi[iact] = xsav;
                            continue 'l100;
                        }

                        // If SDEC is sufficiently small, then RETURN after
                        // setting XNEW to XOPT+D.
                        if sdec > qred * 0.01 {
                            continue 'l120;
                        }
                        break 'l20;
                    }
                }
            }
        }

        // L190: set XNEW to XOPT+D, giving careful attention to the bounds.
        let mut dsq = zero;
        for i in 1..=n {
            self.xnew[i] = (self.xopt[i] + self.d[i]).min(self.su[i]).max(self.sl[i]);
            if xbdi[i] == onemin {
                self.xnew[i] = self.sl[i];
            }
            if xbdi[i] == one {
                self.xnew[i] = self.su[i];
            }
            self.d[i] = self.xnew[i] - self.xopt[i];
            dsq += sq(self.d[i]);
        }
        (dsq, crvmin)
    }

    /// The L210 block of `trsbox_`: multiplies the current S-vector by the
    /// second derivative matrix of the quadratic model, putting the product
    /// in HS.
    fn hsmul(&self, s: &[f64], hs: &mut [f64]) {
        let n = self.n;
        let npt = self.npt;
        let zero = 0.0;
        let mut ih = 0usize;
        for j in 1..=n {
            hs[j] = zero;
            for i in 1..=j {
                ih += 1;
                if i < j {
                    hs[j] += self.hq[ih] * s[i];
                }
                hs[i] += self.hq[ih] * s[j];
            }
        }
        for k in 1..=npt {
            if self.pq[k] != zero {
                let mut temp = zero;
                for j in 1..=n {
                    temp += a2!(self.xpt, k, j, npt) * s[j];
                }
                temp *= self.pq[k];
                for i in 1..=n {
                    hs[i] += temp * a2!(self.xpt, k, i, npt);
                }
            }
        }
    }

    /// Port of `update_`: updates BMAT and ZMAT as required by the new
    /// position of the interpolation point with index KNEW.
    fn update(&mut self, beta: f64, denom: f64, knew: usize) {
        let n = self.n;
        let npt = self.npt;
        let ndim = self.ndim;
        let one = 1.0;
        let zero = 0.0;
        let nptm = npt - n - 1;
        let mut w = vec![0.0; ndim + 1];

        let mut ztest: f64 = zero;
        for k in 1..=npt {
            for j in 1..=nptm {
                ztest = ztest.max(a2!(self.zmat, k, j, npt).abs());
            }
        }
        ztest *= 1e-20;

        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.
        for j in 2..=nptm {
            if a2!(self.zmat, knew, j, npt).abs() > ztest {
                let temp =
                    (sq(a2!(self.zmat, knew, 1, npt)) + sq(a2!(self.zmat, knew, j, npt))).sqrt();
                let tempa = a2!(self.zmat, knew, 1, npt) / temp;
                let tempb = a2!(self.zmat, knew, j, npt) / temp;
                for i in 1..=npt {
                    let temp =
                        tempa * a2!(self.zmat, i, 1, npt) + tempb * a2!(self.zmat, i, j, npt);
                    a2!(self.zmat, i, j, npt) =
                        tempa * a2!(self.zmat, i, j, npt) - tempb * a2!(self.zmat, i, 1, npt);
                    a2!(self.zmat, i, 1, npt) = temp;
                }
            }
            a2!(self.zmat, knew, j, npt) = zero;
        }

        // Put the first NPT components of the KNEW-th column of HLAG into W,
        // and calculate the parameters of the updating formula.
        for i in 1..=npt {
            w[i] = a2!(self.zmat, knew, 1, npt) * a2!(self.zmat, i, 1, npt);
        }
        let alpha = w[knew];
        let tau = self.vlag[knew];
        self.vlag[knew] -= one;

        // Complete the updating of ZMAT.
        let temp = denom.sqrt();
        let tempb = a2!(self.zmat, knew, 1, npt) / temp;
        let tempa = tau / temp;
        for i in 1..=npt {
            a2!(self.zmat, i, 1, npt) = tempa * a2!(self.zmat, i, 1, npt) - tempb * self.vlag[i];
        }

        // Finally, update the matrix BMAT.
        for j in 1..=n {
            let jp = npt + j;
            w[jp] = a2!(self.bmat, knew, j, ndim);
            let tempa = (alpha * self.vlag[jp] - tau * w[jp]) / denom;
            let tempb = (-beta * w[jp] - tau * self.vlag[jp]) / denom;
            for i in 1..=jp {
                a2!(self.bmat, i, j, ndim) =
                    a2!(self.bmat, i, j, ndim) + tempa * self.vlag[i] + tempb * w[i];
                if i > npt {
                    a2!(self.bmat, jp, i - npt, ndim) = a2!(self.bmat, i, j, ndim);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &Matrix<f64>) -> f64 {
        sq(1.0 - x[(0, 0)]) + 100.0 * sq(x[(1, 0)] - sq(x[(0, 0)]))
    }

    fn sphere(x: &Matrix<f64>) -> f64 {
        (0..x.nr()).map(|i| sq(x[(i, 0)])).sum()
    }

    #[test]
    fn rosenbrock_2d() {
        let mut x = Matrix::from_row_vec(2, 1, &[-1.2, 1.0]);
        let (fmin, count) = find_min_bobyqa(rosenbrock, &mut x, 1.0).expect("rosenbrock");
        assert!(
            (x[(0, 0)] - 1.0).abs() < 1e-4 && (x[(1, 0)] - 1.0).abs() < 1e-4,
            "x = {:?}",
            (x[(0, 0)], x[(1, 0)])
        );
        assert!(fmin < 1e-6, "fmin = {fmin}");
        assert!(count > 0);
    }

    #[test]
    fn sphere_4d() {
        let mut x = Matrix::from_row_vec(4, 1, &[1.0, 2.0, 3.0, 4.0]);
        let (fmin, count) = find_min_bobyqa(sphere, &mut x, 1.0).expect("sphere");
        assert!(fmin < 1e-8, "fmin = {fmin}");
        for i in 0..4 {
            assert!(x[(i, 0)].abs() < 1e-4, "x[{i}] = {}", x[(i, 0)]);
        }
        assert!(count > 0);
    }

    #[test]
    fn bounded_problem() {
        let target = [0.3, 0.7];
        let f = |x: &Matrix<f64>| sq(x[(0, 0)] - target[0]) + sq(x[(1, 0)] - target[1]);
        let mut x = Matrix::from_row_vec(2, 1, &[0.5, 0.5]);
        let lower = Matrix::from_row_vec(2, 1, &[0.0, 0.0]);
        let upper = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        let (fmin, count) =
            find_min_bobyqa_with_bounds(f, &mut x, 0.1, &lower, &upper).expect("bounded");
        assert!(
            (x[(0, 0)] - target[0]).abs() < 1e-4 && (x[(1, 0)] - target[1]).abs() < 1e-4,
            "x = {:?}",
            (x[(0, 0)], x[(1, 0)])
        );
        for i in 0..2 {
            assert!(x[(i, 0)] >= 0.0 && x[(i, 0)] <= 1.0);
        }
        assert!(fmin < 1e-8, "fmin = {fmin}");
        assert!(count > 0);
    }

    #[test]
    fn bounded_problem_active_constraint() {
        // Optimum at (0, 0) is inside bounds; optimum of f = |x - (-1, 2)|^2 is
        // clamped to (0, 1).
        let f = |x: &Matrix<f64>| sq(x[(0, 0)] + 1.0) + sq(x[(1, 0)] - 2.0);
        let mut x = Matrix::from_row_vec(2, 1, &[0.5, 0.5]);
        let lower = Matrix::from_row_vec(2, 1, &[0.0, 0.0]);
        let upper = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        let (fmin, _) =
            find_min_bobyqa_with_bounds(f, &mut x, 0.1, &lower, &upper).expect("bounded");
        assert!(
            (x[(0, 0)] - 0.0).abs() < 1e-6 && (x[(1, 0)] - 1.0).abs() < 1e-6,
            "x = {:?}",
            (x[(0, 0)], x[(1, 0)])
        );
        assert!((fmin - (1.0 + 1.0)).abs() < 1e-8, "fmin = {fmin}");
    }

    #[test]
    fn pathological_inputs() {
        let f = |x: &Matrix<f64>| sphere(x);
        let mut x0 = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        assert!(find_min_bobyqa(f, &mut x0, 1.0).is_ok());

        // n = 0
        let mut empty = Matrix::zeros(0, 1);
        assert!(find_min_bobyqa(f, &mut empty, 1.0).is_err());

        // radius <= 0
        let mut x = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        assert!(find_min_bobyqa(f, &mut x, 0.0).is_err());
        assert!(find_min_bobyqa(f, &mut x, -1.0).is_err());

        // bounds tighter than 2*rho_begin
        let mut xb = Matrix::from_row_vec(2, 1, &[0.5, 0.5]);
        let lower = Matrix::from_row_vec(2, 1, &[0.0, 0.0]);
        let upper = Matrix::from_row_vec(2, 1, &[1.0, 1.0]);
        assert!(find_min_bobyqa_with_bounds(f, &mut xb, 0.6, &lower, &upper).is_err());

        // starting point outside the bounds
        let mut xo = Matrix::from_row_vec(2, 1, &[2.0, 0.5]);
        assert!(find_min_bobyqa_with_bounds(f, &mut xo, 0.1, &lower, &upper).is_err());
    }
}

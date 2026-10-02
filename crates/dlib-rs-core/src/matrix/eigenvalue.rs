//! Eigenvalue decomposition, ported line-by-line from
//! `dlib/matrix/matrix_eigenvalue.h` (adapted from the JAMA part of NIST's
//! TNT library).
//!
//! Both dlib code paths are ported: the symmetric fast path (Householder
//! tridiagonalization `tred2` + implicit QL `tql2`) and the general
//! nonsymmetric path (Hessenberg reduction `orthes` + `hqr2`).

use crate::matrix::Matrix;

/// Port of `dlib::eigenvalue_decomposition<double>` from
/// `dlib/matrix/matrix_eigenvalue.h`.
///
/// Since this crate has no complex matrix type, `get_v()` returns the real
/// pseudo-eigenvector matrix (dlib's `get_pseudo_v()`, which for symmetric
/// input is the ordinary real eigenvector matrix) and `get_d()` returns the
/// real block-diagonal pseudo-eigenvalue matrix (dlib's `get_pseudo_d()`,
/// with 2x2 blocks for complex-conjugate pairs).
pub struct EigenvalueDecomposition {
    /// Row and column dimension (square matrix); port of `n`.
    n: usize,
    /// Real parts of the eigenvalues; port of `d`.
    d: Vec<f64>,
    /// Imaginary parts of the eigenvalues; port of `e`.
    e: Vec<f64>,
    /// Eigenvector storage; port of `V`.
    v: Matrix<f64>,
    /// Nonsymmetric Hessenberg form; port of `H`.
    h: Matrix<f64>,
    /// Working storage for the nonsymmetric algorithm; port of `ort`.
    ort: Vec<f64>,
}

impl EigenvalueDecomposition {
    /// Port of the `eigenvalue_decomposition(const matrix_exp&)`
    /// constructor: uses the symmetric path when `A(i,j) == A(j,i)` exactly,
    /// otherwise the Hessenberg/QR path.
    ///
    /// # Panics
    /// Panics if `a` is not a non-empty square matrix (dlib `DLIB_ASSERT`).
    pub fn new(a: &Matrix<f64>) -> Self {
        assert!(
            a.nr() == a.nc() && a.size() > 0,
            "eigenvalue_decomposition::new(A): A must be a non-empty square matrix"
        );

        let n = a.nc();
        let mut issymmetric = true;
        'outer: for j in 0..n {
            for i in 0..n {
                if a[(i, j)] != a[(j, i)] {
                    issymmetric = false;
                    break 'outer;
                }
            }
        }

        let mut this = EigenvalueDecomposition {
            n,
            d: vec![0.0; n],
            e: vec![0.0; n],
            v: Matrix::zeros(n, n),
            h: Matrix::zeros(n, n),
            ort: vec![0.0; n],
        };

        if issymmetric {
            this.v = a.clone();
            // Tridiagonalize.
            this.tred2();
            // Diagonalize.
            this.tql2();
        } else {
            this.h = a.clone();
            // Reduce to Hessenberg form.
            this.orthes();
            // Reduce Hessenberg to real Schur form.
            this.hqr2();
        }
        this
    }

    /// Port of `dim()`.
    pub fn dim(&self) -> usize {
        self.v.nr()
    }

    /// Port of `get_real_eigenvalues()`: real parts, as an `n x 1` matrix.
    pub fn get_real_eigenvalues(&self) -> Matrix<f64> {
        Matrix::from_vec(self.n, 1, self.d.clone()).unwrap()
    }

    /// Port of `get_imag_eigenvalues()`: imaginary parts, as an `n x 1`
    /// matrix.
    pub fn get_imag_eigenvalues(&self) -> Matrix<f64> {
        Matrix::from_vec(self.n, 1, self.e.clone()).unwrap()
    }

    /// Port of `get_pseudo_v()` (real Schur vectors / eigenvectors).
    pub fn get_pseudo_v(&self) -> &Matrix<f64> {
        &self.v
    }

    /// Real eigenvector matrix; alias of [`Self::get_pseudo_v`] standing in
    /// for dlib's complex `get_v()`.
    pub fn get_v(&self) -> &Matrix<f64> {
        &self.v
    }

    /// Port of `get_pseudo_d()`: block-diagonal matrix whose diagonal is
    /// `d` and whose 2x2 blocks carry the complex-conjugate pairs from `e`.
    pub fn get_pseudo_d(&self) -> Matrix<f64> {
        let n = self.n;
        let mut dm = Matrix::zeros(n, n);
        for i in 0..n {
            dm[(i, i)] = self.d[i];
            if self.e[i] > 0.0 {
                dm[(i, i + 1)] = self.e[i];
            } else if self.e[i] < 0.0 {
                dm[(i, i - 1)] = self.e[i];
            }
        }
        dm
    }

    /// Real block-diagonal eigenvalue matrix; alias of [`Self::get_pseudo_d`]
    /// standing in for dlib's complex `get_d()`.
    pub fn get_d(&self) -> Matrix<f64> {
        self.get_pseudo_d()
    }

    // ------------------------------------------------------------------
    // Symmetric Householder reduction to tridiagonal form: tred2()
    // ------------------------------------------------------------------

    /// Port of `tred2()` (derived from the Algol procedure tred2 by Bowdler,
    /// Martin, Reinsch, and Wilkinson; EISPACK).
    fn tred2(&mut self) {
        let n = self.n;

        for j in 0..n {
            self.d[j] = self.v[(n - 1, j)];
        }

        // Householder reduction to tridiagonal form.
        for i in (1..n).rev() {
            // Scale to avoid under/overflow.
            let mut scale = 0.0f64;
            let mut h = 0.0f64;
            for k in 0..i {
                scale += self.d[k].abs();
            }
            if scale == 0.0 {
                self.e[i] = self.d[i - 1];
                for j in 0..i {
                    self.d[j] = self.v[(i - 1, j)];
                    self.v[(i, j)] = 0.0;
                    self.v[(j, i)] = 0.0;
                }
            } else {
                // Generate Householder vector.
                for k in 0..i {
                    self.d[k] /= scale;
                    h += self.d[k] * self.d[k];
                }
                let mut f = self.d[i - 1];
                let mut g = h.sqrt();
                if f > 0.0 {
                    g = -g;
                }
                self.e[i] = scale * g;
                h -= f * g;
                self.d[i - 1] = f - g;
                for j in 0..i {
                    self.e[j] = 0.0;
                }

                // Apply similarity transformation to remaining columns.
                for j in 0..i {
                    f = self.d[j];
                    self.v[(j, i)] = f;
                    g = self.e[j] + self.v[(j, j)] * f;
                    for k in (j + 1)..=(i - 1) {
                        g += self.v[(k, j)] * self.d[k];
                        self.e[k] += self.v[(k, j)] * f;
                    }
                    self.e[j] = g;
                }
                f = 0.0;
                for j in 0..i {
                    self.e[j] /= h;
                    f += self.e[j] * self.d[j];
                }
                let hh = f / (h + h);
                for j in 0..i {
                    self.e[j] -= hh * self.d[j];
                }
                for j in 0..i {
                    f = self.d[j];
                    g = self.e[j];
                    for k in j..=(i - 1) {
                        let v = self.v[(k, j)] - (f * self.e[k] + g * self.d[k]);
                        self.v[(k, j)] = v;
                    }
                    self.d[j] = self.v[(i - 1, j)];
                    self.v[(i, j)] = 0.0;
                }
            }
            self.d[i] = h;
        }

        // Accumulate transformations.
        for i in 0..(n - 1) {
            self.v[(n - 1, i)] = self.v[(i, i)];
            self.v[(i, i)] = 1.0;
            let h = self.d[i + 1];
            if h != 0.0 {
                for k in 0..=i {
                    self.d[k] = self.v[(k, i + 1)] / h;
                }
                for j in 0..=i {
                    let mut g = 0.0f64;
                    for k in 0..=i {
                        g += self.v[(k, i + 1)] * self.v[(k, j)];
                    }
                    for k in 0..=i {
                        let v = self.v[(k, j)] - g * self.d[k];
                        self.v[(k, j)] = v;
                    }
                }
            }
            for k in 0..=i {
                self.v[(k, i + 1)] = 0.0;
            }
        }
        for j in 0..n {
            self.d[j] = self.v[(n - 1, j)];
            self.v[(n - 1, j)] = 0.0;
        }
        self.v[(n - 1, n - 1)] = 1.0;
        self.e[0] = 0.0;
    }

    // ------------------------------------------------------------------
    // Symmetric tridiagonal QL algorithm: tql2()
    // ------------------------------------------------------------------

    /// Port of `tql2()` (derived from the Algol procedure tql2 by Bowdler,
    /// Martin, Reinsch, and Wilkinson; EISPACK). Like dlib, the eigenvalue
    /// sorting code present in the original JAMA version is NOT performed.
    fn tql2(&mut self) {
        let n = self.n;

        for i in 1..n {
            self.e[i - 1] = self.e[i];
        }
        self.e[n - 1] = 0.0;

        let mut f = 0.0f64;
        let mut tst1 = 0.0f64;
        let eps = f64::EPSILON;
        for l in 0..n {
            // Find small subdiagonal element
            tst1 = tst1.max(self.d[l].abs() + self.e[l].abs());
            let mut m = l;

            // Original while-loop from Java code
            while m < n {
                if self.e[m].abs() <= eps * tst1 {
                    break;
                }
                m += 1;
            }
            if m == n {
                m -= 1;
            }

            // If m == l, d(l) is an eigenvalue, otherwise, iterate.
            if m > l {
                let mut _iter = 0;
                loop {
                    _iter += 1; // (Could check iteration count here.)

                    // Compute implicit shift
                    let mut g = self.d[l];
                    let mut p = (self.d[l + 1] - g) / (2.0 * self.e[l]);
                    let mut r = p.hypot(1.0);
                    if p < 0.0 {
                        r = -r;
                    }
                    self.d[l] = self.e[l] / (p + r);
                    self.d[l + 1] = self.e[l] * (p + r);
                    let dl1 = self.d[l + 1];
                    let mut h = g - self.d[l];
                    for i in (l + 2)..n {
                        self.d[i] -= h;
                    }
                    f += h;

                    // Implicit QL transformation.
                    p = self.d[m];
                    let mut c = 1.0f64;
                    let mut c2 = c;
                    let mut c3 = c;
                    let el1 = self.e[l + 1];
                    let mut s = 0.0f64;
                    let mut s2 = 0.0f64;
                    for i in ((l)..=(m - 1)).rev() {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * self.e[i];
                        h = c * p;
                        r = p.hypot(self.e[i]);
                        self.e[i + 1] = s * r;
                        s = self.e[i] / r;
                        c = p / r;
                        p = c * self.d[i] - s * g;
                        self.d[i + 1] = h + s * (c * g + s * self.d[i]);

                        // Accumulate transformation.
                        for k in 0..n {
                            h = self.v[(k, i + 1)];
                            let a = s * self.v[(k, i)] + c * h;
                            let b = c * self.v[(k, i)] - s * h;
                            self.v[(k, i + 1)] = a;
                            self.v[(k, i)] = b;
                        }
                    }
                    p = -s * s2 * c3 * el1 * self.e[l] / dl1;
                    self.e[l] = s * p;
                    self.d[l] = c * p;

                    // Check for convergence.
                    if self.e[l].abs() <= eps * tst1 {
                        break;
                    }
                }
            }
            self.d[l] += f;
            self.e[l] = 0.0;
        }
    }

    // ------------------------------------------------------------------
    // Nonsymmetric reduction to Hessenberg form: orthes()
    // ------------------------------------------------------------------

    /// Port of `orthes()` (derived from the Algol procedures orthes and
    /// ortran by Martin and Wilkinson; EISPACK).
    fn orthes(&mut self) {
        let n = self.n;
        let low: usize = 0;
        let high: usize = n - 1;

        for m in (low + 1)..high {
            // Scale column.
            let mut scale = 0.0f64;
            for i in m..=high {
                scale += self.h[(i, m - 1)].abs();
            }
            if scale != 0.0 {
                // Compute Householder transformation.
                let mut hh = 0.0f64;
                for i in (m..=high).rev() {
                    self.ort[i] = self.h[(i, m - 1)] / scale;
                    hh += self.ort[i] * self.ort[i];
                }
                let mut g = hh.sqrt();
                if self.ort[m] > 0.0 {
                    g = -g;
                }
                hh -= self.ort[m] * g;
                self.ort[m] -= g;

                // Apply Householder similarity transformation
                // H = (I-u*u'/h)*H*(I-u*u')/h)
                for j in m..n {
                    let mut f = 0.0f64;
                    for i in (m..=high).rev() {
                        f += self.ort[i] * self.h[(i, j)];
                    }
                    f /= hh;
                    for i in m..=high {
                        let v = self.h[(i, j)] - f * self.ort[i];
                        self.h[(i, j)] = v;
                    }
                }

                for i in 0..=high {
                    let mut f = 0.0f64;
                    for j in (m..=high).rev() {
                        f += self.ort[j] * self.h[(i, j)];
                    }
                    f /= hh;
                    for j in m..=high {
                        let v = self.h[(i, j)] - f * self.ort[j];
                        self.h[(i, j)] = v;
                    }
                }
                self.ort[m] *= scale;
                self.h[(m, m - 1)] = scale * g;
            }
        }

        // Accumulate transformations (Algol's ortran).
        for i in 0..n {
            for j in 0..n {
                self.v[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }

        if high > low {
            for m in ((low + 1)..=(high - 1)).rev() {
                if self.h[(m, m - 1)] != 0.0 {
                    for i in (m + 1)..=high {
                        self.ort[i] = self.h[(i, m - 1)];
                    }
                    for j in m..=high {
                        let mut g = 0.0f64;
                        for i in m..=high {
                            g += self.ort[i] * self.v[(i, j)];
                        }
                        // Double division avoids possible underflow
                        g = (g / self.ort[m]) / self.h[(m, m - 1)];
                        for i in m..=high {
                            let a = self.v[(i, j)] + g * self.ort[i];
                            self.v[(i, j)] = a;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Complex scalar division: cdiv_()
    // ------------------------------------------------------------------

    /// Port of `cdiv_()`: complex division `(xr + xi*i) / (yr + yi*i)`,
    /// returning `(cdivr, cdivi)`.
    fn cdiv(&self, xr: f64, xi: f64, yr: f64, yi: f64) -> (f64, f64) {
        if yr.abs() > yi.abs() {
            let r = yi / yr;
            let d = yr + r * yi;
            ((xr + r * xi) / d, (xi - r * xr) / d)
        } else {
            let r = yr / yi;
            let d = yi + r * yr;
            ((r * xr + xi) / d, (r * xi - xr) / d)
        }
    }

    // ------------------------------------------------------------------
    // Nonsymmetric reduction from Hessenberg to real Schur form: hqr2()
    // ------------------------------------------------------------------

    /// Port of `hqr2()` (derived from the Algol procedure hqr2 by Martin and
    /// Wilkinson; EISPACK).
    fn hqr2(&mut self) {
        let nn = self.n;
        let low: isize = 0;
        let high: isize = nn as isize - 1;
        let eps = f64::EPSILON;
        let mut exshift = 0.0f64;

        // Store roots isolated by balanc and compute matrix norm
        let mut norm = 0.0f64;
        for i in 0..nn {
            if (i as isize) < low || (i as isize) > high {
                self.d[i] = self.h[(i, i)];
                self.e[i] = 0.0;
            }
            for j in i.saturating_sub(1)..nn {
                norm += self.h[(i, j)].abs();
            }
        }

        let mut p = 0.0f64;
        let mut q = 0.0f64;
        let mut r = 0.0f64;
        let mut s = 0.0f64;
        let mut z = 0.0f64;
        let mut t;
        let mut w;
        let mut x;
        let mut y;

        // Outer loop over eigenvalue index
        let mut n: isize = nn as isize - 1;
        let mut iter = 0;
        while n >= low {
            // Look for single small sub-diagonal element
            let mut l: isize = n;
            loop {
                if l <= low {
                    break;
                }
                s = self.h[(((l - 1) as usize), (l - 1) as usize)].abs()
                    + self.h[(l as usize, l as usize)].abs();
                if s == 0.0 {
                    s = norm;
                }
                if self.h[(l as usize, (l - 1) as usize)].abs() < eps * s {
                    break;
                }
                l -= 1;
            }

            // Check for convergence
            // One root found
            if l == n {
                let un = n as usize;
                let val = self.h[(un, un)] + exshift;
                self.h[(un, un)] = val;
                self.d[un] = val;
                self.e[un] = 0.0;
                n -= 1;
                iter = 0;

                // Two roots found
            } else if l == n - 1 {
                let un = n as usize;
                let um = (n - 1) as usize;
                w = self.h[(un, um)] * self.h[(um, un)];
                p = (self.h[(um, um)] - self.h[(un, un)]) / 2.0;
                q = p * p + w;
                z = q.abs().sqrt();
                let val = self.h[(un, un)] + exshift;
                self.h[(un, un)] = val;
                let valm = self.h[(um, um)] + exshift;
                self.h[(um, um)] = valm;
                x = self.h[(un, un)];

                // type pair
                if q >= 0.0 {
                    if p >= 0.0 {
                        z += p;
                    } else {
                        z = p - z;
                    }
                    self.d[um] = x + z;
                    self.d[un] = self.d[um];
                    if z != 0.0 {
                        self.d[un] = x - w / z;
                    }
                    for j in um..nn {
                        z = self.h[(um, j)];
                        let a = q * z + p * self.h[(un, j)];
                        let b = q * self.h[(un, j)] - p * z;
                        self.h[(um, j)] = a;
                        self.h[(un, j)] = b;
                    }

                    // Column modification
                    for i in 0..=un {
                        z = self.h[(i, um)];
                        let a = q * z + p * self.h[(i, un)];
                        let b = q * self.h[(i, un)] - p * z;
                        self.h[(i, um)] = a;
                        self.h[(i, un)] = b;
                    }

                    // Accumulate transformations
                    for i in low as usize..=high as usize {
                        z = self.v[(i, um)];
                        let a = q * z + p * self.v[(i, un)];
                        let b = q * self.v[(i, un)] - p * z;
                        self.v[(i, um)] = a;
                        self.v[(i, un)] = b;
                    }

                    // Complex pair
                } else {
                    self.d[um] = x + p;
                    self.d[un] = x + p;
                    self.e[um] = z;
                    self.e[un] = -z;
                }
                n -= 2;
                iter = 0;

                // No convergence yet
            } else {
                // Form shift
                x = self.h[(n as usize, n as usize)];
                y = 0.0;
                w = 0.0;
                if l < n {
                    y = self.h[((n - 1) as usize, (n - 1) as usize)];
                    w = self.h[(n as usize, (n - 1) as usize)]
                        * self.h[((n - 1) as usize, n as usize)];
                }

                // Wilkinson's original ad hoc shift
                if iter == 10 {
                    exshift += x;
                    for i in (low as usize)..=(n as usize) {
                        let v = self.h[(i, i)] - x;
                        self.h[(i, i)] = v;
                    }
                    s = self.h[(n as usize, (n - 1) as usize)].abs()
                        + self.h[((n - 1) as usize, (n - 2) as usize)].abs();
                    x = 0.75 * s;
                    y = 0.75 * s;
                    w = -0.4375 * s * s;
                }

                // MATLAB's new ad hoc shift
                if iter == 30 {
                    s = (y - x) / 2.0;
                    s = s * s + w;
                    if s > 0.0 {
                        s = s.sqrt();
                        if y < x {
                            s = -s;
                        }
                        s = x - w / ((y - x) / 2.0 + s);
                        for i in (low as usize)..=(n as usize) {
                            let v = self.h[(i, i)] - s;
                            self.h[(i, i)] = v;
                        }
                        exshift += s;
                        x = 0.964;
                        y = 0.964;
                        w = 0.964;
                    }
                }

                iter += 1; // (Could check iteration count here.)

                // Look for two consecutive small sub-diagonal elements
                let mut m: isize = n - 2;
                while m >= l {
                    let um = m as usize;
                    z = self.h[(um, um)];
                    r = x - z;
                    s = y - z;
                    p = (r * s - w) / self.h[(um + 1, um)] + self.h[(um, um + 1)];
                    q = self.h[(um + 1, um + 1)] - z - r - s;
                    r = self.h[(um + 2, um + 1)];
                    s = p.abs() + q.abs() + r.abs();
                    p /= s;
                    q /= s;
                    r /= s;
                    if m == l {
                        break;
                    }
                    if self.h[(um, um - 1)].abs() * (q.abs() + r.abs())
                        < eps
                            * (p.abs()
                                * (self.h[(um - 1, um - 1)].abs()
                                    + z.abs()
                                    + self.h[(um + 1, um + 1)].abs()))
                    {
                        break;
                    }
                    m -= 1;
                }

                let um = m as usize;
                for i in (um + 2)..=(n as usize) {
                    self.h[(i, i - 2)] = 0.0;
                    if i > um + 2 {
                        self.h[(i, i - 3)] = 0.0;
                    }
                }

                // Double QR step involving rows l:n and columns m:n
                let mut k = m;
                while k < n {
                    let ku = k as usize;
                    let notlast = k != n - 1;
                    if k != m {
                        p = self.h[(ku, ku - 1)];
                        q = self.h[(ku + 1, ku - 1)];
                        r = if notlast {
                            self.h[(ku + 2, ku - 1)]
                        } else {
                            0.0
                        };
                        x = p.abs() + q.abs() + r.abs();
                        if x != 0.0 {
                            p /= x;
                            q /= x;
                            r /= x;
                        }
                    }
                    if x == 0.0 {
                        break;
                    }
                    s = (p * p + q * q + r * r).sqrt();
                    if p < 0.0 {
                        s = -s;
                    }
                    if s != 0.0 {
                        if k != m {
                            self.h[(ku, ku - 1)] = -s * x;
                        } else if l != m {
                            self.h[(ku, ku - 1)] = -self.h[(ku, ku - 1)];
                        }
                        p += s;
                        x = p / s;
                        y = q / s;
                        z = r / s;
                        q /= p;
                        r /= p;

                        // Row modification
                        for j in ku..nn {
                            p = self.h[(ku, j)] + q * self.h[(ku + 1, j)];
                            if notlast {
                                p += r * self.h[(ku + 2, j)];
                                let v = self.h[(ku + 2, j)] - p * z;
                                self.h[(ku + 2, j)] = v;
                            }
                            let a = self.h[(ku, j)] - p * x;
                            let b = self.h[(ku + 1, j)] - p * y;
                            self.h[(ku, j)] = a;
                            self.h[(ku + 1, j)] = b;
                        }

                        // Column modification
                        let colmax = (n as usize).min(ku + 3);
                        for i in 0..=colmax {
                            p = x * self.h[(i, ku)] + y * self.h[(i, ku + 1)];
                            if notlast {
                                p += z * self.h[(i, ku + 2)];
                                let v = self.h[(i, ku + 2)] - p * r;
                                self.h[(i, ku + 2)] = v;
                            }
                            let a = self.h[(i, ku)] - p;
                            let b = self.h[(i, ku + 1)] - p * q;
                            self.h[(i, ku)] = a;
                            self.h[(i, ku + 1)] = b;
                        }

                        // Accumulate transformations
                        for i in (low as usize)..=(high as usize) {
                            p = x * self.v[(i, ku)] + y * self.v[(i, ku + 1)];
                            if notlast {
                                p += z * self.v[(i, ku + 2)];
                                let vv = self.v[(i, ku + 2)] - p * r;
                                self.v[(i, ku + 2)] = vv;
                            }
                            let a = self.v[(i, ku)] - p;
                            let b = self.v[(i, ku + 1)] - p * q;
                            self.v[(i, ku)] = a;
                            self.v[(i, ku + 1)] = b;
                        }
                    } // (s != 0)
                    k += 1;
                } // k loop
            } // check convergence
        } // while (n >= low)

        // Backsubstitute to find vectors of upper triangular form
        if norm == 0.0 {
            return;
        }

        for n in (0..nn).rev() {
            p = self.d[n];
            q = self.e[n];

            // Real vector
            if q == 0.0 {
                let mut l = n;
                self.h[(n, n)] = 1.0;
                for i in (0..n).rev() {
                    w = self.h[(i, i)] - p;
                    let mut rr = 0.0f64;
                    for j in l..=n {
                        rr += self.h[(i, j)] * self.h[(j, n)];
                    }
                    r = rr;
                    if self.e[i] < 0.0 {
                        z = w;
                        s = r;
                    } else {
                        l = i;
                        if self.e[i] == 0.0 {
                            if w != 0.0 {
                                self.h[(i, n)] = -r / w;
                            } else {
                                self.h[(i, n)] = -r / (eps * norm);
                            }
                            // Solve real equations
                        } else {
                            x = self.h[(i, i + 1)];
                            y = self.h[(i + 1, i)];
                            q = (self.d[i] - p) * (self.d[i] - p) + self.e[i] * self.e[i];
                            t = (x * s - z * r) / q;
                            self.h[(i, n)] = t;
                            if x.abs() > z.abs() {
                                self.h[(i + 1, n)] = (-r - w * t) / x;
                            } else {
                                self.h[(i + 1, n)] = (-s - y * t) / z;
                            }
                        }

                        // Overflow control
                        t = self.h[(i, n)].abs();
                        if (eps * t) * t > 1.0 {
                            for j in i..=n {
                                let v = self.h[(j, n)] / t;
                                self.h[(j, n)] = v;
                            }
                        }
                    }
                }

                // Complex vector
            } else if q < 0.0 {
                let mut l = n - 1;

                // Last vector component imaginary so matrix is triangular
                if self.h[(n, n - 1)].abs() > self.h[(n - 1, n)].abs() {
                    let a = q / self.h[(n, n - 1)];
                    self.h[(n - 1, n - 1)] = a;
                    let b = -(self.h[(n, n)] - p) / self.h[(n, n - 1)];
                    self.h[(n - 1, n)] = b;
                } else {
                    let (cdivr, cdivi) =
                        self.cdiv(0.0, -self.h[(n - 1, n)], self.h[(n - 1, n - 1)] - p, q);
                    self.h[(n - 1, n - 1)] = cdivr;
                    self.h[(n - 1, n)] = cdivi;
                }
                self.h[(n, n - 1)] = 0.0;
                self.h[(n, n)] = 1.0;
                if n >= 2 {
                    for i in (0..=(n - 2)).rev() {
                        let mut ra = 0.0f64;
                        let mut sa = 0.0f64;
                        let mut vr;
                        let vi;
                        for j in l..=n {
                            ra += self.h[(i, j)] * self.h[(j, n - 1)];
                            sa += self.h[(i, j)] * self.h[(j, n)];
                        }
                        w = self.h[(i, i)] - p;

                        if self.e[i] < 0.0 {
                            z = w;
                            r = ra;
                            s = sa;
                        } else {
                            l = i;
                            if self.e[i] == 0.0 {
                                let (cdivr, cdivi) = self.cdiv(-ra, -sa, w, q);
                                self.h[(i, n - 1)] = cdivr;
                                self.h[(i, n)] = cdivi;
                            } else {
                                // Solve complex equations
                                x = self.h[(i, i + 1)];
                                y = self.h[(i + 1, i)];
                                vr = (self.d[i] - p) * (self.d[i] - p) + self.e[i] * self.e[i]
                                    - q * q;
                                vi = (self.d[i] - p) * 2.0 * q;
                                if vr == 0.0 && vi == 0.0 {
                                    vr = eps
                                        * norm
                                        * (w.abs() + q.abs() + x.abs() + y.abs() + z.abs());
                                }
                                let (cdivr, cdivi) = self.cdiv(
                                    x * r - z * ra + q * sa,
                                    x * s - z * sa - q * ra,
                                    vr,
                                    vi,
                                );
                                self.h[(i, n - 1)] = cdivr;
                                self.h[(i, n)] = cdivi;
                                if x.abs() > (z.abs() + q.abs()) {
                                    self.h[(i + 1, n - 1)] =
                                        (-ra - w * self.h[(i, n - 1)] + q * self.h[(i, n)]) / x;
                                    self.h[(i + 1, n)] =
                                        (-sa - w * self.h[(i, n)] - q * self.h[(i, n - 1)]) / x;
                                } else {
                                    let (cdivr, cdivi) = self.cdiv(
                                        -r - y * self.h[(i, n - 1)],
                                        -s - y * self.h[(i, n)],
                                        z,
                                        q,
                                    );
                                    self.h[(i + 1, n - 1)] = cdivr;
                                    self.h[(i + 1, n)] = cdivi;
                                }
                            }

                            // Overflow control
                            t = self.h[(i, n - 1)].abs().max(self.h[(i, n)].abs());
                            if (eps * t) * t > 1.0 {
                                for j in i..=n {
                                    let a = self.h[(j, n - 1)] / t;
                                    let b = self.h[(j, n)] / t;
                                    self.h[(j, n - 1)] = a;
                                    self.h[(j, n)] = b;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Vectors of isolated roots
        for i in 0..nn {
            if (i as isize) < low || (i as isize) > high {
                for j in i..nn {
                    self.v[(i, j)] = self.h[(i, j)];
                }
            }
        }

        // Back transformation to get eigenvectors of original matrix
        for j in (low as usize..nn).rev() {
            for i in (low as usize)..=(high as usize) {
                z = 0.0;
                for k in (low as usize)..=(j.min(high as usize)) {
                    z += self.v[(i, k)] * self.h[(k, j)];
                }
                self.v[(i, j)] = z;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_symmetric_known_eigenvalues() {
        // diag(2, 4, 6) via orthogonal similarity of a known symmetric matrix
        // with eigenvalues 2, 4, 6: A = Q * diag(2,4,6) * Q', Q = rotation.
        let theta: f64 = 0.7;
        let (sn, cs) = (theta.sin(), theta.cos());
        // 3x3 rotation about y-axis (orthogonal)
        let mut qm = Matrix::zeros(3, 3);
        qm[(0, 0)] = cs;
        qm[(0, 2)] = sn;
        qm[(1, 1)] = 1.0;
        qm[(2, 0)] = -sn;
        qm[(2, 2)] = cs;
        let mut lam = Matrix::zeros(3, 3);
        lam[(0, 0)] = 2.0;
        lam[(1, 1)] = 4.0;
        lam[(2, 2)] = 6.0;
        let qt = {
            let mut t = Matrix::zeros(3, 3);
            for i in 0..3 {
                for j in 0..3 {
                    t[(i, j)] = qm[(j, i)];
                }
            }
            t
        };
        let a = mat_mul(&mat_mul(&qm, &lam), &qt);

        let eig = EigenvalueDecomposition::new(&a);
        let d = eig.get_real_eigenvalues();
        let e = eig.get_imag_eigenvalues();
        for i in 0..3 {
            assert!(e[(i, 0)].abs() < 1e-12);
        }
        // eigenvalues (unsorted, per dlib tql2) must be a permutation of {2,4,6}
        let mut vals: Vec<f64> = (0..3).map(|i| d[(i, 0)]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in vals.iter().zip([2.0, 4.0, 6.0]) {
            assert!((got - want).abs() < 1e-10, "got {}", got);
        }

        // A * v == lambda * v for each eigenvector column
        let v = eig.get_pseudo_v();
        for j in 0..3 {
            let mut av = Matrix::zeros(3, 1);
            let mut lv = Matrix::zeros(3, 1);
            for i in 0..3 {
                av[(i, 0)] = mat_mul(&a, &v.col(j))[(i, 0)];
                lv[(i, 0)] = d[(j, 0)] * v[(i, j)];
            }
            assert!(max_abs(&av, &lv) < 1e-9);
        }
    }

    #[test]
    fn test_general_complex_pair() {
        // rotation matrix: eigenvalues cos(t) +- i sin(t)
        let t: f64 = 0.9;
        let mut a = Matrix::zeros(2, 2);
        a[(0, 0)] = t.cos();
        a[(0, 1)] = -t.sin();
        a[(1, 0)] = t.sin();
        a[(1, 1)] = t.cos();
        let eig = EigenvalueDecomposition::new(&a);
        let d = eig.get_real_eigenvalues();
        let e = eig.get_imag_eigenvalues();
        assert!((d[(0, 0)] - t.cos()).abs() < 1e-12);
        assert!((d[(1, 0)] - t.cos()).abs() < 1e-12);
        let imag = e[(0, 0)].abs().max(e[(1, 0)].abs());
        assert!((imag - t.sin()).abs() < 1e-12);
        assert!((e[(0, 0)] + e[(1, 0)]).abs() < 1e-12); // conjugate pair

        // general 3x3 with a complex pair: block diag(rotation, [5])
        let mut rot = Matrix::zeros(2, 2);
        rot[(0, 0)] = 0.0;
        rot[(0, 1)] = -2.0;
        rot[(1, 0)] = 2.0;
        rot[(1, 1)] = 0.0;
        let mut a3 = Matrix::zeros(3, 3);
        for i in 0..2 {
            for j in 0..2 {
                a3[(i, j)] = rot[(i, j)];
            }
        }
        a3[(2, 2)] = 5.0;
        let eig3 = EigenvalueDecomposition::new(&a3);
        let d3 = eig3.get_real_eigenvalues();
        let e3 = eig3.get_imag_eigenvalues();
        let mut reals: Vec<f64> = (0..3).map(|i| d3[(i, 0)]).collect();
        reals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((reals[0] + 0.0).abs() < 1e-12);
        assert!((reals[1] - 0.0).abs() < 1e-12);
        assert!((reals[2] - 5.0).abs() < 1e-12);
        let imag_sum: f64 = (0..3).map(|i| e3[(i, 0)]).sum();
        assert!(imag_sum.abs() < 1e-12);
        let max_imag = (0..3).map(|i| e3[(i, 0)].abs()).fold(0.0, f64::max);
        assert!((max_imag - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_pseudo_d_reconstruction_symmetric() {
        // For symmetric matrices: A * V == V * D
        let mut a = Matrix::zeros(3, 3);
        let vals = [4.0, 1.0, 0.5, -1.0, 2.0, 3.0, 0.0, -2.0, 5.0];
        let mut idx = 0;
        for r in 0..3 {
            for c in 0..3 {
                a[(r, c)] = vals[idx];
                idx += 1;
            }
        }
        // symmetrize
        for r in 0..3 {
            for c in 0..r {
                let v = 0.5 * (a[(r, c)] + a[(c, r)]);
                a[(r, c)] = v;
                a[(c, r)] = v;
            }
        }
        let eig = EigenvalueDecomposition::new(&a);
        let v = eig.get_pseudo_v();
        let dmat = eig.get_pseudo_d();
        assert!(max_abs(&mat_mul(&a, v), &mat_mul(v, &dmat)) < 1e-10);
    }
}

//! Golden comparison against C++ dlib output (`golden/src/matrix_golden.cpp`
//! → `tests/golden/matrix.txt`). Verifies the ported decompositions match
//! dlib's own non-LAPACK implementations to tight relative tolerance.

use dlib_rs_core::matrix::{self, Matrix};

fn golden_path() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/golden/matrix.txt").to_string()
}

struct Golden {
    lines: Vec<String>,
    idx: usize,
}

impl Golden {
    fn load() -> Self {
        let text = std::fs::read_to_string(golden_path()).expect("golden matrix.txt present");
        Golden {
            lines: text
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect(),
            idx: 0,
        }
    }

    /// Reads the next section: either "name nr nc" + rows, or "name value".
    fn next(&mut self) -> (String, Matrix<f64>) {
        let header = self.lines[self.idx].clone();
        self.idx += 1;
        let parts: Vec<&str> = header.split_whitespace().collect();
        match parts.len() {
            3 => {
                let nr: usize = parts[1].parse().unwrap();
                let nc: usize = parts[2].parse().unwrap();
                let mut m = Matrix::zeros(nr, nc);
                for r in 0..nr {
                    let line = &self.lines[self.idx];
                    self.idx += 1;
                    let vals: Vec<f64> = line
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(vals.len(), nc, "row width mismatch in {}", parts[0]);
                    for (c, v) in vals.iter().enumerate() {
                        m[(r, c)] = *v;
                    }
                }
                (parts[0].to_string(), m)
            }
            2 => (
                parts[0].to_string(),
                Matrix::from_row_vec(1, 1, &[parts[1].parse().unwrap()]),
            ),
            _ => panic!("bad golden header: {header}"),
        }
    }
}

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn assert_matrices_close(name: &str, got: &Matrix<f64>, want: &Matrix<f64>, tol: f64) {
    assert_eq!((got.nr(), got.nc()), (want.nr(), want.nc()), "{name} shape");
    for r in 0..want.nr() {
        for c in 0..want.nc() {
            assert!(
                rel_close(got[(r, c)], want[(r, c)], tol),
                "{name}[{r},{c}]: got {:.17e} want {:.17e}",
                got[(r, c)],
                want[(r, c)]
            );
        }
    }
}

fn build_inputs() -> (Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let mut a6 = Matrix::zeros(6, 6);
    for r in 0..6usize {
        for c in 0..6usize {
            a6[(r, c)] =
                (0.7 * ((r + 1) as f64) * ((c + 1) as f64) + 0.01 * (r as f64 - c as f64)).sin();
        }
    }
    let mut b85 = Matrix::zeros(8, 5);
    for r in 0..8usize {
        for c in 0..5usize {
            b85[(r, c)] =
                (0.3 * ((r + 2) as f64) * ((c + 1) as f64)).cos() + (r as f64 - c as f64) * 0.05;
        }
    }
    let ident6: Matrix<f64> = Matrix::identity(6);
    let s6 = a6.clone() * a6.transpose() + ident6 * 6.0;
    let mut g5 = Matrix::zeros(5, 5);
    for r in 0..5usize {
        for c in 0..5usize {
            g5[(r, c)] = (0.11 * ((r + 1) as f64) * ((c + 2) as f64)).sin()
                + 0.5 * (0.07 * ((r + 2) as f64) * ((c + 1) as f64)).cos()
                + if r == c { 2.0 } else { 0.0 };
        }
    }
    (a6, b85, s6, g5)
}

fn col_matches_sign_agnostic(got: &Matrix<f64>, want: &Matrix<f64>, col: usize, tol: f64) -> bool {
    (0..got.nr()).all(|r| rel_close(got[(r, col)], want[(r, col)], tol))
        || (0..got.nr()).all(|r| rel_close(got[(r, col)], -want[(r, col)], tol))
}

#[test]
fn golden_matrix_decompositions() {
    let (a6, b85, s6, g5) = build_inputs();
    let mut g = Golden::load();

    // Inputs themselves (guards against input-construction drift).
    let (_, ga6) = g.next();
    let (_, gb85) = g.next();
    let (_, gs6) = g.next();
    let (_, gg5) = g.next();
    assert_matrices_close("A6", &a6, &ga6, 1e-15);
    assert_matrices_close("B85", &b85, &gb85, 1e-15);
    assert_matrices_close("S6", &s6, &gs6, 1e-15);
    assert_matrices_close("G5", &g5, &gg5, 1e-15);

    let tol = 1e-10;

    // LU
    let (name, det) = g.next();
    assert_eq!(name, "LU_det");
    let lu = matrix::lu::LuDecomposition::new(&a6);
    assert!(rel_close(lu.det(), det[(0, 0)], tol), "LU det");
    let (name, gl) = g.next();
    assert_eq!(name, "LU_l");
    assert_matrices_close("LU_l", &lu.get_l(), &gl, tol);
    let (name, gu) = g.next();
    assert_eq!(name, "LU_u");
    assert_matrices_close("LU_u", &lu.get_u(), &gu, tol);
    let (name, gsolve) = g.next();
    assert_eq!(name, "LU_solve");
    let mut b6 = Matrix::zeros(6, 1);
    for r in 0..6usize {
        b6[(r, 0)] = 1.0 + 0.25 * r as f64;
    }
    assert_matrices_close("LU_solve", &lu.solve(&b6).unwrap(), &gsolve, tol);
    let (name, gsingular) = g.next();
    assert_eq!(name, "LU_is_singular");
    assert_eq!(gsingular[(0, 0)] as u32, 0, "A6 is nonsingular");
    assert!(!lu.is_singular());

    // Cholesky
    let ch = matrix::cholesky::CholeskyDecomposition::new(&s6);
    let (name, gl) = g.next();
    assert_eq!(name, "CH_l");
    assert_matrices_close("CH_l", ch.l(), &gl, tol);
    let (name, gspd) = g.next();
    assert_eq!(name, "CH_is_spd");
    assert_eq!(gspd[(0, 0)] as u32, 1);
    assert!(ch.is_spd());
    let (name, gsolve) = g.next();
    assert_eq!(name, "CH_solve");
    let mut b6c = Matrix::zeros(6, 1);
    for r in 0..6usize {
        b6c[(r, 0)] = 1.0 + 0.25 * r as f64;
    }
    assert_matrices_close("CH_solve", &ch.solve(&b6c), &gsolve, tol);

    // QR (dlib qr on 8x5: q() is economy 8x5)
    let qr = matrix::qr::QrDecomposition::new(&b85);
    let (name, gq) = g.next();
    assert_eq!(name, "QR_q");
    assert_matrices_close("QR_q", &qr.q(), &gq, 1e-9);
    let (name, gr) = g.next();
    assert_eq!(name, "QR_r");
    assert_matrices_close("QR_r", &qr.r(), &gr, 1e-9);
    let (name, gfullrank) = g.next();
    assert_eq!(name, "QR_is_full_rank");
    assert_eq!(gfullrank[(0, 0)] as u32, 1);
    assert!(qr.is_full_rank());
    let (name, gsolve) = g.next();
    assert_eq!(name, "QR_solve");
    let mut b81 = Matrix::zeros(8, 1);
    for r in 0..8usize {
        b81[(r, 0)] = (0.2 * ((r + 1) as f64)).sin();
    }
    assert_matrices_close("QR_solve", &qr.solve(&b81), &gsolve, 1e-8);

    // Eigenvalue (symmetric): values + eigenvector matrix
    let eig = matrix::eigenvalue::EigenvalueDecomposition::new(&s6);
    let (name, gev) = g.next();
    assert_eq!(name, "EIG_real");
    assert_matrices_close("EIG_real", &eig.get_real_eigenvalues(), &gev, 1e-9);
    let (name, gvec) = g.next();
    assert_eq!(name, "EIG_v");
    let v = eig.get_v();
    assert_eq!((v.nr(), v.nc()), (gvec.nr(), gvec.nc()), "EIG_v shape");
    for c in 0..v.nc() {
        assert!(
            col_matches_sign_agnostic(v, &gvec, c, 1e-8),
            "EIG_v col {c}: got {:?} want {:?}",
            (0..v.nr()).map(|r| v[(r, c)]).collect::<Vec<_>>(),
            (0..v.nr()).map(|r| gvec[(r, c)]).collect::<Vec<_>>()
        );
    }

    // Eigenvalue (general)
    let eig = matrix::eigenvalue::EigenvalueDecomposition::new(&g5);
    let (name, greal) = g.next();
    assert_eq!(name, "EIGG_real");
    assert_matrices_close("EIGG_real", &eig.get_real_eigenvalues(), &greal, 1e-8);
    let (name, gimag) = g.next();
    assert_eq!(name, "EIGG_imag");
    assert_matrices_close("EIGG_imag", &eig.get_imag_eigenvalues(), &gimag, 1e-8);
    let (name, gd) = g.next();
    assert_eq!(name, "EIGG_d");
    assert_matrices_close("EIGG_d", &eig.get_pseudo_d(), &gd, 1e-8);

    // SVD
    let mut u = Matrix::new();
    let mut w = Matrix::new();
    let mut v = Matrix::new();
    assert!(matrix::svd::svd(&b85, &mut u, &mut w, &mut v));
    let (name, gu) = g.next();
    assert_eq!(name, "SVD_u");
    assert_eq!((u.nr(), u.nc()), (gu.nr(), gu.nc()), "SVD_u shape");
    for c in 0..u.nc() {
        assert!(
            col_matches_sign_agnostic(&u, &gu, c, 1e-9),
            "SVD_u col {c} mismatch"
        );
    }
    let (name, gw) = g.next();
    assert_eq!(name, "SVD_w");
    assert_matrices_close("SVD_w", &w, &gw, 1e-9);
    let (name, gv) = g.next();
    assert_eq!(name, "SVD_v");
    for c in 0..v.nc() {
        assert!(
            col_matches_sign_agnostic(&v, &gv, c, 1e-9),
            "SVD_v col {c} mismatch"
        );
    }

    // inv / det
    let (name, ginv) = g.next();
    assert_eq!(name, "INV_A6");
    assert_matrices_close("INV_A6", &matrix::la::inv(&a6).unwrap(), &ginv, 1e-9);
    let (name, gdet) = g.next();
    assert_eq!(name, "DET_A6");
    assert!(rel_close(matrix::la::det(&a6), gdet[(0, 0)], tol), "DET_A6");

    assert_eq!(g.idx, g.lines.len(), "all golden sections consumed");
}

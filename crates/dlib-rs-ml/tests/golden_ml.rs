//! Golden comparison against C++ dlib ml output (`golden/src/ml_golden.cpp`
//! → `tests/golden/ml.txt`): Rosenbrock minimizers (absolute tolerance 1e-8
//! per the plan) and SVM C-SVC training (deterministic SMO: same support
//! vectors, alphas and bias as C++).

use dlib_rs_core::matrix::Matrix;
use dlib_rs_ml::optimization::bobyqa::find_min_bobyqa;
use dlib_rs_ml::optimization::find_min;
use dlib_rs_ml::optimization::search_strategies::{
    BfgsSearchStrategy, CgSearchStrategy, LbfgsSearchStrategy,
};
use dlib_rs_ml::optimization::stop_strategies::ObjectiveDeltaStopStrategy;
use dlib_rs_ml::svm::kernels::RadialBasisKernel;
use dlib_rs_ml::svm::smo::SvmCTrainer;

fn load() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/ml.txt"
    ))
    .expect("golden ml.txt present")
}

fn rosen(x: &Matrix<f64>) -> f64 {
    100.0 * (x[(1, 0)] - x[(0, 0)] * x[(0, 0)]).powi(2) + (1.0 - x[(0, 0)]).powi(2)
}

fn rosen_der(x: &Matrix<f64>) -> Matrix<f64> {
    let mut d = Matrix::zeros(2, 1);
    d[(0, 0)] = -400.0 * x[(0, 0)] * (x[(1, 0)] - x[(0, 0)] * x[(0, 0)]) - 2.0 * (1.0 - x[(0, 0)]);
    d[(1, 0)] = 200.0 * (x[(1, 0)] - x[(0, 0)] * x[(0, 0)]);
    d
}

fn x0() -> Matrix<f64> {
    Matrix::from_row_vec(2, 1, &[-1.2, 1.0])
}

fn val(text: &str, key: &str) -> Vec<f64> {
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(key) {
            return rest
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
        }
    }
    panic!("key {key} not found");
}

#[test]
fn golden_optimizers() {
    let text = load();

    let (f, x) = find_min(
        &mut LbfgsSearchStrategy::with_size(10),
        &mut ObjectiveDeltaStopStrategy::new(1e-9).with_max_iterations(3000),
        rosen,
        rosen_der,
        x0(),
        -1.0,
    );
    let gf: f64 = val(&text, "LBFGS_f")[0];
    let gx = val(&text, "LBFGS_x");
    assert!((f - gf).abs() < 1e-12, "LBFGS f: {f:e} vs {gf:e}");
    assert!(
        (x[(0, 0)] - gx[0]).abs() < 1e-8,
        "LBFGS x0: {} vs {}",
        x[(0, 0)],
        gx[0]
    );
    assert!(
        (x[(1, 0)] - gx[1]).abs() < 1e-8,
        "LBFGS x1: {} vs {}",
        x[(1, 0)],
        gx[1]
    );

    let (f, x) = find_min(
        &mut BfgsSearchStrategy::default(),
        &mut ObjectiveDeltaStopStrategy::new(1e-9).with_max_iterations(3000),
        rosen,
        rosen_der,
        x0(),
        -1.0,
    );
    let gf: f64 = val(&text, "BFGS_f")[0];
    let gx = val(&text, "BFGS_x");
    assert!((f - gf).abs() < 1e-12, "BFGS f: {f:e} vs {gf:e}");
    assert!((x[(0, 0)] - gx[0]).abs() < 1e-7, "BFGS x0");
    assert!((x[(1, 0)] - gx[1]).abs() < 1e-7, "BFGS x1");

    let (f, x) = find_min(
        &mut CgSearchStrategy::default(),
        &mut ObjectiveDeltaStopStrategy::new(1e-9).with_max_iterations(3000),
        rosen,
        rosen_der,
        x0(),
        -1.0,
    );
    let gf: f64 = val(&text, "CG_f")[0];
    let gx = val(&text, "CG_x");
    // CG converges more slowly; the plan's 1e-8 applies to the L-BFGS run.
    assert!((f - gf).abs() < 1e-8, "CG f: {f:e} vs {gf:e}");
    assert!((x[(0, 0)] - gx[0]).abs() < 1e-4, "CG x0");
    assert!((x[(1, 0)] - gx[1]).abs() < 1e-4, "CG x1");

    // BOBYQA (unconstrained helper: same ±1e100 bounds as dlib tests).
    let mut x = x0();
    let (f, evals) = find_min_bobyqa(rosen, &mut x, 1.0).unwrap();
    assert!(evals > 0);
    let gf: f64 = val(&text, "BOBYQA_result")[0];
    let gx = val(&text, "BOBYQA_x");
    assert!((f - gf).abs() < 1e-9, "BOBYQA f: {f:e} vs {gf:e}");
    assert!((x[(0, 0)] - gx[0]).abs() < 1e-6, "BOBYQA x0");
    assert!((x[(1, 0)] - gx[1]).abs() < 1e-6, "BOBYQA x1");
}

#[test]
fn golden_svm_c_svc_rbf() {
    let text = load();

    let mut samples = Vec::with_capacity(100);
    let mut labels = Vec::with_capacity(100);
    for i in 0..100usize {
        let px = (i as f64 * 1.7).sin() * 2.0 + if i % 2 == 0 { 1.7 } else { -1.7 };
        let py = (i as f64 * 2.3).cos() * 2.0 + if i % 3 == 0 { 1.3 } else { -1.1 };
        samples.push(Matrix::from_row_vec(2, 1, &[px, py]));
        labels.push(if px + 0.5 * py > 0.1 { 1.0 } else { -1.0 });
    }

    let trainer = SvmCTrainer::new(RadialBasisKernel::new(0.5), 5.0);
    let df = trainer.train(&samples, &labels).unwrap();

    let want_count = val(&text, "SVM_sv_count")[0] as usize;
    assert_eq!(
        df.basis_dictionary.len(),
        want_count,
        "support vector count"
    );

    let want_b = val(&text, "SVM_b")[0];
    assert!(
        (df.b - want_b).abs() < 1e-12 || (-df.b - want_b).abs() < 1e-12,
        "SVM b: {} vs {want_b}",
        df.b
    );

    let svs: Vec<(f64, f64, f64)> = text
        .lines()
        .filter(|l| l.trim_start().starts_with("SVM_sv "))
        .map(|l| {
            let p: Vec<&str> = l.split_whitespace().collect();
            (
                p[2].parse::<f64>().unwrap(),
                p[3].parse::<f64>().unwrap(),
                p[4].parse::<f64>().unwrap(),
            )
        })
        .collect();
    assert_eq!(svs.len(), want_count);

    // The SMO solver is deterministic: every support vector (position and
    // alpha) must match C++ exactly.
    let mut matched = vec![false; want_count];
    for (i, sv) in df.basis_dictionary.iter().enumerate() {
        let mut found = None;
        for (j, (wx, wy, _wa)) in svs.iter().enumerate() {
            if !matched[j] && (sv[(0, 0)] - *wx).abs() < 1e-12 && (sv[(1, 0)] - *wy).abs() < 1e-12 {
                matched[j] = true;
                found = Some(j);
                break;
            }
        }
        let j = found
            .unwrap_or_else(|| panic!("unexpected support vector ({},{})", sv[(0, 0)], sv[(1, 0)]));
        let alpha = df.alpha_vector[(i, 0)];
        let wa = svs[j].2;
        assert!(
            (alpha - wa).abs() < 1e-12,
            "alpha for sv ({},{}): {alpha:e} vs {wa:e}",
            svs[j].0,
            svs[j].1
        );
    }
}

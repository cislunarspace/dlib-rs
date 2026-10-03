// Golden outputs for the dlib-rs ml port: Rosenbrock minimization via
// find_min(lbfgs/bfgs/cg), BOBYQA, and an SVM C-SVC training on a fixed
// deterministic sample set with an RBF kernel (SMO must be deterministic).
#include <dlib/optimization.h>
#include <dlib/optimization/optimization_bobyqa.h>
#include <dlib/svm.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace std;
using namespace dlib;

typedef matrix<double, 2, 1> mat2;
typedef radial_basis_kernel<mat2> rbf;

static double rosen(const mat2& x) {
    return 100.0 * pow(x(1) - x(0) * x(0), 2.0) + pow(1.0 - x(0), 2.0);
}
static mat2 rosen_der(const mat2& x) {
    mat2 d;
    d(0) = -400.0 * x(0) * (x(1) - x(0) * x(0)) - 2.0 * (1.0 - x(0));
    d(1) = 200.0 * (x(1) - x(0) * x(0));
    return d;
}

int main() {
    // ---- find_min(lbfgs) on Rosenbrock ----
    {
        mat2 x; x = -1.2, 1.0;
        double f = find_min(
            lbfgs_search_strategy(10),
            objective_delta_stop_strategy(1e-9, 3000),
            rosen, rosen_der, x, -1.0);
        std::printf("LBFGS_f %.17g\n", f);
        std::printf("LBFGS_x %.17g %.17g\n", x(0), x(1));
    }
    // ---- find_min(bfgs) ----
    {
        mat2 x; x = -1.2, 1.0;
        double f = find_min(
            bfgs_search_strategy(),
            objective_delta_stop_strategy(1e-9, 3000),
            rosen, rosen_der, x, -1.0);
        std::printf("BFGS_f %.17g\n", f);
        std::printf("BFGS_x %.17g %.17g\n", x(0), x(1));
    }
    // ---- find_min(cg) ----
    {
        mat2 x; x = -1.2, 1.0;
        double f = find_min(
            cg_search_strategy(),
            objective_delta_stop_strategy(1e-9, 3000),
            rosen, rosen_der, x, -1.0);
        std::printf("CG_f %.17g\n", f);
        std::printf("CG_x %.17g %.17g\n", x(0), x(1));
    }
    // ---- BOBYQA on Rosenbrock ----
    {
        mat2 x; x = -1.2, 1.0;
        mat2 lo; lo = -100.0, -100.0;
        mat2 hi; hi = 100.0, 100.0;
        double result = find_min_bobyqa(
            rosen, x, 5, lo, hi, 1.0, 1e-9, 100000);
        std::printf("BOBYQA_result %.17g\n", result);
        std::printf("BOBYQA_x %.17g %.17g\n", x(0), x(1));
    }

    // ---- SVM C-SVC, RBF kernel, deterministic samples ----
    {
        std::vector<mat2> samples;
        std::vector<double> labels;
        for (int i = 0; i < 100; ++i) {
            double px = std::sin(i * 1.7) * 2.0 + (i % 2 == 0 ? 1.7 : -1.7);
            double py = std::cos(i * 2.3) * 2.0 + (i % 3 == 0 ? 1.3 : -1.1);
            mat2 s; s = px, py;
            samples.push_back(s);
            labels.push_back((px + 0.5 * py > 0.1) ? +1.0 : -1.0);
        }
        svm_c_trainer<rbf> trainer;
        trainer.set_c(5.0);
        trainer.set_kernel(rbf(0.5));
        decision_function<rbf> df = trainer.train(samples, labels);
        std::printf("SVM_sv_count %zu\n", df.basis_vectors.size());
        std::printf("SVM_b %.17g\n", df.b);
        for (unsigned long i = 0; i < df.basis_vectors.size(); ++i)
            std::printf("SVM_sv %lu %.17g %.17g %.17g\n", i,
                        df.basis_vectors(i)(0), df.basis_vectors(i)(1),
                        df.alpha(i));
        int correct = 0;
        for (unsigned long i = 0; i < samples.size(); ++i)
            if ((df(samples[i]) > 0) == (labels[i] > 0)) ++correct;
        std::printf("SVM_correct %d\n", correct);
    }

    return 0;
}

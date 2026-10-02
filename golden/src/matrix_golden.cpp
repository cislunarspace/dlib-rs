// Golden outputs for the dlib-rs matrix port. Prints deterministic matrices
// and decomposition results as text with %.17g precision. The Rust test suite
// parses this file and compares within tight relative tolerances.
#include <dlib/matrix.h>
#include <cstdio>
#include <cmath>

using namespace dlib;

static void print_mat(const char* name, const matrix<double>& m) {
    std::printf("%s %ld %ld\n", name, m.nr(), m.nc());
    for (long r = 0; r < m.nr(); ++r) {
        for (long c = 0; c < m.nc(); ++c)
            std::printf("%.17g ", m(r, c));
        std::printf("\n");
    }
}

int main() {
    // Deterministic inputs.
    matrix<double> A6(6, 6);
    for (long r = 0; r < 6; ++r)
        for (long c = 0; c < 6; ++c)
            A6(r, c) = std::sin(0.7 * (r + 1) * (c + 1) + 0.01 * (r - c));

    matrix<double> B85(8, 5);
    for (long r = 0; r < 8; ++r)
        for (long c = 0; c < 5; ++c)
            B85(r, c) = std::cos(0.3 * (r + 2) * (c + 1)) + (r - c) * 0.05;

    matrix<double> S6 = A6 * trans(A6) + 6.0 * identity_matrix<double>(6);

    matrix<double> G5(5, 5);
    for (long r = 0; r < 5; ++r)
        for (long c = 0; c < 5; ++c)
            G5(r, c) = std::sin(0.11 * (r + 1) * (c + 2)) + 0.5 * std::cos(0.07 * (r + 2) * (c + 1))
                        + ((r == c) ? 2.0 : 0.0);

    print_mat("A6", A6);
    print_mat("B85", B85);
    print_mat("S6", S6);
    print_mat("G5", G5);

    matrix<double> b6(6, 1);
    for (long r = 0; r < 6; ++r) b6(r) = 1.0 + 0.25 * r;

    // LU
    {
        lu_decomposition<matrix<double> > lu(A6);
        std::printf("LU_det %.17g\n", lu.det());
        print_mat("LU_l", lu.get_l());
        print_mat("LU_u", lu.get_u());
        print_mat("LU_solve", lu.solve(b6));
        std::printf("LU_is_singular %d\n", lu.is_singular() ? 1 : 0);
    }

    // Cholesky
    {
        cholesky_decomposition<matrix<double> > ch(S6);
        print_mat("CH_l", ch.get_l());
        std::printf("CH_is_spd %d\n", ch.is_spd() ? 1 : 0);
        print_mat("CH_solve", ch.solve(b6));
    }

    // QR
    {
        matrix<double> b81(8, 1);
        for (long r = 0; r < 8; ++r) b81(r) = std::sin(0.2 * (r + 1));
        qr_decomposition<matrix<double> > qr(B85);
        print_mat("QR_q", qr.get_q());
        print_mat("QR_r", qr.get_r());
        std::printf("QR_is_full_rank %d\n", qr.is_full_rank() ? 1 : 0);
        print_mat("QR_solve", qr.solve(b81));
    }

    // Eigenvalue (symmetric)
    {
        eigenvalue_decomposition<matrix<double> > eig(S6);
        print_mat("EIG_real", eig.get_real_eigenvalues());
        print_mat("EIG_v", dlib::real(eig.get_v()));
    }

    // Eigenvalue (general, nonsymmetric)
    {
        eigenvalue_decomposition<matrix<double> > eig(G5);
        print_mat("EIGG_real", eig.get_real_eigenvalues());
        print_mat("EIGG_imag", eig.get_imag_eigenvalues());
        print_mat("EIGG_d", dlib::real(eig.get_d()));
    }

    // SVD
    {
        matrix<double> u, w, v;
        svd(B85, u, w, v);
        print_mat("SVD_u", u);
        print_mat("SVD_w", w);
        print_mat("SVD_v", v);
    }

    // inv / det
    print_mat("INV_A6", inv(A6));
    std::printf("DET_A6 %.17g\n", det(A6));

    return 0;
}

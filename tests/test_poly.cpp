#include <keel/poly.h>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (std::fabs(got - expected) > tol || std::isnan(got)) {
        std::fprintf(stderr, "FAIL: %s (got %.15g, expected %.15g, tol %.3g)\n", msg, got, expected, tol);
        ++g_failures;
    }
}

void test_roots_and_expand() {
    // (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6
    const keel::c64 expected_roots[3] = {keel::c64(1.0, 0.0), keel::c64(2.0, 0.0), keel::c64(-3.0, 0.0)};
    auto coeffs = keel::poly::from_roots(expected_roots);
    check(coeffs.size() == 4, "from_roots size");
    check_near(coeffs[0], 1.0, 1e-14, "from_roots c0");
    check_near(coeffs[1], 0.0, 1e-14, "from_roots c1");
    check_near(coeffs[2], -7.0, 1e-14, "from_roots c2");
    check_near(coeffs[3], 6.0, 1e-14, "from_roots c3");

    auto rs = keel::poly::roots(keel::ArrayRef<keel::f64>(coeffs.data(), coeffs.size()));
    check(rs.size() == 3, "roots count");
    for (keel::usize i = 0; i < 3; ++i) {
        const keel::c64 val = keel::poly::polyval(
            keel::ArrayRef<keel::c64>(keel::poly::from_roots_c64(expected_roots).data(), 4), rs[i]);
        check_near(keel::abs(val), 0.0, 1e-11, "root evaluates to zero");
    }

    // Leading and trailing zeros: 0*x^3 + x^2 - x = x(x - 1)
    const keel::f64 zpoly[4] = {0.0, 1.0, -1.0, 0.0};
    auto r2 = keel::poly::roots(zpoly);
    check(r2.size() == 2, "roots with leading/trailing zero");
}

void test_arith_and_calculus() {
    const keel::f64 a[2] = {1.0, -1.0};
    const keel::f64 b[2] = {1.0, 1.0};
    auto m = keel::poly::mul(a, b);
    check(m.size() == 3, "mul size");
    check_near(m[0], 1.0, 1e-14, "mul[0]");
    check_near(m[1], 0.0, 1e-14, "mul[1]");
    check_near(m[2], -1.0, 1e-14, "mul[2]");

    const keel::f64 num[4] = {1.0, 0.0, -1.0, 2.0};
    auto dv = keel::poly::div(num, a);
    check(dv.ok(), "div ok");
    check(dv.q.size() == 3, "div q size");
    check_near(dv.q[0], 1.0, 1e-14, "div q0");
    check_near(dv.q[1], 1.0, 1e-14, "div q1");
    check_near(dv.q[2], 0.0, 1e-14, "div q2");
    check_near(dv.r[0], 2.0, 1e-14, "div r0");

    auto dp = keel::poly::deriv(num);
    check(dp.size() == 3, "deriv size");
    check_near(dp[0], 3.0, 1e-14, "deriv[0]");
    check_near(dp[1], 0.0, 1e-14, "deriv[1]");
    check_near(dp[2], -1.0, 1e-14, "deriv[2]");

    auto ip = keel::poly::integ(dp, 2.0);
    check(ip.size() == 4, "integ size");
    check_near(ip[0], 1.0, 1e-14, "integ[0]");
    check_near(ip[3], 2.0, 1e-14, "integ[3]");
}

void test_polyfit_and_pade() {
    const keel::f64 xs[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    keel::f64 ys[5];
    for (keel::usize i = 0; i < 5; ++i) {
        ys[i] = 2.0 * xs[i] * xs[i] - 3.0 * xs[i] + 5.0;
    }
    auto fit = keel::poly::polyfit(xs, ys, 2);
    check(fit.ok(), "polyfit ok");
    check(fit.coeffs.size() == 3, "polyfit size");
    check_near(fit.coeffs[0], 2.0, 1e-12, "polyfit c2");
    check_near(fit.coeffs[1], -3.0, 1e-12, "polyfit c1");
    check_near(fit.coeffs[2], 5.0, 1e-12, "polyfit c0");

    auto pd = keel::poly::pade_delay(1.0, 2);
    check(pd.ok(), "pade_delay ok");
    check(pd.num.size() == 3 && pd.den.size() == 3, "pade_delay size");
    // Pade(1, 2): num = [1, -6, 12], den = [1, 6, 12]
    check_near(pd.num[0], 1.0, 1e-14, "pade num[0]");
    check_near(pd.num[1], -6.0, 1e-14, "pade num[1]");
    check_near(pd.num[2], 12.0, 1e-14, "pade num[2]");
    check_near(pd.den[1], 6.0, 1e-14, "pade den[1]");
}

void test_residue_and_residuez() {
    // Repeated pole: s^2 / (s + 1)^3 = 1/(s+1) - 2/(s+1)^2 + 1/(s+1)^3
    const keel::f64 b[3] = {1.0, 0.0, 0.0};
    const keel::f64 a[4] = {1.0, 3.0, 3.0, 1.0};
    auto res = keel::poly::residue(b, a);
    check(res.ok(), "residue repeated ok");
    check(res.r.size() == 3, "residue r size");
    check_near(res.r[0].real(), 1.0, 1e-9, "residue r[0]");
    check_near(res.r[1].real(), -2.0, 1e-9, "residue r[1]");
    check_near(res.r[2].real(), 1.0, 1e-9, "residue r[2]");
    check_near(res.p[0].real(), -1.0, 1e-9, "residue pole");

    // Simple z-domain: 1 / ((1 - 0.5 z^-1)(1 - 0.25 z^-1)) = 2/(1-0.5z^-1) - 1/(1-0.25z^-1)
    const keel::f64 bz[1] = {1.0};
    const keel::f64 az[3] = {1.0, -0.75, 0.125};
    auto rz = keel::poly::residuez(bz, az);
    check(rz.ok(), "residuez ok");
    check(rz.r.size() == 2, "residuez r size");
    check_near(rz.p[0].real(), 0.5, 1e-11, "residuez p[0]");
    check_near(rz.r[0].real(), 2.0, 1e-11, "residuez r[0]");
    check_near(rz.p[1].real(), 0.25, 1e-11, "residuez p[1]");
    check_near(rz.r[1].real(), -1.0, 1e-11, "residuez r[1]");
}

} // namespace

int main() {
    test_roots_and_expand();
    test_arith_and_calculus();
    test_polyfit_and_pade();
    test_residue_and_residuez();
    if (g_failures == 0) {
        std::printf("ALL keel::poly TESTS PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
    return 1;
}

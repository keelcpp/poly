#include <keel/format.h>
#include <keel/math.h>
#include <keel/poly.h>
#include <keel/string.h>

namespace {

keel::i32 g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        keel::eprintln("FAIL: %s", msg);
        ++g_failures;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* msg) {
    if (keel::abs(got - expected) > tol || keel::is_nan(got)) {
        keel::eprintln("FAIL: %s (got %.15g, expected %.15g, tol %.3g)", msg, got, expected, tol);
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
    auto ccoeffs = keel::poly::from_roots_c64(expected_roots);
    for (keel::usize i = 0; i < 3; ++i) {
        const keel::c64 val = keel::poly::polyval(
            keel::ArrayRef<keel::c64>(ccoeffs.data(), ccoeffs.size()), rs[i]);
        check_near(keel::abs(val), 0.0, 1e-12, "root evaluates to zero");
    }

    // Leading and trailing zeros: 0*x^3 + x^2 - x = x(x - 1)
    const keel::f64 zpoly[4] = {0.0, 1.0, -1.0, 0.0};
    auto r2 = keel::poly::roots(zpoly);
    check(r2.size() == 2, "roots with leading/trailing zero");

    // Large-scale roots (previously overflowed Cauchy radius): (x - 1e4)(x - 2e4)(x - 3e4)(x - 4e4)
    const keel::c64 big_r[4] = {
        keel::c64(1e4, 0.0), keel::c64(2e4, 0.0), keel::c64(3e4, 0.0), keel::c64(4e4, 0.0)
    };
    auto big_p = keel::poly::from_roots_c64(big_r);
    auto big_found = keel::poly::roots(keel::ArrayRef<keel::c64>(big_p.data(), big_p.size()));
    check(big_found.size() == 4, "big roots count");
    for (keel::usize i = 0; i < 4; ++i) {
        keel::f64 best = 1e30;
        for (keel::usize j = 0; j < 4; ++j) {
            best = keel::min(best, keel::abs(big_found[i] - big_r[j]));
        }
        check(best < 1e-8, "big root accuracy");
    }

    // Tiny-scale roots: (x - 1e-8)(x + 2e-8)(x - 3e-8)
    const keel::c64 tiny_r[3] = {keel::c64(1e-8, 0.0), keel::c64(-2e-8, 0.0), keel::c64(3e-8, 0.0)};
    auto tiny_p = keel::poly::from_roots_c64(tiny_r);
    auto tiny_found = keel::poly::roots(keel::ArrayRef<keel::c64>(tiny_p.data(), tiny_p.size()));
    check(tiny_found.size() == 3, "tiny roots count");
    for (keel::usize i = 0; i < 3; ++i) {
        keel::f64 best = 1e30;
        for (keel::usize j = 0; j < 3; ++j) {
            best = keel::min(best, keel::abs(tiny_found[i] - tiny_r[j]));
        }
        check(best < 1e-20, "tiny root accuracy");
    }

    // Non-finite input returns NaN roots without hanging
    const keel::f64 nan_p[3] = {1.0, keel::kNaN, 1.0};
    auto nan_r = keel::poly::roots(nan_p);
    check(nan_r.size() == 2 && keel::is_nan(nan_r[0].real()), "roots NaN guard");
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

    const keel::f64 zero_a[2] = {0.0, 0.0};
    auto dvz = keel::poly::div(num, zero_a);
    check(dvz.status == keel::poly::Status::ZeroDivisor, "div zero divisor");

    auto dp = keel::poly::deriv(num);
    check(dp.size() == 3, "deriv size");
    check_near(dp[0], 3.0, 1e-14, "deriv[0]");
    check_near(dp[1], 0.0, 1e-14, "deriv[1]");
    check_near(dp[2], -1.0, 1e-14, "deriv[2]");

    auto dr = keel::poly::deriv_ratio(b, a);
    // d/dx ((x+1)/(x-1)) = -2 / (x-1)^2
    check(dr.num.size() == 1 && dr.den.size() == 3, "deriv_ratio sizes");
    check_near(dr.num[0], -2.0, 1e-14, "deriv_ratio num");
    check_near(dr.den[0], 1.0, 1e-14, "deriv_ratio den[0]");
    check_near(dr.den[1], -2.0, 1e-14, "deriv_ratio den[1]");
    check_near(dr.den[2], 1.0, 1e-14, "deriv_ratio den[2]");

    auto ip = keel::poly::integ(dp, 2.0);
    check(ip.size() == 4, "integ size");
    check_near(ip[0], 1.0, 1e-14, "integ[0]");
    check_near(ip[3], 2.0, 1e-14, "integ[3]");

    // polyvalm safety on undersized matrix
    const keel::c64 pm_c[2] = {keel::c64(1.0, 0.0), keel::c64(2.0, 0.0)};
    const keel::c64 pm_small[1] = {keel::c64(3.0, 0.0)};
    auto pm_bad = keel::poly::polyvalm(pm_c, pm_small, 100);
    check(pm_bad.empty(), "polyvalm rejects undersized matrix without huge alloc");
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

    // Shifted x conditioning test: x around 1000
    const keel::f64 xsh[5] = {1000.0, 1001.0, 1002.0, 1003.0, 1004.0};
    keel::f64 ysh[5];
    for (keel::usize i = 0; i < 5; ++i) {
        const keel::f64 d = xsh[i] - 1002.0;
        ysh[i] = 1.5 * d * d - 2.0 * d + 3.0;
    }
    auto fit_sh = keel::poly::polyfit(xsh, ysh, 2);
    check(fit_sh.ok(), "polyfit shifted ok");
    for (keel::usize i = 0; i < 5; ++i) {
        const keel::f64 pred = keel::poly::polyval(
            keel::ArrayRef<keel::f64>(fit_sh.coeffs.data(), fit_sh.coeffs.size()), xsh[i]);
        check_near(pred, ysh[i], 1e-8, "polyfit shifted eval");
    }

    // Identical x and NaN guards
    const keel::f64 xdup[3] = {5.0, 5.0, 5.0};
    const keel::f64 ydup[3] = {1.0, 2.0, 3.0};
    auto fit_dup = keel::poly::polyfit(xdup, ydup, 2);
    check(fit_dup.ok(), "polyfit identical x ok");
    check_near(fit_dup.coeffs[2], 2.0, 1e-14, "polyfit identical x mean");

    const keel::f64 xnan[2] = {1.0, keel::kNaN};
    const keel::f64 ynan[2] = {1.0, 2.0};
    auto fit_nan = keel::poly::polyfit(xnan, ynan, 1);
    check(fit_nan.status == keel::poly::Status::InvalidInput, "polyfit NaN rejected");

    // Pade delay: normal, zero delay, negative/NaN rejection
    auto pd = keel::poly::pade_delay(1.0, 2);
    check(pd.ok(), "pade_delay ok");
    check(pd.num.size() == 3 && pd.den.size() == 3, "pade_delay size");
    check_near(pd.num[0], 1.0, 1e-14, "pade num[0]");
    check_near(pd.num[1], -6.0, 1e-14, "pade num[1]");
    check_near(pd.num[2], 12.0, 1e-14, "pade num[2]");
    check_near(pd.den[1], 6.0, 1e-14, "pade den[1]");

    auto pd0 = keel::poly::pade_delay(0.0, 3);
    check(pd0.ok() && pd0.num.size() == 1 && pd0.den.size() == 1, "pade_delay(0,3) -> {1},{1}");
    check_near(pd0.num[0], 1.0, 1e-15, "pade_delay(0,3) num");
    check_near(pd0.den[0], 1.0, 1e-15, "pade_delay(0,3) den");

    check(keel::poly::pade_delay(-1.0, 2).status == keel::poly::Status::InvalidInput, "pade negative t");
    check(keel::poly::pade_delay(keel::kNaN, 2).status == keel::poly::Status::InvalidInput, "pade NaN t");
}

void test_residue_and_residuez() {
    // Repeated pole: s^2 / (s + 1)^3 = 1/(s+1) - 2/(s+1)^2 + 1/(s+1)^3
    const keel::f64 b[3] = {1.0, 0.0, 0.0};
    const keel::f64 a[4] = {1.0, 3.0, 3.0, 1.0};
    auto res = keel::poly::residue(b, a);
    check(res.ok(), "residue repeated ok");
    check(res.r.size() == 3, "residue r size");
    check_near(res.r[0].real(), 1.0, 1e-12, "residue r[0]");
    check_near(res.r[1].real(), -2.0, 1e-12, "residue r[1]");
    check_near(res.r[2].real(), 1.0, 1e-12, "residue r[2]");
    check_near(res.p[0].real(), -1.0, 1e-13, "residue pole");

    // Simple z-domain: 1 / ((1 - 0.5 z^-1)(1 - 0.25 z^-1)) = 2/(1-0.5z^-1) - 1/(1-0.25z^-1)
    const keel::f64 bz[1] = {1.0};
    const keel::f64 az[3] = {1.0, -0.75, 0.125};
    auto rz = keel::poly::residuez(bz, az);
    check(rz.ok(), "residuez ok");
    check(rz.r.size() == 2, "residuez r size");
    check_near(rz.p[0].real(), 0.5, 1e-13, "residuez p[0]");
    check_near(rz.r[0].real(), 2.0, 1e-13, "residuez r[0]");
    check_near(rz.p[1].real(), 0.25, 1e-13, "residuez p[1]");
    check_near(rz.r[1].real(), -1.0, 1e-13, "residuez r[1]");

    // z-domain with leading zero in b (z^-1 delay): z^-1 / (1 - 0.5 z^-1) = -2 + 2 / (1 - 0.5 z^-1)
    const keel::f64 bz_delay[2] = {0.0, 1.0};
    const keel::f64 az_delay[2] = {1.0, -0.5};
    auto rz_delay = keel::poly::residuez(bz_delay, az_delay);
    check(rz_delay.ok(), "residuez leading-zero b ok");
    check(rz_delay.r.size() == 1 && rz_delay.k.size() == 1, "residuez leading-zero sizes");
    check_near(rz_delay.r[0].real(), 2.0, 1e-13, "residuez leading-zero r");
    check_near(rz_delay.p[0].real(), 0.5, 1e-13, "residuez leading-zero p");
    check_near(rz_delay.k[0], -2.0, 1e-13, "residuez leading-zero k");

    // z-domain with trailing zero in a (must not divide by 0)
    const keel::f64 az_trail[3] = {1.0, -0.5, 0.0};
    auto rz_trail = keel::poly::residuez(bz, az_trail);
    check(rz_trail.ok() && rz_trail.r.size() == 1, "residuez trailing-zero a ok");
    check_near(rz_trail.r[0].real(), 1.0, 1e-13, "residuez trailing-zero r");
    check_near(rz_trail.p[0].real(), 0.5, 1e-13, "residuez trailing-zero p");

    // z-domain triple pole: 1 / (1 - 0.5 z^-1)^3
    const keel::f64 az_triple[4] = {1.0, -1.5, 0.75, -0.125};
    auto rz_tri = keel::poly::residuez(bz, az_triple);
    check(rz_tri.ok() && rz_tri.r.size() == 3, "residuez triple pole ok");
    check_near(rz_tri.p[0].real(), 0.5, 1e-13, "residuez triple p");
    check_near(rz_tri.r[0].real(), 0.0, 1e-12, "residuez triple r0");
    check_near(rz_tri.r[1].real(), 0.0, 1e-12, "residuez triple r1");
    check_near(rz_tri.r[2].real(), 1.0, 1e-12, "residuez triple r2");
}

void dump_f64_array(const char* name, keel::ArrayRef<keel::f64> v) {
    keel::print("%s:", name);
    for (keel::usize i = 0; i < v.size(); ++i) {
        keel::print(" %.17g", v[i]);
    }
    keel::println("");
}

void dump_c64_array(const char* name, keel::ArrayRef<keel::c64> v) {
    keel::print("%s:", name);
    for (keel::usize i = 0; i < v.size(); ++i) {
        keel::print(" %.17g %.17g", v[i].real(), v[i].imag());
    }
    keel::println("");
}

void dump_reference() {
    // 1. polyval + polyval_batch + horner_deriv
    const keel::f64 pv_c[5] = {2.5, -1.25, 0.0, 3.75, -4.5};
    const keel::f64 pv_x[4] = {-2.5, 0.0, 1.5, 3.25};
    keel::f64 pv_out[4];
    keel::poly::polyval_batch(pv_c, pv_x, pv_out);
    dump_f64_array("polyval_batch", pv_out);

    const keel::c64 hd_c[4] = {
        keel::c64(1.0, -0.5), keel::c64(-2.0, 1.0), keel::c64(0.5, 0.25), keel::c64(3.0, -1.0)
    };
    keel::c64 h_p;
    keel::c64 h_dp;
    keel::poly::horner_deriv(hd_c, keel::c64(1.2, -0.7), h_p, h_dp);
    const keel::c64 hd_out[2] = {h_p, h_dp};
    dump_c64_array("horner_deriv", hd_out);

    // 2. polyvalm (3x3 column-major)
    const keel::c64 pm_c[4] = {
        keel::c64(2.0, 0.0), keel::c64(-1.5, 0.0), keel::c64(0.5, 0.0), keel::c64(3.0, 0.0)
    };
    const keel::c64 pm_a[9] = {
        keel::c64(1.0, 0.0), keel::c64(2.0, 0.0), keel::c64(-1.0, 0.0),
        keel::c64(0.5, 0.0), keel::c64(-1.0, 0.0), keel::c64(3.0, 0.0),
        keel::c64(2.0, 0.0), keel::c64(1.0, 0.0), keel::c64(0.25, 0.0)
    };
    auto pm_res = keel::poly::polyvalm(pm_c, pm_a, 3);
    dump_c64_array("polyvalm", keel::ArrayRef<keel::c64>(pm_res.data(), pm_res.size()));

    // 3. roots: real mixed, complex coeffs, high-degree Wilkinson-10, scaled huge/tiny
    const keel::f64 r_real[6] = {1.0, -3.0, 4.0, -2.0, 5.0, -1.0};
    auto r_real_out = keel::poly::roots(r_real);
    dump_c64_array("roots_real", keel::ArrayRef<keel::c64>(r_real_out.data(), r_real_out.size()));

    auto r_cx_out = keel::poly::roots(hd_c);
    dump_c64_array("roots_cx", keel::ArrayRef<keel::c64>(r_cx_out.data(), r_cx_out.size()));

    keel::c64 w10_r[10];
    for (keel::usize i = 0; i < 10; ++i) w10_r[i] = keel::c64(keel::as<keel::f64>(i + 1), 0.0);
    auto w10_p = keel::poly::from_roots(w10_r);
    dump_f64_array("from_roots_w10", keel::ArrayRef<keel::f64>(w10_p.data(), w10_p.size()));
    auto w10_found = keel::poly::roots(keel::ArrayRef<keel::f64>(w10_p.data(), w10_p.size()));
    dump_c64_array("roots_w10", keel::ArrayRef<keel::c64>(w10_found.data(), w10_found.size()));

    const keel::c64 big_r[4] = {
        keel::c64(1e5, 0.0), keel::c64(-2e5, 3e5), keel::c64(-2e5, -3e5), keel::c64(5e5, 0.0)
    };
    auto big_p = keel::poly::from_roots(big_r);
    auto big_found = keel::poly::roots(keel::ArrayRef<keel::f64>(big_p.data(), big_p.size()));
    dump_c64_array("roots_big", keel::ArrayRef<keel::c64>(big_found.data(), big_found.size()));

    // 4. mul, div, deriv, deriv_ratio, integ
    const keel::f64 pa[4] = {1.5, -2.0, 0.5, 4.0};
    const keel::f64 pb[3] = {2.0, 1.0, -3.0};
    auto pmul = keel::poly::mul(pa, pb);
    dump_f64_array("mul", keel::ArrayRef<keel::f64>(pmul.data(), pmul.size()));

    auto pdiv = keel::poly::div(pa, pb);
    dump_f64_array("div_q", keel::ArrayRef<keel::f64>(pdiv.q.data(), pdiv.q.size()));
    dump_f64_array("div_r", keel::ArrayRef<keel::f64>(pdiv.r.data(), pdiv.r.size()));

    auto pder = keel::poly::deriv(pa);
    dump_f64_array("deriv", keel::ArrayRef<keel::f64>(pder.data(), pder.size()));

    auto pdr = keel::poly::deriv_ratio(pa, pb);
    dump_f64_array("deriv_ratio_num", keel::ArrayRef<keel::f64>(pdr.num.data(), pdr.num.size()));
    dump_f64_array("deriv_ratio_den", keel::ArrayRef<keel::f64>(pdr.den.data(), pdr.den.size()));

    auto pint = keel::poly::integ(pa, -7.25);
    dump_f64_array("integ", keel::ArrayRef<keel::f64>(pint.data(), pint.size()));

    // 5. polyfit: noisy least-squares + shifted x
    const keel::f64 fx[8] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    const keel::f64 fy[8] = {12.1, 4.9, 1.2, 0.8, 3.1, 8.8, 18.2, 30.9};
    auto pf1 = keel::poly::polyfit(fx, fy, 3);
    dump_f64_array("polyfit_lsq", keel::ArrayRef<keel::f64>(pf1.coeffs.data(), pf1.coeffs.size()));

    const keel::f64 fx2[6] = {100.0, 101.0, 102.0, 103.0, 104.0, 105.0};
    const keel::f64 fy2[6] = {1.0, 2.3, 4.1, 7.2, 11.5, 17.8};
    auto pf2 = keel::poly::polyfit(fx2, fy2, 3);
    dump_f64_array("polyfit_shifted", keel::ArrayRef<keel::f64>(pf2.coeffs.data(), pf2.coeffs.size()));

    // 6. pade_delay: order 0, order 5 (t=0.5), order 8 (t=2.5), zero delay
    auto pade5 = keel::poly::pade_delay(0.5, 5);
    dump_f64_array("pade5_num", keel::ArrayRef<keel::f64>(pade5.num.data(), pade5.num.size()));
    dump_f64_array("pade5_den", keel::ArrayRef<keel::f64>(pade5.den.data(), pade5.den.size()));

    auto pade8 = keel::poly::pade_delay(2.5, 8);
    dump_f64_array("pade8_num", keel::ArrayRef<keel::f64>(pade8.num.data(), pade8.num.size()));
    dump_f64_array("pade8_den", keel::ArrayRef<keel::f64>(pade8.den.data(), pade8.den.size()));

    auto pade0 = keel::poly::pade_delay(0.0, 4);
    dump_f64_array("pade0_num", keel::ArrayRef<keel::f64>(pade0.num.data(), pade0.num.size()));
    dump_f64_array("pade0_den", keel::ArrayRef<keel::f64>(pade0.den.data(), pade0.den.size()));

    // 7. residue: improper simple + complex conjugate poles + repeated pole
    const keel::f64 rs_b1[5] = {2.0, -1.0, 3.0, 1.0, -5.0};
    const keel::f64 rs_a1[4] = {1.0, 3.0, 4.0, 2.0}; // (s+1)(s^2+2s+2)
    auto rs1 = keel::poly::residue(rs_b1, rs_a1);
    dump_c64_array("residue1_r", keel::ArrayRef<keel::c64>(rs1.r.data(), rs1.r.size()));
    dump_c64_array("residue1_p", keel::ArrayRef<keel::c64>(rs1.p.data(), rs1.p.size()));
    dump_f64_array("residue1_k", keel::ArrayRef<keel::f64>(rs1.k.data(), rs1.k.size()));

    // (s+2)^3 * (s-1) = s^4 + 5s^3 + 6s^2 - 4s - 8
    const keel::f64 rs_b2[4] = {3.0, -2.0, 1.0, 4.0};
    const keel::f64 rs_a2[5] = {1.0, 5.0, 6.0, -4.0, -8.0};
    auto rs2 = keel::poly::residue(rs_b2, rs_a2);
    dump_c64_array("residue2_r", keel::ArrayRef<keel::c64>(rs2.r.data(), rs2.r.size()));
    dump_c64_array("residue2_p", keel::ArrayRef<keel::c64>(rs2.p.data(), rs2.p.size()));
    dump_f64_array("residue2_k", keel::ArrayRef<keel::f64>(rs2.k.data(), rs2.k.size()));

    // 8. residuez: improper + leading-zero delay + repeated pole
    const keel::f64 rz_b1[4] = {1.0, 2.0, -1.0, 0.5};
    const keel::f64 rz_a1[3] = {1.0, -0.75, 0.125};
    auto rz1 = keel::poly::residuez(rz_b1, rz_a1);
    dump_c64_array("residuez1_r", keel::ArrayRef<keel::c64>(rz1.r.data(), rz1.r.size()));
    dump_c64_array("residuez1_p", keel::ArrayRef<keel::c64>(rz1.p.data(), rz1.p.size()));
    dump_f64_array("residuez1_k", keel::ArrayRef<keel::f64>(rz1.k.data(), rz1.k.size()));

    const keel::f64 rz_b2[3] = {0.0, 1.0, -0.25};
    const keel::f64 rz_a2[3] = {1.0, -1.0, 0.25}; // (1 - 0.5 z^-1)^2
    auto rz2 = keel::poly::residuez(rz_b2, rz_a2);
    dump_c64_array("residuez2_r", keel::ArrayRef<keel::c64>(rz2.r.data(), rz2.r.size()));
    dump_c64_array("residuez2_p", keel::ArrayRef<keel::c64>(rz2.p.data(), rz2.p.size()));
    dump_f64_array("residuez2_k", keel::ArrayRef<keel::f64>(rz2.k.data(), rz2.k.size()));
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
    test_roots_and_expand();
    test_arith_and_calculus();
    test_polyfit_and_pade();
    test_residue_and_residuez();
    if (g_failures == 0) {
        keel::println("ALL keel::poly TESTS PASSED");
        return 0;
    }
    keel::eprintln("%d FAILURE(S)", g_failures);
    return 1;
}

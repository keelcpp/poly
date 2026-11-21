#include "keel/poly.h"

namespace keel::poly {

namespace {

void trim_leading_zeros(Array<f64>& v) {
    usize lo = 0;
    while (lo + 1 < v.size() && v[lo] == 0.0) ++lo;
    if (lo > 0) {
        const usize new_n = v.size() - lo;
        for (usize i = 0; i < new_n; ++i) v[i] = v[lo + i];
        v.resize(new_n);
    }
}

Array<f64> strip_leading_copy(ArrayRef<f64> in, Allocator* alloc) {
    Array<f64> v(alloc);
    if (in.empty()) return v;
    usize lo = 0;
    while (lo + 1 < in.size() && in[lo] == 0.0) ++lo;
    const usize n = in.size() - lo;
    v.resize(n);
    for (usize i = 0; i < n; ++i) v[i] = in[lo + i];
    return v;
}

Array<f64> deriv_raw(ArrayRef<f64> p, Allocator* alloc) {
    Array<f64> r(alloc);
    if (p.size() <= 1) {
        r.resize(1);
        r[0] = 0.0;
        return r;
    }
    const usize n = p.size() - 1;
    r.resize(n);
    for (usize i = 0; i < n; ++i) {
        r[i] = p[i] * as<f64>(n - i);
    }
    return r;
}

c64 horner_real_at_c64(ArrayRef<f64> coeffs, c64 z) noexcept {
    if (coeffs.empty()) return c64(0.0, 0.0);
    c64 acc(coeffs[0], 0.0);
    for (usize i = 1; i < coeffs.size(); ++i) {
        acc = acc * z + c64(coeffs[i], 0.0);
    }
    return acc;
}

Array<c64> synthetic_divide_c64(ArrayRef<c64> c, c64 p, Allocator* alloc) {
    Array<c64> q(alloc);
    if (c.size() <= 1) return q;
    q.resize(c.size() - 1);
    q[0] = c[0];
    for (usize i = 1; i + 1 < c.size(); ++i) {
        q[i] = c[i] + p * q[i - 1];
    }
    return q;
}

Array<c64> taylor_coeffs_c64(ArrayRef<c64> c, c64 p, usize order, Allocator* alloc) {
    Array<c64> t(alloc);
    t.reserve(order);
    Array<c64> q(alloc);
    q.resize(c.size());
    for (usize i = 0; i < c.size(); ++i) q[i] = c[i];

    c64 fact(1.0, 0.0);
    for (usize d = 0; d < order; ++d) {
        c64 acc(0.0, 0.0);
        if (!q.empty()) {
            acc = q[0];
            for (usize i = 1; i < q.size(); ++i) acc = acc * p + q[i];
        }
        t.push_back(acc / fact);
        fact *= c64(as<f64>(d) + 1.0, 0.0);
        if (q.size() > 1) {
            const usize n = q.size() - 1;
            Array<c64> dq(alloc);
            dq.resize(n);
            for (usize i = 0; i < n; ++i) {
                dq[i] = q[i] * as<f64>(n - i);
            }
            q = move(dq);
        } else {
            q.clear();
        }
    }
    return t;
}

Array<c64> zfilter_c64(ArrayRef<c64> bb, ArrayRef<c64> aa, ArrayRef<c64> xx, Allocator* alloc) {
    Array<c64> y(alloc);
    y.resize(xx.size());
    for (usize n = 0; n < xx.size(); ++n) {
        c64 acc(0.0, 0.0);
        for (usize k = 0; k < bb.size() && k <= n; ++k) acc += bb[k] * xx[n - k];
        for (usize k = 1; k < aa.size() && k <= n; ++k) acc -= aa[k] * y[n - k];
        y[n] = acc;
    }
    return y;
}

} // namespace

// ============================================================================
// Horner Evaluation & Matrix Polynomials
// ============================================================================

f64 polyval(ArrayRef<f64> coeffs, f64 x) noexcept {
    if (coeffs.empty()) return 0.0;
    f64 val = coeffs[0];
    for (usize i = 1; i < coeffs.size(); ++i) {
        val = val * x + coeffs[i];
    }
    return val;
}

c64 polyval(ArrayRef<c64> coeffs, c64 z) noexcept {
    if (coeffs.empty()) return c64(0.0, 0.0);
    c64 val = coeffs[0];
    for (usize i = 1; i < coeffs.size(); ++i) {
        val = val * z + coeffs[i];
    }
    return val;
}

void polyval_batch(ArrayRef<f64> coeffs, ArrayRef<f64> xs, MutArrayRef<f64> out) noexcept {
    const usize n = min(xs.size(), out.size());
    if (coeffs.empty()) {
        for (usize i = 0; i < n; ++i) out[i] = 0.0;
        return;
    }
    const usize np = coeffs.size();
    for (usize i = 0; i < n; ++i) {
        f64 val = coeffs[0];
        const f64 x = xs[i];
        for (usize j = 1; j < np; ++j) {
            val = val * x + coeffs[j];
        }
        out[i] = val;
    }
}

void horner_deriv(ArrayRef<c64> coeffs, c64 z, c64& out_p, c64& out_dp) noexcept {
    if (coeffs.empty()) {
        out_p = c64(0.0, 0.0);
        out_dp = c64(0.0, 0.0);
        return;
    }
    c64 p = coeffs[0];
    c64 dp(0.0, 0.0);
    for (usize i = 1; i < coeffs.size(); ++i) {
        dp = dp * z + p;
        p = p * z + coeffs[i];
    }
    out_p = p;
    out_dp = dp;
}

Array<c64> polyvalm(ArrayRef<c64> coeffs, ArrayRef<c64> a_colmajor, usize n, Allocator* alloc) {
    Array<c64> result(alloc);
    const usize nn = n * n;
    result.resize(nn);
    for (usize i = 0; i < nn; ++i) result[i] = c64(0.0, 0.0);
    if (coeffs.empty() || n == 0 || a_colmajor.size() < nn) {
        return result;
    }

    const c64 p0 = coeffs[0];
    for (usize i = 0; i < n; ++i) result[i * n + i] = p0;

    Array<c64> next(alloc);
    next.resize(nn);
    for (usize step = 1; step < coeffs.size(); ++step) {
        for (usize i = 0; i < n; ++i) {
            for (usize j = 0; j < n; ++j) {
                c64 s(0.0, 0.0);
                for (usize kk = 0; kk < n; ++kk) {
                    s += result[kk * n + i] * a_colmajor[j * n + kk];
                }
                next[j * n + i] = s;
            }
        }
        const c64 pi = coeffs[step];
        for (usize i = 0; i < n; ++i) next[i * n + i] += pi;
        swap(result, next);
    }
    return result;
}

// ============================================================================
// Roots & Expansion
// ============================================================================

Array<c64> roots(ArrayRef<c64> coeffs, Allocator* alloc) {
    usize lo = 0;
    usize hi = coeffs.size();
    while (lo < hi && coeffs[lo] == c64(0.0, 0.0)) ++lo;
    usize orig_zeros = 0;
    while (hi > lo && coeffs[hi - 1] == c64(0.0, 0.0)) {
        --hi;
        ++orig_zeros;
    }
    if (hi - lo <= 1) {
        Array<c64> z(alloc);
        z.resize(orig_zeros);
        for (usize i = 0; i < orig_zeros; ++i) z[i] = c64(0.0, 0.0);
        return z;
    }

    const usize m = hi - lo;
    Array<c64> c(alloc);
    c.resize(m);
    for (usize i = 0; i < m; ++i) c[i] = coeffs[lo + i] / coeffs[lo];
    const usize n = m - 1;

    f64 max_coef = 0.0;
    for (usize i = 1; i <= n; ++i) max_coef = max(max_coef, abs(c[i]));
    const f64 radius = 1.0 + max_coef;

    Array<c64> r(alloc);
    r.resize(n);
    constexpr f64 kTwoPi = 6.28318530717958647692;
    for (usize k = 0; k < n; ++k) {
        const f64 theta = kTwoPi * as<f64>(k) / as<f64>(n) + 0.4;
        r[k] = c64(radius * cos(theta), radius * sin(theta));
    }

    constexpr i32 kMaxIter = 200;
    constexpr f64 kTol = 1e-14;
    for (i32 it = 0; it < kMaxIter; ++it) {
        f64 max_step = 0.0;
        for (usize k = 0; k < n; ++k) {
            c64 pv;
            c64 dpv;
            horner_deriv(ArrayRef<c64>(c.data(), c.size()), r[k], pv, dpv);
            c64 denom(1.0, 0.0);
            for (usize j = 0; j < n; ++j) {
                if (j != k) denom *= (r[k] - r[j]);
            }
            if (abs(denom) < 1e-300) continue;
            const c64 step = pv / denom;
            r[k] -= step;
            const f64 s = abs(step);
            if (s > max_step) max_step = s;
        }
        if (max_step < kTol) break;
    }

    if (orig_zeros > 0) {
        Array<c64> out(alloc);
        out.resize(orig_zeros + n);
        for (usize i = 0; i < orig_zeros; ++i) out[i] = c64(0.0, 0.0);
        for (usize i = 0; i < n; ++i) out[orig_zeros + i] = r[i];
        return out;
    }
    return r;
}

Array<c64> roots(ArrayRef<f64> coeffs, Allocator* alloc) {
    Array<c64> cc(alloc);
    cc.resize(coeffs.size());
    for (usize i = 0; i < coeffs.size(); ++i) cc[i] = c64(coeffs[i], 0.0);
    return roots(ArrayRef<c64>(cc.data(), cc.size()), alloc);
}

Array<c64> from_roots_c64(ArrayRef<c64> root_list, Allocator* alloc) {
    const usize n = root_list.size();
    Array<c64> p(alloc);
    p.resize(n + 1);
    p[0] = c64(1.0, 0.0);
    for (usize i = 1; i <= n; ++i) p[i] = c64(0.0, 0.0);
    for (usize i = 0; i < n; ++i) {
        for (usize j = i + 1; j >= 1; --j) {
            p[j] = p[j] - root_list[i] * p[j - 1];
        }
    }
    return p;
}

Array<f64> from_roots(ArrayRef<c64> root_list, Allocator* alloc) {
    Array<c64> pc = from_roots_c64(root_list, alloc);
    Array<f64> r(alloc);
    r.resize(pc.size());
    for (usize i = 0; i < pc.size(); ++i) r[i] = pc[i].real();
    return r;
}

// ============================================================================
// Arithmetic & Calculus
// ============================================================================

Array<f64> mul(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc) {
    Array<f64> r(alloc);
    if (a.empty() || b.empty()) return r;
    const usize rn = a.size() + b.size() - 1;
    r.resize(rn);
    for (usize i = 0; i < rn; ++i) r[i] = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        for (usize j = 0; j < b.size(); ++j) {
            r[i + j] += a[i] * b[j];
        }
    }
    return r;
}

DivResult div(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc) {
    DivResult res{Status::Ok, Array<f64>(alloc), Array<f64>(alloc)};
    if (a.empty()) {
        res.status = Status::ZeroDivisor;
        return res;
    }
    usize a_off = 0;
    while (a_off + 1 < a.size() && a[a_off] == 0.0) ++a_off;
    const usize a_eff = a.size() - a_off;
    if (a_eff == 0 || a[a_off] == 0.0) {
        res.status = Status::ZeroDivisor;
        return res;
    }

    const usize nb = b.size();
    if (nb < a_eff) {
        res.q.resize(1);
        res.q[0] = 0.0;
        res.r.resize(nb);
        for (usize i = 0; i < nb; ++i) res.r[i] = b[i];
        return res;
    }

    Array<f64> bb(alloc);
    bb.resize(nb);
    for (usize i = 0; i < nb; ++i) bb[i] = b[i];

    const f64 a_lead = a[a_off];
    const usize q_len = nb - a_eff + 1;
    res.q.resize(q_len);
    for (usize i = 0; i < q_len; ++i) {
        const f64 coef = bb[i] / a_lead;
        res.q[i] = coef;
        for (usize j = 0; j < a_eff; ++j) {
            bb[i + j] -= coef * a[a_off + j];
        }
    }

    usize r_off = q_len;
    while (r_off < nb && bb[r_off] == 0.0) ++r_off;
    if (r_off >= nb) {
        res.r.resize(1);
        res.r[0] = 0.0;
    } else {
        const usize r_len = nb - r_off;
        res.r.resize(r_len);
        for (usize i = 0; i < r_len; ++i) res.r[i] = bb[r_off + i];
    }
    return res;
}

Array<f64> deriv(ArrayRef<f64> p, Allocator* alloc) {
    Array<f64> d = deriv_raw(p, alloc);
    trim_leading_zeros(d);
    return d;
}

DerivRatioResult deriv_ratio(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc) {
    Array<f64> bp = deriv_raw(b, alloc);
    Array<f64> ap = deriv_raw(a, alloc);
    Array<f64> t1 = mul(a, ArrayRef<f64>(bp.data(), bp.size()), alloc);
    Array<f64> t2 = mul(b, ArrayRef<f64>(ap.data(), ap.size()), alloc);

    auto prepend_zeros = [](Array<f64>& vec, usize pad) {
        const usize old_n = vec.size();
        vec.resize(old_n + pad);
        for (usize i = old_n; i > 0; --i) vec[i - 1 + pad] = vec[i - 1];
        for (usize i = 0; i < pad; ++i) vec[i] = 0.0;
    };

    if (t1.size() < t2.size()) prepend_zeros(t1, t2.size() - t1.size());
    if (t2.size() < t1.size()) prepend_zeros(t2, t1.size() - t2.size());

    Array<f64> num(alloc);
    num.resize(t1.size());
    for (usize i = 0; i < t1.size(); ++i) num[i] = t1[i] - t2[i];
    trim_leading_zeros(num);

    Array<f64> den = mul(a, a, alloc);
    trim_leading_zeros(den);
    return {move(num), move(den)};
}

Array<f64> integ(ArrayRef<f64> p, f64 k, Allocator* alloc) {
    Array<f64> r(alloc);
    if (p.empty()) {
        r.resize(1);
        r[0] = k;
        return r;
    }
    const usize n = p.size();
    r.resize(n + 1);
    for (usize i = 0; i < n; ++i) {
        r[i] = p[i] / as<f64>(n - i);
    }
    r[n] = k;
    return r;
}

// ============================================================================
// Curve Fitting (Householder QR on Vandermonde)
// ============================================================================

FitResult polyfit(ArrayRef<f64> x, ArrayRef<f64> y, i32 deg, Allocator* alloc) {
    FitResult res{Status::Ok, Array<f64>(alloc)};
    const usize m = x.size();
    if (m == 0 || y.size() != m || deg < 0) {
        res.status = Status::InvalidInput;
        return res;
    }

    const usize np = as<usize>(deg) + 1;
    if (np > m) {
        FitResult sub = polyfit(x, y, as<i32>(m - 1), alloc);
        if (!sub.ok()) return sub;
        res.coeffs.resize(np);
        const usize offset = np - m;
        for (usize j = 0; j < offset; ++j) res.coeffs[j] = 0.0;
        for (usize j = 0; j < m; ++j) res.coeffs[offset + j] = sub.coeffs[j];
        return res;
    }

    // Vandermonde A[j * m + i] = x[i]^(deg - j), column-major.
    Array<f64> a_mat(alloc);
    a_mat.resize(m * np);
    for (usize i = 0; i < m; ++i) {
        for (usize j = 0; j < np; ++j) {
            a_mat[j * m + i] = pow(x[i], as<f64>(deg - as<i32>(j)));
        }
    }

    Array<f64> rhs(alloc);
    rhs.resize(m);
    for (usize i = 0; i < m; ++i) rhs[i] = y[i];

    for (usize k = 0; k < np; ++k) {
        f64 norm2 = 0.0;
        for (usize i = k; i < m; ++i) {
            const f64 val = a_mat[k * m + i];
            norm2 += val * val;
        }
        const f64 norm_x = sqrt(norm2);
        if (norm_x == 0.0) continue;
        const f64 akk = a_mat[k * m + k];
        const f64 alpha = (akk >= 0.0) ? -norm_x : norm_x;
        const f64 v0 = akk - alpha;
        a_mat[k * m + k] = v0;

        f64 v_norm2 = v0 * v0;
        for (usize i = k + 1; i < m; ++i) {
            const f64 vi = a_mat[k * m + i];
            v_norm2 += vi * vi;
        }
        if (v_norm2 > 0.0) {
            const f64 beta = 2.0 / v_norm2;
            for (usize c = k + 1; c < np; ++c) {
                f64 dot = 0.0;
                for (usize i = k; i < m; ++i) dot += a_mat[k * m + i] * a_mat[c * m + i];
                const f64 scale = beta * dot;
                for (usize i = k; i < m; ++i) a_mat[c * m + i] -= scale * a_mat[k * m + i];
            }
            f64 dot_y = 0.0;
            for (usize i = k; i < m; ++i) dot_y += a_mat[k * m + i] * rhs[i];
            const f64 scale_y = beta * dot_y;
            for (usize i = k; i < m; ++i) rhs[i] -= scale_y * a_mat[k * m + i];
        }
        a_mat[k * m + k] = alpha;
    }

    res.coeffs.resize(np);
    for (usize kk = np; kk > 0; --kk) {
        const usize k = kk - 1;
        f64 s = rhs[k];
        for (usize c = k + 1; c < np; ++c) s -= a_mat[c * m + k] * res.coeffs[c];
        const f64 diag = a_mat[k * m + k];
        res.coeffs[k] = (abs(diag) > 0.0) ? (s / diag) : 0.0;
    }
    return res;
}

// ============================================================================
// Pade Time-Delay Approximant
// ============================================================================

PadeResult pade_delay(f64 t, i32 order, Allocator* alloc) {
    PadeResult res{Status::Ok, Array<f64>(alloc), Array<f64>(alloc)};
    if (order < 0) {
        res.status = Status::InvalidInput;
        return res;
    }
    const usize n = as<usize>(order) + 1;
    Array<f64> a(alloc);
    a.resize(n);
    a[0] = 1.0;
    for (i32 k = 0; k < order; ++k) {
        a[as<usize>(k + 1)] =
            a[as<usize>(k)] * as<f64>(order - k) /
            (as<f64>(2 * order - k) * as<f64>(k + 1));
    }

    res.num.resize(n);
    res.den.resize(n);
    f64 pow_neg_t = 1.0;
    f64 pow_t = 1.0;
    for (i32 k = 0; k <= order; ++k) {
        const usize idx = as<usize>(order - k);
        res.num[idx] = a[as<usize>(k)] * pow_neg_t;
        res.den[idx] = a[as<usize>(k)] * pow_t;
        pow_neg_t *= -t;
        pow_t *= t;
    }

    const f64 scale = res.den[0];
    if (scale != 0.0) {
        for (usize i = 0; i < n; ++i) {
            res.num[i] /= scale;
            res.den[i] /= scale;
        }
    }
    return res;
}

// ============================================================================
// Partial Fraction Expansion: s-domain residue
// ============================================================================

ResidueResult residue(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc) {
    ResidueResult out{Status::Ok, Array<c64>(alloc), Array<c64>(alloc), Array<f64>(alloc)};
    Array<f64> a_poly = strip_leading_copy(a, alloc);
    Array<f64> b_poly = strip_leading_copy(b, alloc);
    if (a_poly.empty() || a_poly[0] == 0.0) {
        out.status = Status::ZeroDivisor;
        return out;
    }

    // Polynomial division: B = K * A + R (deg R < deg A).
    Array<f64> r_poly(alloc);
    if (b_poly.size() >= a_poly.size()) {
        const usize q_len = b_poly.size() - a_poly.size() + 1;
        out.k.resize(q_len);
        Array<f64> bb(alloc);
        bb.resize(b_poly.size());
        for (usize i = 0; i < b_poly.size(); ++i) bb[i] = b_poly[i];
        const f64 a_lead = a_poly[0];
        for (usize i = 0; i < q_len; ++i) {
            const f64 coef = bb[i] / a_lead;
            out.k[i] = coef;
            for (usize j = 0; j < a_poly.size(); ++j) bb[i + j] -= coef * a_poly[j];
        }
        usize r_off = q_len;
        while (r_off < bb.size() && bb[r_off] == 0.0) ++r_off;
        if (r_off < bb.size()) {
            const usize r_len = bb.size() - r_off;
            r_poly.resize(r_len);
            for (usize i = 0; i < r_len; ++i) r_poly[i] = bb[r_off + i];
        }
    } else {
        r_poly.resize(b_poly.size());
        for (usize i = 0; i < b_poly.size(); ++i) r_poly[i] = b_poly[i];
    }

    Array<c64> poles = roots(ArrayRef<f64>(a_poly.data(), a_poly.size()), alloc);

    struct PoleGroup {
        c64 p;
        usize m;
    };
    Array<PoleGroup> groups(alloc);
    {
        Array<c64> a_cx(alloc);
        a_cx.resize(a_poly.size());
        for (usize i = 0; i < a_poly.size(); ++i) a_cx[i] = c64(a_poly[i], 0.0);

        Array<u8> grouped(alloc);
        grouped.resize(poles.size());
        for (usize i = 0; i < poles.size(); ++i) grouped[i] = 0;

        for (usize i = 0; i < poles.size(); ++i) {
            if (grouped[i]) continue;
            grouped[i] = 1;
            PoleGroup g{poles[i], 1};
            for (usize j = i + 1; j < poles.size(); ++j) {
                if (grouped[j]) continue;
                const f64 scale = max(1.0, max(abs(poles[i]), abs(poles[j])));
                if (abs(poles[j] - poles[i]) < 5e-4 * scale) {
                    grouped[j] = 1;
                    ++g.m;
                    g.p += poles[j];
                }
            }
            if (g.m > 1) {
                g.p /= as<f64>(g.m);
                if (abs(g.p.imag()) < 5e-4 * (1.0 + abs(g.p))) {
                    g.p = c64(g.p.real(), 0.0);
                }
                const f64 m_d = as<f64>(g.m);
                for (i32 it = 0; it < 30; ++it) {
                    const c64 fv = polyval(ArrayRef<c64>(a_cx.data(), a_cx.size()), g.p);
                    c64 dacc(0.0, 0.0);
                    for (usize k = 0; k + 1 < a_cx.size(); ++k) {
                        dacc = dacc * g.p + a_cx[k] * as<f64>(a_cx.size() - 1 - k);
                    }
                    if (abs(dacc) < 1e-300) break;
                    const c64 step = m_d * fv / dacc;
                    g.p -= step;
                    if (abs(step) < 1e-15 * (1.0 + abs(g.p))) break;
                }
            }
            groups.push_back(g);
        }
    }

    const bool any_repeated = (groups.size() != poles.size());

    Array<f64> a_prime(alloc);
    if (a_poly.size() > 1) {
        const usize n = a_poly.size() - 1;
        a_prime.resize(n);
        for (usize i = 0; i < n; ++i) a_prime[i] = a_poly[i] * as<f64>(n - i);
    }

    out.r.reserve(poles.size());
    out.p.reserve(poles.size());

    if (!any_repeated) {
        if (!r_poly.empty()) {
            for (usize i = 0; i < poles.size(); ++i) {
                const c64 num = horner_real_at_c64(ArrayRef<f64>(r_poly.data(), r_poly.size()), poles[i]);
                const c64 den = horner_real_at_c64(ArrayRef<f64>(a_prime.data(), a_prime.size()), poles[i]);
                if (abs(den) < 1e-300) {
                    out.status = Status::DegeneratePole;
                    return out;
                }
                out.r.push_back(num / den);
            }
        } else {
            out.r.resize(poles.size());
            for (usize i = 0; i < poles.size(); ++i) out.r[i] = c64(0.0, 0.0);
        }
        out.p = move(poles);
        return out;
    }

    Array<c64> r_cx(alloc);
    r_cx.resize(r_poly.size());
    for (usize i = 0; i < r_poly.size(); ++i) r_cx[i] = c64(r_poly[i], 0.0);

    for (usize gi = 0; gi < groups.size(); ++gi) {
        const PoleGroup& g = groups[gi];
        if (g.m == 1) {
            const c64 num = r_poly.empty() ? c64(0.0, 0.0)
                                           : horner_real_at_c64(ArrayRef<f64>(r_poly.data(), r_poly.size()), g.p);
            const c64 den = horner_real_at_c64(ArrayRef<f64>(a_prime.data(), a_prime.size()), g.p);
            if (abs(den) < 1e-300) {
                out.status = Status::DegeneratePole;
                return out;
            }
            out.r.push_back(r_poly.empty() ? c64(0.0, 0.0) : (num / den));
            out.p.push_back(g.p);
            continue;
        }

        Array<c64> dk(alloc);
        dk.resize(a_poly.size());
        for (usize i = 0; i < a_poly.size(); ++i) dk[i] = c64(a_poly[i], 0.0);
        for (usize d = 0; d < g.m && dk.size() > 1; ++d) {
            dk = synthetic_divide_c64(ArrayRef<c64>(dk.data(), dk.size()), g.p, alloc);
        }
        Array<c64> dser = taylor_coeffs_c64(ArrayRef<c64>(dk.data(), dk.size()), g.p, g.m, alloc);
        if (dser.empty() || abs(dser[0]) < 1e-300) {
            out.status = Status::DegeneratePole;
            return out;
        }

        Array<c64> recip(alloc);
        recip.resize(g.m);
        for (usize j = 0; j < g.m; ++j) recip[j] = c64(0.0, 0.0);
        recip[0] = c64(1.0, 0.0) / dser[0];
        for (usize j = 1; j < g.m; ++j) {
            c64 s(0.0, 0.0);
            for (usize i = 1; i <= j; ++i) {
                const c64 di = (i < dser.size()) ? dser[i] : c64(0.0, 0.0);
                s += di * recip[j - i];
            }
            recip[j] = -s / dser[0];
        }

        Array<c64> rser(alloc);
        rser.resize(g.m);
        for (usize j = 0; j < g.m; ++j) rser[j] = c64(0.0, 0.0);
        if (!r_cx.empty()) {
            Array<c64> rt = taylor_coeffs_c64(ArrayRef<c64>(r_cx.data(), r_cx.size()), g.p, g.m, alloc);
            for (usize j = 0; j < g.m && j < rt.size(); ++j) rser[j] = rt[j];
        }

        Array<c64> h(alloc);
        h.resize(g.m);
        for (usize j = 0; j < g.m; ++j) {
            c64 acc(0.0, 0.0);
            for (usize i = 0; i <= j; ++i) acc += rser[i] * recip[j - i];
            h[j] = acc;
        }
        for (usize i = 0; i < g.m; ++i) {
            out.r.push_back(h[g.m - 1 - i]);
            out.p.push_back(g.p);
        }
    }
    return out;
}

// ============================================================================
// Partial Fraction Expansion: z-domain residuez
// ============================================================================

ResidueResult residuez(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc) {
    ResidueResult out{Status::Ok, Array<c64>(alloc), Array<c64>(alloc), Array<f64>(alloc)};
    Array<f64> a_poly = strip_leading_copy(a, alloc);
    Array<f64> b_poly = strip_leading_copy(b, alloc);
    if (a_poly.empty() || a_poly[0] == 0.0) {
        out.status = Status::ZeroDivisor;
        return out;
    }

    const f64 a0 = a_poly[0];
    for (usize i = 0; i < a_poly.size(); ++i) a_poly[i] /= a0;
    for (usize i = 0; i < b_poly.size(); ++i) b_poly[i] /= a0;

    const usize la = a_poly.size();
    const usize lb0 = b_poly.size();
    if (la == 1) {
        out.k = move(b_poly);
        return out;
    }

    const usize nres = la - 1;
    Array<f64> bp(alloc);
    bp.resize(lb0);
    for (usize i = 0; i < lb0; ++i) bp[i] = b_poly[i];

    if (lb0 > nres) {
        Array<f64> bf(alloc);
        bf.resize(lb0);
        for (usize i = 0; i < lb0; ++i) bf[i] = bp[lb0 - 1 - i];
        Array<f64> af(alloc);
        af.resize(la);
        for (usize i = 0; i < la; ++i) af[i] = a_poly[la - 1 - i];

        const usize q_len = lb0 - la + 1;
        out.k.resize(q_len);
        for (usize i = 0; i < q_len; ++i) {
            const f64 coef = bf[i] / af[0];
            out.k[i] = coef;
            for (usize j = 0; j < la; ++j) bf[i + j] -= coef * af[j];
        }
        // Reverse out.k
        for (usize i = 0; i < q_len / 2; ++i) {
            swap(out.k[i], out.k[q_len - 1 - i]);
        }
        usize r_off = q_len;
        while (r_off < lb0 && bf[r_off] == 0.0) ++r_off;
        bp.clear();
        for (usize idx = lb0; idx > r_off; --idx) bp.push_back(bf[idx - 1]);
        if (bp.empty()) bp.push_back(0.0);
    }

    Array<c64> poles = roots(ArrayRef<f64>(a_poly.data(), a_poly.size()), alloc);
    for (usize i = 0; i < poles.size(); ++i) {
        if (abs(poles[i].imag()) < 1e-4 * (1.0 + abs(poles[i]))) {
            poles[i] = c64(poles[i].real(), 0.0);
        }
    }

    // Stable descending sort by |p| (insertion sort is naturally stable).
    for (usize i = 1; i < poles.size(); ++i) {
        const c64 key = poles[i];
        const f64 key_mag = abs(key);
        usize j = i;
        while (j > 0 && key_mag > abs(poles[j - 1])) {
            poles[j] = poles[j - 1];
            --j;
        }
        poles[j] = key;
    }

    constexpr f64 kMpolesTol = 1e-3;
    usize m = poles.size();
    Array<usize> mults(alloc);
    mults.reserve(m);
    {
        Array<u8> used(alloc);
        used.resize(m);
        for (usize i = 0; i < m; ++i) used[i] = 0;
        Array<c64> grouped(alloc);
        grouped.reserve(m);

        for (usize i = 0; i < m; ++i) {
            if (used[i]) continue;
            bool all_nonzero = true;
            for (usize k = i; k < m; ++k) {
                if (!used[k] && abs(poles[k]) == 0.0) {
                    all_nonzero = false;
                    break;
                }
            }
            const f64 thresh = all_nonzero ? (kMpolesTol * abs(poles[i])) : kMpolesTol;
            usize count = 0;
            for (usize k = i; k < m; ++k) {
                if (!used[k] && abs(poles[k] - poles[i]) < thresh) {
                    used[k] = 1;
                    grouped.push_back(poles[k]);
                    ++count;
                    mults.push_back(count);
                }
            }
        }
        poles = move(grouped);
        m = poles.size();
    }

    const usize nh = m + 1;
    Array<c64> imp(alloc);
    imp.resize(nh);
    for (usize i = 0; i < nh; ++i) imp[i] = c64(0.0, 0.0);
    imp[0] = c64(1.0, 0.0);

    Array<c64> bc(alloc);
    bc.resize(bp.size());
    for (usize i = 0; i < bp.size(); ++i) bc[i] = c64(bp[i], 0.0);
    Array<c64> ac(alloc);
    ac.resize(a_poly.size());
    for (usize i = 0; i < a_poly.size(); ++i) ac[i] = c64(a_poly[i], 0.0);

    Array<c64> h = zfilter_c64(ArrayRef<c64>(bc.data(), bc.size()), ArrayRef<c64>(ac.data(), ac.size()),
                               ArrayRef<c64>(imp.data(), imp.size()), alloc);

    // S matrix columns stored flat: s_cols[j * nh + n].
    Array<c64> s_cols(alloc);
    s_cols.resize(m * nh);
    for (usize j = 0; j < m; ++j) {
        const c64 num1[1] = {c64(1.0, 0.0)};
        const c64 den1[2] = {c64(1.0, 0.0), -poles[j]};
        ArrayRef<c64> inp = (mults[j] > 1) ? ArrayRef<c64>(s_cols.data() + (j - 1) * nh, nh)
                                           : ArrayRef<c64>(imp.data(), nh);
        Array<c64> col = zfilter_c64(ArrayRef<c64>(num1, 1), ArrayRef<c64>(den1, 2), inp, alloc);
        for (usize n = 0; n < nh; ++n) s_cols[j * nh + n] = col[n];
    }

    out.r.resize(m);
    for (usize i = 0; i < m; ++i) out.r[i] = c64(0.0, 0.0);
    Array<u8> jdone(alloc);
    jdone.resize(m);
    for (usize i = 0; i < m; ++i) jdone[i] = 0;

    Array<c64> bflip(alloc);
    bflip.resize(bp.size());
    for (usize i = 0; i < bp.size(); ++i) bflip[i] = c64(bp[bp.size() - 1 - i], 0.0);

    for (usize i = 0; i < m; ++i) {
        usize ii = 0;
        if (i == m - 1) {
            ii = m;
        } else if (mults[i + 1] == 1) {
            ii = i + 1;
        }
        if (ii == 0) continue;

        Array<c64> others(alloc);
        others.reserve(m);
        for (usize j = 0; j < m; ++j) {
            if (j < i - mults[i] + 1 || j > i) others.push_back(poles[j]);
        }

        Array<c64> temp = from_roots_c64(ArrayRef<c64>(others.data(), others.size()), alloc);
        Array<c64> temp_flip(alloc);
        temp_flip.resize(temp.size());
        for (usize t = 0; t < temp.size(); ++t) temp_flip[t] = temp[temp.size() - 1 - t];

        const c64 x_inv = c64(1.0, 0.0) / poles[i];
        const c64 den = polyval(ArrayRef<c64>(temp_flip.data(), temp_flip.size()), x_inv);
        if (abs(den) < 1e-300) {
            out.status = Status::DegeneratePole;
            return out;
        }
        out.r[i] = polyval(ArrayRef<c64>(bflip.data(), bflip.size()), x_inv) / den;
        jdone[i] = 1;
    }

    Array<usize> jkl(alloc);
    for (usize j = 0; j < m; ++j) {
        if (!jdone[j]) jkl.push_back(j);
    }

    if (!jkl.empty()) {
        for (usize j = 0; j < m; ++j) {
            if (jdone[j]) {
                for (usize n = 0; n < nh; ++n) h[n] -= s_cols[j * nh + n] * out.r[j];
            }
        }
        const usize nm = jkl.size();
        Array<c64> ata(alloc);
        ata.resize(nm * nm);
        Array<c64> atb(alloc);
        atb.resize(nm);
        for (usize r = 0; r < nm; ++r) {
            c64 b_acc(0.0, 0.0);
            for (usize n = 0; n < nh; ++n) b_acc += conj(s_cols[jkl[r] * nh + n]) * h[n];
            atb[r] = b_acc;
            for (usize c = 0; c < nm; ++c) {
                c64 a_acc(0.0, 0.0);
                for (usize n = 0; n < nh; ++n) a_acc += conj(s_cols[jkl[r] * nh + n]) * s_cols[jkl[c] * nh + n];
                ata[r * nm + c] = a_acc;
            }
        }

        // Gaussian elimination with partial pivoting on ATA * x = ATb.
        for (usize col = 0; col < nm; ++col) {
            usize piv = col;
            for (usize r = col + 1; r < nm; ++r) {
                if (abs(ata[r * nm + col]) > abs(ata[piv * nm + col])) piv = r;
            }
            if (piv != col) {
                for (usize c = 0; c < nm; ++c) swap(ata[col * nm + c], ata[piv * nm + c]);
                swap(atb[col], atb[piv]);
            }
            const c64 d = ata[col * nm + col];
            for (usize c = col; c < nm; ++c) ata[col * nm + c] /= d;
            atb[col] /= d;
            for (usize r = 0; r < nm; ++r) {
                if (r == col) continue;
                const c64 f = ata[r * nm + col];
                for (usize c = col; c < nm; ++c) ata[r * nm + c] -= f * ata[col * nm + c];
                atb[r] -= f * atb[col];
            }
        }
        for (usize r = 0; r < nm; ++r) out.r[jkl[r]] = atb[r];
    }

    out.p = move(poles);
    return out;
}

} // namespace keel::poly

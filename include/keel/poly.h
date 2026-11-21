#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/core.h"
#include "keel/math.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel::poly {

enum class Status : u8 {
    Ok = 0,
    InvalidInput,   ///< Empty/mismatched input or negative degree/order
    ZeroDivisor,    ///< Zero or empty denominator polynomial
    DegeneratePole, ///< Vanishing derivative / deflated denominator at a pole
};

// ============================================================================
// Horner Evaluation & Matrix Polynomials
// (Descending-power convention: c[0] * x^n + c[1] * x^(n-1) + ... + c[n])
// ============================================================================

[[nodiscard]] f64 polyval(ArrayRef<f64> coeffs, f64 x) noexcept;
[[nodiscard]] c64 polyval(ArrayRef<c64> coeffs, c64 z) noexcept;
void polyval_batch(ArrayRef<f64> coeffs, ArrayRef<f64> xs, MutArrayRef<f64> out) noexcept;

/// Evaluate p(z) and p'(z) simultaneously via Horner's method.
void horner_deriv(ArrayRef<c64> coeffs, c64 z, c64& out_p, c64& out_dp) noexcept;

/// Evaluate matrix polynomial p(A) for an n x n column-major matrix A.
[[nodiscard]] Array<c64> polyvalm(ArrayRef<c64> coeffs, ArrayRef<c64> a_colmajor, usize n, Allocator* alloc = nullptr);

// ============================================================================
// Roots & Expansion
// ============================================================================

/// Durand-Kerner simultaneous root finder. Leading zeros are stripped; trailing
/// zeros are returned as roots at the origin.
[[nodiscard]] Array<c64> roots(ArrayRef<c64> coeffs, Allocator* alloc = nullptr);
[[nodiscard]] Array<c64> roots(ArrayRef<f64> coeffs, Allocator* alloc = nullptr);

/// Expand monic polynomial prod_i (x - r_i) returning complex coefficients.
[[nodiscard]] Array<c64> from_roots_c64(ArrayRef<c64> root_list, Allocator* alloc = nullptr);

/// Expand monic polynomial prod_i (x - r_i) returning real parts of coefficients.
[[nodiscard]] Array<f64> from_roots(ArrayRef<c64> root_list, Allocator* alloc = nullptr);

// ============================================================================
// Polynomial Arithmetic & Calculus
// ============================================================================

[[nodiscard]] Array<f64> mul(ArrayRef<f64> a, ArrayRef<f64> b, Allocator* alloc = nullptr);

struct DivResult {
    Status status = Status::Ok;
    Array<f64> q;
    Array<f64> r;

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }
};

/// Polynomial Euclidean division b = q * a + r.
[[nodiscard]] DivResult div(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc = nullptr);

/// Canonical derivative d/dx p(x) (leading zeros trimmed).
[[nodiscard]] Array<f64> deriv(ArrayRef<f64> p, Allocator* alloc = nullptr);

struct DerivRatioResult {
    Array<f64> num;
    Array<f64> den;
};

/// Quotient rule derivative d/dx (b(x) / a(x)) = num(x) / den(x).
[[nodiscard]] DerivRatioResult deriv_ratio(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc = nullptr);

/// Antiderivative of p(x) with integration constant `k`.
[[nodiscard]] Array<f64> integ(ArrayRef<f64> p, f64 k = 0.0, Allocator* alloc = nullptr);

// ============================================================================
// Curve Fitting (Householder QR on Vandermonde)
// ============================================================================

struct FitResult {
    Status status = Status::Ok;
    Array<f64> coeffs;

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }
};

[[nodiscard]] FitResult polyfit(ArrayRef<f64> x, ArrayRef<f64> y, i32 deg, Allocator* alloc = nullptr);

// ============================================================================
// Pade Time-Delay Approximant of exp(-t * s)
// ============================================================================

struct PadeResult {
    Status status = Status::Ok;
    Array<f64> num;
    Array<f64> den;

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }
};

[[nodiscard]] PadeResult pade_delay(f64 t, i32 order, Allocator* alloc = nullptr);

// ============================================================================
// Partial Fraction Expansion (s-domain residue & z-domain residuez)
// ============================================================================

struct ResidueResult {
    Status status = Status::Ok;
    Array<c64> r; ///< Residues
    Array<c64> p; ///< Poles
    Array<f64> k; ///< Direct term polynomial (empty if strictly proper)

    [[nodiscard]] bool ok() const noexcept { return status == Status::Ok; }
};

/// s-domain partial fraction expansion of B(s) / A(s) (descending powers of s).
/// Supports both simple and repeated poles.
[[nodiscard]] ResidueResult residue(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc = nullptr);

/// z-domain partial fraction expansion of B(z^-1) / A(z^-1) (ascending powers of z^-1).
/// Supports both simple and repeated poles.
[[nodiscard]] ResidueResult residuez(ArrayRef<f64> b, ArrayRef<f64> a, Allocator* alloc = nullptr);

} // namespace keel::poly

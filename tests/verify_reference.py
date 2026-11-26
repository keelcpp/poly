#!/usr/bin/env python3
"""Differential reference verification for keel::poly against NumPy, SciPy, and MATLAB."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import signal


def parse_dump(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = [float(tok) for tok in rest.strip().split()] if rest.strip() else []
        out[key.strip()] = vals
    return out


def as_c64(vals: list[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    return arr[0::2] + 1j * arr[1::2]


def sort_complex(z: np.ndarray) -> np.ndarray:
    idx = np.lexsort((np.round(z.imag, 9), np.round(z.real, 9)))
    return z[idx]


def eval_pfe_s(r: np.ndarray, p: np.ndarray, k: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Evaluate partial fraction sum_i r_i / (s - p_i)^m_i + polyval(k, s)."""
    val = np.polyval(k, s) if len(k) > 0 else np.zeros_like(s, dtype=np.complex128)
    used_mult: dict[int, int] = {}
    reps: list[complex] = []
    for ri, pi in zip(r, p):
        matched = -1
        for idx, prev in enumerate(reps):
            if abs(pi - prev) < 1e-6 * max(1.0, abs(prev)):
                matched = idx
                break
        if matched < 0:
            matched = len(reps)
            reps.append(pi)
            used_mult[matched] = 1
        else:
            used_mult[matched] += 1
        m = used_mult[matched]
        val = val + ri / ((s - pi) ** m)
    return val


def eval_pfe_z(r: np.ndarray, p: np.ndarray, k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate z-domain PFE sum_i r_i / (1 - p_i z^-1)^m_i + sum_j k_j z^-j."""
    zinv = 1.0 / z
    val = np.zeros_like(z, dtype=np.complex128)
    for j, kj in enumerate(k):
        val = val + kj * (zinv ** j)
    used_mult: dict[int, int] = {}
    reps: list[complex] = []
    for ri, pi in zip(r, p):
        matched = -1
        for idx, prev in enumerate(reps):
            if abs(pi - prev) < 1e-6 * max(1.0, abs(prev)):
                matched = idx
                break
        if matched < 0:
            matched = len(reps)
            reps.append(pi)
            used_mult[matched] = 1
        else:
            used_mult[matched] += 1
        m = used_mult[matched]
        val = val + ri / ((1.0 - pi * zinv) ** m)
    return val


def check_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, failures: list[str]) -> None:
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        failures.append(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    err = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
    status = "PASS" if err <= tol else "FAIL"
    print(f"  [{status}] {name:<28s} rel_err={err:.3e} (tol={tol:.1e})")
    if err > tol:
        failures.append(f"{name}: rel_err={err:.3e} > tol={tol:.1e}")


def run_matlab_checks(dump: dict[str, list[float]], failures: list[str]) -> None:
    if not shutil.which("matlab"):
        print("  [SKIP] MATLAB not found on PATH")
        return

    script = r"""
format longg
[n5, d5] = pade(0.5, 5);
[n8, d8] = pade(2.5, 8);
[n0, d0] = pade(0.0, 4);
pm_a = [1 0.5 2; 2 -1 1; -1 3 0.25];
pm_res = polyvalm([2 -1.5 0.5 3], pm_a);
[r1, p1, k1] = residue([2 -1 3 1 -5], [1 3 4 2]);
[rz_r2, rz_p2, rz_k2] = residuez([0 1 -0.25], [1 -1 0.25]);
fprintf('m_pade5_num: %s\n', sprintf('%.17g ', n5));
fprintf('m_pade5_den: %s\n', sprintf('%.17g ', d5));
fprintf('m_pade8_num: %s\n', sprintf('%.17g ', n8));
fprintf('m_pade8_den: %s\n', sprintf('%.17g ', d8));
fprintf('m_pade0_num: %s\n', sprintf('%.17g ', n0));
fprintf('m_pade0_den: %s\n', sprintf('%.17g ', d0));
fprintf('m_polyvalm: %s\n', sprintf('%.17g ', pm_res(:)));
fprintf('m_res1_k: %s\n', sprintf('%.17g ', k1));
fprintf('m_rz2_k: %s\n', sprintf('%.17g ', rz_k2));
fprintf('m_rz2_r: %s\n', sprintf('%.17g %.17g ', [real(rz_r2(:)).'; imag(rz_r2(:)).']));
fprintf('m_rz2_p: %s\n', sprintf('%.17g %.17g ', [real(rz_p2(:)).'; imag(rz_p2(:)).']));
"""
    proc = subprocess.run(["matlab", "-batch", script], capture_output=True, text=True, check=True)
    md = parse_dump(proc.stdout)
    check_close("matlab/pade5_num", dump["pade5_num"], md["m_pade5_num"], 1e-13, failures)
    check_close("matlab/pade5_den", dump["pade5_den"], md["m_pade5_den"], 1e-13, failures)
    check_close("matlab/pade8_num", dump["pade8_num"], md["m_pade8_num"], 1e-12, failures)
    check_close("matlab/pade8_den", dump["pade8_den"], md["m_pade8_den"], 1e-12, failures)
    check_close("matlab/pade0_num", dump["pade0_num"], md["m_pade0_num"], 1e-15, failures)
    check_close("matlab/pade0_den", dump["pade0_den"], md["m_pade0_den"], 1e-15, failures)
    check_close("matlab/polyvalm", as_c64(dump["polyvalm"]), np.asarray(md["m_polyvalm"]), 1e-14, failures)
    check_close("matlab/residue1_k", dump["residue1_k"], md["m_res1_k"], 1e-14, failures)
    check_close("matlab/residuez2_k", dump["residuez2_k"], md["m_rz2_k"], 1e-13, failures)

    # Compare MATLAB residuez2 transfer function against keel residuez2
    z_grid = np.exp(1j * np.linspace(0.2, 2.9, 12)) * 1.4
    tf_keel = eval_pfe_z(as_c64(dump["residuez2_r"]), as_c64(dump["residuez2_p"]), np.asarray(dump["residuez2_k"]), z_grid)
    tf_mat = eval_pfe_z(as_c64(md["m_rz2_r"]), as_c64(md["m_rz2_p"]), np.asarray(md["m_rz2_k"]), z_grid)
    check_close("matlab/residuez2_tf", tf_keel, tf_mat, 1e-12, failures)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_poly>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    dump = parse_dump(proc.stdout)
    failures: list[str] = []

    print("Running keel::poly reference verification against NumPy/SciPy/MATLAB...")

    # 1. polyval_batch & horner_deriv
    pv_c = np.array([2.5, -1.25, 0.0, 3.75, -4.5])
    pv_x = np.array([-2.5, 0.0, 1.5, 3.25])
    check_close("numpy/polyval_batch", dump["polyval_batch"], np.polyval(pv_c, pv_x), 1e-14, failures)

    hd_c = np.array([1.0 - 0.5j, -2.0 + 1.0j, 0.5 + 0.25j, 3.0 - 1.0j])
    z0 = 1.2 - 0.7j
    hd_ref = np.array([np.polyval(hd_c, z0), np.polyval(np.polyder(hd_c), z0)])
    check_close("numpy/horner_deriv", as_c64(dump["horner_deriv"]), hd_ref, 1e-14, failures)

    # 2. polyvalm
    pm_a = np.array([[1.0, 0.5, 2.0], [2.0, -1.0, 1.0], [-1.0, 3.0, 0.25]], dtype=np.complex128)
    pm_ref = 2.0 * (pm_a @ pm_a @ pm_a) - 1.5 * (pm_a @ pm_a) + 0.5 * pm_a + 3.0 * np.eye(3)
    check_close("numpy/polyvalm", as_c64(dump["polyvalm"]), pm_ref.flatten(order="F"), 1e-14, failures)

    # 3. roots & from_roots
    r_real_ref = sort_complex(np.roots([1.0, -3.0, 4.0, -2.0, 5.0, -1.0]))
    check_close("numpy/roots_real", sort_complex(as_c64(dump["roots_real"])), r_real_ref, 1e-12, failures)

    r_cx_ref = sort_complex(np.roots(hd_c))
    check_close("numpy/roots_cx", sort_complex(as_c64(dump["roots_cx"])), r_cx_ref, 1e-12, failures)

    w10_p_ref = np.poly(np.arange(1.0, 11.0))
    check_close("numpy/from_roots_w10", dump["from_roots_w10"], w10_p_ref, 1e-14, failures)
    check_close("numpy/roots_w10", sort_complex(as_c64(dump["roots_w10"])), np.arange(1.0, 11.0), 1e-8, failures)

    big_ref = sort_complex(np.array([1e5, -2e5 + 3e5j, -2e5 - 3e5j, 5e5]))
    check_close("numpy/roots_big", sort_complex(as_c64(dump["roots_big"])), big_ref, 1e-12, failures)

    # 4. Arithmetic & calculus
    pa = np.array([1.5, -2.0, 0.5, 4.0])
    pb = np.array([2.0, 1.0, -3.0])
    check_close("numpy/mul", dump["mul"], np.polymul(pa, pb), 1e-14, failures)
    q_ref, r_ref = np.polydiv(pa, pb)
    check_close("numpy/div_q", dump["div_q"], q_ref, 1e-14, failures)
    check_close("numpy/div_r", dump["div_r"], r_ref, 1e-14, failures)
    check_close("numpy/deriv", dump["deriv"], np.polyder(pa), 1e-14, failures)
    num_ref = np.polysub(np.polymul(pb, np.polyder(pa)), np.polymul(pa, np.polyder(pb)))
    den_ref = np.polymul(pb, pb)
    check_close("numpy/deriv_ratio_num", dump["deriv_ratio_num"], num_ref, 1e-14, failures)
    check_close("numpy/deriv_ratio_den", dump["deriv_ratio_den"], den_ref, 1e-14, failures)
    check_close("numpy/integ", dump["integ"], np.polyint(pa, k=-7.25), 1e-14, failures)

    # 5. polyfit
    fx = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    fy = np.array([12.1, 4.9, 1.2, 0.8, 3.1, 8.8, 18.2, 30.9])
    check_close("numpy/polyfit_lsq", dump["polyfit_lsq"], np.polyfit(fx, fy, 3), 1e-12, failures)

    fx2 = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    fy2 = np.array([1.0, 2.3, 4.1, 7.2, 11.5, 17.8])
    check_close("numpy/polyfit_shifted", dump["polyfit_shifted"], np.polyfit(fx2, fy2, 3), 1e-8, failures)

    # 6. residue (verify both transfer-function identity & SciPy match)
    s_grid = np.array([0.3 + 0.7j, -0.4 + 1.5j, 2.1 - 0.8j, -3.5 + 0.2j, 0.0 + 3.0j])
    rs_b1, rs_a1 = np.array([2.0, -1.0, 3.0, 1.0, -5.0]), np.array([1.0, 3.0, 4.0, 2.0])
    r1_ref, p1_ref, k1_ref = signal.residue(rs_b1, rs_a1)
    check_close("scipy/residue1_k", dump["residue1_k"], k1_ref, 1e-13, failures)
    tf1_got = eval_pfe_s(as_c64(dump["residue1_r"]), as_c64(dump["residue1_p"]), np.asarray(dump["residue1_k"]), s_grid)
    tf1_ref = np.polyval(rs_b1, s_grid) / np.polyval(rs_a1, s_grid)
    check_close("scipy/residue1_tf", tf1_got, tf1_ref, 1e-12, failures)

    rs_b2, rs_a2 = np.array([3.0, -2.0, 1.0, 4.0]), np.array([1.0, 5.0, 6.0, -4.0, -8.0])
    tf2_got = eval_pfe_s(as_c64(dump["residue2_r"]), as_c64(dump["residue2_p"]), np.asarray(dump["residue2_k"]), s_grid)
    tf2_ref = np.polyval(rs_b2, s_grid) / np.polyval(rs_a2, s_grid)
    check_close("scipy/residue2_repeated_tf", tf2_got, tf2_ref, 1e-11, failures)

    # 7. residuez (verify both transfer-function identity & SciPy match)
    z_grid = np.exp(1j * np.linspace(0.2, 2.9, 10)) * 1.4
    rz_b1, rz_a1 = np.array([1.0, 2.0, -1.0, 0.5]), np.array([1.0, -0.75, 0.125])
    rz1_r_ref, rz1_p_ref, rz1_k_ref = signal.residuez(rz_b1, rz_a1)
    check_close("scipy/residuez1_k", dump["residuez1_k"], rz1_k_ref, 1e-13, failures)
    tz1_got = eval_pfe_z(as_c64(dump["residuez1_r"]), as_c64(dump["residuez1_p"]), np.asarray(dump["residuez1_k"]), z_grid)
    tz1_ref = eval_pfe_z(rz1_r_ref, rz1_p_ref, rz1_k_ref, z_grid)
    check_close("scipy/residuez1_tf", tz1_got, tz1_ref, 1e-12, failures)

    rz_b2, rz_a2 = np.array([0.0, 1.0, -0.25]), np.array([1.0, -1.0, 0.25])
    rz2_r_ref, rz2_p_ref, rz2_k_ref = signal.residuez(rz_b2, rz_a2)
    check_close("scipy/residuez2_k", dump["residuez2_k"], rz2_k_ref, 1e-13, failures)
    tz2_got = eval_pfe_z(as_c64(dump["residuez2_r"]), as_c64(dump["residuez2_p"]), np.asarray(dump["residuez2_k"]), z_grid)
    tz2_ref = eval_pfe_z(rz2_r_ref, rz2_p_ref, rz2_k_ref, z_grid)
    check_close("scipy/residuez2_delay_rep_tf", tz2_got, tz2_ref, 1e-12, failures)

    run_matlab_checks(dump, failures)

    if failures:
        print(f"\nFAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL keel::poly REFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

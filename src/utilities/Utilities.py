# Copyright (C) 2025 Sukanta Basu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
File: Utilities.py
===========================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic) and Codex (OpenAI) are used for documentation,
                code restructuring, and performance optimization
:Date: 2025-10-20
:Description: miscellaneous functions
"""


# ============================================================
# Imports
# ============================================================

import jax
import numpy as np
import jax.numpy as jnp


# ============================================================
#  Compute planar averaged values of a 3D field
# ============================================================

@jax.jit
def PlanarMean(F):
    """
    Computes horizontal average of a 3D field at each vertical level.

    Parameters:
    -----------
    F : jax.numpy.ndarray
        3D array with shape (nx, ny, nz)

    Returns:
    --------
    jax.numpy.ndarray
        1D array of length nz containing planar-averaged values
    """

    return jnp.mean(F, axis=(0, 1))


# ============================================================
#  Compute averaging of a 3D array along the z-direction
# ============================================================

@jax.jit
def StagGridAvg(F):
    """
    Computes averaging of a 3D array along the z-direction.

    Parameters:
    -----------
    F : jax.numpy.ndarray
        3D array with shape (nx, ny, nz)

    Returns:
    --------
    jax.numpy.ndarray
        Averaged field with shape (nx, ny, nz-1)
    """

    return 0.5 * (F[:, :, :-1] + F[:, :, 1:])


# ============================================================
# Compute moving average filtering of 2D fields
# ============================================================

@jax.jit
def Imfilter(x):
    """
    Applies a 3x3 box filter to a 2D field with periodic boundaries.

    Parameters:
    -----------
    x : jax.numpy.ndarray
        2D array representing a horizontal slice

    Returns:
    --------
    jax.numpy.ndarray
        Filtered 2D array with the same shape as input
    """

    kernel = jnp.ones((3, 3)) / 9.0  # 3x3 mean filter (box filter)

    # Apply periodic padding (wrap around)
    x_padded = jnp.pad(x, ((1, 1), (1, 1)), mode='wrap')

    # Perform 2D convolution
    x_filtered = jax.lax.conv_general_dilated(
        x_padded[None, None, :, :],  # Add batch & channel dims
        kernel[None, None, :, :],    # Kernel shape: [Out_C, In_C, H, W]
        (1, 1),  # Stride (1,1) ensures all pixels are processed
        'VALID'  # No extra padding applied beyond what we added
    )

    return x_filtered[0, 0, :, :]  # Remove batch & channel dims


# ============================================================
# Scale-dependence parameter from the LASDD polynomial
# ============================================================

def _TwoSum(a, b):
    """Error-free sum: a + b = s + e exactly (Knuth)."""
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _TwoProd(a, b):
    """Error-free product: a * b = p + e exactly (Dekker splitting, no FMA needed)."""
    f = 134217729.0 if a.dtype == jnp.float64 else 4097.0

    def split(x):
        c = f * x
        h = c - (c - x)
        return h, x - h

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _CompHorner(Cs, x):
    """
    Compensated Horner evaluation (Graillat et al. 2005) of the real polynomials
    Cs (nz, n+1; descending) at real points x (nz, m): as accurate as Horner in
    twice the working precision, so sign tests near clustered roots are reliable
    in single precision.
    """
    n = Cs.shape[1] - 1
    c = Cs[:, None, :]
    s = jnp.broadcast_to(c[..., 0], x.shape)
    comp = jnp.zeros_like(x)
    for i in range(1, n + 1):
        p, pe = _TwoProd(s, x)
        s, se = _TwoSum(p, c[..., i])
        comp = comp * x + (pe + se)
    return s + comp


def _TwoProdC(ar, ai, br, bi):
    """Error-free complex product (Graillat & Menissier-Morain 2012):
    (ar + i ai)(br + i bi) = (pr + i pi) + (er + i ei) exactly."""
    p1, e1 = _TwoProd(ar, br)
    p2, e2 = _TwoProd(ai, bi)
    p3, e3 = _TwoProd(ar, bi)
    p4, e4 = _TwoProd(ai, br)
    pr, f1 = _TwoSum(p1, -p2)
    pi_, f2 = _TwoSum(p3, p4)
    return pr, pi_, (e1 - e2) + f1, (e3 + e4) + f2


def _CompHornerC(c, zr, zi):
    """
    Compensated Horner evaluation of the real polynomial c (list of arrays,
    descending, broadcastable) and of its derivative at complex points
    zr + i zi. Both P and P' are carried as double-word values (error-free
    transformations), so each is about as accurate as Horner in twice the
    working precision; this matters for P' near clustered roots, where it is a
    cancellation of large terms.
    Returns (Pr, Pi, dPr, dPi).
    """
    n = len(c) - 1
    sr = jnp.broadcast_to(c[0], zr.shape)        # P, high part
    si = jnp.zeros_like(zr)
    cr = jnp.zeros_like(zr)                      # P, low part
    ci = jnp.zeros_like(zr)
    dr = jnp.zeros_like(zr)                      # P', high part
    di = jnp.zeros_like(zr)
    lr = jnp.zeros_like(zr)                      # P', low part
    li = jnp.zeros_like(zr)
    for k in range(1, n + 1):
        # P' <- P' z + P   (uses P before its update)
        qr, qi, er, ei = _TwoProdC(dr, di, zr, zi)
        tr, fr = _TwoSum(qr, sr)
        ti, fi = _TwoSum(qi, si)
        lr, li = (lr * zr - li * zi) + (er + fr + cr), (lr * zi + li * zr) + (ei + fi + ci)
        dr, di = tr, ti
        # P <- P z + c_k
        pr, pi_, er, ei = _TwoProdC(sr, si, zr, zi)
        sr, gr = _TwoSum(pr, c[k])
        si = pi_
        cr, ci = (cr * zr - ci * zi) + (er + gr), (cr * zi + ci * zr) + ei
    return sr + cr, si + ci, dr + lr, di + li


def _PolyRatio(Cs, z):
    """
    Newton correction P(z)/P'(z) (and the backward-error bound) of the
    polynomials with descending coefficients Cs (nz, n+1) at the points z
    (nz, n), with compensated Horner. For |z| > 1 the reversed polynomial
    R(y) = y^n P(1/y) is used, P/P' = z R / (n R - y R'), so that large roots
    never overflow.

    Returns:
    --------
    w : complex ndarray (nz, n)
    absP : ndarray (nz, n)
        |P(z)| (or |R(1/z)| for |z| > 1)
    bound : ndarray (nz, n)
        sum_i |c_i| |z|^(n-i) (or the reversed analogue)
    """
    n = Cs.shape[1] - 1
    az = jnp.abs(z)
    inside = az <= 1.0
    y = 1.0 / jnp.where(z == 0, 1.0, z)
    u = jnp.where(inside, z, y)
    ur, ui = u.real, u.imag
    cols = [Cs[:, None, i] for i in range(n + 1)]
    fwd = cols
    rev = cols[::-1]
    c = [jnp.where(inside, f, r) for f, r in zip(fwd, rev)]
    Pr, Pi, dPr, dPi = _CompHornerC(c, ur, ui)
    P = Pr + 1j * Pi
    dP = dPr + 1j * dPi
    w_in = P / dP
    w_out = z * P / (n * P - u * dP)
    w = jnp.where(inside, w_in, w_out)
    au = jnp.abs(u)
    bound = jnp.zeros_like(au)
    for ck in c:
        bound = bound * au + jnp.abs(ck)
    return w, jnp.abs(P), bound


def _NewtonPolygonRadii(Cs):
    """
    Starting radii for the n roots of each row of Cs (nz, n+1; descending,
    Cs[:, 0] != 0) from the upper convex hull of (j, log|a_j|), a_j the
    coefficient of z^j (Bini 1996): a hull segment from j1 to j2 holds j2 - j1
    roots of modulus about (|a_j1| / |a_j2|)^(1/(j2 - j1)). Roots at 0 (zero
    trailing coefficients) start at a small radius.
    """
    n = Cs.shape[1] - 1
    A = Cs[:, ::-1]                                   # A[:, j] multiplies z^j
    fin = A != 0
    L = jnp.log(jnp.where(fin, jnp.abs(A), 1.0))
    hull = []
    for jj in range(n + 1):
        on = fin[:, jj]
        for ii in range(jj):
            for kk in range(jj + 1, n + 1):
                pair = fin[:, ii] & fin[:, kk]
                line = L[:, ii] + (L[:, kk] - L[:, ii]) * (jj - ii) / (kk - ii)
                on = on & ~(pair & (L[:, jj] < line - 1e-6 * (1.0 + jnp.abs(line))))
        hull.append(on)
    hull = jnp.stack(hull, axis=1)                    # (nz, n+1)
    jidx = jnp.arange(n + 1)
    rad = []
    for s_ in range(1, n + 1):
        prev = jnp.max(jnp.where(hull & (jidx <= s_ - 1), jidx, -1), axis=1)
        nxt = jnp.min(jnp.where(hull & (jidx >= s_), jidx, n + 1), axis=1)
        Lp = jnp.take_along_axis(L, jnp.clip(prev, 0, n)[:, None], axis=1)[:, 0]
        Ln = jnp.take_along_axis(L, jnp.clip(nxt, 0, n)[:, None], axis=1)[:, 0]
        r = jnp.exp((Lp - Ln) / jnp.maximum(nxt - prev, 1))
        rad.append(jnp.where(prev >= 0, r, jnp.nan))  # nan: root at 0
    rad = jnp.stack(rad, axis=1)
    rmin = jnp.nanmin(jnp.where(jnp.isfinite(rad) & (rad > 0), rad, jnp.nan), axis=1)
    rmin = jnp.where(jnp.isfinite(rmin), rmin, 1.0)
    return jnp.where(jnp.isfinite(rad) & (rad > 0), rad, 1e-3 * rmin[:, None])


def PolyRootsAberth(C, n_iter=45):
    """
    All roots of real polynomials, one per row of C (nz, n+1; descending order),
    by the Aberth-Ehrlich simultaneous iteration, in the working precision on the
    device (vectorized over rows; no deflation, no host callback).

    - Degree reduction with the rule of polyroots.m (original MATLAB code): while
      some |c_i/c_0| exceeds the single-precision range, the polynomial restarts
      at the first such c_i. The removed roots are effectively at infinity; the
      shift leaves roots at 0, which do not affect the beta selection rule.
    - Starting points on Newton-polygon radii (_NewtonPolygonRadii), rotated
      off the real axis.
    - P and P' by compensated Horner (error-free transformations, about twice
      the working precision), so that clustered roots of float32 coefficients
      are resolved as in float64.
    - Iterates until, for every root, the Newton correction is at round-off
      (64 eps |z|) or |P(z)| <= 1024 eps^2 sum_i |c_i||z|^(n-i); rows are
      frozen once converged; fixed n_iter iterations (no host sync). Rows not
      converged are flagged (ok = False). Known roots at 0 (trailing zero
      coefficients) are set exactly.

    Returns:
    --------
    z : complex ndarray (nz, n)
        Roots.
    Cs : ndarray (nz, n+1)
        Normalized, degree-reduced coefficients whose roots are z.
    ok : bool ndarray (nz,)
        False for rows that are non-finite, all zero, of degree 0, or not
        converged within n_iter iterations.
    """
    nlev, ncoef = C.shape
    n = ncoef - 1
    cdt = jnp.complex128 if C.dtype == jnp.float64 else jnp.complex64
    eps = jnp.finfo(C.dtype).eps

    finite = jnp.all(jnp.isfinite(C), axis=1)
    Cn = jnp.where(finite[:, None], C, 0.0)
    cmax = jnp.max(jnp.abs(Cn), axis=1)
    # scale by a power of two (exact: dividing by cmax itself would round every
    # coefficient and move clustered roots by ~sqrt(eps) in single precision)
    _, ex = jnp.frexp(jnp.where(cmax > 0, cmax, 1.0))
    Cn = jnp.ldexp(Cn, -ex[:, None])                         # roots unchanged
    cmax = jnp.where(cmax > 0, 1.0, 0.0)

    # degree reduction (polyroots.m rule)
    BIG = float(np.finfo(np.float32).max)
    idx = jnp.arange(ncoef)
    s = jnp.zeros(nlev, dtype=jnp.int32)
    for _ in range(n):
        lead = jnp.abs(jnp.take_along_axis(Cn, s[:, None], axis=1))
        off = (idx[None, :] > s[:, None]) & (jnp.abs(Cn) > BIG * lead)
        s = jnp.where(jnp.any(off, axis=1), jnp.argmax(off, axis=1), s)
    valid = finite & (cmax > 0) & (s < n)
    jj = idx[None, :] + s[:, None]
    Cs = jnp.where(jj <= n, jnp.take_along_axis(Cn, jnp.clip(jj, 0, n), axis=1), 0.0)
    benign = jnp.zeros(ncoef, Cn.dtype).at[0].set(1.0).at[n].set(-1.0)   # x^n - 1
    Cs = jnp.where(valid[:, None], Cs, benign[None, :])

    rad = _NewtonPolygonRadii(Cs)
    ang = 2 * np.pi * np.arange(n) / n + 0.4
    z0 = (rad * jnp.exp(1j * jnp.asarray(ang))[None, :]).astype(cdt)

    eye = jnp.eye(n, dtype=bool)[None]

    # m trailing zero coefficients = m roots exactly at 0 (shift of the degree
    # reduction, or zero constant terms). They converge only linearly, so the
    # m smallest iterates are set to 0 and excused from the convergence test.
    m = jnp.sum(jnp.cumprod((Cs[:, ::-1] == 0).astype(jnp.int32), axis=1), axis=1)
    jn = jnp.arange(n)

    def zero_slots(z):
        a = jnp.abs(z)
        smaller = (a[:, None, :] < a[:, :, None]) | (
            (a[:, None, :] == a[:, :, None]) & (jn[None, None, :] < jn[None, :, None]))
        return jnp.sum(smaller, axis=2) < m[:, None]

    def aberth(z, w, frozen):
        d = z[:, :, None] - z[:, None, :]
        S = jnp.sum(jnp.where(eye, 0.0, 1.0 / jnp.where(eye, 1.0, d)), axis=2)
        zn = z - w / (1.0 - w * S)
        return jnp.where(jnp.isfinite(zn) & ~frozen, zn, z)

    def comp(_, state):
        z, done = state
        w, aP, bnd = _PolyRatio(Cs, z)
        # converged: Newton correction or |P| at the attainable round-off level
        # (a working-precision iterate cannot get closer than a few ulps)
        conv = ((jnp.abs(w) <= 64 * eps * jnp.abs(z)) | (aP <= 1024 * eps * eps * bnd)
                | ~jnp.isfinite(w) | zero_slots(z))
        done = done | jnp.all(conv, axis=1)
        return aberth(z, w, done[:, None]), done

    # fixed trip count: a while_loop would copy its predicate to the host every
    # iteration on GPUs (a stream sync). Compensated Horner from the start:
    # plain-Horner iterations park clustered roots in float32 noise. All 9000
    # test polynomials (and edge cases) converge within 45 in both precisions.
    z, done = jax.lax.fori_loop(0, n_iter, comp, (z0, ~valid))
    # rows not frozen (e.g. exact multiple roots, which converge linearly) are
    # accepted if every root is backward stable at working precision, the
    # guarantee of numpy.roots / MATLAB roots: |P(z)| <= 64 eps sum|c_i||z|^(n-i)
    _, aP, bnd = _PolyRatio(Cs, z)
    stable = jnp.all((aP <= 64 * eps * bnd) | zero_slots(z), axis=1)
    z = jnp.where(zero_slots(z), 0.0, z)
    return z, Cs, valid & (done | stable)


@jax.jit
def MaxRealRootBeta(C):
    """
    Scale-dependence parameter beta(z) from the fifth-order LASDD polynomials,
    one per vertical level, with the selection rule of the original MATLAB
    code (MLp_LocSMAG.m, MLp_LocScSMAG.m): complex roots count as 0, beta is
    the largest real root, and beta = 1 (scale invariance) if that root is not
    in (0, 5) or the polynomial is degenerate.

    Roots: PolyRootsAberth, on the device in the working precision (2026-11-19;
    a host float64 callback stalled the GPU stream twice per step, up to 9x
    slower). Real or complex is decided without a tolerance on Im z:
      - |Im z| <= 64 eps |z|: real;   |Im z| > 0.05 |z|: complex;
      - in between, the sign of P on the real axis at Re z + |Im z| *
        {-4, -2, -1, 0, 1, 2, 4} (compensated Horner, about twice the working
        precision): a sign change means real roots there (the largest is
        located by bisection). Without one, the critical point x* of P in that
        interval (bisection on P') decides: P(x*) = 0 to compensated round-off
        (<= 1024 eps^2 sum|c_i||x*|^(n-i)) means a real multiple root, otherwise
        a complex pair.
    Unconverged rows (none seen in tests) also give beta = 1.

    Parameters:
    -----------
    C : ndarray of shape (nz, 6)
        Polynomial coefficients per level, descending order of degree.

    Returns:
    --------
    beta : ndarray of shape (nz,)
    """
    z, Cs, ok = PolyRootsAberth(C)
    eps = jnp.finfo(C.dtype).eps
    x = z.real
    h = jnp.abs(z.imag)
    az = jnp.abs(z)
    strict = h <= 64 * eps * az
    amb = ~strict & (h <= 0.05 * az) & (az <= 1e6)

    # sign test on the real axis around ambiguous roots
    k = jnp.asarray([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], dtype=C.dtype)
    hs = jnp.where(amb, jnp.maximum(h, 64 * eps * az), 0.0)
    pts = x[..., None] + hs[..., None] * k                    # (nz, n, 7)
    nlev, n = z.shape
    v = _CompHorner(Cs, pts.reshape(nlev, -1)).reshape(pts.shape)
    sg = jnp.where(v >= 0, 1.0, -1.0)
    chg = sg[..., 1:] != sg[..., :-1]                         # (nz, n, 6)
    real_amb = amb & jnp.any(chg, axis=-1)
    last = (chg.shape[-1] - 1) - jnp.argmax(chg[..., ::-1], axis=-1)
    lo = jnp.take_along_axis(pts, last[..., None], axis=-1)[..., 0]
    hi = jnp.take_along_axis(pts, last[..., None] + 1, axis=-1)[..., 0]
    slo = jnp.take_along_axis(sg, last[..., None], axis=-1)[..., 0]

    def bisect(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        sm = jnp.where(_CompHorner(Cs, mid) >= 0, 1.0, -1.0)
        same = sm == slo
        return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 60, bisect, (lo, hi))
    xr = jnp.where(real_amb, 0.5 * (lo + hi), x)

    # touching test: even-multiplicity real roots give no sign change
    cols = [Cs[:, None, i] for i in range(n + 1)]

    def PdP(xx):
        Pr, _, dPr, _ = _CompHornerC(cols, xx, jnp.zeros_like(xx))
        ax = jnp.abs(xx)
        bnd = jnp.zeros_like(ax)
        for ck in cols:
            bnd = bnd * ax + jnp.abs(ck)
        return Pr, dPr, bnd

    tlo = x - 4 * hs
    thi = x + 4 * hs
    _, dlo, _ = PdP(tlo)
    _, dhi, _ = PdP(thi)
    crit = amb & ~real_amb & ((dlo >= 0) != (dhi >= 0))
    sdlo = jnp.where(dlo >= 0, 1.0, -1.0)

    def bisect_d(_, lh):
        lo, hi = lh
        mid = 0.5 * (lo + hi)
        _, dm, _ = PdP(mid)
        same = jnp.where(dm >= 0, 1.0, -1.0) == sdlo
        return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

    tlo, thi = jax.lax.fori_loop(0, 60, bisect_d, (tlo, thi))
    xs = 0.5 * (tlo + thi)
    Ps, _, bs = PdP(xs)
    touch = crit & (jnp.abs(Ps) <= 1024 * eps * eps * bs)
    xr = jnp.where(touch, xs, xr)

    real = strict | real_amb | touch
    bmax = jnp.max(jnp.where(real, xr, 0.0), axis=1)
    valid = ok & jnp.isfinite(bmax) & (bmax > 0.0) & (bmax < 5.0)
    return jnp.where(valid, bmax, 1.0).astype(C.dtype)


# ============================================================
# Compute roots of a polynomial using Laguerre's method
# ============================================================

@jax.jit
def Roots(coeffs, init_guess=1.0 + 0j, tol=1e-6, max_iter=20):
    """
    Find a root of a polynomial using Laguerre's method.

    Laguerre's method is a root-finding algorithm that works well for
    polynomials and converges cubically for simple roots. It works in
    the complex plane and can find complex roots.

    Parameters:
    -----------
    coeffs : ndarray
        Polynomial coefficients in descending order of degree.
        For a 5th degree polynomial: [a5, a4, a3, a2, a1, a0]
        represents: a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    init_guess : complex, optional
        Initial guess for the root. Default: 1.0+0j
        For real polynomials, can use real guess, but complex arithmetic
        is used internally to handle complex roots.
    tol : float, optional
        Relative step tolerance. Iteration stops when
        |x_new - x_old| < tol * (1 + |x_old|), or when the residual of the
        normalized polynomial is at round-off level (|f(x)| < 10*eps).
        Default: 1e-6
    max_iter : int, optional
        Maximum number of iterations. Default: 20

    Returns:
    --------
    complex
        Root of the polynomial if converged, or nan+0j if not converged
        within max_iter iterations.

    Notes:
    ------
    - Uses complex arithmetic internally to handle complex roots
    - Includes numerical safeguards against division by zero
    - The coefficients are normalized by their largest magnitude, so the
      result does not depend on the overall scale of the polynomial (the
      dynamic-model coefficients scale with powers of the resolved fields).
      Before 2026-11-19 an absolute residual test |f(x)| < tol made the root
      depend on field amplitude.
    - The algorithm is JIT-compiled for performance

    """

    # Convert to complex arithmetic to handle complex roots
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    rdtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    coeffs = coeffs.astype(cdtype)
    # Normalize: roots are unchanged, the residual test becomes scale-free
    scale = jnp.max(jnp.abs(coeffs))
    coeffs = coeffs / jnp.where(scale > 0, scale, 1.0)
    res_tol = 10 * jnp.finfo(rdtype).eps
    x = jnp.asarray(init_guess, dtype=cdtype)

    # Polynomial degree
    n = coeffs.shape[0] - 1

    # Helper functions for polynomial evaluation
    def polynomial(p, x):
        """Evaluate polynomial at x using Horner's method."""
        return jnp.polyval(p, x)

    def derivative(p, x):
        """Evaluate first derivative at x."""
        dp = jnp.polyder(p)
        return jnp.polyval(dp, x)

    def second_derivative(p, x):
        """Evaluate second derivative at x."""
        d2p = jnp.polyder(jnp.polyder(p))
        return jnp.polyval(d2p, x)

    # Loop continuation condition
    def cond_fn(state):
        x, iteration, converged = state
        return (iteration < max_iter) & (~converged)

    # Laguerre iteration step
    def body_fn(state):
        x, iteration, _ = state

        # Evaluate polynomial and derivatives
        f_x = polynomial(coeffs, x)
        df_x = derivative(coeffs, x)
        d2f_x = second_derivative(coeffs, x)

        # Numerical safeguard: small epsilon for complex arithmetic
        eps = 1e-16 + 0j

        # Protect against division by zero
        f_safe = jnp.where(jnp.abs(f_x) < eps, eps, f_x)

        # Laguerre's method formulas
        G = df_x / f_safe
        H = G ** 2 - d2f_x / f_safe

        # Compute discriminant: (n-1) * (n*H - G²)
        # This is the CORRECTED formula
        discriminant = (n - 1) * (n * H - G ** 2)
        sqrt_disc = jnp.sqrt(discriminant)

        # Choose denominator with larger absolute value for stability
        denom1 = G + sqrt_disc
        denom2 = G - sqrt_disc
        denom = jnp.where(jnp.abs(denom1) > jnp.abs(denom2), denom1, denom2)

        # Protect against division by zero
        denom = jnp.where(jnp.abs(denom) < eps, eps, denom)

        # Laguerre update step
        x_new = x - n / denom

        # Check convergence using both relative step size and function value
        # This handles both large and small roots effectively
        converged_step = jnp.abs(x_new - x) < tol * (1 + jnp.abs(x))
        converged_value = jnp.abs(f_x) < res_tol
        new_converged = converged_step | converged_value

        return x_new, iteration + 1, new_converged

    # Initialize state and run iteration loop
    init_state = (x, jnp.array(0), jnp.array(False))
    final_x, final_iter, converged = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    # Return root if converged, otherwise return nan
    return jnp.where(converged, final_x, jnp.nan + 0j)


# ============================================================
#  Measure memory usage
# ============================================================

@jax.jit
def LogMemory():
    """
    Prints memory usage statistics for all available JAX devices.
    Converts memory values to MB for readability.
    """

    devices = jax.devices()
    for device in devices:
        print(f"\nDevice: {device}")
        stats = device.memory_stats()

        if stats is None:
            print("Memory stats not available for this device")
            continue

        # Convert to MB for readability
        bytes_in_use = stats.get('bytes_in_use', 0) / (1024 * 1024)
        peak_bytes = stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
        allocated_bytes = stats.get('bytes_allocated', 0) / (1024 * 1024)

        print(f"Current memory usage: {bytes_in_use:.2f} MB")
        print(f"Peak memory usage: {peak_bytes:.2f} MB")
        print(f"Allocated memory: {allocated_bytes:.2f} MB")

        # Print all available stats for debugging
        print("\nAll available memory stats:")
        for key, value in stats.items():
            print(f"{key}: {value / (1024 * 1024):.2f} MB")

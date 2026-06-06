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
File: NSE_PressureTerms_Thomas.py
====================================

:Author: Sukanta Basu
:AI Assistance: Claude Code (Anthropic)
:Date: 2026-6-6
:Description: Alternative pressure solver using the Thomas (tridiagonal) algorithm.
              Drop-in replacement for PressureInit / PressureSolve in
              NSE_PressureTerms.py.

The pressure Poisson system is tridiagonal in z.  The existing PressureSolve
builds a dense (nz+1)×(nz+1) matrix for every horizontal wavenumber pair and
calls jnp.linalg.solve — O(N³) per pair.  The Thomas algorithm exploits the
tridiagonal structure directly: O(N) per pair.  The LU factorisation of the
main diagonal is precomputed once at startup by ThomasPressureInit, so each
time-step solve requires only the O(N) forward sweep of the RHS and
back-substitution.

Usage — three changes in Main.py (no other files touched):

  1. Add to imports:
       from .pde.NSE_PressureTerms_Thomas import ThomasPressureInit, ThomasPressureSolve

  2. Replace PressureInit call (before the main loop):
       (kr2_pressure, kc2_pressure,
        a_pressure, b_pressure, c_pressure,
        b_thomas, m_thomas) = ThomasPressureInit()

  3. Replace PressureSolve call (inside the main loop):
       (p, dpdx, dpdy, dpdz) = ThomasPressureSolve(
           RC_real, RC_imag, fRz_real,
           b_thomas, m_thomas, c_pressure)

  PressureRC is unchanged.
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

from ..config.ConfigLoader import *
from ..config.DerivedVars import *
from ..operations.FFT import FFT
from ..operations.Derivatives import Derivxy, Derivz_Generic_uvp
from ..initialization.Preprocess import Constant, Wavenumber, ZeRo3DIni

mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ============================================================
#  Internal helpers (not called directly from Main.py)
# ============================================================

@jax.jit
def _thomas_factorize(a, b, c):
    """
    Precompute Thomas forward-sweep factors for a batch of tridiagonal systems.

    Parameters
    ----------
    a : (N,)                  subdiagonal  (a[0] is unused)
    b : (nx, ny_rfft, N)      main diagonal (different for each wavenumber pair)
    c : (N,)                  superdiagonal (c[N-1] is unused)

    Returns
    -------
    b_mod : (nx, ny_rfft, N)      modified main diagonal after forward sweep
    m_all : (N-1, nx, ny_rfft)    multipliers  m[k] = a[k] / b_mod[k-1]
    """
    def step(b_mod, k):
        mk    = a[k] / b_mod[..., k - 1]                          # (nx, ny_rfft)
        b_new = b_mod.at[..., k].set(b_mod[..., k] - mk * c[k - 1])
        return b_new, mk

    b_mod, m_all = jax.lax.scan(step, b, jnp.arange(1, b.shape[-1]))
    return b_mod, m_all          # m_all : (N-1, nx, ny_rfft)


@jax.jit
def _thomas_solve(m_all, b_mod, c, d):
    """
    Solve a batch of tridiagonal systems using precomputed Thomas factors.

    Parameters
    ----------
    m_all : (N-1, nx, ny_rfft)    precomputed multipliers from _thomas_factorize
    b_mod : (nx, ny_rfft, N)      modified main diagonal from _thomas_factorize
    c     : (N,)                  superdiagonal
    d     : (nx, ny_rfft, N)      right-hand side

    Returns
    -------
    x : (nx, ny_rfft, N)          solution vector
    """
    N = d.shape[-1]

    # ------------------------------------------------------------------
    # Forward sweep of RHS:  d[k] -= m[k-1] * d[k-1]  for k = 1..N-1
    # m_all[i] is the multiplier for step k = i+1, shape (nx, ny_rfft)
    # ------------------------------------------------------------------
    def fwd_step(d_acc, km):
        k, mk = km                                                 # k: scalar, mk: (nx, ny_rfft)
        d_new = d_acc.at[..., k].set(d_acc[..., k] - mk * d_acc[..., k - 1])
        return d_new, None

    d_mod, _ = jax.lax.scan(fwd_step, d, (jnp.arange(1, N), m_all))

    # ------------------------------------------------------------------
    # Back substitution:
    #   x[N-1] = d_mod[N-1] / b_mod[N-1]
    #   x[k]   = (d_mod[k] - c[k] * x[k+1]) / b_mod[k]   for k = N-2..0
    # d_mod and b_mod are closures here (constant across the scan).
    # ------------------------------------------------------------------
    x_init = jnp.zeros_like(d_mod)
    x_init = x_init.at[..., N - 1].set(d_mod[..., N - 1] / b_mod[..., N - 1])

    def bwd_step(x, k):
        x_new = x.at[..., k].set(
            (d_mod[..., k] - c[k] * x[..., k + 1]) / b_mod[..., k]
        )
        return x_new, None

    x_final, _ = jax.lax.scan(bwd_step, x_init, jnp.arange(N - 2, -1, -1))
    return x_final


# ============================================================
#  Public API: ThomasPressureInit  and  ThomasPressureSolve
# ============================================================

@jax.jit
def ThomasPressureInit():
    """
    Initialise all pressure-solver data structures and precompute the
    Thomas factorisation of the tridiagonal main diagonal.

    Returns (same first five as PressureInit, plus two Thomas factors)
    -------
    kr2_pressure : (nx, ny_rfft)           x-wavenumber array
    kc2_pressure : (nx, ny_rfft)           y-wavenumber array
    a_pressure   : (nz+1,)                subdiagonal
    b_pressure   : (nx, ny_rfft, nz+1)    original main diagonal
    c_pressure   : (nz+1,)                superdiagonal
    b_thomas     : (nx, ny_rfft, nz+1)    main diagonal after Thomas forward sweep
    m_thomas     : (nz, nx, ny_rfft)      Thomas multipliers (one per elimination step)
    """

    # ------------------------------------------------------------------
    # Wavenumber arrays  (identical to PressureInit)
    # ------------------------------------------------------------------
    kr2_pressure = jnp.zeros((nx, ny_rfft))
    for i in range(nx):
        if i < nx // 2:
            kr2_pressure = kr2_pressure.at[i, :].set(i)
        else:
            kr2_pressure = kr2_pressure.at[i, :].set(i - nx)

    kc2_pressure = jnp.zeros((nx, ny_rfft))
    for j in range(ny_rfft):
        kc2_pressure = kc2_pressure.at[:, j].set(j)

    # ------------------------------------------------------------------
    # Tridiagonal coefficients  (identical to PressureInit)
    # ------------------------------------------------------------------
    a_pressure = jnp.ones(nz + 1) / (dz ** 2)
    c_pressure = jnp.ones(nz + 1) / (dz ** 2)

    a_pressure = a_pressure.at[0].set(0)     # bottom BC row: no subdiagonal entry
    a_pressure = a_pressure.at[nz].set(-1)   # top BC row
    c_pressure = c_pressure.at[0].set(1)     # bottom BC row
    c_pressure = c_pressure.at[nz].set(0)    # top BC row: no superdiagonal entry

    bb = -(kr2_pressure ** 2 + kc2_pressure ** 2 + 2.0 / (dz ** 2))
    b_pressure = jnp.repeat(bb[:, :, jnp.newaxis], nz + 1, axis=2)
    b_pressure = b_pressure.at[:, :, 0].set(-1.0)   # bottom BC
    b_pressure = b_pressure.at[:, :, nz].set(1.0)   # top BC

    # ------------------------------------------------------------------
    # Thomas factorisation  (new — computed once at startup)
    #
    # The (i,j)=(0,0) zero mode has kx=ky=0, which makes the interior
    # main-diagonal entries equal to -2/dz² and the Neumann-Neumann
    # system singular.  The Thomas forward sweep would produce a zero
    # (or near-zero) final pivot, causing division by zero in the back
    # substitution.  Fix: replace the interior rows of b[0,0,:] with
    # -(1 + 2/dz²) — a non-singular Helmholtz system — before factorising.
    # The (0,0) Thomas solution is never used: ThomasPressureSolve
    # supplies the correct zero-mode pressure via direct integration.
    # ------------------------------------------------------------------
    b_safe = b_pressure.at[0, 0, 1:nz].set(-(1.0 + 2.0 / dz ** 2))
    b_thomas, m_thomas = _thomas_factorize(a_pressure, b_safe, c_pressure)

    return (kr2_pressure, kc2_pressure,
            a_pressure, b_pressure, c_pressure,
            b_thomas, m_thomas)


@jax.jit
def ThomasPressureSolve(RC_real, RC_imag, fRz_real,
                        b_thomas, m_thomas, c_pressure):
    """
    Solve the pressure Poisson equation using the Thomas algorithm.
    Drop-in replacement for PressureSolve in NSE_PressureTerms.py.

    Parameters
    ----------
    RC_real, RC_imag : (nx, ny_rfft, nz+1)   real/imag RHS from PressureRC
    fRz_real         : (nx, ny_rfft, nz)      for zero-mode direct integration
    b_thomas         : (nx, ny_rfft, nz+1)   pre-factorised main diagonal
    m_thomas         : (nz, nx, ny_rfft)     Thomas multipliers
    c_pressure       : (nz+1,)              superdiagonal

    Returns
    -------
    p, dpdx, dpdy, dpdz : same shapes and semantics as PressureSolve
    """

    # ------------------------------------------------------------------
    # Mask RHS for special modes before solving.
    #
    # Zero mode (0,0): singular — handled by direct integration below.
    # Nyquist modes (x-Nyquist row and y-Nyquist column): aliased — must
    # be zero in the output.  Zeroing the RHS here means Thomas solves
    # these rows with d=0 and produces x=0, avoiding any reliance on
    # solver behaviour for modes whose results are discarded.
    # ------------------------------------------------------------------
    RC_r = (RC_real
            .at[0,          0,          :].set(0.0)   # zero mode
            .at[nx_rfft - 1, :,         :].set(0.0)   # x-Nyquist row
            .at[:,          ny_rfft - 1, :].set(0.0))  # y-Nyquist column
    RC_i = (RC_imag
            .at[0,          0,          :].set(0.0)
            .at[nx_rfft - 1, :,         :].set(0.0)
            .at[:,          ny_rfft - 1, :].set(0.0))

    # ------------------------------------------------------------------
    # Solve all (nx × ny_rfft) tridiagonal systems in one pass.
    # Special modes have zero RHS so they produce zero solutions.
    # ------------------------------------------------------------------
    x_real = _thomas_solve(m_thomas, b_thomas, c_pressure, RC_r)
    x_imag = _thomas_solve(m_thomas, b_thomas, c_pressure, RC_i)

    # The boundary row k=0 encodes the Neumann BC; interior solution is [1:]
    fp_real = x_real[..., 1:]   # (nx, ny_rfft, nz)
    fp_imag = x_imag[..., 1:]

    # ------------------------------------------------------------------
    # Zero mode (i=0, j=0): overwrite the zero Thomas solution with the
    # correct pressure obtained by direct vertical integration of fRz,
    # identical to PressureSolve.  Uses the original (unmasked) RC_real.
    # ------------------------------------------------------------------
    zero_first = RC_real[0, 0, 0]
    zero_rest  = zero_first + jnp.cumsum(fRz_real[0, 0, 1:nz] * dz)
    zero_mode  = jnp.concatenate([jnp.array([zero_first]), zero_rest])
    fp_real = fp_real.at[0, 0].set(zero_mode)
    fp_imag = fp_imag.at[0, 0].set(jnp.zeros(nz))
    # Nyquist entries are already zero (RHS was zeroed above).

    # ------------------------------------------------------------------
    # Reconstruct pressure and compute its gradients
    # (identical to PressureSolve)
    # ------------------------------------------------------------------
    fp   = fp_real + 1j * fp_imag
    p    = jnp.fft.irfft2(fp, axes=(0, 1), s=(nx, ny))

    kx2, ky2 = Wavenumber()
    p_fft    = FFT(p)
    dpdx     = Derivxy(p_fft, kx2)
    dpdy     = Derivxy(p_fft, ky2)
    dum      = ZeRo3DIni()
    dpdz     = Derivz_Generic_uvp(p, dum)

    return p, dpdx, dpdy, dpdz

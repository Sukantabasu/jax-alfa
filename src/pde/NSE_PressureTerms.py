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
File: NSE_PressureTerms.py
=============================

:Author: Sukanta Basu
:AI Assistance: Claude.AI (Anthropic) is used for documentation,
                code restructuring, and performance optimization
:Date: 2025-4-3
:Description: computes pressure terms by solving the Poisson equation using rfft2
"""

# ============================================================
#  Imports
# ============================================================

import jax
import jax.numpy as jnp

# Import configuration from namelist
from ..config.Config import *

# Import derived variables
from ..config.DerivedVars import *

# Import FFT modules
from ..operations.FFT import FFT

# Import derivative functions
from ..operations.Derivatives import Derivxy, Derivz_Generic_uvp

from ..initialization.Preprocess import Constant, Wavenumber, ZeRo3DIni
mx, my, nx_rfft, ny_rfft, mx_rfft, my_rfft = Constant()


# ============================================================
#  Initialize static variables for pressure solver
# ============================================================

@jax.jit
def PressureInit():
    """
    Returns:
    --------
    kr2_pressure : ndarray, shape (nx, ny_rfft)
        Wavenumber array for x-direction derivatives, for rfft2
    kc2_pressure : ndarray, shape (nx, ny_rfft)
        Wavenumber array for y-direction derivatives, for rfft2
    a_pressure : ndarray, shape (nz+1)
        Coefficient array for tridiagonal matrix (sub-diagonal)
    b_pressure : ndarray, shape (nx, ny_rfft, nz+1)
        Main diagonal coefficients for each grid point
    c_pressure : ndarray, shape (nz+1)
        Coefficient array for tridiagonal matrix (super-diagonal)
    """
    # Create 2D wavenumber arrays
    # For x-direction, use the full range
    kr2_pressure = jnp.zeros((nx, ny_rfft))
    for i in range(nx):
        if i < nx // 2:
            kr2_pressure = kr2_pressure.at[i, :].set(i)
        else:
            kr2_pressure = kr2_pressure.at[i, :].set(i - nx)

    # For y-direction, use only positive frequencies
    kc2_pressure = jnp.zeros((nx, ny_rfft))
    for j in range(ny_rfft):
        kc2_pressure = kc2_pressure.at[:, j].set(j)

    # Create tridiagonal matrix coefficients
    a_pressure = jnp.ones(nz + 1) / (dz ** 2)  # sub-diagonal
    c_pressure = jnp.ones(nz + 1) / (dz ** 2)  # super-diagonal

    # Set boundary conditions for tridiagonal matrix
    a_pressure = a_pressure.at[0].set(0)  # bottom boundary
    a_pressure = a_pressure.at[nz].set(-1)  # top boundary
    c_pressure = c_pressure.at[0].set(1)  # bottom boundary
    c_pressure = c_pressure.at[nz].set(0)  # top boundary

    # Calculate the main diagonal values for each grid point
    bb = -(kr2_pressure ** 2 + kc2_pressure ** 2 + 2 / (dz ** 2))
    b_pressure = jnp.repeat(bb[:, :, jnp.newaxis], nz + 1, axis=2)
    b_pressure = b_pressure.at[:, :, 0].set(-1)
    b_pressure = b_pressure.at[:, :, nz].set(1)

    return (kr2_pressure, kc2_pressure,
            a_pressure, b_pressure, c_pressure)


# ============================================================
#  Compute right-hand side for pressure equation
# ============================================================

@jax.jit
def PressureRC(
        u, v, w,
        RHS_u, RHS_v, RHS_w,
        RHS_u_previous, RHS_v_previous, RHS_w_previous,
        divtz, kr2_pressure, kc2_pressure):
    """
    Calculate the right-hand side of the Poisson equation for pressure using rfft2.

    Parameters:
    -----------
    u, v, w : ndarray of shape (nx, ny, nz)
        Velocity components at current time step
    RHS_u, RHS_v, RHS_w : ndarray of shape (nx, ny, nz)
        Right-hand side terms for momentum equations at current time step
    RHS_u_previous, RHS_v_previous, RHS_w_previous : ndarray of shape (nx, ny, nz)
        Right-hand side terms for momentum equations at previous time step
    divtz : ndarray of shape (nx, ny, nz)
        Divergence of the SGS stress tensor in z-direction
    kr2_pressure, kc2_pressure : ndarray
        Wavenumber arrays for spectral derivatives with rfft2

    Returns:
    --------
    RC_real : ndarray of shape (nx, ny_rfft, nz+1)
        Real part of the right-hand side for pressure Poisson equation
    RC_imag : ndarray of shape (nx, ny_rfft, nz+1)
        Imaginary part of the right-hand side for pressure Poisson equation
    fRz_real : ndarray
        Real part of fRz for zero mode processing
    """
    # Compute intermediate terms
    Rx = RHS_u - (1 / 3) * RHS_u_previous + (2 / (3 * dt_nondim)) * u
    Ry = RHS_v - (1 / 3) * RHS_v_previous + (2 / (3 * dt_nondim)) * v
    Rz = RHS_w - (1 / 3) * RHS_w_previous + (2 / (3 * dt_nondim)) * w

    # Calculate rfft2 transforms
    fRx = jnp.fft.rfft2(Rx, axes=(0, 1))
    fRy = jnp.fft.rfft2(Ry, axes=(0, 1))
    fRz = jnp.fft.rfft2(Rz, axes=(0, 1))

    # Initialize arrays for RC_real and RC_imag with zeros
    RC_real = jnp.zeros((nx, ny_rfft, nz + 1))
    RC_imag = jnp.zeros((nx, ny_rfft, nz + 1))

    # Get boundary terms
    fbot = jnp.fft.rfft2(divtz[:, :, 0], axes=(0, 1))
    ftop = jnp.fft.rfft2(divtz[:, :, -1], axes=(0, 1))

    # Set boundary conditions
    RC_real = RC_real.at[:, :, 0].set(-dz * jnp.real(fbot))
    RC_imag = RC_imag.at[:, :, 0].set(-dz * jnp.imag(fbot))
    RC_real = RC_real.at[:, :, nz].set(-dz * jnp.real(ftop))
    RC_imag = RC_imag.at[:, :, nz].set(-dz * jnp.imag(ftop))

    # Prepare wavenumbers for broadcasting
    kr2_3d = kr2_pressure[:, :, jnp.newaxis]
    kc2_3d = kc2_pressure[:, :, jnp.newaxis]

    # Compute horizontal derivatives for interior points
    horiz_deriv_real = -kr2_3d * jnp.imag(fRx[:, :, :-1])
    horiz_deriv_real -= kc2_3d * jnp.imag(fRy[:, :, :-1])

    horiz_deriv_imag = kr2_3d * jnp.real(fRx[:, :, :-1])
    horiz_deriv_imag += kc2_3d * jnp.real(fRy[:, :, :-1])

    # Compute vertical derivatives
    vert_diff_real = jnp.diff(jnp.real(fRz), axis=2) / dz
    vert_diff_imag = jnp.diff(jnp.imag(fRz), axis=2) / dz

    # Update interior points
    RC_real = RC_real.at[:, :, 1:nz].set(horiz_deriv_real + vert_diff_real)
    RC_imag = RC_imag.at[:, :, 1:nz].set(horiz_deriv_imag + vert_diff_imag)

    return RC_real, RC_imag, jnp.real(fRz)


@jax.jit
def PressureMatrix(a_pressure, b_pressure_ij, c_pressure):
    """
    Create a tridiagonal matrix for pressure solver for a specific (i,j) wavenumber.

    Parameters:
    -----------
    a_pressure : ndarray, shape (nz+1)
        Lower diagonal coefficients of the tridiagonal matrix
    b_pressure_ij : ndarray, shape (nz+1)
        Main diagonal coefficients for a specific wavenumber pair (i,j)
    c_pressure : ndarray, shape (nz+1)
        Upper diagonal coefficients of the tridiagonal matrix

    Returns:
    --------
    L_matrix : ndarray, shape (nz+1, nz+1)
        Tridiagonal matrix for the specific wavenumber pair
    """
    # Initialize the tridiagonal matrix
    L_matrix = jnp.zeros((nz + 1, nz + 1))

    # Set main diagonal (b values)
    L_matrix = L_matrix.at[jnp.arange(nz + 1), jnp.arange(nz + 1)].set(b_pressure_ij)

    # Set lower diagonal (a values) - skip first row
    L_matrix = L_matrix.at[jnp.arange(1, nz + 1), jnp.arange(nz)].set(a_pressure[1:])

    # Set upper diagonal (c values) - skip last row
    L_matrix = L_matrix.at[jnp.arange(nz), jnp.arange(1, nz + 1)].set(c_pressure[:nz])

    return L_matrix


@jax.jit
def PressureSolve(RC_real, RC_imag, fRz_real, a_pressure, b_pressure, c_pressure):
    """
    Solve the pressure Poisson equation using rfft2 with batched processing.

    Parameters:
    -----------
    RC_real : ndarray of shape (nx, ny_rfft, nz+1)
        Real part of the right-hand side
    RC_imag : ndarray of shape (nx, ny_rfft, nz+1)
        Imaginary part of the right-hand side
    fRz_real : ndarray
        Real part of fRz for zero mode processing
    a_pressure, c_pressure : ndarray of shape (nz+1)
        Coefficients for tridiagonal matrix
    b_pressure : ndarray of shape (nx, ny_rfft, nz+1)
        Main diagonal coefficients for all wavenumbers

    Returns:
    --------
    p : ndarray of shape (nx, ny, nz)
        Pressure field
    dpdx, dpdy, dpdz : ndarray of shape (nx, ny, nz)
        Pressure gradient components
    """
    # Function to solve for a single (i, j) pair
    def solve_system(i, j, b_ij, rc_real, rc_imag):
        # In rfft2, special cases are: zero mode (0,0) and Nyquist (nx//2, 0)
        is_zero_mode = (i == 0) & (j == 0)
        is_nyquist_x = (i == nx_rfft - 1)  # This is nx//2 for even nx
        is_nyquist_y = (j == ny_rfft - 1)  # This is ny//2 for even ny
        is_special = is_zero_mode | is_nyquist_x | is_nyquist_y

        # Function to solve the tridiagonal system
        def solve_case(_):
            L_matrix = PressureMatrix(a_pressure, b_ij, c_pressure)
            return jnp.linalg.solve(L_matrix, rc_real)[1:], jnp.linalg.solve(L_matrix, rc_imag)[1:]

        # Function to return zeros for special cases
        def skip_case(_):
            return jnp.zeros(nz), jnp.zeros(nz)

        return jax.lax.cond(is_special, skip_case, solve_case, None)

    # Define batch size - number of rows to process at a time
    batch_size = 8

    # Function to process a batch of rows at once using vmap
    def process_batch(carry, i_batch):
        # Apply vmap to solve each (i,j) pair in the batch
        fp_real_batch, fp_imag_batch = jax.vmap(
            jax.vmap(solve_system, in_axes=(None, 0, 0, 0, 0)),
            in_axes=(0, None, 0, 0, 0)
        )(i_batch, jnp.arange(ny_rfft), b_pressure[i_batch], RC_real[i_batch], RC_imag[i_batch])

        return carry, (fp_real_batch, fp_imag_batch)

    # Create batches of row indices
    num_full_batches = nx // batch_size
    last_batch_size = nx % batch_size

    # Process full batches first
    i_batches = jnp.arange(num_full_batches * batch_size).reshape(-1, batch_size)
    _, (fp_real_batches, fp_imag_batches) = jax.lax.scan(process_batch, None, i_batches)

    # Reshape the batches to correct dimensions
    fp_real = fp_real_batches.reshape(num_full_batches * batch_size, ny_rfft, nz)
    fp_imag = fp_imag_batches.reshape(num_full_batches * batch_size, ny_rfft, nz)

    # Handle last partial batch if needed
    if last_batch_size > 0:
        last_batch = jnp.arange(num_full_batches * batch_size, nx)

        # Pad the last batch to match batch_size (ensuring consistent processing)
        padded_last_batch = jnp.pad(last_batch, (0, batch_size - last_batch_size),
                                    mode='constant', constant_values=last_batch[-1])

        # Process the padded last batch
        _, (fp_real_last_padded, fp_imag_last_padded) = process_batch(None, padded_last_batch)

        # Extract only the valid results (discard padding results)
        fp_real_last = fp_real_last_padded[:last_batch_size]
        fp_imag_last = fp_imag_last_padded[:last_batch_size]

        # Combine results
        fp_real = jnp.concatenate([fp_real, fp_real_last], axis=0)
        fp_imag = jnp.concatenate([fp_imag, fp_imag_last], axis=0)

    # Handle the zero mode (i=0, j=0) special case
    zero_mode_first = RC_real[0, 0, 0]
    zero_mode_rest = zero_mode_first + jnp.cumsum(fRz_real[0, 0, 1:nz] * dz)
    zero_mode = jnp.concatenate([jnp.array([zero_mode_first]), zero_mode_rest])

    # Set the zero mode values
    fp_real = fp_real.at[0, 0].set(zero_mode)
    fp_imag = fp_imag.at[0, 0].set(jnp.zeros(nz))

    # Combine real and imaginary parts into complex values
    fp = fp_real + 1j * fp_imag

    # Compute inverse FFT using irfft2 to get back to physical space
    p = jnp.fft.irfft2(fp, axes=(0, 1), s=(nx, ny))

    # Compute pressure derivatives
    p_fft = FFT(p)  # Fourier transform of pressure

    # Get wavenumbers for spectral derivatives
    kx2, ky2 = Wavenumber()

    dpdx = Derivxy(p_fft, kx2)  # x-derivative
    dpdy = Derivxy(p_fft, ky2)  # y-derivative

    # z-derivative using finite differences
    dum = ZeRo3DIni()
    dpdz = Derivz_Generic_uvp(p, dum)

    return p, dpdx, dpdy, dpdz

import numpy as np
from numba import njit

EPS = 1.0e-14

BC_DIRICHLET = 1
BC_ZEROGRADIENT = 3

# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always", cache=True, fastmath=True)
def compute_diffusive_flux_matrix_entry(f, grad_phi, mesh, mu):
    """
    Over‑relaxed implicit conductance for one internal face.
    Optimized with pre-fetched mesh data and manual norm calculations.
    """
    # Pre-fetch connectivity and mesh data (better cache locality)
    P = mesh.owner_cells[f]
    N = mesh.neighbor_cells[f]
    mu_f = mu 

    # Pre-fetch and cache vector components (single memory access per component)
    vector_E_f = mesh.vector_S_f[f]
    vector_d_CE = mesh.vector_d_CE[f]
    E_f_0 = vector_E_f[0]
    E_f_1 = vector_E_f[1]
    d_CE_0 = vector_d_CE[0]
    d_CE_1 = vector_d_CE[1]

    # Manual norm calculations (faster than np.linalg.norm)
    E_mag = np.sqrt(E_f_0 * E_f_0 + E_f_1 * E_f_1)
    d_mag = np.sqrt(d_CE_0 * d_CE_0 + d_CE_1 * d_CE_1)

    # Over‑relaxed orthogonal conductance (Eq 8.58)
    geoDiff = E_mag / d_mag
    Flux_P_f = mu_f * geoDiff
    Flux_N_f = -mu_f * geoDiff

    return Flux_P_f, Flux_N_f


@njit(inline="always", cache=True, fastmath=True)
def compute_diffusive_correction(f, grad_phi, mesh, mu):
    """
    Compute diffusive correction term.
    For orthogonal Cartesian grids, T_f = 0 and skewness = 0, so correction is always 0.
    """
    # For orthogonal grids, there is no non-orthogonal correction
    return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Boundary faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always", cache=True, fastmath=True)
def compute_boundary_diffusive_correction(
        f, U, grad_phi, mesh, mu, p_b, bc_type, bc_val, component_idx):
    """
    Return (diffFlux_P_b, diffFlux_N_b) for boundary diffusion.

    diffFlux_P_b : diagonal coefficient to add to owner cell
    diffFlux_N_b : RHS increment (will be subtracted: b[P] -= diffFlux_N_b)

    Supports only BC_DIRICHLET (all velocity boundaries use this).
    For BC_ZEROGRADIENT (used for pressure), no diffusive flux is applied.
    """
    muF = mu
    E_f = np.ascontiguousarray(mesh.vector_S_f[f])
    d_PB = mesh.d_Cb[f]

    if bc_type == BC_DIRICHLET:
        # Dirichlet BC: fixed value at boundary
        E_mag = np.linalg.norm(E_f)
        diffFlux_P_b = muF * E_mag / d_PB
        diffFlux_N_b = -diffFlux_P_b * bc_val
        return diffFlux_P_b, diffFlux_N_b

    # BC_ZEROGRADIENT or any other type: no diffusive flux
    return 0.0, 0.0

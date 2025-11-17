import numpy as np
from numba import njit

EPS = 1.0e-14

# ──────────────────────────────────────────────────────────────────────────────
# Internal faces
# ──────────────────────────────────────────────────────────────────────────────
@njit(inline="always", cache=True, fastmath=True)
def compute_diffusive_flux_matrix_entry(f, mesh, mu):
    """
    Orthogonal implicit conductance for one internal face.
    Optimized with pre-fetched mesh data and manual norm calculations.
    """
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

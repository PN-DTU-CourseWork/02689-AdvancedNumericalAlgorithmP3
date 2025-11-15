"""Scipy-based linear solver using direct method (spsolve)."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def scipy_solver(
    A_csr: csr_matrix,
    b_np: np.ndarray,
    remove_nullspace: bool = False,
    **kwargs  # Accept and ignore extra parameters for flexible calling
):
    """
    Solve A x = b using SciPy sparse direct solver (spsolve).

    Works well for small to medium-sized problems (< 100k unknowns).

    Parameters
    ----------
    A_csr : csr_matrix
        Sparse matrix in CSR format.
    b_np : np.ndarray
        Right-hand side vector.
    remove_nullspace : bool, optional
        Whether to handle the nullspace (default: False).
        If True, constrains the solution to have zero at first DOF.
    **kwargs
        Additional parameters (ignored, for compatibility).

    Returns
    -------
    x_np : np.ndarray
        Solution vector x.
    residual_norm : float
        Final 2-norm of the residual: ||b - A x||_2.
    ksp : None
        Always returns None (for API compatibility).
    """
    n = A_csr.shape[0]

    # Handle nullspace if requested (pressure correction equation)
    if remove_nullspace:
        # Pin first value to zero to remove nullspace
        A_modified = A_csr.tolil()  # Convert to LIL for efficient row modification
        A_modified[0, :] = 0.0
        A_modified[0, 0] = 1.0
        A_work = A_modified.tocsr()

        b_modified = b_np.copy()
        b_modified[0] = 0.0
        b_work = b_modified
    else:
        A_work = A_csr
        b_work = b_np

    # Solve using direct solver (LU factorization)
    try:
        x_np = spsolve(A_work, b_work)
    except Exception as e:
        raise RuntimeError(f"Direct solver failed: {e}")

    # Ensure x_np is a 1D array (spsolve sometimes returns matrices)
    if hasattr(x_np, 'A1'):
        x_np = x_np.A1

    # Compute residual norm
    residual = b_work - A_work @ x_np
    residual_norm = np.linalg.norm(residual)

    return x_np, residual_norm, None

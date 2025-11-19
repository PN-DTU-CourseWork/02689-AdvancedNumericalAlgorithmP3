"""
Method of Manufactured Solutions (MMS) Test for Spectral Solver
================================================================

This script verifies the spectral convergence of our pseudospectral
implementation by solving a Navier-Stokes problem with known analytical solution.

Manufactured Solution:
- u(x,y) = sin(π*x) * cos(π*y)
- v(x,y) = -cos(π*x) * sin(π*y)
- p(x,y) = cos(2*π*x) * sin(2*π*y)

This solution is divergence-free by construction: ∂u/∂x + ∂v/∂y = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from spectral.spectral import LegendreLobattoBasis


class ManufacturedSolution:
    """Defines the manufactured solution and computes source terms."""

    def __init__(self, Re=100.0):
        self.Re = Re
        self.nu = 1.0 / Re

    def exact_u(self, x, y):
        """Exact u velocity: sin(πx)cos(πy)"""
        return np.sin(np.pi * x) * np.cos(np.pi * y)

    def exact_v(self, x, y):
        """Exact v velocity: -cos(πx)sin(πy)"""
        return -np.cos(np.pi * x) * np.sin(np.pi * y)

    def exact_p(self, x, y):
        """Exact pressure: cos(2πx)sin(2πy)"""
        return np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def source_u(self, x, y):
        """Source term for u-momentum equation.

        From Navier-Stokes: ∂u/∂t + u·∇u = -∇p + ν∇²u + S_u
        For steady state (∂u/∂t = 0): S_u = u·∇u + ∇p - ν∇²u
        """
        u = self.exact_u(x, y)
        v = self.exact_v(x, y)

        # Derivatives of u
        du_dx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        du_dy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        d2u_dx2 = -np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)
        d2u_dy2 = -np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y)

        # Pressure gradient
        dp_dx = -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

        # Convection
        conv = u * du_dx + v * du_dy

        # Laplacian
        lap_u = d2u_dx2 + d2u_dy2

        return conv + dp_dx - self.nu * lap_u

    def source_v(self, x, y):
        """Source term for v-momentum equation.

        From Navier-Stokes: S_v = u·∇v + ∇p - ν∇²v
        """
        u = self.exact_u(x, y)
        v = self.exact_v(x, y)

        # Derivatives of v
        dv_dx = np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        dv_dy = -np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        d2v_dx2 = np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * y)
        d2v_dy2 = np.pi**2 * np.cos(np.pi * x) * np.sin(np.pi * y)

        # Pressure gradient
        dp_dy = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        # Convection
        conv = u * dv_dx + v * dv_dy

        # Laplacian
        lap_v = d2v_dx2 + d2v_dy2

        return conv + dp_dy - self.nu * lap_v


class SpectralMMSSolver:
    """Simplified spectral solver for MMS testing with source terms."""

    def __init__(self, N, Re=100.0, manufactured_solution=None):
        """
        Parameters
        ----------
        N : int
            Polynomial order (nodes = N+1)
        Re : float
            Reynolds number
        manufactured_solution : ManufacturedSolution
            Object providing exact solution and source terms
        """
        self.N = N
        self.Re = Re
        self.nu = 1.0 / Re
        self.mms = manufactured_solution

        # Setup grid and operators
        self._setup_grid()
        self._build_operators()

    def _setup_grid(self):
        """Setup Legendre-Gauss-Lobatto grid on [0,1] × [0,1]"""
        basis = LegendreLobattoBasis(domain=(0.0, 1.0))
        # Get nodes in [0, 1]
        x_nodes = basis.nodes(self.N + 1)
        y_nodes = basis.nodes(self.N + 1)

        self.x, self.y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
        self.x_flat = self.x.ravel()
        self.y_flat = self.y.ravel()

        # Store 1D nodes for timestep calculation
        self.x_1d = x_nodes
        self.y_1d = y_nodes

    def _build_operators(self):
        """Build spectral differentiation matrices"""
        basis = LegendreLobattoBasis(domain=(0.0, 1.0))

        # Get nodes
        nodes = basis.nodes(self.N + 1)

        # 1D differentiation matrix already scaled to physical domain [0,1]
        D_1d = basis.diff_matrix(nodes)

        # Build 2D operators via Kronecker products
        # For meshgrid with indexing='ij': first index is x, second is y
        Ix = np.eye(self.N + 1)
        Iy = np.eye(self.N + 1)
        self.Dx = np.kron(D_1d, Iy)  # d/dx: differentiate along first index
        self.Dy = np.kron(Ix, D_1d)  # d/dy: differentiate along second index

        # Second derivatives
        D2_1d = D_1d @ D_1d
        self.Dxx = np.kron(D2_1d, Iy)
        self.Dyy = np.kron(Ix, D2_1d)
        self.Laplacian = self.Dxx + self.Dyy

    def _compute_residuals(self, u, v, p, S_u, S_v, beta_squared):
        """Compute residuals R from equations (6) and (7).

        R_p = -β²(∇·u)  (continuity)
        R_u = -(u·∇)u - ∇p + Re^(-1)∇²u + S_u  (momentum)
        R_v = -(u·∇)v - ∇p + Re^(-1)∇²v + S_v  (momentum)
        """
        # Compute spatial derivatives
        du_dx = self.Dx @ u
        du_dy = self.Dy @ u
        dv_dx = self.Dx @ v
        dv_dy = self.Dy @ v
        dp_dx = self.Dx @ p
        dp_dy = self.Dy @ p

        lap_u = self.Laplacian @ u
        lap_v = self.Laplacian @ v

        # Convective terms
        conv_u = u * du_dx + v * du_dy
        conv_v = u * dv_dx + v * dv_dy

        # Momentum residuals (equation 7)
        R_u = -conv_u - dp_dx + self.nu * lap_u + S_u
        R_v = -conv_v - dp_dy + self.nu * lap_v + S_v

        # Continuity residual (equation 6)
        divergence = du_dx + dv_dy
        R_p = -beta_squared * divergence

        return R_u, R_v, R_p

    def _apply_bcs(self, u, v):
        """Apply Dirichlet boundary conditions from exact solution."""
        u_2d = u.reshape((self.N+1, self.N+1))
        v_2d = v.reshape((self.N+1, self.N+1))

        # West (x=0)
        u_2d[0, :] = self.mms.exact_u(0.0, self.y[0, :])
        v_2d[0, :] = self.mms.exact_v(0.0, self.y[0, :])
        # East (x=1)
        u_2d[-1, :] = self.mms.exact_u(1.0, self.y[-1, :])
        v_2d[-1, :] = self.mms.exact_v(1.0, self.y[-1, :])
        # South (y=0)
        u_2d[:, 0] = self.mms.exact_u(self.x[:, 0], 0.0)
        v_2d[:, 0] = self.mms.exact_v(self.x[:, 0], 0.0)
        # North (y=1)
        u_2d[:, -1] = self.mms.exact_u(self.x[:, -1], 1.0)
        v_2d[:, -1] = self.mms.exact_v(self.x[:, -1], 1.0)

    def solve_steady_state(self, max_iter=5000, tolerance=1e-10):
        """Solve steady Navier-Stokes with source terms using pseudo time-stepping.

        Returns
        -------
        u, v, p : np.ndarray
            Computed velocity and pressure fields
        error_u, error_v, error_p : float
            L2 errors compared to exact solution
        """
        # Initialize with exact solution (perfect initialization)
        u = self.mms.exact_u(self.x_flat, self.y_flat)
        v = self.mms.exact_v(self.x_flat, self.y_flat)
        p = self.mms.exact_p(self.x_flat, self.y_flat)

        # Compute source terms on grid
        S_u = self.mms.source_u(self.x_flat, self.y_flat)
        S_v = self.mms.source_v(self.x_flat, self.y_flat)

        # Artificial compressibility parameter
        beta_squared = 5.0

        # Adaptive timestep based on grid spacing
        dx_min = np.min(np.diff(self.x_1d))
        dy_min = np.min(np.diff(self.y_1d))
        h_min = min(dx_min, dy_min)
        CFL = 0.2  # RK4 is more stable than Forward Euler
        dt = CFL * h_min**2 / self.nu

        print(f"\nMMS Test: N={self.N}, Re={self.Re}")
        print(f"Grid: {self.N+1}×{self.N+1} = {(self.N+1)**2} nodes")
        print(f"Timestep: {dt:.6e}")

        for iter in range(max_iter):
            u_old = u.copy()
            v_old = v.copy()

            # Store initial state for RK4
            u_n = u.copy()
            v_n = v.copy()
            p_n = p.copy()

            # 4-stage RK4 from equation (5)
            # Stage 1: φ^(1) = φ^n + (1/4)Δτ R(φ^n)
            R_u, R_v, R_p = self._compute_residuals(u, v, p, S_u, S_v, beta_squared)
            u = u_n + 0.25 * dt * R_u
            v = v_n + 0.25 * dt * R_v
            p = p_n + 0.25 * dt * R_p
            self._apply_bcs(u, v)

            # Stage 2: φ^(2) = φ^n + (1/3)Δτ R(φ^(1))
            R_u, R_v, R_p = self._compute_residuals(u, v, p, S_u, S_v, beta_squared)
            u = u_n + (1.0/3.0) * dt * R_u
            v = v_n + (1.0/3.0) * dt * R_v
            p = p_n + (1.0/3.0) * dt * R_p
            self._apply_bcs(u, v)

            # Stage 3: φ^(3) = φ^n + (1/2)Δτ R(φ^(2))
            R_u, R_v, R_p = self._compute_residuals(u, v, p, S_u, S_v, beta_squared)
            u = u_n + 0.5 * dt * R_u
            v = v_n + 0.5 * dt * R_v
            p = p_n + 0.5 * dt * R_p
            self._apply_bcs(u, v)

            # Stage 4: φ^(n+1) = φ^n + Δτ R(φ^(3))
            R_u, R_v, R_p = self._compute_residuals(u, v, p, S_u, S_v, beta_squared)
            u = u_n + dt * R_u
            v = v_n + dt * R_v
            p = p_n + dt * R_p
            self._apply_bcs(u, v)

            # Check convergence
            if iter % 100 == 0:
                du = np.linalg.norm(u - u_old)
                dv = np.linalg.norm(v - v_old)
                residual = max(du, dv)

                if iter % 500 == 0:
                    print(f"  Iter {iter:4d}: residual = {residual:.6e}")

                if residual < tolerance:
                    print(f"  Converged at iteration {iter}")
                    break

        # Compute L2 errors
        u_exact = self.mms.exact_u(self.x_flat, self.y_flat)
        v_exact = self.mms.exact_v(self.x_flat, self.y_flat)
        p_exact = self.mms.exact_p(self.x_flat, self.y_flat)

        error_u = np.sqrt(np.mean((u - u_exact)**2))
        error_v = np.sqrt(np.mean((v - v_exact)**2))
        error_p = np.sqrt(np.mean((p - p_exact)**2))

        print(f"  L2 errors: u={error_u:.6e}, v={error_v:.6e}, p={error_p:.6e}")

        return u, v, p, error_u, error_v, error_p


def run_convergence_study():
    """Run MMS convergence study for increasing polynomial orders."""

    print("=" * 70)
    print("Spectral MMS Convergence Study")
    print("=" * 70)

    Re = 100.0
    mms = ManufacturedSolution(Re=Re)

    # Test polynomial orders - extended range to show spectral convergence
    N_values = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    errors_u = []
    errors_v = []
    errors_p = []

    for N in N_values:
        solver = SpectralMMSSolver(N=N, Re=Re, manufactured_solution=mms)
        _, _, _, error_u, error_v, error_p = solver.solve_steady_state(max_iter=5000, tolerance=1e-10)

        errors_u.append(error_u)
        errors_v.append(error_v)
        errors_p.append(error_p)

    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))

    nodes = [N+1 for N in N_values]
    ax.semilogy(nodes, errors_u, 'o-', label='u velocity', linewidth=2, markersize=8)
    ax.semilogy(nodes, errors_v, 's-', label='v velocity', linewidth=2, markersize=8)
    ax.semilogy(nodes, errors_p, '^-', label='pressure', linewidth=2, markersize=8)

    # Add reference lines for algebraic convergence
    N_ref = np.array([5, 21])
    # N^-4 (4th order algebraic)
    p4_ref = 1e-2 * (N_ref / 5)**(-4)
    ax.loglog(N_ref, p4_ref, 'k--', alpha=0.3, label='N⁻⁴ (4th order)')
    # N^-8 (8th order algebraic)
    p8_ref = 1e-3 * (N_ref / 5)**(-8)
    ax.loglog(N_ref, p8_ref, 'k:', alpha=0.3, label='N⁻⁸ (8th order)')

    ax.set_xlabel('Number of nodes (N+1)', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title(f'Spectral Convergence Study (Re={Re})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / "mms_convergence.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {output_path}")

    # Print convergence table
    print("\n" + "=" * 70)
    print("Convergence Table")
    print("=" * 70)
    print(f"{'N':>3} {'Nodes':>6} {'Error(u)':>12} {'Error(v)':>12} {'Error(p)':>12}")
    print("-" * 70)
    for i, N in enumerate(N_values):
        print(f"{N:3d} {N+1:6d} {errors_u[i]:12.6e} {errors_v[i]:12.6e} {errors_p[i]:12.6e}")

    # Estimate convergence rates
    print("\n" + "=" * 70)
    print("Convergence Rates (between consecutive refinements)")
    print("=" * 70)
    print(f"{'N1->N2':>10} {'Rate(u)':>10} {'Rate(v)':>10} {'Rate(p)':>10}")
    print("-" * 70)
    for i in range(len(N_values)-1):
        N1, N2 = N_values[i], N_values[i+1]
        rate_u = -np.log(errors_u[i+1] / errors_u[i]) / np.log((N2+1) / (N1+1))
        rate_v = -np.log(errors_v[i+1] / errors_v[i]) / np.log((N2+1) / (N1+1))
        rate_p = -np.log(errors_p[i+1] / errors_p[i]) / np.log((N2+1) / (N1+1))
        print(f"{N1:3d}->{N2:3d} {rate_u:10.2f} {rate_v:10.2f} {rate_p:10.2f}")

    print("\n" + "=" * 70)
    print("Expected: Spectral convergence (exponential decay)")
    print("For smooth solutions, error ~ exp(-cN) faster than any polynomial N^-p")
    print("=" * 70)


if __name__ == "__main__":
    run_convergence_study()

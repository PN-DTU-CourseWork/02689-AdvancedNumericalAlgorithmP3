"""Diagnose spectral solver issues."""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ldc import SpectralSolver

# Create minimal solver
print("="*70)
print("DIAGNOSING SPECTRAL SOLVER")
print("="*70)

solver = SpectralSolver(Re=100.0, Nx=16, Ny=16, CFL=0.1)

print(f"\nConfiguration:")
print(f"  Grid: {solver.config.Nx+1} x {solver.config.Ny+1}")
print(f"  Re: {solver.config.Re}")
print(f"  nu: {1.0/solver.config.Re}")
print(f"  lid_velocity: {solver.config.lid_velocity}")
print(f"  beta^2: {solver.config.beta_squared}")

# Check initial state
print(f"\nInitial state:")
print(f"  u: min={np.min(solver.arrays.u):.6e}, max={np.max(solver.arrays.u):.6e}")
print(f"  v: min={np.min(solver.arrays.v):.6e}, max={np.max(solver.arrays.v):.6e}")
print(f"  p: min={np.min(solver.arrays.p):.6e}, max={np.max(solver.arrays.p):.6e}")

# Take ONE step and check residuals
print(f"\nTaking one RK4 step...")
solver.step()

print(f"\nAfter 1 step:")
print(f"  u: min={np.min(solver.arrays.u):.6e}, max={np.max(solver.arrays.u):.6e}")
print(f"  v: min={np.min(solver.arrays.v):.6e}, max={np.max(solver.arrays.v):.6e}")
print(f"  p: min={np.min(solver.arrays.p):.6e}, max={np.max(solver.arrays.p):.6e}")

# Compute residuals manually
solver._compute_residuals(solver.arrays.u, solver.arrays.v, solver.arrays.p)
print(f"\nResiduals:")
print(f"  R_u: min={np.min(solver.arrays.R_u):.6e}, max={np.max(solver.arrays.R_u):.6e}, norm={np.linalg.norm(solver.arrays.R_u):.6e}")
print(f"  R_v: min={np.min(solver.arrays.R_v):.6e}, max={np.max(solver.arrays.R_v):.6e}, norm={np.linalg.norm(solver.arrays.R_v):.6e}")
print(f"  R_p: min={np.min(solver.arrays.R_p):.6e}, max={np.max(solver.arrays.R_p):.6e}, norm={np.linalg.norm(solver.arrays.R_p):.6e}")

# Check if boundary conditions are applied
Nx, Ny = solver.config.Nx, solver.config.Ny
u_2d = solver.arrays.u.reshape((Nx+1, Ny+1))
v_2d = solver.arrays.v.reshape((Nx+1, Ny+1))

print(f"\nBoundary conditions after step:")
print(f"  Top (lid): u_mean={np.mean(u_2d[:, -1]):.6f}, u_max={np.max(u_2d[:, -1]):.6f}")
print(f"  Bottom: u_max={np.max(np.abs(u_2d[:, 0])):.6e}, v_max={np.max(np.abs(v_2d[:, 0])):.6e}")
print(f"  Left: u_max={np.max(np.abs(u_2d[0, :])):.6e}, v_max={np.max(np.abs(v_2d[0, :])):.6e}")
print(f"  Right: u_max={np.max(np.abs(u_2d[-1, :])):.6e}, v_max={np.max(np.abs(v_2d[-1, :])):.6e}")

# Run for 100 iterations
print(f"\nRunning for 100 iterations...")
for i in range(99):
    solver.step()
    if i % 20 == 19:
        solver._compute_residuals(solver.arrays.u, solver.arrays.v, solver.arrays.p)
        u_res = np.linalg.norm(solver.arrays.R_u)
        v_res = np.linalg.norm(solver.arrays.R_v)
        p_res = np.linalg.norm(solver.arrays.R_p)
        print(f"  Iter {i+1}: u_res={u_res:.6e}, v_res={v_res:.6e}, p_res={p_res:.6e}")

print(f"\nAfter 100 steps:")
print(f"  u at center: {u_2d[Nx//2, Ny//2]:.6e}")
print(f"  v at center: {v_2d[Nx//2, Ny//2]:.6e}")
print(f"  u_max (should approach 1.0): {np.max(u_2d):.6f}")
print(f"  v_min, v_max: {np.min(v_2d):.6f}, {np.max(v_2d):.6f}")

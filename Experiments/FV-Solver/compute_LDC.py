#!/usr/bin/env python3
"""Compute lid-driven cavity solution using FV solver."""

# %% Imports
from ldc import FVSolver
from utils import get_project_root

# %% Setup paths
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
data_dir.mkdir(parents=True, exist_ok=True)

# %% Configure and solve
solver = FVSolver(Re=100.0, nx=64, ny=64, alpha_uv=0.6, alpha_p=0.2)
solver.solve(tolerance=1e-5, max_iter=500)

# Access results via dataclass attributes
print(f"Converged: {solver.metadata.converged}")
print(f"Iterations: {solver.metadata.iterations}")
print(f"Final residual: {solver.metadata.final_residual:.6e}")

# %% Save results
output_file = data_dir / "LDC_Re100.h5"
solver.save(output_file)

print(f"\nResults saved to: {output_file}")

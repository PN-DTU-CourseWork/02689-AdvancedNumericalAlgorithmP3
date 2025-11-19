"""Compare Spectral vs FV solver accuracy against Ghia benchmark."""

import numpy as np
import h5py
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils import get_project_root

project_root = get_project_root()

# Load Ghia benchmark data
ghia_u = np.loadtxt(project_root / "data/validation/ghia/ghia_Re100_u_centerline.csv",
                     delimiter=",", skiprows=1)
ghia_v = np.loadtxt(project_root / "data/validation/ghia/ghia_Re100_v_centerline.csv",
                     delimiter=",", skiprows=1)

ghia_y_u = ghia_u[:, 0]
ghia_u_vals = ghia_u[:, 1]
ghia_x_v = ghia_v[:, 0]
ghia_v_vals = ghia_v[:, 1]

def compute_errors(h5_path, name):
    """Compute L2 and Linf errors against Ghia benchmark."""
    with h5py.File(h5_path, "r") as f:
        grid_points = f["grid_points"][:]
        u_flat = f["fields/u"][:]
        v_flat = f["fields/v"][:]

        x_coords = grid_points[:, 0]
        y_coords = grid_points[:, 1]

        # Infer grid size
        n_points = len(grid_points)
        N = int(np.sqrt(n_points))

        # Find centerline profiles
        # U along vertical centerline (x ≈ 0.5)
        tol = 1 / (2 * (N - 1))  # Half cell width tolerance
        vertical_mask = np.abs(x_coords - 0.5) < tol
        y_u = y_coords[vertical_mask]
        u_centerline = u_flat[vertical_mask]

        # Sort by y coordinate
        sort_idx = np.argsort(y_u)
        y_u = y_u[sort_idx]
        u_centerline = u_centerline[sort_idx]

        # V along horizontal centerline (y ≈ 0.5)
        horizontal_mask = np.abs(y_coords - 0.5) < tol
        x_v = x_coords[horizontal_mask]
        v_centerline = v_flat[horizontal_mask]

        # Sort by x coordinate
        sort_idx = np.argsort(x_v)
        x_v = x_v[sort_idx]
        v_centerline = v_centerline[sort_idx]

        # Interpolate to Ghia points
        u_interp = np.interp(ghia_y_u, y_u, u_centerline)
        v_interp = np.interp(ghia_x_v, x_v, v_centerline)

        # Compute errors
        u_l2 = np.sqrt(np.mean((u_interp - ghia_u_vals)**2))
        u_linf = np.max(np.abs(u_interp - ghia_u_vals))
        v_l2 = np.sqrt(np.mean((v_interp - ghia_v_vals)**2))
        v_linf = np.max(np.abs(v_interp - ghia_v_vals))

        combined_l2 = np.sqrt((u_l2**2 + v_l2**2) / 2)
        combined_linf = max(u_linf, v_linf)

        print(f"\n{name}")
        print(f"{'='*60}")
        print(f"Grid size: {N} × {N} = {N*N} nodes")
        print(f"\nU-velocity (vertical centerline at x=0.5):")
        print(f"  L2 error:   {u_l2:.6e}")
        print(f"  L∞ error:   {u_linf:.6e}")
        print(f"\nV-velocity (horizontal centerline at y=0.5):")
        print(f"  L2 error:   {v_l2:.6e}")
        print(f"  L∞ error:   {v_linf:.6e}")
        print(f"\nCombined:")
        print(f"  L2 error:   {combined_l2:.6e}")
        print(f"  L∞ error:   {combined_linf:.6e}")

        return {
            'name': name,
            'grid_size': N * N,
            'u_l2': u_l2,
            'u_linf': u_linf,
            'v_l2': v_l2,
            'v_linf': v_linf,
            'combined_l2': combined_l2,
            'combined_linf': combined_linf
        }

# Compare solvers
print("="*60)
print("GHIA BENCHMARK COMPARISON: Spectral vs Finite Volume")
print("="*60)

spectral_errors = compute_errors(
    project_root / "data/Spectral-Solver/LDC_Spectral_Re100.h5",
    "Spectral Solver (PN-PN-2)"
)

fv_errors = compute_errors(
    project_root / "data/FV-Solver/LDC_Re100.h5",
    "Finite Volume Solver"
)

# Summary comparison
print("\n" + "="*60)
print("SUMMARY: Error Ratios (FV / Spectral)")
print("="*60)
print(f"Grid size ratio:     {fv_errors['grid_size'] / spectral_errors['grid_size']:.1f}x")
print(f"                     (FV: {fv_errors['grid_size']}, Spectral: {spectral_errors['grid_size']})")
print(f"\nU-velocity L2:       {fv_errors['u_l2'] / spectral_errors['u_l2']:.2f}x worse")
print(f"U-velocity L∞:       {fv_errors['u_linf'] / spectral_errors['u_linf']:.2f}x worse")
print(f"V-velocity L2:       {fv_errors['v_l2'] / spectral_errors['v_l2']:.2f}x worse")
print(f"V-velocity L∞:       {fv_errors['v_linf'] / spectral_errors['v_linf']:.2f}x worse")
print(f"\nCombined L2 error:   {fv_errors['combined_l2'] / spectral_errors['combined_l2']:.2f}x worse")
print(f"Combined L∞ error:   {fv_errors['combined_linf'] / spectral_errors['combined_linf']:.2f}x worse")

# Accuracy per degree of freedom
spectral_acc = spectral_errors['combined_l2'] * spectral_errors['grid_size']
fv_acc = fv_errors['combined_l2'] * fv_errors['grid_size']

print(f"\nAccuracy per DOF (error × grid_size):")
print(f"  Spectral: {spectral_acc:.6e}")
print(f"  FV:       {fv_acc:.6e}")
print(f"  Spectral is {fv_acc / spectral_acc:.2f}x more efficient")

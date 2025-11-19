"""Test spectral differentiation matrices."""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from spectral.spectral import LegendreLobattoBasis

# Test 1D differentiation
print("=" * 70)
print("Test 1D Differentiation")
print("=" * 70)

N = 8
basis = LegendreLobattoBasis(domain=(0.0, 1.0))
nodes = basis.nodes(N + 1)
D = basis.diff_matrix(nodes)

# Test function: f(x) = sin(2πx)
# Exact derivative: f'(x) = 2π cos(2πx)
f = np.sin(2 * np.pi * nodes)
df_dx_exact = 2 * np.pi * np.cos(2 * np.pi * nodes)
df_dx_numerical = D @ f

error = np.linalg.norm(df_dx_numerical - df_dx_exact) / np.linalg.norm(df_dx_exact)
print(f"N = {N}")
print(f"Test function: f(x) = sin(2πx)")
print(f"Relative L2 error in df/dx: {error:.6e}")
print(f"df/dx at x=0: exact={df_dx_exact[0]:.6f}, numerical={df_dx_numerical[0]:.6f}")
print(f"df/dx at x=0.5: exact={df_dx_exact[N//2]:.6f}, numerical={df_dx_numerical[N//2]:.6f}")
print(f"df/dx at x=1: exact={df_dx_exact[-1]:.6f}, numerical={df_dx_numerical[-1]:.6f}")

# Test 2D differentiation
print("\n" + "=" * 70)
print("Test 2D Differentiation")
print("=" * 70)

# Build 2D operators
x_nodes = basis.nodes(N + 1)
y_nodes = basis.nodes(N + 1)
x, y = np.meshgrid(x_nodes, y_nodes, indexing='ij')

Dx_1d = basis.diff_matrix(x_nodes)
Dy_1d = basis.diff_matrix(y_nodes)

Ix = np.eye(N + 1)
Iy = np.eye(N + 1)
# Try swapped ordering
Dx = np.kron(Dx_1d, Iy)
Dy = np.kron(Ix, Dy_1d)

# Test function: f(x,y) = sin(πx)cos(πy)
# df/dx = π cos(πx)cos(πy)
# df/dy = -π sin(πx)sin(πy)
f_2d = np.sin(np.pi * x) * np.cos(np.pi * y)
df_dx_exact_2d = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
df_dy_exact_2d = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)

f_flat = f_2d.ravel()
df_dx_numerical_2d = (Dx @ f_flat).reshape(x.shape)
df_dy_numerical_2d = (Dy @ f_flat).reshape(x.shape)

error_x = np.linalg.norm(df_dx_numerical_2d - df_dx_exact_2d) / np.linalg.norm(df_dx_exact_2d)
error_y = np.linalg.norm(df_dy_numerical_2d - df_dy_exact_2d) / np.linalg.norm(df_dy_exact_2d)

print(f"Test function: f(x,y) = sin(πx)cos(πy)")
print(f"Relative L2 error in ∂f/∂x: {error_x:.6e}")
print(f"Relative L2 error in ∂f/∂y: {error_y:.6e}")

# Check at a specific point
i, j = N//2, N//2
print(f"\nAt point (x,y) = ({x[i,j]:.3f}, {y[i,j]:.3f}):")
print(f"  ∂f/∂x: exact={df_dx_exact_2d[i,j]:.6f}, numerical={df_dx_numerical_2d[i,j]:.6f}")
print(f"  ∂f/∂y: exact={df_dy_exact_2d[i,j]:.6f}, numerical={df_dy_numerical_2d[i,j]:.6f}")

if error_x < 1e-10 and error_y < 1e-10:
    print("\n✓ Differentiation matrices are working correctly!")
else:
    print("\n✗ ERROR: Differentiation matrices have significant errors!")

"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
import numpy as np


#========================================================
# Core Data Structures
#========================================================

@dataclass
class Mesh:
    """Structured grid geometry."""
    nx: int
    ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    def __post_init__(self):
        """Create mesh geometry."""
        # 1D coordinates
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)

        # Grid spacing
        self.dx = self.Lx / (self.nx - 1) if self.nx > 1 else self.Lx
        self.dy = self.Ly / (self.ny - 1) if self.ny > 1 else self.Ly

        # 2D meshgrid
        X, Y = np.meshgrid(self.x, self.y)
        self.grid_points = np.column_stack([X.flatten(), Y.flatten()])

        # Number of cells
        self.n_cells = self.nx * self.ny


@dataclass
class Fields:
    """Solution fields: velocity and pressure."""
    n_cells: int

    def __post_init__(self):
        """Initialize zero fields."""
        # Solution fields
        self.u = np.zeros(self.n_cells)
        self.v = np.zeros(self.n_cells)
        self.p = np.zeros(self.n_cells)

        # Previous iteration (for under-relaxation/residuals)
        self.u_prev = np.zeros(self.n_cells)
        self.v_prev = np.zeros(self.n_cells)


@dataclass
class TimeSeries:
    """Convergence history."""
    u_residuals: list[float] = field(default_factory=list)
    v_residuals: list[float] = field(default_factory=list)


@dataclass
class Meta:
    """Configuration and results metadata."""
    # Physics parameters (required)
    Re: float

    # Physics parameters (with defaults)
    lid_velocity: float = 1.0

    # Grid parameters (with defaults)
    nx: int = 64
    ny: int = 64
    Lx: float = 1.0
    Ly: float = 1.0

    # Solver settings
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = ""

    # Results (filled after solve)
    iterations: int = 0
    converged: bool = False
    final_residual: float = 0.0


#========================================================
# Finite Volume Extensions
#========================================================

@dataclass
class FVMeta(Meta):
    """FV-specific configuration."""
    convection_scheme: str = "Upwind"
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4


@dataclass
class FVFields(Fields):
    """FV-specific fields with face mass flux."""
    n_faces: int = 0

    def __post_init__(self):
        """Initialize FV fields including mass flux."""
        super().__post_init__()
        if self.n_faces > 0:
            self.mdot = np.zeros(self.n_faces)




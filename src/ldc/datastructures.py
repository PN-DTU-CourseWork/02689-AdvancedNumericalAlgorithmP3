"""Data structures for solver configuration and results.

This module defines the configuration and result data structures
for lid-driven cavity solvers (both FV and spectral).
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np

#========================================================
# Shared Data Classes 
# =======================================================


@dataclass
class Fields:
    """Base spatial solution fields."""
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    y: np.ndarray
    grid_points: np.ndarray


@dataclass
class TimeSeries:
    """Time series data common to all solvers."""
    residual: List[float]
    u_residual: List[float] = None
    v_residual: List[float] = None
    continuity_residual: List[float] = None
    #TODO: Add the quantities stuff from the paper


@dataclass
class Info:
    """Base solver metadata, config and convergence info."""
    # Physics parameters (required)
    Re: float

    # Grid parameters (with defaults)
    nx: int = 64
    ny: int = 64

    # Physics parameters (with defaults)
    lid_velocity: float = 1
    Lx: float = 1
    Ly: float = 1

    # Solver config
    max_iterations: int = 500
    tolerance: float = 1e-4
    method: str = None

    # Convergence info
    iterations: int = None
    converged: bool = False
    final_residual: float = None


#=============================================================
# Finite Volume specific data classes
# ============================================================

@dataclass
class FVinfo(Info):
    """FV-specific metadata with discretization parameters."""
    convection_scheme: str = "Upwind" 
    limiter: str = "MUSCL"
    alpha_uv: float = 0.6
    alpha_p: float = 0.4

@dataclass
class FVFields(Fields):
    """FV-specific fields with mass flux."""
    mdot: np.ndarray = None

#=====================================================
# Spectral Data Classes
#=====================================================

@dataclass
class SpectralInfo(Info):
    """Spectral-specific metadata with discretization parameters."""
    Nx: int = 64
    Ny: int = 64
    differentiation_method: str = "fft"  # 'fft', 'chebyshev', 'matrix'
    time_scheme: str = "rk4"
    dt: float = 0.001
    dealiasing: bool = True
    multigrid: bool = False
    mg_levels: int = 3



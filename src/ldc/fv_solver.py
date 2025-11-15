"""Finite volume solver for lid-driven cavity.

This module implements a collocated finite volume solver using
SIMPLE algorithm for pressure-velocity coupling.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple

from .base_solver import LidDrivenCavitySolver
from .datastructures import FVinfo


class FVSolver(LidDrivenCavitySolver):
    """Finite volume solver for lid-driven cavity problem.

    This solver uses a collocated grid arrangement with Rhie-Chow interpolation
    for pressure-velocity coupling using the SIMPLE algorithm.

    Parameters
    ----------
    config : FVConfig
        Configuration with physics (Re, lid velocity, domain size) and
        FV-specific parameters (nx, ny, convection scheme, etc.).
    """

    # Make config class accessible via solver
    Config = FVinfo

    def __init__(self, **kwargs):
        """Initialize FV solver.

        Parameters
        ----------
        **kwargs
            Configuration parameters passed to FVConfig.
            Can also pass config=FVConfig(...) directly.
        """
        super().__init__(**kwargs)

    def _initialize_fields(self):
        """Initialize velocity and pressure fields for SIMPLE algorithm."""
        from fv.core.simple_iteration import initialize_simple_state

        # Initialize all SIMPLE state variables
        self.simple_state = initialize_simple_state(self.mesh, self.config)

        # Set u, v, p as aliases to the state (for base class residual calculation)
        self.u = self.simple_state['u']
        self.v = self.simple_state['v']
        self.p = self.simple_state['p']

    def step(self):
        """Perform one SIMPLE iteration.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocity and pressure fields
        """
        from fv.core.simple_iteration import simple_step

        # Perform one SIMPLE iteration
        self.simple_state = simple_step(self.mesh, self.config, self.simple_state)

        # Update u, v, p references
        self.u = self.simple_state['u']
        self.v = self.simple_state['v']
        self.p = self.simple_state['p']

        return self.u, self.v, self.p

    def _create_output_dataclasses(self, residual_history, final_iter_count, is_converged):
        """Create FV-specific output dataclasses."""
        from .datastructures import FVFields, TimeSeries, FVinfo

        # Extract residuals
        u_residuals = [r['u'] for r in residual_history]
        v_residuals = [r['v'] for r in residual_history]
        combined_residual = [max(r['u'], r['v']) for r in residual_history]

        fields = FVFields(
            u=self.u,
            v=self.v,
            p=self.p,
            x=self.mesh.cell_centers[:, 0],
            y=self.mesh.cell_centers[:, 1],
            grid_points=self.mesh.cell_centers,
            mdot=self.simple_state['mdot'],
        )

        time_series = TimeSeries(
            residual=combined_residual,
            u_residual=u_residuals,
            v_residual=v_residuals,
            continuity_residual=None,  # Can add this later if needed
        )

        metadata = FVinfo(
            Re=self.config.Re,
            lid_velocity=self.config.lid_velocity,
            Lx=self.config.Lx,
            Ly=self.config.Ly,
            nx=self.config.nx,
            ny=self.config.ny,
            iterations=final_iter_count,
            converged=is_converged,
            final_residual=combined_residual[-1] if combined_residual else float('inf'),
            convection_scheme=self.config.convection_scheme,
            limiter=self.config.limiter,
            alpha_uv=self.config.alpha_uv,
            alpha_p=self.config.alpha_p,
        )

        return fields, time_series, metadata

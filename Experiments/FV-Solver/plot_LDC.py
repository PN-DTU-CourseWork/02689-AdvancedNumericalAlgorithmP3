#!/usr/bin/env python3
"""Plot lid-driven cavity solution."""

# %% Imports
from utils import get_project_root, LDCPlotter, GhiaValidator

# %% Setup paths
project_root = get_project_root()
data_dir = project_root / "data" / "FV-Solver"
fig_dir = project_root / "figures" / "FV-Solver"
fig_dir.mkdir(parents=True, exist_ok=True)

# %% Create plotter (single run)
plotter = LDCPlotter(data_dir / "LDC_Re100.h5")

# %% Generate field plots
plotter.plot_convergence(output_path=fig_dir / "LDC_Re100_convergence.pdf")
plotter.plot_velocity_fields(output_path=fig_dir / "LDC_Re100_velocity.pdf")
plotter.plot_pressure(output_path=fig_dir / "LDC_Re100_pressure.pdf")

# %% Ghia validation (separate class)
validator = GhiaValidator(h5_path=data_dir / "LDC_Re100.h5")
validator.plot_validation(output_path=fig_dir / "LDC_Re100_ghia_validation.pdf")

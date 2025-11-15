Finite Volume Solver Examples
==============================

This directory contains examples demonstrating the finite volume solver for
the lid-driven cavity (LDC) flow problem.

The solver uses the SIMPLE algorithm for pressure-velocity coupling on a
collocated grid. The examples show how to:

- Compute lid-driven cavity flow at various Reynolds numbers
- Visualize velocity and pressure fields
- Validate results against Ghia benchmark data

Example Scripts
---------------

* ``compute_LDC.py``: Run the finite volume solver for lid-driven cavity flow
* ``plot_LDC.py``: Visualize the computed solution and validate against benchmarks

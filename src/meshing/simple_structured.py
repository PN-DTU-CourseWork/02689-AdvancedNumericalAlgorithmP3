"""Simplified structured quad mesh builder for FV solver.

This is a minimal implementation specifically for structured quad meshes,
avoiding the complexity of the general-purpose mesh_loader.
"""

import numpy as np
import gmsh
from numba import njit
from .mesh_data import MeshData2D


@njit
def _build_face_connectivity(cells):
    """Build face connectivity from quad cells.

    Returns owner/neighbor arrays where internal faces have both,
    boundary faces have neighbor=-1.
    """
    n_cells = cells.shape[0]

    # Estimate face count (4 faces per cell, but internal shared)
    face_dict_size = n_cells * 4
    face_keys = np.zeros((face_dict_size, 2), dtype=np.int64)
    face_owners = np.zeros(face_dict_size, dtype=np.int64)
    face_neighbors = np.full(face_dict_size, -1, dtype=np.int64)
    n_faces = 0

    # Build faces from cells
    for cell_id in range(n_cells):
        cell_nodes = cells[cell_id]

        # Four edges of quad (in order)
        edges = [
            (cell_nodes[0], cell_nodes[1]),
            (cell_nodes[1], cell_nodes[2]),
            (cell_nodes[2], cell_nodes[3]),
            (cell_nodes[3], cell_nodes[0])
        ]

        for n0, n1 in edges:
            # Canonical edge ordering (smaller index first)
            if n0 > n1:
                n0, n1 = n1, n0

            # Check if face already exists
            found = False
            for f in range(n_faces):
                if face_keys[f, 0] == n0 and face_keys[f, 1] == n1:
                    # Face exists, this cell is neighbor
                    face_neighbors[f] = cell_id
                    found = True
                    break

            if not found:
                # New face, this cell is owner
                face_keys[n_faces, 0] = n0
                face_keys[n_faces, 1] = n1
                face_owners[n_faces] = cell_id
                n_faces += 1

    return face_keys[:n_faces], face_owners[:n_faces], face_neighbors[:n_faces]


@njit
def _compute_face_geometry(points, face_vertices, owner_cells, neighbor_cells, cell_centers):
    """Compute face centers, areas, and normal vectors."""
    n_faces = face_vertices.shape[0]

    face_centers = np.zeros((n_faces, 2), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)
    vector_S_f = np.zeros((n_faces, 2), dtype=np.float64)

    for f in range(n_faces):
        v0_idx = face_vertices[f, 0]
        v1_idx = face_vertices[f, 1]

        v0 = points[v0_idx]
        v1 = points[v1_idx]

        # Face center
        face_centers[f] = 0.5 * (v0 + v1)

        # Edge vector and length
        edge = v1 - v0
        length = np.sqrt(edge[0]**2 + edge[1]**2)
        face_areas[f] = length

        # Normal vector (rotate edge by 90 degrees)
        # Convention: normal points from owner to neighbor
        normal = np.array([edge[1], -edge[0]])

        # Check orientation: normal should point outward from owner
        owner = owner_cells[f]
        neighbor = neighbor_cells[f]

        if neighbor >= 0:
            # Internal face: normal points from owner to neighbor
            d = cell_centers[neighbor] - cell_centers[owner]
            if normal[0] * d[0] + normal[1] * d[1] < 0:
                normal = -normal
        else:
            # Boundary face: normal points outward from owner
            d = face_centers[f] - cell_centers[owner]
            if normal[0] * d[0] + normal[1] * d[1] < 0:
                normal = -normal

        vector_S_f[f] = normal

    return face_centers, face_areas, vector_S_f


@njit
def _compute_geometric_factors(n_faces, owner_cells, neighbor_cells,
                                 cell_centers, face_centers, vector_S_f, face_areas):
    """Compute geometric factors for FV discretization."""
    vector_d_CE = np.zeros((n_faces, 2), dtype=np.float64)
    unit_vector_n = np.zeros((n_faces, 2), dtype=np.float64)
    unit_vector_e = np.zeros((n_faces, 2), dtype=np.float64)
    face_interp_factors = np.zeros(n_faces, dtype=np.float64)
    d_Cb = np.zeros(n_faces, dtype=np.float64)

    for f in range(n_faces):
        owner = owner_cells[f]
        neighbor = neighbor_cells[f]

        # Unit normal
        if face_areas[f] > 1e-12:
            unit_vector_n[f] = vector_S_f[f] / face_areas[f]

        if neighbor >= 0:
            # Internal face
            vector_d_CE[f] = cell_centers[neighbor] - cell_centers[owner]
            d_mag = np.sqrt(vector_d_CE[f, 0]**2 + vector_d_CE[f, 1]**2)

            if d_mag > 1e-12:
                unit_vector_e[f] = vector_d_CE[f] / d_mag

                # Distance from owner to face
                d_Pf = face_centers[f] - cell_centers[owner]
                delta_Pf = np.sqrt(d_Pf[0]**2 + d_Pf[1]**2)

                # Interpolation factor
                face_interp_factors[f] = delta_Pf / d_mag
        else:
            # Boundary face
            d_boundary = face_centers[f] - cell_centers[owner]
            d_Cb[f] = np.sqrt(d_boundary[0]**2 + d_boundary[1]**2)
            vector_d_CE[f] = d_boundary

            if d_Cb[f] > 1e-12:
                unit_vector_e[f] = d_boundary / d_Cb[f]

    return vector_d_CE, unit_vector_n, unit_vector_e, face_interp_factors, d_Cb


def create_structured_mesh_2d(nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0,
                                lid_velocity: float = 1.0) -> MeshData2D:
    """Create structured quad mesh using gmsh, simplified for structured grids only.

    This is a minimal implementation that:
    - Uses gmsh to generate geometry and quad mesh
    - Builds FV connectivity with simple, focused algorithms
    - Hard-codes lid-driven cavity boundary conditions
    - Skips non-orthogonality corrections (assumes structured orthogonal grid)
    """
    # 1. Generate mesh with gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.clear()
    gmsh.model.add("cavity")
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress gmsh output

    # Geometry
    lc = Lx / max(nx, ny)
    p = [gmsh.model.geo.addPoint(x, y, 0.0, lc)
         for x, y in [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]]

    lines = [gmsh.model.geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    loop = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # Structured mesh
    for line, n in zip([lines[0], lines[2], lines[1], lines[3]],
                        [nx + 1, nx + 1, ny + 1, ny + 1]):
        gmsh.model.geo.mesh.setTransfiniteCurve(line, n)
    gmsh.model.geo.mesh.setTransfiniteSurface(surf)
    gmsh.model.geo.mesh.setRecombine(2, surf)

    # Physical groups
    for i, name in enumerate(["bottom", "right", "top", "left"]):
        gmsh.model.addPhysicalGroup(1, [lines[i]], i + 1)
        gmsh.model.setPhysicalName(1, i + 1, name)
    gmsh.model.addPhysicalGroup(2, [surf], 10)
    gmsh.model.setPhysicalName(2, 10, "fluid")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # 2. Extract mesh data
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)[:, :2]
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}

    # Get quad cells
    _, _, elem_nodes = gmsh.model.mesh.getElements(2)
    cells = np.array([[node_map[int(n)] for n in elem_nodes[0][i:i+4]]
                      for i in range(0, len(elem_nodes[0]), 4)], dtype=np.int64)

    # Get boundary edges
    boundary_lines_dict = {}
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        name = gmsh.model.getPhysicalName(dim, tag)
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

        for entity in entities:
            _, _, edge_nodes = gmsh.model.mesh.getElements(dim, entity)
            if len(edge_nodes) > 0:
                edges = edge_nodes[0].reshape(-1, 2)
                for edge in edges:
                    e = tuple(sorted([node_map[int(edge[0])], node_map[int(edge[1])]]))
                    boundary_lines_dict[e] = (tag, name)

    gmsh.finalize()

    # 3. Build FV connectivity
    n_cells = len(cells)
    cell_centers = np.mean(points[cells], axis=1)
    cell_volumes = np.zeros(n_cells)

    # Cell volumes (quad area = |cross product of diagonals| / 2)
    for i, cell in enumerate(cells):
        p0, p1, p2, p3 = points[cell]
        # Shoelace formula for quad area
        area = 0.5 * abs((p0[0] - p2[0]) * (p1[1] - p3[1]) -
                         (p1[0] - p3[0]) * (p0[1] - p2[1]))
        cell_volumes[i] = area

    # Build face connectivity
    face_vertices, owner_cells, neighbor_cells = _build_face_connectivity(cells)
    n_faces = len(face_vertices)

    # Face geometry
    face_centers, face_areas, vector_S_f = _compute_face_geometry(
        points, face_vertices, owner_cells, neighbor_cells, cell_centers
    )

    # Geometric factors
    vector_d_CE, unit_vector_n, unit_vector_e, face_interp_factors, d_Cb = \
        _compute_geometric_factors(n_faces, owner_cells, neighbor_cells,
                                     cell_centers, face_centers, vector_S_f, face_areas)

    # 4. Boundary conditions
    internal_faces = np.where(neighbor_cells >= 0)[0].astype(np.int64)
    boundary_faces_list = []
    boundary_patches = np.full(n_faces, -1, dtype=np.int64)
    boundary_types = np.full((n_faces, 2), -1, dtype=np.int64)
    boundary_values = np.zeros((n_faces, 3), dtype=np.float64)

    # Map boundary edges to faces
    for f in range(n_faces):
        if neighbor_cells[f] >= 0:
            continue  # Internal face

        edge = tuple(sorted(face_vertices[f]))
        if edge in boundary_lines_dict:
            patch_tag, patch_name = boundary_lines_dict[edge]
            boundary_faces_list.append(f)
            boundary_patches[f] = patch_tag

            # Lid-driven cavity BCs
            if patch_name == "top":
                boundary_types[f] = [0, 3]  # wall velocity, zeroGradient pressure
                boundary_values[f] = [lid_velocity, 0.0, 0.0]
            else:
                boundary_types[f] = [0, 3]  # wall velocity, zeroGradient pressure
                boundary_values[f] = [0.0, 0.0, 0.0]

    boundary_faces = np.array(boundary_faces_list, dtype=np.int64)

    # 5. Simplified structured mesh: skip non-orthogonality terms
    vector_E_f = vector_S_f.copy()  # E_f = S_f for orthogonal mesh
    vector_T_f = np.zeros_like(vector_S_f)  # T_f = 0 for orthogonal mesh
    vector_skewness = np.zeros_like(face_centers)  # No skewness

    # Rhie-Chow weights
    rc_interp_weights = np.zeros(n_faces)
    for f in internal_faces:
        g_f = face_interp_factors[f]
        delta_PN = np.linalg.norm(vector_d_CE[f])
        if delta_PN > 1e-12 and g_f > 1e-12 and (1 - g_f) > 1e-12:
            rc_interp_weights[f] = 1.0 / (g_f * (1 - g_f) * delta_PN)

    # Cell-face connectivity
    max_faces = 4  # Quads have 4 faces
    cell_faces = np.full((n_cells, max_faces), -1, dtype=np.int64)
    face_count = np.zeros(n_cells, dtype=np.int32)

    for f in range(n_faces):
        owner = owner_cells[f]
        cell_faces[owner, face_count[owner]] = f
        face_count[owner] += 1

        neighbor = neighbor_cells[f]
        if neighbor >= 0:
            cell_faces[neighbor, face_count[neighbor]] = f
            face_count[neighbor] += 1

    # Masks
    face_boundary_mask = np.zeros(n_faces, dtype=np.int64)
    face_boundary_mask[boundary_faces] = 1
    face_flux_mask = np.ones(n_faces, dtype=np.int64)

    # Build MeshData2D
    return MeshData2D(
        cell_volumes=np.ascontiguousarray(cell_volumes),
        cell_centers=np.ascontiguousarray(cell_centers),
        face_areas=np.ascontiguousarray(face_areas),
        face_centers=np.ascontiguousarray(face_centers),
        owner_cells=np.ascontiguousarray(owner_cells),
        neighbor_cells=np.ascontiguousarray(neighbor_cells),
        cell_faces=np.ascontiguousarray(cell_faces),
        face_vertices=np.ascontiguousarray(face_vertices),
        vertices=np.ascontiguousarray(points),
        vector_S_f=np.ascontiguousarray(vector_S_f),
        vector_d_CE=np.ascontiguousarray(vector_d_CE),
        unit_vector_n=np.ascontiguousarray(unit_vector_n),
        unit_vector_e=np.ascontiguousarray(unit_vector_e),
        vector_E_f=np.ascontiguousarray(vector_E_f),
        vector_T_f=np.ascontiguousarray(vector_T_f),
        vector_skewness=np.ascontiguousarray(vector_skewness),
        face_interp_factors=np.ascontiguousarray(face_interp_factors),
        rc_interp_weights=np.ascontiguousarray(rc_interp_weights),
        internal_faces=np.ascontiguousarray(internal_faces),
        boundary_faces=np.ascontiguousarray(boundary_faces),
        boundary_patches=np.ascontiguousarray(boundary_patches),
        boundary_types=np.ascontiguousarray(boundary_types),
        boundary_values=np.ascontiguousarray(boundary_values),
        d_Cb=np.ascontiguousarray(d_Cb),
        face_boundary_mask=np.ascontiguousarray(face_boundary_mask),
        face_flux_mask=np.ascontiguousarray(face_flux_mask),
    )

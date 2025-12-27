#!/usr/bin/env python3
"""
Specialized solver for: poisson_simple
PDE Type: Poisson
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np

# dolfinx imports
import dolfinx
from dolfinx import fem, mesh, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

def main():
    # ===== CLI Interface =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--degree', type=int, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    comm = MPI.COMM_WORLD
    
    # ===== Problem-Specific Parameters =====
    kappa = 1.0
    nx_output, ny_output = 50, 50
    
    # ===== Create Mesh =====
    domain = mesh.create_unit_square(
        comm,
        args.resolution,
        args.resolution,
        mesh.CellType.triangle
    )
    
    # ===== Function Space =====
    V = fem.functionspace(domain, ("Lagrange", args.degree))
    
    # ===== Define Boundary Conditions =====
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_D = fem.Constant(domain, PETSc.ScalarType(0))
    bc = fem.dirichletbc(u_D, boundary_dofs, V)
    
    # ===== Define Variational Problem =====
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = 2 * np.pi**2 * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # ===== Assemble and Solve =====
    problem = petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "amg"})
    
    start_time = time.time()
    uh = problem.solve()
    solve_time = time.time() - start_time
    
    solver = problem.solver
    num_iters = solver.getIterationNumber()

    # ===== Interpolate to Output Grid (Robust Parallel Method) =====
    x_out = np.linspace(0, 1, nx_output)
    y_out = np.linspace(0, 1, ny_output)
    xx, yy = np.meshgrid(x_out, y_out)
    points = np.vstack((xx.ravel(), yy.ravel(), np.zeros(xx.size))).T
    
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_spec(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_grid_local = np.zeros(points.shape[0], dtype=PETSc.ScalarType)
    points_on_proc = []
    cells_on_proc = []
    point_indices_on_proc = []

    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells_on_proc.append(colliding_cells.links(i)[0])
            point_indices_on_proc.append(i)

    if points_on_proc:
        u_on_proc = uh.eval(np.array(points_on_proc), np.array(cells_on_proc))
        u_grid_local[point_indices_on_proc] = u_on_proc.flatten()
        
    u_grid_global = np.zeros(points.shape[0], dtype=PETSc.ScalarType)
    comm.Allreduce(u_grid_local, u_grid_global, op=MPI.SUM)
    
    u_grid = u_grid_global.reshape((ny_output, nx_output))

    # ===== Save Outputs =====
    if comm.rank == 0:
        np.savez(outdir / 'solution.npz', x=x_out, y=y_out, u=u_grid)
        
        meta = {
            'wall_time_sec': solve_time,
            'solver_info': {
                'ksp_type': 'cg',
                'pc_type': 'hypre',
                'iterations': num_iters
            }
        }
        with open(outdir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Solved in {solve_time:.3f}s, {num_iters} iterations")

if __name__ == '__main__':
    main()

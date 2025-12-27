#!/usr/bin/env python3
"""
Specialized solver for: heat_simple
PDE Type: Heat Equation (Time-dependent)
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
    parser.add_argument('--dt', type=float, required=False, default=0.01)
    
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    comm = MPI.COMM_WORLD
    
    # ===== Problem-Specific Parameters =====
    kappa = 1.0
    T_final = 0.1
    dt_val = args.dt
    nx_output, ny_output = 40, 40
    
    # ===== Create Mesh and Function Space =====
    domain = mesh.create_unit_square(comm, args.resolution, args.resolution, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", args.degree))
    
    # ===== Define Boundary Conditions =====
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0), boundary_dofs, V)
    
    # ===== Define Variational Problem =====
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0))
    f = (2 * np.pi**2 - 1) * ufl.exp(-t) * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))

    a = ufl.inner(u, v) * ufl.dx + dt * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n + dt * f, v) * ufl.dx
    
    # ===== Set up Solver (Manual KSP) =====
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = fem.Function(V)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")

    # ===== Initial Condition =====
    u_n.interpolate(lambda p: np.sin(np.pi * p[0]) * np.sin(np.pi * p[1]))
    
    # ===== Time Stepping Loop =====
    num_steps = int(round(T_final / dt_val))
    total_iters = 0
    uh = fem.Function(V)

    start_time = time.time()
    
    for i in range(num_steps):
        t.value = (i + 1) * dt_val
        
        with b.x.petsc_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b.x.petsc_vec, L_form)
        
        petsc.apply_lifting(b.x.petsc_vec, [a_form], [[bc]])
        b.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b.x.petsc_vec, [bc])
        
        solver.solve(b.x.petsc_vec, uh.x.petsc_vec)
        uh.x.scatter_forward()
        
        u_n.x.array[:] = uh.x.array
        
        total_iters += solver.getIterationNumber()

    solve_time = time.time() - start_time
    
    # ===== Interpolate to Output Grid (Corrected) =====
    x_out = np.linspace(0, 1, nx_output)
    y_out = np.linspace(0, 1, ny_output)
    xx, yy = np.meshgrid(x_out, y_out)
    points = np.vstack((xx.ravel(), yy.ravel(), np.zeros(xx.size))).T
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_grid_local = np.zeros(points.shape[0], dtype=PETSc.ScalarType)
    points_on_proc_idx = []
    points_on_proc = []
    cells_on_proc = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc_idx.append(i)
            points_on_proc.append(point)
            cells_on_proc.append(colliding_cells.links(i)[0])
    
    if points_on_proc:
        u_on_proc = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc))
        u_grid_local[points_on_proc_idx] = u_on_proc.flatten()

    u_grid_global = np.zeros_like(u_grid_local)
    comm.Allreduce(u_grid_local, u_grid_global, op=MPI.SUM)
    u_grid = u_grid_global.reshape((ny_output, nx_output))

    # ===== Save Outputs (Corrected) =====
    if comm.rank == 0:
        np.savez(outdir / 'solution.npz', x=x_out, y=y_out, u=u_grid, t_final=T_final)
        
        meta = {
            'wall_time_sec': solve_time,
            'solver_info': {
                'ksp_type': 'cg',
                'pc_type': 'hypre',
                'iterations': total_iters
            }
        }
        with open(outdir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"✅ Solved in {solve_time:.3f}s, total iterations: {total_iters}")

if __name__ == '__main__':
    main()

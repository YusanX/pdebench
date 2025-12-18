# FEniCSx (dolfinx v0.10.0) Quick Reference Guide

This guide provides the correct syntax for `dolfinx v0.10.0`. Use this reference to avoid common API errors.

## 1. Imports

```python
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
```

## 2. Mesh Generation

```python
# Create a unit square mesh with triangular elements
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, 
    nx=32, ny=32, 
    cell_type=mesh.CellType.triangle
)
```

## 3. Function Space

**Note:** Use `fem.functionspace`, NOT `FunctionSpace`.

```python
# Lagrange element, degree 1 (P1)
V = fem.functionspace(domain, ("Lagrange", 1))

# Vector element (e.g., for velocity)
V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
```

## 4. Boundary Conditions (Dirichlet)

**Step 1: Locate Boundary Dofs**

```python
def boundary_check(x):
    # Example: all boundaries (x[0] approx 0 or 1, x[1] approx 0 or 1)
    return np.full(x.shape[1], True)

# Locate facets on the boundary
tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_check)

# Find DOFs associated with these facets
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
```

**Step 2: Create BC Object**

```python
# Constant value BC
u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0
bc = fem.dirichletbc(u_bc, boundary_dofs)

# OR precise value using interpolation
u_exact = lambda x: np.sin(np.pi * x[0])
u_bc.interpolate(u_exact)
bc = fem.dirichletbc(u_bc, boundary_dofs)
```

## 5. Variational Problem (Weak Form)

Use `ufl` for symbolic representation.

```python
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Poisson: -div(grad(u)) = f
f = fem.Constant(domain, PETSc.ScalarType(1.0))
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
```

## 6. Linear Solver

```python
# Set up Linear Problem
problem = petsc.LinearProblem(
    a, L, bcs=[bc], 
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
u_sol = problem.solve()
```

## 7. 🚨 Point Evaluation / Interpolation (CRITICAL)

**Do NOT use `BoundingBoxTree(mesh, dim)` constructor directly.**

**Correct Pattern for v0.8+ / v0.10.0:**

```python
def evaluate_on_grid(u_fem, nx, ny):
    # 1. Create target grid points (z=0 for 2D)
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    points = np.zeros((3, X.size))
    points[0] = X.ravel()
    points[1] = Y.ravel()
    
    # 2. Build BoundingBoxTree (Use factory function!)
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # 3. Compute collisions
    # cell_candidates: AdjacencyList containing candidate cells for each point
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    
    # 4. Resolve exact collisions
    # colliding_cells: AdjacencyList containing actual cell for each point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # 5. Evaluate function
    # u_values shape: (num_points, value_size)
    u_values = u_fem.eval(points.T, colliding_cells.array)
    
    return X, Y, u_values.reshape((ny, nx))
```


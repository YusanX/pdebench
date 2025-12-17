"""Baseline Krylov linear solver."""
import time
from petsc4py import PETSc

def solve_linear(A, b, ksp_params=None, problem_meta=None):
    """
    Solve Ax=b using Krylov method (baseline solver).
    
    Args:
        A: PETSc Mat
        b: PETSc Vec
        ksp_params: dict with keys 'type', 'pc_type', 'rtol', 'atol', 'max_it' (optional override)
        problem_meta: dict with problem metadata (NEW in V2):
            - 'pde_type': str, e.g., 'poisson', 'heat', 'convection_diffusion'
            - 'num_dofs': int, number of degrees of freedom
            - 'is_symmetric': bool, whether the matrix is symmetric
            - 'is_time_dependent': bool, whether it's a time-dependent problem
            - Additional PDE-specific metadata:
                * For convection_diffusion: 'epsilon', 'beta', 'peclet_number'
                * For heat: 'time_step', 'total_steps'
    
    Returns:
        (x, info): solution vector and solver info dict
            info contains: ksp_type, pc_type, rtol, iters, converged, wall_time_sec
    """
    if ksp_params is None:
        ksp_params = {}
    
    if problem_meta is None:
        problem_meta = {}

    # ============================================================
    # 🚀 AGENT OPTIMIZATION ZONE 🚀
    # ============================================================
    pde_type = problem_meta.get('pde_type')
    num_dofs = problem_meta.get('num_dofs', 0)
    is_symmetric = problem_meta.get('is_symmetric', True)
    peclet = problem_meta.get('peclet_number', 0)
    
    pc_factor_levels = 1 # Default ILU/ICC fill level

    # Rule for poisson_p2 (large DoFs, ICC is unstable, SOR is robust)
    if pde_type == 'poisson' and num_dofs > 150000:
        ksp_type = 'cg'
        pc_type = 'sor'
    # Rule for convdiff_hard (high Peclet, needs stronger ILU)
    elif pde_type == 'convection_diffusion' and peclet > 500:
        ksp_type = 'gmres'
        pc_type = 'ilu'
        pc_factor_levels = 2 # Increase fill-in for robustness
    # General rule for symmetric problems
    elif is_symmetric:
        ksp_type = 'cg'
        pc_type = 'icc'
    # General rule for non-symmetric problems
    else:
        ksp_type = 'gmres'
        pc_type = 'ilu'

    rtol = ksp_params.get('rtol', 1e-10)
    atol = ksp_params.get('atol', 1e-12)
    max_it = ksp_params.get('max_it', 10000)
    
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    
    pc = ksp.getPC()
    pc.setType(pc_type)
    if pc_type in ['ilu', 'icc']:
        pc.setFactorLevels(pc_factor_levels)

    x = b.duplicate()
    
    t0 = time.time()
    ksp.solve(b, x)
    wall_time = time.time() - t0
    
    reason = ksp.getConvergedReason()
    iters = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()
    
    info = {
        'ksp_type': ksp_type,
        'pc_type': pc_type,
        'rtol': rtol,
        'atol': atol,
        'iters': iters,
        'converged': reason > 0,
        'converged_reason': reason,
        'residual_norm': float(rnorm),
        'wall_time_sec': wall_time,
    }
    
    return x, info


def solve_linear_direct(A, b):
    """
    Solve Ax=b using direct LU (for reference solution).
    
    Args:
        A: PETSc Mat
        b: PETSc Vec
    
    Returns:
        (x, info): solution vector and solver info dict
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType('preonly')
    
    pc = ksp.getPC()
    pc.setType('lu')
    
    ksp.setFromOptions()
    
    x = b.duplicate()
    
    t0 = time.time()
    ksp.solve(b, x)
    wall_time = time.time() - t0
    
    reason = ksp.getConvergedReason()
    
    info = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'rtol': 0.0,
        'atol': 0.0,
        'iters': 0,
        'converged': reason > 0,
        'converged_reason': reason,
        'wall_time_sec': wall_time,
    }
    
    return x, info

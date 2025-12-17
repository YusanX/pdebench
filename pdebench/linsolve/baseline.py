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
    # This is where Code Agents should implement intelligent solver selection.
    # The baseline below is intentionally WEAK (CG + Jacobi), providing a
    # low bar for agents to beat.
    #
    # Available metadata in problem_meta:
    #   - pde_type: 'poisson' | 'heat' | 'convection_diffusion'
    #   - num_dofs: problem size (e.g., 40000, 160000)
    #   - is_symmetric: True/False
    #   - is_time_dependent: True/False
    #   - peclet_number: (for convection_diffusion only, indicates difficulty)
    #
    # 📚 STRATEGY EXAMPLES FOR AGENTS:
    #
    # Example 1: Adaptive solver based on symmetry
    #   if problem_meta.get('is_symmetric', True):
    #       ksp_type = 'cg'      # Conjugate Gradient for SPD matrices
    #       pc_type = 'icc'      # Incomplete Cholesky
    #   else:
    #       ksp_type = 'gmres'   # GMRES for non-symmetric
    #       pc_type = 'ilu'      # Incomplete LU
    #
    # Example 2: Scale-aware preconditioning
    #   num_dofs = problem_meta.get('num_dofs', 0)
    #   if num_dofs > 100000:
    #       pc_type = 'gamg'     # Algebraic Multigrid for large problems
    #   elif num_dofs > 10000:
    #       pc_type = 'ilu'      # ILU for medium problems
    #   else:
    #       pc_type = 'jacobi'   # Jacobi for small problems
    #
    # Example 3: PDE-specific tuning
    #   pde_type = problem_meta.get('pde_type')
    #   if pde_type == 'convection_diffusion':
    #       peclet = problem_meta.get('peclet_number', 0)
    #       if peclet > 100:  # Convection-dominated
    #           ksp_type = 'gmres'
    #           pc_type = 'ilu'
    #           rtol = 1e-8  # Relax tolerance for stability
    #   elif pde_type == 'heat':
    #       # Heat equation benefits from ICC
    #       ksp_type = 'cg'
    #       pc_type = 'icc'
    #
    # 🎯 CURRENT BASELINE (INTENTIONALLY WEAK):
    # ============================================================
    
    ksp_type = 'cg'
    pc_type = 'jacobi'  # Simplest preconditioner, ~1000 iterations expected
    
    rtol = ksp_params.get('rtol', 1e-10)
    atol = ksp_params.get('atol', 1e-12)
    max_it = ksp_params.get('max_it', 10000)
    
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    
    pc = ksp.getPC()
    pc.setType(pc_type)
    
    # ksp.setFromOptions() # Commented out to ensure our settings are applied.
    
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

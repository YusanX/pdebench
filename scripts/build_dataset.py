#!/usr/bin/env python3
"""Build benchmark dataset from existing Oracle case configurations.

This script converts Oracle case JSON files into agent-facing JSONL dataset entries.
It uses templates to generate natural language prompts from case specifications.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdebench.datasets.schema import DatasetEntry, save_dataset, LEVELS


# Prompt templates for different PDE types
POISSON_TEMPLATE = """Solve the Poisson equation on a unit square domain [0,1]×[0,1]:

  -∇·(κ ∇u) = f   in Ω
  u = g           on ∂Ω

**Problem Parameters:**
{parameters}

**Boundary Conditions:**
{boundary_conditions}

**Requirements:**
Your implementation must:
1. Use `dolfinx` (FEniCSx) for finite element assembly and solving
2. Accept command-line arguments: `--resolution N` (mesh resolution) and `--degree P` (polynomial degree)
3. Save the solution to `solution.npz` with fields: `x` (1D array), `y` (1D array), `u` (2D array, shape [ny, nx])
4. Save solver metadata to `meta.json` with fields: `wall_time_sec`, `solver_info` (dict with `ksp_type`, `pc_type`, `iters`)

**Output Grid:**
Sample the solution on a uniform {nx}×{ny} grid spanning the domain.
"""

HEAT_TEMPLATE = """Solve the transient Heat equation on a unit square domain [0,1]×[0,1]:

  ∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, T]
  u = g                    on ∂Ω × (0, T]
  u(x, y, 0) = u₀(x, y)    in Ω

**Problem Parameters:**
{parameters}

**Time Discretization:**
- Final time T = {t_end}
- Time step Δt = {dt}
- Use backward Euler scheme

**Boundary Conditions:**
{boundary_conditions}

**Requirements:**
Your implementation must:
1. Use `dolfinx` (FEniCSx) for finite element assembly and solving
2. Accept command-line arguments: `--resolution N` and `--degree P`
3. Save the final solution (at t=T) to `solution.npz` with fields: `x`, `y`, `u`, `t_final`
4. Save solver metadata to `meta.json` with total iteration count and wall time

**Output Grid:**
Sample the final solution on a uniform {nx}×{ny} grid.
"""

CONVDIFF_TEMPLATE = """Solve the steady-state Convection-Diffusion equation on a unit square domain [0,1]×[0,1]:

  -ε ∇²u + β·∇u = f   in Ω
  u = g                on ∂Ω

**Problem Parameters:**
{parameters}

**Physical Context:**
This is a {regime} problem (Péclet number Pe ≈ {peclet:.1f}).
{stability_hint}

**Boundary Conditions:**
{boundary_conditions}

**Requirements:**
Your implementation must:
1. Use `dolfinx` (FEniCSx) for finite element assembly and solving
2. Accept command-line arguments: `--resolution N` and `--degree P`
3. **Handle numerical stability appropriately** (consider SUPG/streamline stabilization for high Péclet)
4. Save solution to `solution.npz` with fields: `x`, `y`, `u`
5. Save metadata to `meta.json`

**Output Grid:**
Sample the solution on a uniform {nx}×{ny} grid.
"""


def extract_parameters_description(case_spec: Dict[str, Any]) -> str:
    """Generate parameter description from case spec."""
    pde = case_spec['pde']
    pde_type = pde['type']
    
    lines = []
    
    # Diffusion coefficient
    kappa = pde.get('coefficients', {}).get('kappa', {})
    if kappa.get('type') == 'constant':
        lines.append(f"- Diffusion coefficient: κ = {kappa['value']}")
    elif kappa.get('type') == 'piecewise_x':
        lines.append(f"- Diffusion coefficient: κ = {kappa['left']} (x < {kappa['x_split']}), κ = {kappa['right']} (x ≥ {kappa['x_split']})")
    
    # Source term
    if 'manufactured_solution' in pde:
        lines.append(f"- Manufactured solution: u = {pde['manufactured_solution']['u']}")
        lines.append("- Source term f and boundary data g are derived from the manufactured solution")
    elif 'source_term' in pde:
        lines.append(f"- Source term: f = {pde['source_term'].get('f', '0')}")
    
    # Convection-diffusion specific
    if pde_type == 'convection_diffusion' and 'pde_params' in pde:
        params = pde['pde_params']
        lines.append(f"- Diffusion coefficient: ε = {params.get('epsilon', 0.01)}")
        beta = params.get('beta', [1.0, 1.0])
        lines.append(f"- Convection velocity: β = ({beta[0]}, {beta[1]})")
    
    return '\n'.join(lines)


def extract_bc_description(case_spec: Dict[str, Any]) -> str:
    """Generate boundary condition description."""
    bc = case_spec.get('bc', {})
    dirichlet = bc.get('dirichlet', {})
    
    bc_on = dirichlet.get('on', 'all')
    bc_value = dirichlet.get('value', 'u')
    
    if bc_value == 'u':
        return f"- Dirichlet BC on {bc_on} boundaries: u = u_exact (from manufactured solution)"
    else:
        return f"- Dirichlet BC on {bc_on} boundaries: u = {bc_value}"


def determine_level(case_spec: Dict[str, Any]) -> str:
    """Determine difficulty level based on case characteristics."""
    pde_type = case_spec['pde']['type']
    
    if pde_type in ['poisson', 'heat']:
        return "2.1"
    elif pde_type == 'convection_diffusion':
        # Check Péclet number
        pde_params = case_spec['pde'].get('pde_params', {})
        epsilon = pde_params.get('epsilon', 0.01)
        beta = pde_params.get('beta', [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5
        peclet = beta_norm / epsilon if epsilon > 0 else float('inf')
        
        # High Péclet requires stabilization
        return "2.2"
    else:
        return "2.3"


def generate_prompt(case_spec: Dict[str, Any]) -> str:
    """Generate natural language prompt from case specification."""
    pde_type = case_spec['pde']['type']
    
    # Common data
    grid_spec = case_spec['output']['grid']
    nx, ny = grid_spec['nx'], grid_spec['ny']
    
    params_desc = extract_parameters_description(case_spec)
    bc_desc = extract_bc_description(case_spec)
    
    if pde_type == 'poisson':
        return POISSON_TEMPLATE.format(
            parameters=params_desc,
            boundary_conditions=bc_desc,
            nx=nx,
            ny=ny
        )
    
    elif pde_type == 'heat':
        time_spec = case_spec['pde']['time']
        return HEAT_TEMPLATE.format(
            parameters=params_desc,
            boundary_conditions=bc_desc,
            t_end=time_spec['t_end'],
            dt=time_spec['dt'],
            nx=nx,
            ny=ny
        )
    
    elif pde_type == 'convection_diffusion':
        pde_params = case_spec['pde'].get('pde_params', {})
        epsilon = pde_params.get('epsilon', 0.01)
        beta = pde_params.get('beta', [1.0, 1.0])
        beta_norm = (beta[0]**2 + beta[1]**2)**0.5
        peclet = beta_norm / epsilon if epsilon > 0 else float('inf')
        
        if peclet > 10:
            regime = "convection-dominated"
            stability_hint = "⚠️ Standard Galerkin may produce oscillations. Consider stabilization techniques."
        elif peclet < 1:
            regime = "diffusion-dominated"
            stability_hint = "Standard Galerkin should work well."
        else:
            regime = "balanced convection-diffusion"
            stability_hint = "Standard Galerkin should be adequate."
        
        return CONVDIFF_TEMPLATE.format(
            parameters=params_desc,
            boundary_conditions=bc_desc,
            regime=regime,
            peclet=peclet,
            stability_hint=stability_hint,
            nx=nx,
            ny=ny
        )
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def generate_requirements(case_spec: Dict[str, Any]) -> List[str]:
    """Generate requirement list."""
    requirements = [
        "Use dolfinx (FEniCSx) for FEM implementation",
        "Accept CLI arguments: --resolution N, --degree P",
        "Save solution to solution.npz with fields: x, y, u",
        "Save metadata to meta.json with wall_time_sec and solver_info",
    ]
    
    pde_type = case_spec['pde']['type']
    if pde_type == 'convection_diffusion':
        requirements.append("Handle numerical stability for convection-dominated flows")
    
    return requirements


def generate_evaluation_config(case_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate evaluation configuration."""
    targets = case_spec.get('targets', {})
    
    return {
        'target_metric': targets.get('metric', 'rel_L2_grid'),
        'target_error': targets.get('target_error', 0.01),
        'timeout_sec': 300,  # 5 minutes
        'memory_limit_mb': 4096,  # 4 GB
    }


def convert_case_to_dataset_entry(case_file: Path) -> DatasetEntry:
    """Convert a single case JSON to a dataset entry."""
    with open(case_file, 'r') as f:
        case_spec = json.load(f)
    
    case_id = case_spec['id']
    level = determine_level(case_spec)
    prompt = generate_prompt(case_spec)
    requirements = generate_requirements(case_spec)
    evaluation_config = generate_evaluation_config(case_spec)
    
    return DatasetEntry(
        id=case_id,
        level=level,
        prompt=prompt,
        requirements=requirements,
        oracle_config=case_spec,
        evaluation_config=evaluation_config
    )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build benchmark dataset from Oracle cases')
    parser.add_argument(
        '--cases-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'cases' / 'demo',
        help='Directory containing Oracle case JSON files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file path (e.g., datasets/level_2_1_basic.jsonl)'
    )
    parser.add_argument(
        '--filter-level',
        type=str,
        help='Only include cases of specified level (e.g., "2.1")'
    )
    parser.add_argument(
        '--filter-pde',
        type=str,
        help='Only include cases of specified PDE type (e.g., "poisson")'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all case files
    case_files = sorted(args.cases_dir.glob('*.json'))
    
    print(f"Found {len(case_files)} case files in {args.cases_dir}")
    
    entries = []
    for case_file in case_files:
        try:
            entry = convert_case_to_dataset_entry(case_file)
            
            # Apply filters
            if args.filter_level and entry.level != args.filter_level:
                continue
            if args.filter_pde and entry.oracle_config['pde']['type'] != args.filter_pde:
                continue
            
            entries.append(entry)
            print(f"  ✓ {entry.id} (level {entry.level})")
        
        except Exception as e:
            print(f"  ✗ {case_file.name}: {e}")
    
    # Save dataset
    save_dataset(entries, str(args.output))
    
    print(f"\n✅ Generated dataset with {len(entries)} entries")
    print(f"   Saved to: {args.output}")
    
    # Print statistics
    level_counts = {}
    for entry in entries:
        level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
    
    print("\nLevel distribution:")
    for level, count in sorted(level_counts.items()):
        print(f"  Level {level}: {count} cases - {LEVELS.get(level, 'Unknown')}")


if __name__ == '__main__':
    main()


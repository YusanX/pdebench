"""Execution sandbox for agent-generated scripts.

This module provides isolated execution environment with:
- Resource limits (timeout, memory)
- Automatic CLI argument injection
- Output capture and validation
"""

import subprocess
import tempfile
import shutil
import json
import time
import signal
import resource
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of executing an agent script."""
    
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    
    # 分离的时间统计
    t_agent_run: float  # Agent 脚本执行时间
    t_oracle_run: float = 0.0  # Oracle 生成时间（如果有）
    t_validation: float = 0.0  # 验证计算时间（如果有）
    wall_time_sec: float = 0.0  # 总时间（向后兼容）
    
    timeout_occurred: bool = False
    memory_exceeded: bool = False
    
    # Output files (if successful)
    solution_file: Optional[Path] = None
    meta_file: Optional[Path] = None
    
    # Error information
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'wall_time_sec': self.wall_time_sec,
            'timeout_occurred': self.timeout_occurred,
            'memory_exceeded': self.memory_exceeded,
            'error_message': self.error_message,
        }


def execute_agent_script(
    script_path: Path,
    outdir: Path,
    timeout_sec: int = 300,
    **test_params
) -> ExecutionResult:
    """
    Execute agent-generated script in a sandbox environment.
    
    All test parameters (resolution, degree, dt, etc.) are passed via **test_params
    and converted to CLI arguments automatically.
    
    Args:
        script_path: Path to the Python script to execute
        outdir: Output directory for solution files
        timeout_sec: Maximum execution time in seconds
        **test_params: Test parameters (resolution, degree, dt, etc.)
            Common parameters:
            - resolution: int - Mesh resolution
            - degree: int - Polynomial degree
            - dt: float - Time step (for time-dependent PDEs)
            - velocity_degree: int - Velocity space degree (for Stokes)
            - pressure_degree: int - Pressure space degree (for Stokes)
    
    Returns:
        ExecutionResult containing execution status and outputs
    
    Example:
        >>> result = execute_agent_script(
        ...     script_path=Path('solver.py'),
        ...     outdir=Path('output'),
        ...     timeout_sec=300,
        ...     resolution=128,
        ...     degree=2,
        ...     dt=0.01
        ... )
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Prepare command with all test parameters
    cmd = ['python', str(script_path), '--outdir', str(outdir)]
    
    # Add all test parameters as CLI arguments
    for key, value in test_params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Start timing
    t_start = time.time()
    
    timeout_occurred = False
    memory_exceeded = False
    
    try:
        # Run with timeout
        # Use absolute paths to avoid issues with cwd
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        
        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        
    except subprocess.TimeoutExpired:
        timeout_occurred = True
        exit_code = -1
        stdout = ""
        stderr = f"Execution timeout after {timeout_sec} seconds"
    
    except Exception as e:
        exit_code = -1
        stdout = ""
        stderr = f"Execution error: {str(e)}"
    
    t_end = time.time()
    wall_time = t_end - t_start
    
    # Check if execution was successful
    success = (exit_code == 0) and not timeout_occurred
    
    # Locate output files
    solution_file = outdir / 'solution.npz'
    meta_file = outdir / 'meta.json'
    
    if success:
        # Verify required output files exist
        if not solution_file.exists():
            success = False
            error_message = "Required output file 'solution.npz' not found"
        elif not meta_file.exists():
            success = False
            error_message = "Required output file 'meta.json' not found"
        else:
            error_message = None
    else:
        error_message = stderr if stderr else "Unknown execution failure"
        solution_file = None
        meta_file = None
    
    return ExecutionResult(
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        t_agent_run=wall_time,
        wall_time_sec=wall_time,  # 向后兼容
        timeout_occurred=timeout_occurred,
        memory_exceeded=memory_exceeded,
        solution_file=solution_file if success else None,
        meta_file=meta_file if success else None,
        error_message=error_message,
    )


def execute_agent_script_with_oracle(
    script_path: Path,
    oracle_config: Dict[str, Any],
    base_outdir: Path,
    evaluation_config: Dict[str, Any]
) -> tuple[ExecutionResult, Path, Path]:
    """
    Execute agent script and prepare for comparison with oracle.
    
    This function:
    1. Executes the agent script with parameters from oracle_config
    2. Generates oracle ground truth using the same configuration
    3. Returns both results for evaluation
    
    Args:
        script_path: Path to agent-generated script
        oracle_config: Oracle configuration (case spec)
        base_outdir: Base output directory
        evaluation_config: Evaluation configuration (timeout, etc.)
    
    Returns:
        (agent_result, agent_outdir, oracle_outdir)
    """
    import time
    from ..oracle import generate
    
    # Create output directories
    agent_outdir = base_outdir / 'agent_output'
    oracle_outdir = base_outdir / 'oracle_output'
    
    agent_outdir.mkdir(parents=True, exist_ok=True)
    oracle_outdir.mkdir(parents=True, exist_ok=True)
    
    # Extract parameters from oracle config
    mesh_spec = oracle_config['mesh']
    fem_spec = oracle_config['fem']
    
    resolution = mesh_spec['resolution']
    degree = fem_spec['degree']
    
    # Execute agent script
    agent_result = execute_agent_script(
        script_path=script_path,
        outdir=agent_outdir,
        resolution=resolution,
        degree=degree,
        timeout_sec=evaluation_config.get('timeout_sec', 300),
        memory_limit_mb=evaluation_config.get('memory_limit_mb', 4096),
    )
    
    # Generate oracle ground truth (计时)
    t_oracle_start = time.time()
    if agent_result.success:
        try:
            generate(oracle_config, oracle_outdir)
            agent_result.t_oracle_run = time.time() - t_oracle_start
        except Exception as e:
            agent_result.success = False
            agent_result.error_message = f"Oracle generation failed: {str(e)}"
            agent_result.t_oracle_run = time.time() - t_oracle_start
    
    return agent_result, agent_outdir, oracle_outdir


def validate_agent_code_syntax(script_path: Path) -> tuple[bool, Optional[str]]:
    """
    Validate that agent code has valid Python syntax.
    
    Args:
        script_path: Path to Python script
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        
        compile(code, str(script_path), 'exec')
        return True, None
    
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def create_agent_script_template(
    prompt: str,
    requirements: List[str],
    output_path: Path
):
    """
    Create a template agent script with prompt and requirements as comments.
    
    This is useful for manual testing or as a starting point for agents.
    
    Args:
        prompt: Problem description
        requirements: List of requirements
        output_path: Path to save template script
    """
    template = f'''#!/usr/bin/env python3
"""
Agent-generated script for PDE solving task.

PROBLEM DESCRIPTION:
{prompt}

REQUIREMENTS:
{chr(10).join(f"{i+1}. {req}" for i, req in enumerate(requirements))}
"""

import argparse
import numpy as np
from dolfinx import mesh, fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='PDE solver')
    parser.add_argument('--resolution', type=int, required=True, help='Mesh resolution')
    parser.add_argument('--degree', type=int, required=True, help='Polynomial degree')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # TODO: Implement solver here
    
    # Example: Save solution
    # x_grid = np.linspace(0, 1, 100)
    # y_grid = np.linspace(0, 1, 100)
    # u_grid = np.zeros((100, 100))
    # 
    # np.savez(
    #     f"{{args.outdir}}/solution.npz",
    #     x=x_grid,
    #     y=y_grid,
    #     u=u_grid,
    # )
    
    # Save metadata
    # meta = {{
    #     'wall_time_sec': 0.0,
    #     'solver_info': {{
    #         'ksp_type': 'cg',
    #         'pc_type': 'jacobi',
    #         'iters': 0,
    #     }}
    # }}
    # with open(f"{{args.outdir}}/meta.json", 'w') as f:
    #     json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(template)


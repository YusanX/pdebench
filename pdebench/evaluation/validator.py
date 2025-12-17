"""Mesh-agnostic solution validator.

This module provides validation logic that works even when:
- Agent and Oracle use different mesh resolutions
- Agent and Oracle use different FE spaces (degree, family)

Key technique: Interpolate both solutions to a common reference grid for comparison.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validating an agent solution."""
    
    is_valid: bool
    reason: str
    
    # Accuracy metrics
    rel_L2_error: float
    rel_Linf_error: float
    abs_L2_error: float
    
    # Target checking
    target_metric: str
    target_threshold: float
    achieved_value: float
    meets_target: bool
    
    # Physical constraints
    mass_conservation_error: Optional[float] = None
    
    # Additional metrics
    metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'is_valid': self.is_valid,
            'reason': self.reason,
            'accuracy': {
                'rel_L2_error': float(self.rel_L2_error),
                'rel_Linf_error': float(self.rel_Linf_error),
                'abs_L2_error': float(self.abs_L2_error),
            },
            'target': {
                'metric': self.target_metric,
                'threshold': self.target_threshold,
                'achieved': float(self.achieved_value),
                'meets_target': self.meets_target,
            },
        }
        
        if self.mass_conservation_error is not None:
            result['mass_conservation_error'] = float(self.mass_conservation_error)
        
        if self.metrics:
            result['additional_metrics'] = self.metrics
        
        return result


def validate_solution(
    agent_outdir: Path,
    oracle_outdir: Path,
    evaluation_config: Dict[str, Any]
) -> ValidationResult:
    """
    Validate agent solution against oracle ground truth.
    
    This function performs mesh-agnostic validation by:
    1. Loading both solutions on their respective grids
    2. Computing error metrics on a common reference grid
    3. Checking against target thresholds
    
    Args:
        agent_outdir: Directory containing agent solution files
        oracle_outdir: Directory containing oracle reference files
        evaluation_config: Evaluation configuration (target metric, threshold)
    
    Returns:
        ValidationResult with detailed metrics
    """
    try:
        # Load agent solution
        agent_sol = np.load(agent_outdir / 'solution.npz')
        x_agent = agent_sol['x']
        y_agent = agent_sol['y']
        u_agent = agent_sol['u']
        
        # Load oracle reference
        oracle_ref = np.load(oracle_outdir / 'reference.npz')
        x_oracle = oracle_ref['x']
        y_oracle = oracle_ref['y']
        u_oracle = oracle_ref['u_star']
        
    except FileNotFoundError as e:
        return ValidationResult(
            is_valid=False,
            reason=f"Missing output file: {e.filename}",
            rel_L2_error=np.nan,
            rel_Linf_error=np.nan,
            abs_L2_error=np.nan,
            target_metric='unknown',
            target_threshold=0.0,
            achieved_value=np.nan,
            meets_target=False,
        )
    
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            reason=f"Error loading solution: {str(e)}",
            rel_L2_error=np.nan,
            rel_Linf_error=np.nan,
            abs_L2_error=np.nan,
            target_metric='unknown',
            target_threshold=0.0,
            achieved_value=np.nan,
            meets_target=False,
        )
    
    # Compute metrics on common grid (use oracle grid as reference)
    metrics = compute_metrics(
        u_agent, x_agent, y_agent,
        u_oracle, x_oracle, y_oracle
    )
    
    # Extract target configuration
    target_metric = evaluation_config.get('target_metric', 'rel_L2_grid')
    target_threshold = evaluation_config.get('target_error', 0.01)
    
    # Map target metric to computed value
    if target_metric == 'rel_L2_grid' or target_metric == 'rel_L2_error':
        achieved_value = metrics['rel_L2_error']
    elif target_metric == 'rel_Linf_error':
        achieved_value = metrics['rel_Linf_error']
    else:
        achieved_value = metrics['rel_L2_error']  # Default
    
    meets_target = achieved_value <= target_threshold
    
    # Determine validity
    is_valid = meets_target and not np.isnan(achieved_value)
    
    if is_valid:
        reason = f"{target_metric}={achieved_value:.3e} ≤ target={target_threshold:.3e}"
    else:
        if np.isnan(achieved_value):
            reason = "Solution contains NaN or invalid values"
        else:
            reason = f"{target_metric}={achieved_value:.3e} > target={target_threshold:.3e}"
    
    return ValidationResult(
        is_valid=is_valid,
        reason=reason,
        rel_L2_error=metrics['rel_L2_error'],
        rel_Linf_error=metrics['rel_Linf_error'],
        abs_L2_error=metrics['abs_L2_error'],
        target_metric=target_metric,
        target_threshold=target_threshold,
        achieved_value=achieved_value,
        meets_target=meets_target,
        metrics=metrics,
    )


def compute_metrics(
    u_agent: np.ndarray,
    x_agent: np.ndarray,
    y_agent: np.ndarray,
    u_oracle: np.ndarray,
    x_oracle: np.ndarray,
    y_oracle: np.ndarray,
) -> Dict[str, float]:
    """
    Compute error metrics between agent and oracle solutions.
    
    Strategy: Interpolate agent solution onto oracle grid for comparison.
    
    Args:
        u_agent: Agent solution (ny_agent, nx_agent)
        x_agent: Agent x-coordinates (nx_agent,)
        y_agent: Agent y-coordinates (ny_agent,)
        u_oracle: Oracle solution (ny_oracle, nx_oracle)
        x_oracle: Oracle x-coordinates (nx_oracle,)
        y_oracle: Oracle y-coordinates (ny_oracle,)
    
    Returns:
        Dictionary of error metrics
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Check for NaN or inf
    if not np.all(np.isfinite(u_agent)):
        return {
            'rel_L2_error': np.nan,
            'rel_Linf_error': np.nan,
            'abs_L2_error': np.nan,
        }
    
    if not np.all(np.isfinite(u_oracle)):
        return {
            'rel_L2_error': np.nan,
            'rel_Linf_error': np.nan,
            'abs_L2_error': np.nan,
        }
    
    # Interpolate agent solution onto oracle grid
    try:
        # Create interpolator for agent solution
        # Note: RegularGridInterpolator expects (y, x) ordering for 2D grids
        interp_agent = RegularGridInterpolator(
            (y_agent, x_agent),
            u_agent,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # Create oracle grid points
        X_oracle, Y_oracle = np.meshgrid(x_oracle, y_oracle, indexing='xy')
        points_oracle = np.stack([Y_oracle.ravel(), X_oracle.ravel()], axis=1)
        
        # Interpolate
        u_agent_interp = interp_agent(points_oracle).reshape(u_oracle.shape)
        
    except Exception as e:
        return {
            'rel_L2_error': np.nan,
            'rel_Linf_error': np.nan,
            'abs_L2_error': np.nan,
            'error': f"Interpolation failed: {str(e)}"
        }
    
    # Filter out NaN values from interpolation (points outside agent domain)
    mask = np.isfinite(u_agent_interp)
    
    if not np.any(mask):
        return {
            'rel_L2_error': np.nan,
            'rel_Linf_error': np.nan,
            'abs_L2_error': np.nan,
            'error': 'No valid interpolation points'
        }
    
    u_agent_valid = u_agent_interp[mask]
    u_oracle_valid = u_oracle[mask]
    
    # Compute error
    error = u_agent_valid - u_oracle_valid
    
    # L2 error (discrete approximation)
    abs_L2_error = np.sqrt(np.mean(error**2))
    oracle_L2_norm = np.sqrt(np.mean(u_oracle_valid**2))
    
    if oracle_L2_norm < 1e-15:
        rel_L2_error = abs_L2_error
    else:
        rel_L2_error = abs_L2_error / oracle_L2_norm
    
    # L-infinity error
    abs_Linf_error = np.max(np.abs(error))
    oracle_Linf_norm = np.max(np.abs(u_oracle_valid))
    
    if oracle_Linf_norm < 1e-15:
        rel_Linf_error = abs_Linf_error
    else:
        rel_Linf_error = abs_Linf_error / oracle_Linf_norm
    
    return {
        'rel_L2_error': float(rel_L2_error),
        'rel_Linf_error': float(rel_Linf_error),
        'abs_L2_error': float(abs_L2_error),
        'num_valid_points': int(np.sum(mask)),
        'num_total_points': int(mask.size),
    }


def compute_mass_conservation_error(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    expected_mass: Optional[float] = None
) -> float:
    """
    Compute mass conservation error.
    
    Mass = ∫∫ u dA ≈ sum(u) * dx * dy
    
    Args:
        u: Solution field (ny, nx)
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        expected_mass: Expected total mass (if known)
    
    Returns:
        Mass conservation error
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    
    computed_mass = np.sum(u) * dx * dy
    
    if expected_mass is not None:
        if abs(expected_mass) > 1e-15:
            return abs(computed_mass - expected_mass) / abs(expected_mass)
        else:
            return abs(computed_mass - expected_mass)
    else:
        return computed_mass


def check_physical_constraints(
    solution_data: Dict[str, np.ndarray],
    pde_type: str
) -> Dict[str, Any]:
    """
    Check physical constraints specific to PDE type.
    
    Args:
        solution_data: Dictionary with 'x', 'y', 'u'
        pde_type: Type of PDE ('poisson', 'heat', 'convection_diffusion')
    
    Returns:
        Dictionary of constraint check results
    """
    u = solution_data['u']
    
    checks = {
        'has_nan': bool(np.any(np.isnan(u))),
        'has_inf': bool(np.any(np.isinf(u))),
        'is_finite': bool(np.all(np.isfinite(u))),
    }
    
    # PDE-specific checks
    if pde_type == 'heat':
        # For heat equation, solution should remain bounded
        checks['max_value'] = float(np.max(u))
        checks['min_value'] = float(np.min(u))
    
    elif pde_type == 'convection_diffusion':
        # Check for oscillations (sign of instability)
        # Compute discrete second derivative
        if u.shape[0] > 2 and u.shape[1] > 2:
            d2u_dx2 = np.diff(u, n=2, axis=1)
            d2u_dy2 = np.diff(u, n=2, axis=0)
            
            # Large second derivatives indicate oscillations
            max_d2 = max(np.max(np.abs(d2u_dx2)), np.max(np.abs(d2u_dy2)))
            checks['max_second_derivative'] = float(max_d2)
            checks['likely_oscillatory'] = bool(max_d2 > 10 * np.abs(u).max())
    
    return checks


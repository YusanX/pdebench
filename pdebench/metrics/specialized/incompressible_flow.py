"""Incompressible Flow (Stokes/Navier-Stokes) specialized metrics computation.

Metrics for incompressible flow equations (Stokes, Navier-Stokes):
- Solver information (linear solver type, preconditioner, iterations)
- Block preconditioner info
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class IncompressibleFlowMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for incompressible flow PDEs.
    
    Key metrics:
    - linear_solver_type, preconditioner_type: Solver information
    - linear_iterations_mean, linear_iterations_max: Iteration counts
    - block_preconditioner: Block preconditioner info
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute incompressible flow-specific metrics."""
        metrics = {}
        
        try:
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute incompressible flow metrics: {str(e)}"
        
        return metrics
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """Read solver information from meta.json."""
        solver_info = {}
        
        try:
            meta_file = self.agent_output_dir / 'meta.json'
            if not meta_file.exists():
                return solver_info
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # Read linear solver information
            if 'linear_solver' in meta:
                ls = meta['linear_solver']
                if isinstance(ls, dict):
                    solver_info['linear_solver_type'] = ls.get('type', 'unknown')
                    solver_info['preconditioner_type'] = ls.get('preconditioner', 'none')
                    
                    if 'iterations' in ls:
                        iters = ls['iterations']
                        if isinstance(iters, list):
                            solver_info['linear_iterations_mean'] = float(np.mean(iters))
                            solver_info['linear_iterations_max'] = int(np.max(iters))
                        else:
                            solver_info['linear_iterations'] = iters
            
            # Block preconditioner info
            if 'block_preconditioner' in meta:
                solver_info['block_preconditioner'] = meta['block_preconditioner']
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info


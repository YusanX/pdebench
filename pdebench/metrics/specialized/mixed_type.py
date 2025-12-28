"""Mixed-type PDE specialized metrics computation.

Metrics for mixed-type equations (convection-diffusion):
- Péclet number: Flow regime characterization
- Total variation: TV norm
- Solver information (stabilization method, upwind parameter)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from . import SpecializedMetricsComputer


class MixedTypeMetricsComputer(SpecializedMetricsComputer):
    """
    Compute specialized metrics for mixed-type PDEs.
    
    Key metrics:
    - peclet_number: Pe = ||b||L/ε
    - total_variation: TV norm
    - stabilization_method, upwind_parameter: Solver information
    """
    
    def compute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute mixed-type-specific metrics."""
        metrics = {}
        
        try:
            # Read Péclet number
            pde_config = self.config.get('oracle_config', {}).get('pde', {})
            peclet = pde_config.get('peclet', None)
            if peclet is not None:
                metrics['peclet_number'] = float(peclet)
            
            # Read solution fields
            agent_u_file = self.agent_output_dir / 'u.npy'
            
            if agent_u_file.exists():
                u_agent = np.load(agent_u_file)
                
                # Total variation (physical property, not an error metric)
                tv_agent = self._compute_total_variation(u_agent)
                metrics['total_variation'] = float(tv_agent)
            
            # Read solver information
            solver_info = self._read_solver_info()
            if solver_info:
                metrics.update(solver_info)
            
        except Exception as e:
            metrics['error'] = f"Failed to compute mixed-type metrics: {str(e)}"
        
        return metrics
    
    def _compute_total_variation(self, u: np.ndarray) -> float:
        """Compute total variation TV(u)."""
        if u.ndim == 1:
            return float(np.sum(np.abs(np.diff(u))))
        elif u.ndim == 2:
            tv_x = np.sum(np.abs(np.diff(u, axis=0)))
            tv_y = np.sum(np.abs(np.diff(u, axis=1)))
            return float(tv_x + tv_y)
        else:
            return 0.0
    
    def _read_solver_info(self) -> Dict[str, Any]:
        """Read solver information from meta.json."""
        solver_info = {}
        
        try:
            meta_file = self.agent_output_dir / 'meta.json'
            if not meta_file.exists():
                return solver_info
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            if 'solver_info' in meta:
                si = meta['solver_info']
                if isinstance(si, dict):
                    if 'stabilization' in si:
                        solver_info['stabilization_method'] = si['stabilization']
                    if 'upwind_parameter' in si:
                        solver_info['upwind_parameter'] = float(si['upwind_parameter'])
            
        except Exception as e:
            solver_info['read_error'] = f"Failed to read solver info: {str(e)}"
        
        return solver_info


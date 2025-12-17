"""Oracle module: generates ground truth solutions for benchmark cases.

This module contains all the PDE solvers and core logic used to generate
reference solutions. Agent-generated code should NOT import from this module.
"""

from .core.generate import generate
from .core.solve import solve_case  
from .core.evaluate import evaluate

__all__ = ['generate', 'solve_case', 'evaluate']


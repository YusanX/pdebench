"""Evaluation module for mesh-agnostic solution validation."""

from .validator import validate_solution, ValidationResult, compute_metrics

__all__ = ['validate_solution', 'ValidationResult', 'compute_metrics']


"""Dataset schema definitions for the benchmark.

Each dataset entry is a JSON object with the following fields:
- id: Unique identifier (e.g., "poisson_simple_01")
- level: Difficulty level (e.g., "2.1", "2.2", "2.3")
- prompt: Natural language problem description (given to the agent)
- requirements: List of technical requirements (interface contract)
- oracle_config: Hidden configuration used to generate ground truth
- evaluation_config: Parameters for solution validation
"""

from typing import Dict, List, Any, Literal
from dataclasses import dataclass, asdict
import json


@dataclass
class DatasetEntry:
    """A single benchmark problem."""
    
    id: str
    level: str
    prompt: str
    requirements: List[str]
    oracle_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string (single line for JSONL)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetEntry':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DatasetEntry':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def load_dataset(jsonl_path: str) -> List[DatasetEntry]:
    """Load dataset from JSONL file."""
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(DatasetEntry.from_json(line))
    return entries


def save_dataset(entries: List[DatasetEntry], jsonl_path: str):
    """Save dataset to JSONL file."""
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(entry.to_json() + '\n')


# Level definitions
LEVELS = {
    "2.1": "Basic: Linear symmetric PDEs (Poisson, Heat)",
    "2.2": "Stability: Convection-Diffusion (High Peclet), Stokes",
    "2.3": "Complex: Navier-Stokes (Non-linear, Transient)",
}


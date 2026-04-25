"""
Simulation configuration dataclass.

All parameters that control a simulation run are encapsulated here so
experiments are reproducible and easy to serialise/deserialise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SimulationConfig:
    """Complete configuration for a Hotelling simulation run.

    Parameters
    ----------
    n_steps:
        Number of simulation periods to run.
    city_width:
        Width of the 2-D market area.
    city_height:
        Height of the 2-D market area.
    n_consumers:
        Number of consumers uniformly distributed in the city.
    transport_cost:
        Per-unit-distance transport cost (applies to all consumers).
    seed:
        Global random seed for reproducibility.
    log_every:
        Record detailed metrics every *log_every* steps (0 = never).
    output_dir:
        Directory where results and model checkpoints are saved.
    extra:
        Free-form dict for experiment-specific parameters.
    """

    n_steps: int = 100
    city_width: float = 1.0
    city_height: float = 1.0
    n_consumers: int = 1000
    transport_cost: float = 1.0
    seed: Optional[int] = 42
    log_every: int = 10
    output_dir: str = "results"
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a plain dictionary."""
        import dataclasses  # noqa: PLC0415

        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        """Construct a :class:`SimulationConfig` from a dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml  # noqa: PLC0415

            with open(path) as fh:
                data = yaml.safe_load(fh)
            return cls.from_dict(data or {})
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs.") from exc

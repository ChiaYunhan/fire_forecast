"""Configuration loader for FIRE forecast scenarios."""
from dataclasses import dataclass
from typing import Any
import yaml


@dataclass
class ScenarioConfig:
    """Represents a loaded scenario configuration."""

    profile_data: dict[str, Any]
    portfolio_data: dict[str, Any]
    simulation_params: dict[str, Any]

    def __post_init__(self):
        """Validate required fields."""
        required_profile = ["age", "target_age", "income", "expenses_rate", "savings_rate"]
        for field in required_profile:
            if field not in self.profile_data:
                raise ValueError(f"Missing required profile field: {field}")

        if not self.portfolio_data.get("assets"):
            raise ValueError("Portfolio must have at least one asset")

    @classmethod
    def from_yaml(cls, file_path: str) -> "ScenarioConfig":
        """Load scenario from YAML file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            profile_data=data.get("profile", {}),
            portfolio_data=data.get("portfolio", {}),
            simulation_params=data.get("simulation", {})
        )

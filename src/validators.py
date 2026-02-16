"""Validation utilities for financial models."""

from typing import Dict, Optional


def validate_asset_costs(costs: Optional[Dict[str, float]]) -> None:
    """
    Validate asset cost structure.

    Args:
        costs: Dictionary with "ter" and/or "trading_fee" keys

    Raises:
        ValueError: If costs are invalid (negative or unreasonably high)
    """
    if costs is None:
        return

    if "ter" in costs:
        ter = costs["ter"]
        if not (0 <= ter <= 0.05):
            raise ValueError(f"TER must be between 0 and 0.05 (0-5%), got {ter}")

    if "trading_fee" in costs:
        trading_fee = costs["trading_fee"]
        if not (0 <= trading_fee <= 0.05):
            raise ValueError(
                f"trading_fee must be between 0 and 0.05 (0-5%), got {trading_fee}"
            )

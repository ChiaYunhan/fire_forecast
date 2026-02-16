from dataclasses import dataclass
from typing import List, Dict, Optional


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


@dataclass
class Asset:
    name: str
    allocation: float
    expected_return: float
    volatility: float
    costs: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        validate_asset_costs(self.costs)


@dataclass
class Portfolio:
    composition: List[Asset]
    total_value: float
    allocation_methods: str


@dataclass
class FinancialProfile:
    income: float
    expenses_rate: float
    savings_rate: float
    portfolio: Portfolio
    age: int
    target_age: int

    def __post_init__(self):
        self._validate()

    def annual_savings(self) -> float:
        return self.income * self.savings_rate

    def annual_expenses(self) -> float:
        return self.income * self.expenses_rate

    def _validate(self) -> None:
        total = self.expenses_rate + self.savings_rate
        if abs(total - 1) > 0.01:
            raise ValueError("Expenses and Savings rate do not equal to 1.0")


@dataclass
class MonteCarloSimResults:
    success_rate: float
    median_fire_age: Optional[int]
    average_years_to_fire: Optional[float]

    # Distribution metrics
    portfolio_percentiles: Dict[int, float]
    fire_age_percentiles: Dict[int, int]
    worst_case_portfolio: float
    best_case_portfolio: float

    # risk
    shortfall_amount: Optional[float]
    max_drawdown: float

    # Sensitivity analysis
    strategy_name: str
    input_params: FinancialProfile
    n_simulations: int
    np_seed: int

    # charting
    annual_trajectories: List[List[float]]

    failure_rate: float = 0.0

    def __post_init__(self):
        self.failure_rate = 1 - self.success_rate

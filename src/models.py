from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class Asset:
    name: str
    allocation: float
    expected_return: float
    volatility: float
    costs: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        self._validate_costs()

    def _validate_costs(self) -> None:
        """
        Validate asset cost structure.

        Raises:
            ValueError: If costs are invalid (negative or unreasonably high)
        """
        if self.costs is None:
            return

        if "ter" in self.costs:
            ter = self.costs["ter"]
            if not (0 <= ter <= 0.05):
                raise ValueError(f"TER must be between 0 and 0.05 (0-5%), got {ter}")

        if "trading_fee" in self.costs:
            trading_fee = self.costs["trading_fee"]
            if not (0 <= trading_fee <= 0.05):
                raise ValueError(
                    f"trading_fee must be between 0 and 0.05 (0-5%), got {trading_fee}"
                )


@dataclass
class Portfolio:
    composition: List[Asset]
    total_value: float
    allocation_methods: str

    def expected_return(self) -> float:
        """
        Calculate weighted average expected return of the portfolio.

        Returns:
            Expected annual return as decimal (e.g., 0.0565 for 5.65%)
        """
        return sum(asset.allocation * asset.expected_return for asset in self.composition)

    def volatility(self) -> float:
        """
        Calculate portfolio volatility using simplified approach.

        Assumes zero correlation between assets (conservative estimate).
        Formula: sqrt(sum of (allocation^2 * volatility^2))

        Returns:
            Portfolio volatility as decimal (e.g., 0.1045 for 10.45%)
        """
        variance = sum(
            (asset.allocation ** 2) * (asset.volatility ** 2)
            for asset in self.composition
        )
        return math.sqrt(variance)


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
    input_params: FinancialProfile
    n_simulations: int
    np_seed: int

    # charting
    annual_trajectories: List[List[float]]

    # Portfolio metrics
    expected_portfolio_return: float
    portfolio_volatility: float

    failure_rate: float = 0.0

    def __post_init__(self):
        self.failure_rate = 1 - self.success_rate

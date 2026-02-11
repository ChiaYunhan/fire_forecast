from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Asset:
    name: str
    allocation: float
    expected_return: float
    volatility: float


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

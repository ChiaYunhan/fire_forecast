from dataclasses import dataclass
from typing import List


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

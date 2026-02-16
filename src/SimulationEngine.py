from typing import List, Optional
import numpy as np

from .models import FinancialProfile


def apply_ter(portfolio_value: float, ter: float) -> float:
    """
    Apply Total Expense Ratio (TER) to portfolio value.

    Args:
        portfolio_value: Current portfolio value
        ter: Total Expense Ratio as decimal (e.g., 0.0019 for 0.19%)

    Returns:
        Portfolio value after TER deduction
    """
    return portfolio_value * (1 - ter)


def apply_trading_fee(contribution: float, trading_fee: float) -> float:
    """
    Apply trading fee to a contribution/purchase amount.

    Args:
        contribution: Amount being invested
        trading_fee: Trading fee as decimal (e.g., 0.0005 for 0.05%)

    Returns:
        Net contribution after trading fee deduction
    """
    return contribution * (1 - trading_fee)


class SimulationEngine:
    """
    Runs a single financial independence simulation using the Template Method pattern.

    The run() method defines the simulation lifecycle:
    1. setup() - initialize state from profile
    2. simulate_year() - apply returns, contributions (loop until target age)
    3. collect_results() - gather final outcomes
    """

    def __init__(self, profile: FinancialProfile):
        self.profile = profile

        # Simulation state (initialized in setup())
        self.current_portfolio_value: float = 0.0
        self.current_age: int = 0
        self.year_count: int = 0

        self.portfolio_history: List[float] = []
        self.fire_age: Optional[int] = None
        self.fire_target = self.profile.annual_expenses() / 0.04

    def run(self) -> dict:
        """
        Template Method: orchestrates the simulation lifecycle.

        Returns:
            dict with simulation results
        """
        self.setup()

        while self.current_age < self.profile.target_age:
            self.simulate_year()

        return self.collect_results()

    def setup(self) -> None:
        """Initialize simulation state from the financial profile."""
        self.current_portfolio_value = self.profile.portfolio.total_value
        self.current_age = self.profile.age
        self.year_count = 0

    def _calculate_portfolio_return(self) -> float:
        """
        Calculate weighted average return with volatility for the portfolio.

        Generates a randomized return using normal distribution based on
        each asset's expected return and volatility.

        Returns:
            The portfolio return for this year (e.g., 0.08 for 8%)
        """
        total_return = 0.0

        for asset in self.profile.portfolio.composition:
            # Base expected return
            expected = asset.expected_return

            # Add randomness based on asset's natural volatility
            random_shock = np.random.normal(0, asset.volatility)

            # Weighted by allocation
            asset_return = (expected + random_shock) * asset.allocation
            total_return += asset_return

        return total_return

    def simulate_year(self) -> None:
        """
        Simulate one year of the financial journey.

        Steps:
        1. Calculate annual return
        2. Apply return to portfolio
        3. Add annual savings
        4. Increment age and year counter
        """
        annual_return = self._calculate_portfolio_return()
        self.current_portfolio_value *= 1 + annual_return
        self.current_portfolio_value += self.profile.annual_savings()
        self.current_age += 1
        self.year_count += 1

        self.portfolio_history.append(self.current_portfolio_value)

        if self.current_portfolio_value >= self.fire_target and self.fire_age is None:
            self.fire_age = self.current_age

    def collect_results(self) -> dict:
        """
        Gather final simulation outcomes.

        Returns:
            dict containing simulation results
        """
        return {
            "final_portfolio_value": self.current_portfolio_value,
            "fire_target": self.fire_target,
            "fire_achieved": True if self.fire_age else False,
            "years_simulated": self.year_count,
            "final_age": self.current_age,
            "fire_age": self.fire_age,
            "portfolio_history": self.portfolio_history,
        }

    def reset(self):
        """Reset simulation state for next run."""
        self.current_portfolio_value = 0.0
        self.current_age = 0
        self.year_count = 0
        self.portfolio_history = []
        self.fire_age = None

from typing import List, Optional

from .Strategy.base import InvestmentStrategy
from .models import FinancialProfile


class SimulationEngine:
    """
    Runs a single financial independence simulation using the Template Method pattern.

    The run() method defines the simulation lifecycle:
    1. setup() - initialize state from profile
    2. simulate_year() - apply returns, contributions (loop until target age)
    3. collect_results() - gather final outcomes
    """

    def __init__(self, profile: FinancialProfile, strategy: InvestmentStrategy):
        self.profile = profile
        self.strategy = strategy

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
            dict with simulation results (final_value, fire_achieved, years_simulated)
        """
        self.setup()

        # Simulate each year until target age
        while self.current_age < self.profile.target_age:
            self.simulate_year()

        return self.collect_results()

    def setup(self) -> None:
        """Initialize simulation state from the financial profile."""
        self.current_portfolio_value = self.profile.portfolio.total_value
        self.current_age = self.profile.age
        self.year_count = 0

    def simulate_year(self) -> None:
        """
        Simulate one year of the financial journey.

        Steps:
        1. Calculate annual return using the strategy
        2. Apply return to current portfolio value
        3. Add annual savings contribution
        4. Increment age and year counter
        """
        annual_return = self.strategy.calculate_annual_return(
            self.profile.portfolio, self.year_count
        )
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
            dict containing:
            - final_portfolio_value: ending portfolio value
            - fire_achieved: whether FIRE goal was reached
            - years_simulated: number of years simulated
            - final_age: age at end of simulation
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
        # Simulation state (initialized in setup())
        self.current_portfolio_value: float = 0.0
        self.current_age: int = 0
        self.year_count: int = 0

        self.portfolio_history: List[float] = []
        self.fire_age: Optional[int] = None

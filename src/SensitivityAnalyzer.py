from typing import List
from dataclasses import replace

from .models import FinancialProfile, MonteCarloSimResults
from .SimulationEngine import SimulationEngine
from .MonteCarloRunner import MonteCarloRunner


class SensitivityAnalyzer:
    """
    Analyzes how changes to input parameters affect FIRE outcomes.

    Allows sweeping savings rate, income, or other parameters to understand
    their impact on success probability and FIRE timeline.
    """

    def __init__(self, base_profile: FinancialProfile):
        """
        Initialize analyzer with a base financial profile.

        Args:
            base_profile: The baseline financial profile
        """
        self.base_profile = base_profile

    def sweep_savings_rate(
        self, rates: List[float], n_simulations: int, seed: int
    ) -> List[MonteCarloSimResults]:
        """
        Sweep across different savings rates and run Monte Carlo for each.

        Args:
            rates: List of savings rates to test (e.g., [0.2, 0.3, 0.4])
            n_simulations: Number of Monte Carlo runs per rate
            seed: Random seed for reproducibility

        Returns:
            List of MonteCarloSimResults, one for each savings rate
        """
        results = []

        for rate in rates:
            # Create modified profile with new savings rate
            # Expenses rate must adjust to maintain sum = 1.0
            new_expenses_rate = 1.0 - rate

            modified_profile = replace(
                self.base_profile,
                savings_rate=rate,
                expenses_rate=new_expenses_rate
            )

            # Run Monte Carlo with this profile
            engine = SimulationEngine(modified_profile)
            runner = MonteCarloRunner(engine, n_simulations=n_simulations, seed=seed)
            runner.run_simulations()
            result = runner.aggregate_results()

            results.append(result)

        return results

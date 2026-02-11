from typing import List, Dict
import numpy as np

from .SimulationEngine import SimulationEngine
from .models import MonteCarloSimResults


class MonteCarloRunner:
    def __init__(self, engine: SimulationEngine, n_simulations: int, seed: int):
        np.random.seed(seed)
        self.engine = engine
        self.n_simulations = n_simulations
        self.results = []

    def run_simulations(self) -> List[Dict]:
        self.results = []
        for i in range(self.n_simulations):
            self.engine.reset()

            result = self.engine.run()
            self.results.append(result)

        return self.results

    def aggregate_results(self) -> "MonteCarloSimResults":
        """
        Aggregate results from all simulation runs.

        Returns:
            MonteCarloSimResults with percentile calculations, success rates, and metrics
        """
        # Extract final portfolio values from all runs
        final_values = [result["final_portfolio_value"] for result in self.results]

        # Calculate portfolio percentiles
        portfolio_percentiles = {
            10: float(np.percentile(final_values, 10)),
            25: float(np.percentile(final_values, 25)),
            50: float(np.percentile(final_values, 50)),
            75: float(np.percentile(final_values, 75)),
            90: float(np.percentile(final_values, 90)),
        }

        # Placeholder for remaining fields (will implement in next tasks)
        return MonteCarloSimResults(
            success_rate=0.0,  # TODO: Task 2
            median_fire_age=None,  # TODO: Task 2
            average_years_to_fire=None,  # TODO: Task 2
            portfolio_percentiles=portfolio_percentiles,
            fire_age_percentiles={},  # TODO: Task 2
            worst_case_portfolio=min(final_values),
            best_case_portfolio=max(final_values),
            shortfall_amount=None,  # TODO: Task 3
            max_drawdown=0.0,  # TODO: Task 3
            strategy_name=self.engine.strategy.name,
            input_params=self.engine.profile,
            n_simulations=self.n_simulations,
            np_seed=42,  # Will fix in later task
            annual_trajectories=[],  # TODO: Task 4
        )

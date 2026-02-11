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

        # Calculate FIRE success metrics
        successful_runs = [r for r in self.results if r["fire_achieved"]]
        success_rate = len(successful_runs) / len(self.results)

        # Calculate FIRE age statistics (only for successful runs)
        fire_age_percentiles = {}
        median_fire_age = None
        average_years_to_fire = None

        if successful_runs:
            fire_ages = [r["fire_age"] for r in successful_runs]
            fire_age_percentiles = {
                10: round(np.percentile(fire_ages, 10)),
                25: round(np.percentile(fire_ages, 25)),
                50: round(np.percentile(fire_ages, 50)),
                75: round(np.percentile(fire_ages, 75)),
                90: round(np.percentile(fire_ages, 90)),
            }
            median_fire_age = fire_age_percentiles[50]

            # Average years to FIRE from starting age
            starting_age = self.engine.profile.age
            years_to_fire = [r["fire_age"] - starting_age for r in successful_runs]
            average_years_to_fire = float(np.mean(years_to_fire))

        return MonteCarloSimResults(
            success_rate=success_rate,
            median_fire_age=median_fire_age,
            average_years_to_fire=average_years_to_fire,
            portfolio_percentiles=portfolio_percentiles,
            fire_age_percentiles=fire_age_percentiles,
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

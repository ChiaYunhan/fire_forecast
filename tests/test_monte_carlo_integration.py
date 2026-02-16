import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner


class TestMonteCarloIntegration:
    def test_aggregate_results_includes_portfolio_metrics(self):
        """Test that aggregate_results calculates and includes portfolio metrics."""
        assets = [
            Asset(name="VWRA", allocation=0.65, expected_return=0.07, volatility=0.15),
            Asset(name="IGLN", allocation=0.25, expected_return=0.03, volatility=0.15),
            Asset(name="CASH", allocation=0.10, expected_return=0.035, volatility=0.005),
        ]
        portfolio = Portfolio(
            composition=assets,
            total_value=0.0,
            allocation_methods="proportional"
        )
        profile = FinancialProfile(
            income=50000,
            expenses_rate=0.65,
            savings_rate=0.35,
            portfolio=portfolio,
            age=25,
            target_age=27  # Short simulation for speed
        )

        engine = SimulationEngine(profile)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)
        runner.run_simulations()
        results = runner.aggregate_results()

        # Check that metrics are calculated correctly
        expected_return = portfolio.expected_return()
        expected_vol = portfolio.volatility()

        assert abs(results.expected_portfolio_return - expected_return) < 0.0001
        assert abs(results.portfolio_volatility - expected_vol) < 0.0001

        # Verify approximate values
        assert 0.056 < results.expected_portfolio_return < 0.057
        assert 0.104 < results.portfolio_volatility < 0.105

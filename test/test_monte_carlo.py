import pytest
import numpy as np
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.Strategy.balanced import BalancedStrategy


@pytest.fixture
def sample_portfolio():
    """Test portfolio for Monte Carlo runs"""
    stock = Asset(
        name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05)
    return Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="70/30"
    )


@pytest.fixture
def sample_profile(sample_portfolio):
    """Financial profile for testing"""
    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=sample_portfolio,
        age=30,
        target_age=45,
    )


class TestMonteCarloPortfolioPercentiles:
    def test_portfolio_percentiles_structure(self, sample_profile):
        """Percentiles dict contains all required keys (10, 25, 50, 75, 90)"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        expected_percentiles = [10, 25, 50, 75, 90]
        assert all(p in results.portfolio_percentiles for p in expected_percentiles)

    def test_portfolio_percentiles_ordering(self, sample_profile):
        """Percentiles are in ascending order (10th < 25th < 50th < 75th < 90th)"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        p = results.portfolio_percentiles
        assert p[10] < p[25] < p[50] < p[75] < p[90]

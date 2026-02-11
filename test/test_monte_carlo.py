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


class TestMonteCarloFIREMetrics:
    def test_success_rate_calculation(self, sample_profile):
        """Success rate is between 0 and 1"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        assert 0.0 <= results.success_rate <= 1.0
        assert results.failure_rate == 1.0 - results.success_rate
        # Verify it's actually calculating, not just returning 0.0
        assert results.success_rate > 0, "Expected some successful runs with this profile"

    def test_fire_age_percentiles_for_successful_runs(self, sample_portfolio):
        """FIRE age percentiles calculated only from successful runs"""
        # Create profile likely to succeed
        success_profile = FinancialProfile(
            income=100000.0,
            expenses_rate=0.3,
            savings_rate=0.7,
            portfolio=sample_portfolio,
            age=25,
            target_age=55,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(success_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # Should have high success rate with this aggressive savings profile
        assert results.success_rate > 0.5, "Expected high success rate with 70% savings rate"

        # Should have fire_age_percentiles if any runs succeeded
        expected_percentiles = [10, 25, 50, 75, 90]
        assert all(p in results.fire_age_percentiles for p in expected_percentiles)

        # Median fire age should be set
        assert results.median_fire_age is not None
        # Should achieve FIRE before target age
        assert 25 <= results.median_fire_age <= 55

        # Average years to FIRE should be calculated
        assert results.average_years_to_fire is not None
        assert results.average_years_to_fire > 0

    def test_median_fire_age_none_when_all_fail(self, sample_portfolio):
        """Median fire age is None when no runs achieve FIRE"""
        # Create profile unlikely to succeed
        fail_profile = FinancialProfile(
            income=50000.0,
            expenses_rate=0.95,
            savings_rate=0.05,
            portfolio=sample_portfolio,
            age=40,
            target_age=45,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(fail_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # With 5% savings rate and short timeframe, should have very low success
        assert results.success_rate < 0.5, "Expected low success rate with 5% savings rate"

        # If no runs succeeded, these should be None
        if results.success_rate == 0:
            assert results.median_fire_age is None
            assert results.average_years_to_fire is None
            assert results.fire_age_percentiles == {}

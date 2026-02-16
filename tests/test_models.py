import pytest
from src.models import Asset, Portfolio, FinancialProfile, MonteCarloSimResults


class TestPortfolio:
    def test_expected_return_single_asset(self):
        """Test expected return with a single asset."""
        asset = Asset(
            name="VWRA",
            allocation=1.0,
            expected_return=0.07,
            volatility=0.15
        )
        portfolio = Portfolio(
            composition=[asset],
            total_value=10000.0,
            allocation_methods="proportional"
        )

        assert portfolio.expected_return() == 0.07

    def test_expected_return_multiple_assets(self):
        """Test expected return with multiple assets - matches user calculation."""
        assets = [
            Asset(name="VWRA", allocation=0.65, expected_return=0.07, volatility=0.15),
            Asset(name="IGLN", allocation=0.25, expected_return=0.03, volatility=0.15),
            Asset(name="CASH", allocation=0.10, expected_return=0.035, volatility=0.005),
        ]
        portfolio = Portfolio(
            composition=assets,
            total_value=10000.0,
            allocation_methods="proportional"
        )

        # Expected: 0.65 * 0.07 + 0.25 * 0.03 + 0.10 * 0.035 = 0.0565
        expected = 0.65 * 0.07 + 0.25 * 0.03 + 0.10 * 0.035
        assert abs(portfolio.expected_return() - expected) < 0.0001

    def test_volatility_single_asset(self):
        """Test volatility with a single asset."""
        asset = Asset(
            name="VWRA",
            allocation=1.0,
            expected_return=0.07,
            volatility=0.15
        )
        portfolio = Portfolio(
            composition=[asset],
            total_value=10000.0,
            allocation_methods="proportional"
        )

        assert portfolio.volatility() == 0.15

    def test_volatility_multiple_assets(self):
        """Test volatility with multiple assets - simplified calculation."""
        assets = [
            Asset(name="VWRA", allocation=0.65, expected_return=0.07, volatility=0.15),
            Asset(name="IGLN", allocation=0.25, expected_return=0.03, volatility=0.15),
            Asset(name="CASH", allocation=0.10, expected_return=0.035, volatility=0.005),
        ]
        portfolio = Portfolio(
            composition=assets,
            total_value=10000.0,
            allocation_methods="proportional"
        )

        # Simplified: sqrt(sum of (allocation^2 * volatility^2))
        # = sqrt(0.65^2 * 0.15^2 + 0.25^2 * 0.15^2 + 0.10^2 * 0.005^2)
        # = sqrt(0.009506 + 0.001406 + 0.0000003)
        # = sqrt(0.010912) = 0.1045 ≈ 0.1045
        import math
        expected = math.sqrt(
            0.65**2 * 0.15**2 +
            0.25**2 * 0.15**2 +
            0.10**2 * 0.005**2
        )
        result = portfolio.volatility()
        assert abs(result - expected) < 0.0001
        # Verify it's approximately 10.45%
        assert 0.104 < result < 0.105


class TestMonteCarloSimResults:
    def test_results_include_portfolio_metrics(self):
        """Test that MonteCarloSimResults can store portfolio metrics."""
        assets = [
            Asset(name="VWRA", allocation=0.65, expected_return=0.07, volatility=0.15),
            Asset(name="IGLN", allocation=0.25, expected_return=0.03, volatility=0.15),
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
            target_age=55
        )

        results = MonteCarloSimResults(
            success_rate=0.85,
            median_fire_age=45,
            average_years_to_fire=20.0,
            portfolio_percentiles={10: 100000, 50: 500000, 90: 1000000},
            fire_age_percentiles={10: 40, 50: 45, 90: 50},
            worst_case_portfolio=50000.0,
            best_case_portfolio=2000000.0,
            shortfall_amount=10000.0,
            max_drawdown=0.35,
            input_params=profile,
            n_simulations=10000,
            np_seed=42,
            annual_trajectories=[],
            expected_portfolio_return=0.0565,
            portfolio_volatility=0.1045
        )

        assert results.expected_portfolio_return == 0.0565
        assert results.portfolio_volatility == 0.1045

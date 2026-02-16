import pytest
from src.models import Asset, Portfolio, FinancialProfile


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

import pytest
from src.models import Asset, Portfolio
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


# Fixtures
@pytest.fixture
def sample_portfolio():
    """A simple test portfolio"""
    stock = Asset(
        name="Stock ETF", allocation=0.8, expected_return=0.10, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.2, expected_return=0.04, volatility=0.05)
    return Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="80/20"
    )


# Strategy Tests
class TestAggressiveStrategy:
    def test_risk_multiplier(self):
        """Aggressive strategy has 1.5x risk multiplier"""
        strategy = AggressiveStrategy()
        assert strategy.get_risk_multiplier() == 1.5

    def test_name(self):
        """Strategy name is correct"""
        strategy = AggressiveStrategy()
        assert strategy.name == "Aggressive"

    def test_calculate_annual_return(self, sample_portfolio):
        strategy = AggressiveStrategy()
        assert isinstance(strategy.calculate_annual_return(sample_portfolio, 0), float)
        prev = strategy.calculate_annual_return(sample_portfolio, 0)
        assert strategy.calculate_annual_return(sample_portfolio, 0) != prev


class TestBalancedStrategy:
    def test_risk_multiplier(self):
        """Aggressive strategy has 1.5x risk multiplier"""
        strategy = BalancedStrategy()
        assert strategy.get_risk_multiplier() == 1.1

    def test_name(self):
        """Strategy name is correct"""
        strategy = BalancedStrategy()
        assert strategy.name == "Balanced"

    def test_calculate_annual_return(self, sample_portfolio):
        strategy = BalancedStrategy()
        assert isinstance(strategy.calculate_annual_return(sample_portfolio, 0), float)
        prev = strategy.calculate_annual_return(sample_portfolio, 0)
        assert strategy.calculate_annual_return(sample_portfolio, 0) != prev


class TestConservativeStrategy:
    def test_risk_multiplier(self):
        """Aggressive strategy has 1.5x risk multiplier"""
        strategy = ConservativeStrategy()
        assert strategy.get_risk_multiplier() == 0.8

    def test_name(self):
        """Strategy name is correct"""
        strategy = ConservativeStrategy()
        assert strategy.name == "Conservative"

    def test_calculate_annual_return(self, sample_portfolio):
        strategy = ConservativeStrategy()
        assert isinstance(strategy.calculate_annual_return(sample_portfolio, 0), float)
        prev = strategy.calculate_annual_return(sample_portfolio, 0)
        assert strategy.calculate_annual_return(sample_portfolio, 0) != prev


class TestStrategyPolymorphism:
    """Test that all strategies implement the same interface"""

    def test_all_strategies_have_same_interface(self):
        """All strategies can be used interchangeably"""
        strategies = [
            AggressiveStrategy(),
            BalancedStrategy(),
            ConservativeStrategy(),
        ]

        for strategy in strategies:
            # All should have these methods/properties
            assert hasattr(strategy, "calculate_annual_return")
            assert hasattr(strategy, "get_risk_multiplier")
            assert hasattr(strategy, "name")
            assert callable(strategy.calculate_annual_return)
            assert callable(strategy.get_risk_multiplier)

    def test_all_strategies_have_different_risk(self):
        aggressive = AggressiveStrategy()
        balanced = BalancedStrategy()
        conservative = ConservativeStrategy()

        assert aggressive.get_risk_multiplier() > balanced.get_risk_multiplier()
        assert balanced.get_risk_multiplier() > conservative.get_risk_multiplier()

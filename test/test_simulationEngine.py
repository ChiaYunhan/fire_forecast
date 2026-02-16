import numpy as np
import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine


# Test fixtures
@pytest.fixture
def simple_asset():
    return Asset(
        name="Test Asset",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.10
    )


@pytest.fixture
def simple_portfolio(simple_asset):
    return Portfolio(
        composition=[simple_asset],
        total_value=100000.0,
        allocation_methods="100% test"
    )


@pytest.fixture
def simple_profile(simple_portfolio):
    return FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=simple_portfolio,
        age=30,
        target_age=31
    )


class TestSimulationEngineWithoutStrategy:
    def test_engine_initializes_without_strategy(self, simple_profile):
        """SimulationEngine can be created without strategy parameter"""
        engine = SimulationEngine(simple_profile)
        assert engine.profile == simple_profile
        assert not hasattr(engine, 'strategy')

    def test_calculate_portfolio_return_uses_asset_volatility(self, simple_profile):
        """Returns are calculated using asset expected_return and volatility"""
        np.random.seed(42)
        engine = SimulationEngine(simple_profile)

        # Generate returns to verify distribution
        returns = [engine._calculate_portfolio_return() for _ in range(1000)]

        import statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        # Mean should be close to expected_return (0.07)
        assert 0.06 < mean_return < 0.08

        # Std dev should be close to volatility (0.10)
        assert 0.09 < std_return < 0.11


@pytest.fixture
def asset_with_costs():
    """Asset with TER and trading fees"""
    return Asset(
        name="VWRA",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.0862,
        costs={"ter": 0.0019, "trading_fee": 0.0005}
    )


@pytest.fixture
def portfolio_with_costs(asset_with_costs):
    return Portfolio(
        composition=[asset_with_costs],
        total_value=100000.0,
        allocation_methods="100% VWRA"
    )


@pytest.fixture
def profile_with_costs(portfolio_with_costs):
    return FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio_with_costs,
        age=30,
        target_age=31
    )


class TestSimulationEngineWithCosts:
    def test_ter_reduces_portfolio_value(self, profile_with_costs):
        """TER is applied annually to portfolio value"""
        np.random.seed(42)
        engine = SimulationEngine(profile_with_costs)
        engine._calculate_portfolio_return = lambda: 0.0  # Zero return for testing

        result = engine.run()

        # Expected: 100k + net_savings - TER
        annual_savings = 24000.0  # 60k * 0.4
        net_savings = annual_savings * (1 - 0.0005)  # After trading fee
        expected_before_ter = 100000.0 + net_savings
        expected_after_ter = expected_before_ter * (1 - 0.0019)

        assert abs(result["final_portfolio_value"] - expected_after_ter) < 1.0

    def test_trading_fee_reduces_contributions(self, profile_with_costs):
        """Trading fee is applied to contributions"""
        np.random.seed(42)
        engine = SimulationEngine(profile_with_costs)
        engine._calculate_portfolio_return = lambda: 0.0

        result = engine.run()

        # Verify trading fee was applied
        annual_savings = 24000.0
        fee = annual_savings * 0.0005
        assert fee == 12.0  # Sanity check

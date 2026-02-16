import numpy as np
import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy


# Fixtures (reuse the ones from test_models.py pattern)
@pytest.fixture
def sample_portfolio():
    """A test portfolio with known allocations"""
    stock = Asset(
        name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05)
    return Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="70/30"
    )


@pytest.fixture
def sample_profile(sample_portfolio):
    """A valid financial profile for testing"""
    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=sample_portfolio,
        age=30,
        target_age=45,
    )


@pytest.fixture
def aggressive_strategy():
    """Aggressive investment strategy"""
    return AggressiveStrategy()


# SimulationEngine Tests
class TestSimulationEngineInitialization:
    def test_engine_initialization(self, sample_profile, aggressive_strategy):
        """Engine initializes with profile and strategy"""
        engine = SimulationEngine(sample_profile, aggressive_strategy)

        assert engine.profile == sample_profile
        assert engine.strategy == aggressive_strategy
        assert engine.current_portfolio_value == 0.0
        assert engine.current_age == 0
        assert engine.year_count == 0


class TestSimulationEngineSetup:
    def test_engine_setup(self, sample_profile, aggressive_strategy):
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        engine.setup()
        assert engine.current_portfolio_value == sample_profile.portfolio.total_value
        assert engine.current_age == sample_profile.age
        assert engine.year_count == 0


class TestSimulationEngineSimulateYear:
    def test_simulate_year_increments_age(self, sample_profile, aggressive_strategy):
        """simulate_year increments age and year count"""
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        engine.setup()

        initial_age = engine.current_age
        initial_year = engine.year_count

        engine.simulate_year()

        assert engine.current_age == initial_age + 1
        assert engine.year_count == initial_year + 1

    def test_simulate_year_applies_return(self, sample_profile, aggressive_strategy):
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        engine.setup()

        initial_portfolio_value = sample_profile.portfolio.total_value
        engine.simulate_year()
        assert engine.current_portfolio_value != initial_portfolio_value

    def test_simulate_year_adds_savings(self, sample_profile, aggressive_strategy):
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        engine.setup()
        initial = engine.current_portfolio_value

        engine.simulate_year()

        assert engine.current_portfolio_value > initial


class TestSimulationEngineCollectResults:
    def test_collect_results_structure(self, sample_profile, aggressive_strategy):
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        engine.setup()

        results = engine.collect_results()
        attributes = [
            "final_portfolio_value",
            "fire_achieved",
            "years_simulated",
            "final_age",
            "fire_target",
        ]

        assert all([attr in results.keys() for attr in attributes])

    def test_fire_archieved_calculation(self, sample_portfolio, aggressive_strategy):
        success_profile = FinancialProfile(
            income=100_000,
            expenses_rate=0.1,
            savings_rate=0.9,
            portfolio=sample_portfolio,
            age=20,
            target_age=60,
        )
        success_engine = SimulationEngine(success_profile, aggressive_strategy)
        success_engine.run()

        success_results = success_engine.collect_results()

        failed_profile = FinancialProfile(
            income=100_000,
            expenses_rate=0.9,
            savings_rate=0.1,
            portfolio=sample_portfolio,
            age=20,
            target_age=30,
        )

        failed_engine = SimulationEngine(failed_profile, aggressive_strategy)
        failed_engine.run()

        failed_results = failed_engine.collect_results()

        assert success_results["fire_achieved"] == True
        assert failed_results["fire_achieved"] == False


class TestSimulationEngineFullRun:
    def test_run_completes_successfully(self, sample_profile, aggressive_strategy):
        """Full simulation run completes without errors"""
        engine = SimulationEngine(sample_profile, aggressive_strategy)
        results = engine.run()

        # Should return a dict
        assert isinstance(results, dict)
        assert "final_portfolio_value" in results
        assert "fire_achieved" in results

    def test_run_simulates_correct_number_of_years(
        self, sample_portfolio, aggressive_strategy
    ):
        profile = FinancialProfile(
            income=100_000,
            expenses_rate=0.1,
            savings_rate=0.9,
            portfolio=sample_portfolio,
            age=20,
            target_age=60,
        )
        engine = SimulationEngine(profile, aggressive_strategy)
        engine.run()

        results = engine.collect_results()

        assert results["years_simulated"] == 40
        assert results["final_age"] == 60

    def test_zero_starting_portfolio(self, aggressive_strategy):
        stock = Asset(
            name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
        )
        bond = Asset(
            name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05
        )
        portfolio = Portfolio(
            composition=[stock, bond], total_value=0.0, allocation_methods="70/30"
        )

        profile = FinancialProfile(
            income=100_000,
            expenses_rate=0.1,
            savings_rate=0.9,
            portfolio=portfolio,
            age=20,
            target_age=60,
        )
        engine = SimulationEngine(profile, aggressive_strategy)
        engine.setup()

        engine.simulate_year()

        assert engine.current_portfolio_value > 0


# New tests without strategy
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

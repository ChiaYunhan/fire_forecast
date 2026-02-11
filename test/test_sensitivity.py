import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.SensitivityAnalyzer import SensitivityAnalyzer
from src.Strategy.balanced import BalancedStrategy


@pytest.fixture
def base_profile():
    """Base financial profile for sensitivity analysis"""
    stock = Asset(
        name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05)
    portfolio = Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="70/30"
    )

    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio,
        age=30,
        target_age=45,
    )


class TestSensitivityAnalyzer:
    def test_savings_rate_sweep(self, base_profile):
        """Sweeping savings rate produces multiple results"""
        analyzer = SensitivityAnalyzer(base_profile, BalancedStrategy())

        # Sweep savings rates from 0.2 to 0.6 in steps of 0.1
        results = analyzer.sweep_savings_rate(
            rates=[0.2, 0.3, 0.4, 0.5, 0.6],
            n_simulations=50,
            seed=42
        )

        assert len(results) == 5
        # Each result should have a success_rate
        assert all(hasattr(r, "success_rate") for r in results)

    def test_savings_rate_impact_on_success(self, base_profile):
        """Higher savings rates increase FIRE success probability"""
        analyzer = SensitivityAnalyzer(base_profile, BalancedStrategy())

        results = analyzer.sweep_savings_rate(
            rates=[0.2, 0.4, 0.6],
            n_simulations=100,
            seed=42
        )

        # Higher savings should generally lead to higher success
        # (This is probabilistic, but should hold with 100 sims)
        success_rates = [r.success_rate for r in results]
        assert success_rates[1] >= success_rates[0]  # 0.4 >= 0.2
        assert success_rates[2] >= success_rates[1]  # 0.6 >= 0.4

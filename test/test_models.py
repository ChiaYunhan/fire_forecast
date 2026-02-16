import pytest
import math
from src.models import Asset, Portfolio, FinancialProfile, MonteCarloSimResults


# Fixtures for reusable test data
@pytest.fixture
def sample_asset():
    """A typical ETF asset"""
    return Asset(
        name="Vanguard Total Stock Market",
        allocation=0.7,
        expected_return=0.08,
        volatility=0.15,
    )


@pytest.fixture
def sample_portfolio(sample_asset):
    """A basic portfolio with one asset"""
    bond_asset = Asset(
        name="Vanguard Total Bond Market",
        allocation=0.3,
        expected_return=0.04,
        volatility=0.05,
    )
    return Portfolio(
        composition=[sample_asset, bond_asset],
        total_value=100000.0,
        allocation_methods="70/30 stock/bond split",
    )


@pytest.fixture
def sample_profile(sample_portfolio):
    """A valid financial profile"""
    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=sample_portfolio,
        age=30,
        target_age=45,
    )


# Asset tests
class TestAsset:
    def test_asset_creation(self, sample_asset):
        """Asset can be created with all required fields"""
        assert sample_asset.name == "Vanguard Total Stock Market"
        assert sample_asset.allocation == 0.7
        assert sample_asset.expected_return == 0.08
        assert sample_asset.volatility == 0.15

    def test_asset_creation_with_costs(self):
        """Asset can be created with optional cost structure"""
        asset = Asset(
            name="VWRA",
            allocation=0.75,
            expected_return=0.07,
            volatility=0.0862,
            costs={"ter": 0.0019, "trading_fee": 0.0005}
        )
        assert asset.costs["ter"] == 0.0019
        assert asset.costs["trading_fee"] == 0.0005

    def test_asset_creation_without_costs(self):
        """Asset defaults to None for costs if not provided"""
        asset = Asset(
            name="Simple Asset",
            allocation=0.5,
            expected_return=0.06,
            volatility=0.10
        )
        assert asset.costs is None

    def test_asset_creation_with_risk_metrics(self):
        """Asset can store optional risk metrics"""
        asset = Asset(
            name="VWRA",
            allocation=1.0,
            expected_return=0.07,
            volatility=0.0862,
            risk_metrics={"semi_deviation": 0.0278, "correlation": 0.99}
        )
        assert asset.risk_metrics["semi_deviation"] == 0.0278
        assert asset.risk_metrics["correlation"] == 0.99

    def test_asset_with_invalid_ter_fails(self):
        """Asset creation fails with invalid TER"""
        with pytest.raises(ValueError, match="TER must be between"):
            Asset(
                name="Invalid",
                allocation=1.0,
                expected_return=0.07,
                volatility=0.10,
                costs={"ter": 0.10}  # 10% is too high
            )


# Portfolio tests
class TestPortfolio:
    def test_portfolio_creation(self, sample_portfolio):
        """Portfolio can be created with composition of assets"""
        assert len(sample_portfolio.composition) == 2
        assert sample_portfolio.total_value == 100000.0

    def test_portfolio_composition(self, sample_portfolio):
        """Portfolio correctly contains Asset objects"""
        assert all(isinstance(asset, Asset) for asset in sample_portfolio.composition)
        assert sample_portfolio.composition[0].name == "Vanguard Total Stock Market"
        assert sample_portfolio.composition[1].name == "Vanguard Total Bond Market"

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
        expected = math.sqrt(
            0.65**2 * 0.15**2 +
            0.25**2 * 0.15**2 +
            0.10**2 * 0.005**2
        )
        result = portfolio.volatility()
        assert abs(result - expected) < 0.0001
        # Verify it's approximately 10.45%
        assert 0.104 < result < 0.105


# FinancialProfile tests
class TestFinancialProfile:
    def test_profile_creation(self, sample_profile):
        """FinancialProfile can be created with all required fields"""
        assert sample_profile.income == 100000.0
        assert sample_profile.age == 30
        assert sample_profile.target_age == 45

    def test_annual_savings_calculation(self, sample_profile):
        """annual_savings() correctly calculates income * savings_rate"""
        expected = 100000.0 * 0.4
        assert sample_profile.annual_savings() == expected

    def test_annual_expenses_calculation(self, sample_profile):
        """annual_expenses() correctly calculates income * expenses_rate"""
        expected = 100000.0 * 0.6
        assert sample_profile.annual_expenses() == expected

    def test_portfolio_composition_relationship(self, sample_profile):
        """FinancialProfile correctly contains a Portfolio with Assets"""
        assert isinstance(sample_profile.portfolio, Portfolio)
        assert len(sample_profile.portfolio.composition) == 2

    def test_validation_invalid_rates_sum(self, sample_portfolio):
        """Validation fails when expenses_rate + savings_rate != 1.0"""
        with pytest.raises(
            ValueError, match="Expenses and Savings rate do not equal to 1.0"
        ):
            FinancialProfile(
                income=10_000,
                expenses_rate=1.0,
                savings_rate=1.0,
                portfolio=sample_portfolio,
                age=30,
                target_age=45,
            )

    def test_validation_valid_exact_sum(self, sample_portfolio):
        """Validation passes when rates sum to exactly 1.0"""
        profile = FinancialProfile(
            income=10_000,
            expenses_rate=0.7,
            savings_rate=0.3,
            portfolio=sample_portfolio,
            age=30,
            target_age=45,
        )
        assert profile.expenses_rate + profile.savings_rate == 1.0

    def test_validation_within_tolerance(self, sample_portfolio):
        """Validation passes when rates are within 0.01 tolerance"""
        # 0.505 + 0.495 = 1.0 (within tolerance)
        profile = FinancialProfile(
            income=10_000,
            expenses_rate=0.505,
            savings_rate=0.495,
            portfolio=sample_portfolio,
            age=30,
            target_age=45,
        )
        assert abs((profile.expenses_rate + profile.savings_rate) - 1.0) <= 0.01

    def test_validation_outside_tolerance(self, sample_portfolio):
        """Validation fails when rates are outside 0.01 tolerance"""
        with pytest.raises(
            ValueError, match="Expenses and Savings rate do not equal to 1.0"
        ):
            FinancialProfile(
                income=10_000,
                expenses_rate=0.5,
                savings_rate=0.48,  # Sum is 0.98, outside tolerance
                portfolio=sample_portfolio,
                age=30,
                target_age=45,
            )


# MonteCarloSimResults tests
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

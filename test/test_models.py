import pytest
from src.models import Asset, Portfolio, FinancialProfile


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

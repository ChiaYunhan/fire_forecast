import pytest
from src.validators import validate_asset_costs


class TestAssetCostValidation:
    def test_valid_costs_pass(self):
        """Valid cost values pass validation"""
        costs = {"ter": 0.0019, "trading_fee": 0.0005}
        validate_asset_costs(costs)  # Should not raise

    def test_ter_too_high_fails(self):
        """TER above 5% fails validation"""
        costs = {"ter": 0.06, "trading_fee": 0.0005}
        with pytest.raises(ValueError, match="TER must be between 0 and 0.05"):
            validate_asset_costs(costs)

    def test_ter_negative_fails(self):
        """Negative TER fails validation"""
        costs = {"ter": -0.001, "trading_fee": 0.0005}
        with pytest.raises(ValueError, match="TER must be between 0 and 0.05"):
            validate_asset_costs(costs)

    def test_trading_fee_too_high_fails(self):
        """Trading fee above 5% fails validation"""
        costs = {"ter": 0.0019, "trading_fee": 0.06}
        with pytest.raises(ValueError, match="trading_fee must be between 0 and 0.05"):
            validate_asset_costs(costs)

    def test_none_costs_pass(self):
        """None costs pass validation"""
        validate_asset_costs(None)  # Should not raise

    def test_zero_costs_pass(self):
        """Zero costs pass validation"""
        costs = {"ter": 0.0, "trading_fee": 0.0}
        validate_asset_costs(costs)  # Should not raise

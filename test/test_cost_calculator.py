import pytest
from src.SimulationEngine import apply_ter, apply_trading_fee


class TestCostCalculator:
    def test_apply_ter_reduces_portfolio_value(self):
        """TER reduces portfolio value by percentage annually"""
        initial_value = 100000.0
        ter = 0.0019  # 0.19%
        result = apply_ter(initial_value, ter)
        expected = 100000.0 * (1 - 0.0019)
        assert result == expected
        assert result == 99810.0

    def test_apply_ter_with_zero_ter(self):
        """Zero TER returns original value"""
        initial_value = 100000.0
        result = apply_ter(initial_value, 0.0)
        assert result == 100000.0

    def test_apply_trading_fee_reduces_contribution(self):
        """Trading fee reduces contribution amount"""
        contribution = 1000.0
        trading_fee = 0.0005  # 0.05%
        result = apply_trading_fee(contribution, trading_fee)
        expected = 1000.0 * (1 - 0.0005)
        assert result == expected
        assert result == 999.50

    def test_apply_trading_fee_with_zero_fee(self):
        """Zero trading fee returns original contribution"""
        contribution = 1000.0
        result = apply_trading_fee(contribution, 0.0)
        assert result == 1000.0

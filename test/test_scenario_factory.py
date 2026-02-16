import pytest
from src.ScenarioFactory import ScenarioFactory
from src.ScenarioConfig import ScenarioConfig
from src.models import FinancialProfile
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


def test_create_profile_from_config(tmp_path):
    """Test creating FinancialProfile from config."""
    config_content = """
profile:
  age: 35
  target_age: 55
  income: 80000
  expenses_rate: 0.55
  savings_rate: 0.45
portfolio:
  total_value: 50000
  allocation_method: balanced
  assets:
    - name: "Stocks"
      allocation: 0.60
      expected_return: 0.08
      volatility: 0.12
strategy: balanced
simulation:
  n_simulations: 100
"""
    config_file = tmp_path / "scenario.yaml"
    config_file.write_text(config_content)

    config = ScenarioConfig.from_yaml(str(config_file))
    factory = ScenarioFactory()

    profile = factory.create_profile(config)

    assert isinstance(profile, FinancialProfile)
    assert profile.age == 35
    assert profile.target_age == 55
    assert profile.income == 80000
    assert len(profile.portfolio.composition) == 1


def test_create_strategy_from_config():
    """Test creating strategy from config."""
    factory = ScenarioFactory()

    aggressive = factory.create_strategy("aggressive")
    balanced = factory.create_strategy("balanced")
    conservative = factory.create_strategy("conservative")

    assert isinstance(aggressive, AggressiveStrategy)
    assert isinstance(balanced, BalancedStrategy)
    assert isinstance(conservative, ConservativeStrategy)


def test_create_strategy_invalid_name():
    """Test error on invalid strategy name."""
    factory = ScenarioFactory()

    with pytest.raises(ValueError, match="Unknown strategy"):
        factory.create_strategy("invalid_strategy")

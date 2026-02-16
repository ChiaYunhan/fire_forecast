import pytest
from src.ScenarioConfig import ScenarioConfig


def test_load_scenario_from_yaml(tmp_path):
    """Test loading scenario from YAML file."""
    config_content = """
profile:
  age: 30
  target_age: 50
  income: 60000
  expenses_rate: 0.60
  savings_rate: 0.40
portfolio:
  total_value: 10000
  allocation_method: balanced
  assets:
    - name: "Stock ETF"
      allocation: 0.70
      expected_return: 0.09
      volatility: 0.14
strategy: aggressive
simulation:
  n_simulations: 1000
  seed: 123
"""
    config_file = tmp_path / "test_scenario.yaml"
    config_file.write_text(config_content)

    config = ScenarioConfig.from_yaml(str(config_file))

    assert config.profile_data["age"] == 30
    assert config.profile_data["target_age"] == 50
    assert config.strategy_name == "aggressive"
    assert config.simulation_params["n_simulations"] == 1000


def test_validate_scenario_config():
    """Test validation of scenario config data."""
    invalid_config = {
        "profile": {"age": 30},  # Missing required fields
        "portfolio": {"assets": []},
        "strategy": "balanced",
        "simulation": {}
    }

    with pytest.raises(ValueError, match="Missing required profile field"):
        ScenarioConfig(
            profile_data=invalid_config["profile"],
            portfolio_data=invalid_config["portfolio"],
            strategy_name=invalid_config["strategy"],
            simulation_params=invalid_config["simulation"]
        )

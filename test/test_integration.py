"""Integration tests for full workflow."""
import pytest
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')

from src.ScenarioConfig import ScenarioConfig
from src.ScenarioFactory import ScenarioFactory
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.visualization import create_projection_fan_chart, create_fire_age_distribution


def test_full_workflow_from_yaml_to_charts(tmp_path):
    """Test complete workflow: YAML → Factory → Simulation → Charts."""
    # Create test scenario
    scenario_content = """
profile:
  age: 30
  target_age: 50
  income: 60000
  expenses_rate: 0.60
  savings_rate: 0.40
portfolio:
  total_value: 10000
  assets:
    - name: "Stocks"
      allocation: 0.70
      expected_return: 0.08
      volatility: 0.12
    - name: "Bonds"
      allocation: 0.30
      expected_return: 0.04
      volatility: 0.05
simulation:
  n_simulations: 100
  seed: 123
"""
    scenario_file = tmp_path / "test.yaml"
    scenario_file.write_text(scenario_content)

    # Load config
    config = ScenarioConfig.from_yaml(str(scenario_file))

    # Create objects via factory
    factory = ScenarioFactory()
    profile = factory.create_profile(config)

    # Run simulation
    engine = SimulationEngine(profile)
    runner = MonteCarloRunner(
        engine,
        n_simulations=config.simulation_params["n_simulations"],
        seed=config.simulation_params["seed"]
    )
    runner.run_simulations()
    results = runner.aggregate_results()

    # Verify results
    assert results.success_rate >= 0.0
    assert results.success_rate <= 1.0
    assert len(results.portfolio_percentiles) > 0

    # Generate charts
    fig1 = create_projection_fan_chart(runner, title="Test Chart")
    assert fig1 is not None

    # Calculate age distribution
    fire_ages = []
    for result in runner.results:
        if result.get("fire_achieved", False):
            fire_ages.append(result["fire_age"])

    fig2 = create_fire_age_distribution(
        fire_ages=fire_ages,
        target_age=50,
        median_fire_age=results.median_fire_age,
        success_rate=results.success_rate
    )
    assert fig2 is not None

    print(f"✓ Integration test passed: {results.success_rate:.1%} success rate")


def test_cost_modeling_integration(tmp_path):
    """Test that costs (TER and trading fees) reduce portfolio growth."""
    # Scenario without costs
    scenario_no_costs = """
profile:
  age: 30
  target_age: 35
  income: 60000
  expenses_rate: 0.60
  savings_rate: 0.40
portfolio:
  total_value: 100000
  assets:
    - name: "Test Asset"
      allocation: 1.0
      expected_return: 0.07
      volatility: 0.10
simulation:
  n_simulations: 100
  seed: 42
"""

    # Scenario with costs
    scenario_with_costs = """
profile:
  age: 30
  target_age: 35
  income: 60000
  expenses_rate: 0.60
  savings_rate: 0.40
portfolio:
  total_value: 100000
  assets:
    - name: "Test Asset"
      allocation: 1.0
      expected_return: 0.07
      volatility: 0.10
      costs:
        ter: 0.0020
        trading_fee: 0.0010
simulation:
  n_simulations: 100
  seed: 42
"""

    # Run simulation without costs
    file_no_costs = tmp_path / "no_costs.yaml"
    file_no_costs.write_text(scenario_no_costs)

    config_no_costs = ScenarioConfig.from_yaml(str(file_no_costs))
    factory = ScenarioFactory()
    profile_no_costs = factory.create_profile(config_no_costs)

    engine_no_costs = SimulationEngine(profile_no_costs)
    runner_no_costs = MonteCarloRunner(engine_no_costs, n_simulations=100, seed=42)
    runner_no_costs.run_simulations()
    results_no_costs = runner_no_costs.aggregate_results()

    # Run simulation with costs
    file_with_costs = tmp_path / "with_costs.yaml"
    file_with_costs.write_text(scenario_with_costs)

    config_with_costs = ScenarioConfig.from_yaml(str(file_with_costs))
    profile_with_costs = factory.create_profile(config_with_costs)

    engine_with_costs = SimulationEngine(profile_with_costs)
    runner_with_costs = MonteCarloRunner(engine_with_costs, n_simulations=100, seed=42)
    runner_with_costs.run_simulations()
    results_with_costs = runner_with_costs.aggregate_results()

    # Verify costs reduce portfolio values
    assert results_with_costs.portfolio_percentiles[50] < results_no_costs.portfolio_percentiles[50]
    assert results_with_costs.portfolio_percentiles[90] < results_no_costs.portfolio_percentiles[90]

    # Calculate impact magnitude
    impact = (results_no_costs.portfolio_percentiles[50] - results_with_costs.portfolio_percentiles[50]) / results_no_costs.portfolio_percentiles[50]
    assert 0.005 < impact < 0.15  # Costs should reduce portfolio by 0.5-15% over 5 years

    print(f"✓ Cost impact test passed: {impact:.1%} portfolio reduction from costs")


def test_real_world_scenario_with_costs(tmp_path):
    """Test realistic scenario with VWRA ETF including actual costs."""
    scenario_content = """
profile:
  age: 25
  target_age: 45
  income: 75000
  expenses_rate: 0.70
  savings_rate: 0.30
portfolio:
  total_value: 50000
  assets:
    - name: "VWRA"
      allocation: 1.0
      expected_return: 0.07
      volatility: 0.15
      costs:
        ter: 0.0022
        trading_fee: 0.0005
simulation:
  n_simulations: 1000
  seed: 123
"""

    scenario_file = tmp_path / "real_world.yaml"
    scenario_file.write_text(scenario_content)

    # Load and run
    config = ScenarioConfig.from_yaml(str(scenario_file))
    factory = ScenarioFactory()
    profile = factory.create_profile(config)

    # Verify costs are loaded correctly
    vwra = profile.portfolio.composition[0]
    assert vwra.costs is not None
    assert vwra.costs["ter"] == 0.0022
    assert vwra.costs["trading_fee"] == 0.0005

    # Run Monte Carlo
    engine = SimulationEngine(profile)
    runner = MonteCarloRunner(engine, n_simulations=1000, seed=123)
    runner.run_simulations()
    results = runner.aggregate_results()

    # Verify results are reasonable
    assert results.success_rate > 0.0
    assert results.portfolio_percentiles[50] > 0

    # With 20 years, $75k income, 30% savings = $22.5k/year
    # Should accumulate significant wealth even with costs
    assert results.portfolio_percentiles[50] > 300000  # At least $300k median

    print(f"✓ Real-world scenario test passed: {results.success_rate:.1%} success rate")

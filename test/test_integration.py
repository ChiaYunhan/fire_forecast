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

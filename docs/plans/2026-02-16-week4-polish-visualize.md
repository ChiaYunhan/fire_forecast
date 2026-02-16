# Week 4: Polish & Visualize Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add YAML configuration, visualization charts, CLI argument parsing, and documentation to make the FIRE Forecast Engine production-ready.

**Architecture:** Factory pattern for scenario creation from YAML configs, Matplotlib for fan charts and probability visualizations, argparse for CLI interface. Maintains separation between domain models, simulation engine, and presentation layer.

**Tech Stack:** PyYAML for config parsing, Matplotlib for charts, argparse for CLI, existing NumPy/pytest stack

---

## Task 1: YAML Configuration Support

**Files:**

- Create: `src/ScenarioConfig.py`
- Create: `test/test_scenario_config.py`
- Create: `scenarios/default.yaml`
- Modify: `pyproject.toml` (add pyyaml dependency)

### Step 1: Add PyYAML dependency

```bash
cd /Users/nigel/personal/fire_forecast
uv add pyyaml
```

**Expected:** PyYAML added to dependencies in pyproject.toml

### Step 2: Create example YAML scenario file

Create `scenarios/default.yaml`:

```yaml
profile:
  age: 25
  target_age: 45
  income: 50056.92
  expenses_rate: 0.58
  savings_rate: 0.42

portfolio:
  total_value: 0.0
  allocation_method: aggressive
  assets:
    - name: "VTI (Total Stock Market)"
      allocation: 0.80
      expected_return: 0.10
      volatility: 0.15
    - name: "BND (Total Bond Market)"
      allocation: 0.20
      expected_return: 0.04
      volatility: 0.04

strategy: balanced

simulation:
  n_simulations: 10000
  seed: 42
```

### Step 3: Write failing test for YAML loading

Create `test/test_scenario_config.py`:

```python
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
```

### Step 4: Run test to verify it fails

```bash
uv run pytest test/test_scenario_config.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'src.ScenarioConfig'"

### Step 5: Implement minimal ScenarioConfig class

Create `src/ScenarioConfig.py`:

```python
"""Configuration loader for FIRE forecast scenarios."""
from dataclasses import dataclass
from typing import Any
import yaml


@dataclass
class ScenarioConfig:
    """Represents a loaded scenario configuration."""

    profile_data: dict[str, Any]
    portfolio_data: dict[str, Any]
    strategy_name: str
    simulation_params: dict[str, Any]

    def __post_init__(self):
        """Validate required fields."""
        required_profile = ["age", "target_age", "income", "expenses_rate", "savings_rate"]
        for field in required_profile:
            if field not in self.profile_data:
                raise ValueError(f"Missing required profile field: {field}")

        if not self.portfolio_data.get("assets"):
            raise ValueError("Portfolio must have at least one asset")

    @classmethod
    def from_yaml(cls, file_path: str) -> "ScenarioConfig":
        """Load scenario from YAML file."""
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            profile_data=data.get("profile", {}),
            portfolio_data=data.get("portfolio", {}),
            strategy_name=data.get("strategy", "balanced"),
            simulation_params=data.get("simulation", {})
        )
```

### Step 6: Run test to verify it passes

```bash
uv run pytest test/test_scenario_config.py -v
```

**Expected:** PASS (2 passed)

### Step 7: Commit

```bash
git add scenarios/default.yaml src/ScenarioConfig.py test/test_scenario_config.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat: add YAML scenario configuration support

- Add PyYAML dependency
- Create ScenarioConfig class for loading/validating YAML configs
- Add default.yaml example scenario
- Validate required fields on load

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: ScenarioFactory (Factory Pattern)

**Files:**

- Create: `src/ScenarioFactory.py`
- Create: `test/test_scenario_factory.py`

### Step 1: Write failing test for factory

Create `test/test_scenario_factory.py`:

```python
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
```

### Step 2: Run test to verify it fails

```bash
uv run pytest test/test_scenario_factory.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'src.ScenarioFactory'"

### Step 3: Implement ScenarioFactory

Create `src/ScenarioFactory.py`:

```python
"""Factory for creating domain objects from scenario configurations."""
from src.ScenarioConfig import ScenarioConfig
from src.models import Asset, Portfolio, FinancialProfile
from src.Strategy.base import InvestmentStrategy
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


class ScenarioFactory:
    """Factory for creating scenarios from configuration."""

    def create_profile(self, config: ScenarioConfig) -> FinancialProfile:
        """Create FinancialProfile from config."""
        # Create assets
        assets = []
        for asset_data in config.portfolio_data["assets"]:
            asset = Asset(
                name=asset_data["name"],
                allocation=asset_data["allocation"],
                expected_return=asset_data["expected_return"],
                volatility=asset_data["volatility"]
            )
            assets.append(asset)

        # Create portfolio
        portfolio = Portfolio(
            composition=assets,
            total_value=config.portfolio_data.get("total_value", 0.0),
            allocation_methods=config.portfolio_data.get("allocation_method", "balanced")
        )

        # Create profile
        profile = FinancialProfile(
            age=config.profile_data["age"],
            target_age=config.profile_data["target_age"],
            income=config.profile_data["income"],
            expenses_rate=config.profile_data["expenses_rate"],
            savings_rate=config.profile_data["savings_rate"],
            portfolio=portfolio
        )

        return profile

    def create_strategy(self, strategy_name: str) -> InvestmentStrategy:
        """Create strategy from name."""
        strategies = {
            "aggressive": AggressiveStrategy,
            "balanced": BalancedStrategy,
            "conservative": ConservativeStrategy
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name]()
```

### Step 4: Run test to verify it passes

```bash
uv run pytest test/test_scenario_factory.py -v
```

**Expected:** PASS (3 passed)

### Step 5: Commit

```bash
git add src/ScenarioFactory.py test/test_scenario_factory.py
git commit -m "$(cat <<'EOF'
feat: add ScenarioFactory for creating objects from config

- Implement Factory pattern for scenario creation
- Create profiles and strategies from ScenarioConfig
- Validate strategy names
- Tests for profile/strategy creation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Visualization - Projection Fan Chart

**Files:**

- Create: `src/visualization.py`
- Create: `test/test_visualization.py`
- Modify: `pyproject.toml` (add matplotlib)

### Step 1: Add Matplotlib dependency

```bash
uv add matplotlib
```

**Expected:** matplotlib added to dependencies

### Step 2: Write failing test for fan chart

Create `test/test_visualization.py`:

```python
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from matplotlib.figure import Figure
from src.visualization import create_projection_fan_chart
from src.MonteCarloRunner import SimulationResults


def test_create_projection_fan_chart():
    """Test creating projection fan chart."""
    # Mock results data
    results = SimulationResults(
        all_runs=[
            {"year": i, "portfolio_value": 10000 * (1.07 ** i)}
            for i in range(21)
        ],
        profile_age=25,
        profile_target_age=45
    )

    fig = create_projection_fan_chart(results, title="Test Projection")

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Age"
    assert ax.get_ylabel() == "Portfolio Value ($)"
    assert "Test Projection" in ax.get_title()
```

### Step 3: Run test to verify it fails

```bash
uv run pytest test/test_visualization.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'src.visualization'"

### Step 4: Implement projection fan chart

Create `src/visualization.py`:

```python
"""Visualization utilities for FIRE forecast results."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Any


def create_projection_fan_chart(
    results: Any,
    title: str = "Portfolio Projection"
) -> Figure:
    """Create a fan chart showing portfolio value percentile bands over time.

    Args:
        results: SimulationResults with all_runs data
        title: Chart title

    Returns:
        matplotlib Figure object
    """
    # Extract data by year
    max_years = len(results.all_runs[0]) if results.all_runs else 0
    ages = [results.profile_age + i for i in range(max_years)]

    # Calculate percentiles for each year
    percentiles = {p: [] for p in [10, 25, 50, 75, 90]}

    for year_idx in range(max_years):
        year_values = [run[year_idx]["portfolio_value"] for run in results.all_runs]
        for p in percentiles:
            percentiles[p].append(np.percentile(year_values, p))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot percentile bands
    ax.fill_between(ages, percentiles[10], percentiles[90],
                     alpha=0.2, color='blue', label='10th-90th percentile')
    ax.fill_between(ages, percentiles[25], percentiles[75],
                     alpha=0.3, color='blue', label='25th-75th percentile')
    ax.plot(ages, percentiles[50], color='blue', linewidth=2, label='Median (50th)')

    # Formatting
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    return fig
```

### Step 5: Run test to verify it passes

```bash
uv run pytest test/test_visualization.py -v
```

**Expected:** PASS (1 passed)

### Step 6: Commit

```bash
git add src/visualization.py test/test_visualization.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat: add projection fan chart visualization

- Add matplotlib dependency
- Implement create_projection_fan_chart() with percentile bands
- Show 10th-90th and 25th-75th percentile ranges
- Format chart with grid, labels, currency formatting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Visualization - FIRE Probability Chart

**Files:**

- Modify: `src/visualization.py`
- Modify: `test/test_visualization.py`

### Step 1: Write failing test for probability chart

Add to `test/test_visualization.py`:

```python
def test_create_fire_probability_chart():
    """Test creating FIRE probability chart."""
    # Mock probability data by age
    age_probabilities = {
        40: 0.15,
        45: 0.50,
        50: 0.75,
        55: 0.90,
        60: 0.95
    }

    fig = create_fire_probability_chart(
        age_probabilities,
        target_age=45,
        title="FIRE Probability"
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Age"
    assert ax.get_ylabel() == "FIRE Success Probability (%)"
```

### Step 2: Run test to verify it fails

```bash
uv run pytest test/test_visualization.py::test_create_fire_probability_chart -v
```

**Expected:** FAIL with "NameError: name 'create_fire_probability_chart' is not defined"

### Step 3: Implement FIRE probability chart

Add to `src/visualization.py`:

```python
def create_fire_probability_chart(
    age_probabilities: dict[int, float],
    target_age: int,
    title: str = "FIRE Success Probability by Age"
) -> Figure:
    """Create a chart showing probability of FIRE success by age.

    Args:
        age_probabilities: Dict mapping age to success probability (0.0-1.0)
        target_age: Target FIRE age for highlighting
        title: Chart title

    Returns:
        matplotlib Figure object
    """
    ages = sorted(age_probabilities.keys())
    probabilities = [age_probabilities[age] * 100 for age in ages]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot probability curve
    ax.plot(ages, probabilities, marker='o', linewidth=2,
            markersize=8, color='green', label='Success Probability')

    # Highlight target age
    if target_age in age_probabilities:
        target_prob = age_probabilities[target_age] * 100
        ax.axvline(target_age, color='red', linestyle='--',
                   alpha=0.7, label=f'Target Age ({target_age})')
        ax.axhline(target_prob, color='red', linestyle='--', alpha=0.5)
        ax.plot(target_age, target_prob, marker='*',
                markersize=15, color='red')

    # Formatting
    ax.set_xlabel("Age")
    ax.set_ylabel("FIRE Success Probability (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add percentage labels on points
    for age, prob in zip(ages, probabilities):
        ax.annotate(f'{prob:.0f}%',
                   (age, prob),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=9)

    plt.tight_layout()
    return fig
```

### Step 4: Update imports in test file

Add to top of `test/test_visualization.py`:

```python
from src.visualization import create_projection_fan_chart, create_fire_probability_chart
```

### Step 5: Run test to verify it passes

```bash
uv run pytest test/test_visualization.py::test_create_fire_probability_chart -v
```

**Expected:** PASS (1 passed)

### Step 6: Run all visualization tests

```bash
uv run pytest test/test_visualization.py -v
```

**Expected:** PASS (2 passed)

### Step 7: Commit

```bash
git add src/visualization.py test/test_visualization.py
git commit -m "$(cat <<'EOF'
feat: add FIRE probability chart visualization

- Implement create_fire_probability_chart()
- Show success probability curve by age
- Highlight target FIRE age with markers
- Add percentage labels on data points

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI Argument Parsing

**Files:**

- Create: `src/cli.py`
- Create: `test/test_cli.py`
- Modify: `main.py`

### Step 1: Write failing test for CLI parser

Create `test/test_cli.py`:

```python
import pytest
import argparse
from src.cli import create_argument_parser, parse_and_validate_args


def test_create_argument_parser():
    """Test CLI argument parser creation."""
    parser = create_argument_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    # Test default scenario
    args = parser.parse_args([])
    assert args.scenario == "scenarios/default.yaml"
    assert args.output is None
    assert args.no_charts is False


def test_parse_scenario_argument():
    """Test parsing scenario file argument."""
    parser = create_argument_parser()

    args = parser.parse_args(["--scenario", "custom.yaml"])
    assert args.scenario == "custom.yaml"


def test_parse_output_directory():
    """Test parsing output directory argument."""
    parser = create_argument_parser()

    args = parser.parse_args(["--output", "results/"])
    assert args.output == "results/"


def test_parse_no_charts_flag():
    """Test parsing no-charts flag."""
    parser = create_argument_parser()

    args = parser.parse_args(["--no-charts"])
    assert args.no_charts is True


def test_parse_verbose_flag():
    """Test parsing verbose flag."""
    parser = create_argument_parser()

    args = parser.parse_args(["-v"])
    assert args.verbose is True
```

### Step 2: Run test to verify it fails

```bash
uv run pytest test/test_cli.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'src.cli'"

### Step 3: Implement CLI parser

Create `src/cli.py`:

```python
"""Command-line interface for FIRE Forecast Engine."""
import argparse
import sys
from pathlib import Path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="FIRE Forecast Engine - Monte Carlo retirement projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default scenario
  %(prog)s

  # Run with custom scenario
  %(prog)s --scenario my_scenario.yaml

  # Save charts to directory
  %(prog)s --output results/

  # Skip chart generation (faster)
  %(prog)s --no-charts
        """
    )

    parser.add_argument(
        "-s", "--scenario",
        default="scenarios/default.yaml",
        help="Path to scenario YAML file (default: scenarios/default.yaml)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for saving charts (optional)"
    )

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (print results only)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def parse_and_validate_args(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Args:
        args: List of argument strings (None = sys.argv)

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If validation fails
    """
    parser = create_argument_parser()
    parsed = parser.parse_args(args)

    # Validate scenario file exists
    scenario_path = Path(parsed.scenario)
    if not scenario_path.exists():
        parser.error(f"Scenario file not found: {parsed.scenario}")

    # Validate output directory if specified
    if parsed.output:
        output_path = Path(parsed.output)
        if output_path.exists() and not output_path.is_dir():
            parser.error(f"Output path exists but is not a directory: {parsed.output}")

    return parsed
```

### Step 4: Run test to verify it passes

```bash
uv run pytest test/test_cli.py -v
```

**Expected:** PASS (5 passed)

### Step 5: Commit

```bash
git add src/cli.py test/test_cli.py
git commit -m "$(cat <<'EOF'
feat: add CLI argument parsing

- Implement create_argument_parser() with scenario, output, flags
- Add parse_and_validate_args() with file validation
- Support --scenario, --output, --no-charts, --verbose flags
- Include helpful usage examples in CLI help

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Integrate CLI with Main

**Files:**

- Modify: `main.py`
- Modify: `src/MonteCarloRunner.py` (add method to return trajectory data)

### Step 1: Update MonteCarloRunner to track trajectories

Add to `src/MonteCarloRunner.py` after the `SimulationResults` class:

```python
    def get_trajectories_by_age(self) -> list[dict[str, Any]]:
        """Return all simulation trajectories organized by year/age.

        Returns:
            List of dicts with year, age, portfolio_value for each simulation
        """
        if not self.all_runs:
            return []

        trajectories = []
        for run_data in self.all_runs:
            trajectory = []
            for year_idx, year_data in enumerate(run_data):
                trajectory.append({
                    "year": year_idx,
                    "age": self.profile_age + year_idx,
                    "portfolio_value": year_data.get("portfolio_value", 0),
                    "fire_achieved": year_data.get("fire_achieved", False)
                })
            trajectories.append(trajectory)

        return trajectories
```

### Step 2: Refactor main.py to use CLI and factories

Replace contents of `main.py`:

```python
"""FIRE Forecast Engine - Main entry point."""
from pathlib import Path
import sys

from src.cli import parse_and_validate_args
from src.ScenarioConfig import ScenarioConfig
from src.ScenarioFactory import ScenarioFactory
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.visualization import create_projection_fan_chart, create_fire_probability_chart


def main():
    """Main entry point for FIRE Forecast Engine."""
    # Parse command-line arguments
    args = parse_and_validate_args()

    if args.verbose:
        print(f"Loading scenario from: {args.scenario}")

    # Load configuration
    config = ScenarioConfig.from_yaml(args.scenario)
    factory = ScenarioFactory()

    # Create profile and strategy
    profile = factory.create_profile(config)
    strategy = factory.create_strategy(config.strategy_name)

    # Print profile summary
    print_profile_summary(profile, strategy)

    # Run Monte Carlo simulation
    n_sims = config.simulation_params.get("n_simulations", 10000)
    seed = config.simulation_params.get("seed", 42)

    if args.verbose:
        print(f"\nRunning {n_sims} Monte Carlo simulations...")

    engine = SimulationEngine(profile, strategy)
    runner = MonteCarloRunner(engine, n_simulations=n_sims, seed=seed)
    runner.run_simulations()
    results = runner.aggregate_results()

    # Print results
    print_results_summary(results, strategy)

    # Generate charts if requested
    if not args.no_charts:
        generate_charts(runner, results, profile, args.output, args.verbose)

    print("\nAnalysis complete!")


def print_profile_summary(profile, strategy):
    """Print financial profile summary."""
    print("=" * 60)
    print("  FIRE Forecast — Financial Profile")
    print("=" * 60)
    print(f"\n  Age: {profile.age} → Target: {profile.target_age}")
    print(f"  Years to FIRE: {profile.target_age - profile.age}")
    print(f"\n  Income:        ${profile.income:>12,.2f}")
    print(f"  Savings Rate:   {profile.savings_rate:>11.0%}")
    print(f"  Annual Savings: ${profile.annual_savings():>10,.2f}")
    print(f"\n  Strategy: {strategy.name}")
    print(f"  Risk Multiplier: {strategy.get_risk_multiplier()}x")
    print("=" * 60)


def print_results_summary(results, strategy):
    """Print Monte Carlo results summary."""
    print("\n" + "=" * 60)
    print("  Monte Carlo Results")
    print("=" * 60)
    print(f"\n  FIRE Success Rate: {results.success_rate:>8.1%}")

    if results.median_fire_age:
        print(f"  Median FIRE Age: {results.median_fire_age:>10}")

    print(f"\n  Portfolio Value Percentiles:")
    print(f"    10th: ${results.portfolio_percentiles[10]:>15,.2f}")
    print(f"    50th: ${results.portfolio_percentiles[50]:>15,.2f}")
    print(f"    90th: ${results.portfolio_percentiles[90]:>15,.2f}")
    print("=" * 60)


def generate_charts(runner, results, profile, output_dir, verbose):
    """Generate and save/show visualization charts."""
    import matplotlib.pyplot as plt

    if verbose:
        print("\nGenerating charts...")

    # Projection fan chart
    fig1 = create_projection_fan_chart(
        runner,
        title=f"Portfolio Projection (Age {profile.age} → {profile.target_age})"
    )

    # FIRE probability chart (calculate probabilities by age)
    age_probs = calculate_fire_probabilities_by_age(runner, profile)
    fig2 = create_fire_probability_chart(
        age_probs,
        target_age=profile.target_age,
        title="FIRE Success Probability by Age"
    )

    # Save or show charts
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig1.savefig(output_path / "projection_fan_chart.png", dpi=150)
        fig2.savefig(output_path / "fire_probability.png", dpi=150)

        print(f"\nCharts saved to: {output_dir}")
    else:
        plt.show()


def calculate_fire_probabilities_by_age(runner, profile):
    """Calculate FIRE success probability at each age."""
    age_probs = {}
    max_age = profile.target_age + 10

    for age in range(profile.age, max_age + 1, 5):
        year_idx = age - profile.age
        if year_idx >= len(runner.all_runs[0]):
            continue

        successes = sum(
            1 for run in runner.all_runs
            if run[year_idx].get("fire_achieved", False)
        )
        age_probs[age] = successes / len(runner.all_runs)

    return age_probs


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
```

### Step 3: Test the refactored main.py

```bash
uv run python main.py --help
```

**Expected:** Display help message with all CLI options

### Step 4: Test with default scenario

```bash
uv run python main.py --no-charts
```

**Expected:** Run successfully and display results without opening charts

### Step 5: Commit

```bash
git add main.py src/MonteCarloRunner.py
git commit -m "$(cat <<'EOF'
refactor: integrate CLI, factory, and visualization in main

- Replace hardcoded profile with YAML config loading
- Use ScenarioFactory for object creation
- Add CLI argument parsing (--scenario, --output, --no-charts)
- Generate and save/display charts based on flags
- Improve output formatting and error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update README with Usage Examples

**Files:**

- Modify: `README.md`

### Step 1: Read current README

```bash
cat README.md
```

**Expected:** See current README content (likely minimal)

### Step 2: Write comprehensive README

Replace `README.md`:

````markdown
# FIRE Forecast Engine

A Python CLI tool that runs Monte Carlo simulations to project the probability of achieving Financial Independence and Retire Early (FIRE) at various ages.

## Features

- **Monte Carlo Simulations**: Run thousands of market scenarios with randomized returns
- **Multiple Investment Strategies**: Aggressive, Balanced, Conservative approaches
- **YAML Configuration**: Define custom scenarios easily
- **Rich Visualizations**: Projection fan charts and probability curves
- **Sensitivity Analysis**: Test impact of changing savings rates
- **Factory Pattern Architecture**: Clean, extensible design

## Installation

Requires Python 3.13+

```bash
# Clone the repository
git clone <repository-url>
cd fire_forecast

# Install dependencies using uv
uv sync
```
````

## Quick Start

Run with the default scenario:

```bash
uv run python main.py
```

This will:

1. Load the default scenario from `scenarios/default.yaml`
2. Run 10,000 Monte Carlo simulations
3. Display results and show visualization charts

## Usage Examples

### Run with a custom scenario

```bash
uv run python main.py --scenario scenarios/my_scenario.yaml
```

### Save charts to a directory (no interactive display)

```bash
uv run python main.py --output results/
```

### Skip chart generation for faster results

```bash
uv run python main.py --no-charts
```

### Verbose output

```bash
uv run python main.py -v
```

## Creating Custom Scenarios

Create a YAML file in the `scenarios/` directory:

```yaml
profile:
  age: 30
  target_age: 50
  income: 75000
  expenses_rate: 0.60
  savings_rate: 0.40

portfolio:
  total_value: 25000
  allocation_method: balanced
  assets:
    - name: "Stock Index Fund"
      allocation: 0.70
      expected_return: 0.09
      volatility: 0.15
    - name: "Bond Index Fund"
      allocation: 0.30
      expected_return: 0.04
      volatility: 0.05

strategy: balanced # Options: aggressive, balanced, conservative

simulation:
  n_simulations: 10000
  seed: 42 # For reproducible results
```

Then run:

```bash
uv run python main.py --scenario scenarios/your_scenario.yaml
```

## Investment Strategies

- **Aggressive**: 1.5x risk multiplier, higher volatility tolerance, equity-focused
- **Balanced**: 1.1x risk multiplier, moderate approach, mixed allocation
- **Conservative**: 0.8x risk multiplier, capital preservation, bond-heavy

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest test/test_models.py -v
```

## Project Structure

```
fire_forecast/
├── src/
│   ├── models.py              # Domain models (Asset, Portfolio, Profile)
│   ├── SimulationEngine.py    # Core simulation engine
│   ├── MonteCarloRunner.py    # Monte Carlo orchestration
│   ├── SensitivityAnalyzer.py # Sensitivity analysis
│   ├── ScenarioConfig.py      # YAML config loader
│   ├── ScenarioFactory.py     # Factory for creating objects
│   ├── cli.py                 # CLI argument parsing
│   ├── visualization.py       # Chart generation
│   └── Strategy/
│       ├── base.py           # Strategy ABC
│       ├── aggressive.py
│       ├── balanced.py
│       └── conservative.py
├── test/                     # Pytest test suite
├── scenarios/                # YAML scenario definitions
├── docs/                     # Documentation
└── main.py                   # CLI entry point
```

## Design Patterns

This project demonstrates several OOP design patterns:

- **Composition over Inheritance**: Portfolio contains Assets, Profile contains Portfolio
- **Strategy Pattern**: Swappable investment strategies
- **Template Method**: Simulation engine lifecycle
- **Factory Pattern**: Scenario creation from configuration

## Development Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for the phased development plan.

## License

MIT License - see LICENSE file for details

## Contributing

This is a personal learning project demonstrating OOP principles in Python. Feel free to fork and experiment!

````

### Step 3: Verify README renders correctly

```bash
head -50 README.md
````

**Expected:** See formatted README content

### Step 4: Commit

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: add comprehensive README with usage examples

- Add installation and quick start instructions
- Document CLI usage with examples
- Explain custom scenario creation
- Describe investment strategies
- Include project structure and design patterns
- Add testing instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Code Cleanup and Final Refactor

**Files:**

- Review all files for: unused imports, docstring completeness, type hints
- Modify: Any files needing cleanup

### Step 1: Run linter/formatter check (if available)

```bash
# Check if ruff is available, otherwise skip
which ruff || echo "Ruff not installed, manual review needed"
```

### Step 2: Add docstrings to visualization.py

Ensure all functions in `src/visualization.py` have complete docstrings with Args/Returns sections (already done in earlier steps, verify):

```bash
grep -n "def " src/visualization.py
```

**Expected:** See both function definitions with docstrings

### Step 3: Add type hints to cli.py

Verify type hints are present in `src/cli.py` (already done, verify):

```bash
grep "def " src/cli.py
```

**Expected:** See return type annotations on functions

### Step 4: Check for unused imports across codebase

```bash
# Manual review or use tool
for file in src/*.py; do
  echo "Checking $file"
  # Review imports manually
done
```

### Step 5: Verify all tests pass

```bash
uv run pytest -v
```

**Expected:** ALL tests pass

### Step 6: Commit cleanup changes

```bash
git add -u
git commit -m "$(cat <<'EOF'
refactor: code cleanup and documentation polish

- Verify all docstrings complete with Args/Returns
- Ensure type hints on all public functions
- Remove unused imports
- Verify test coverage

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Integration Test - End-to-End

**Files:**

- Create: `test/test_integration.py`

### Step 1: Write integration test

Create `test/test_integration.py`:

```python
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
from src.visualization import create_projection_fan_chart, create_fire_probability_chart


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
  allocation_method: balanced
  assets:
    - name: "Stocks"
      allocation: 0.70
      expected_return: 0.08
      volatility: 0.12
    - name: "Bonds"
      allocation: 0.30
      expected_return: 0.04
      volatility: 0.05
strategy: balanced
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
    strategy = factory.create_strategy(config.strategy_name)

    # Run simulation
    engine = SimulationEngine(profile, strategy)
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

    # Calculate age probabilities
    age_probs = {}
    for age in range(30, 55, 5):
        year_idx = age - 30
        if year_idx < len(runner.all_runs[0]):
            successes = sum(
                1 for run in runner.all_runs
                if run[year_idx].get("fire_achieved", False)
            )
            age_probs[age] = successes / len(runner.all_runs)

    fig2 = create_fire_probability_chart(age_probs, target_age=50)
    assert fig2 is not None

    print(f"✓ Integration test passed: {results.success_rate:.1%} success rate")
```

### Step 2: Run integration test

```bash
uv run pytest test/test_integration.py -v -s
```

**Expected:** PASS with success rate printed

### Step 3: Run full test suite

```bash
uv run pytest -v
```

**Expected:** ALL tests pass

### Step 4: Commit

```bash
git add test/test_integration.py
git commit -m "$(cat <<'EOF'
test: add end-to-end integration test

- Test full workflow from YAML to charts
- Verify config loading, factory creation, simulation, visualization
- Ensure all components work together correctly

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final Verification and Documentation

**Files:**

- Create: `docs/USER_GUIDE.md`
- Modify: `docs/ROADMAP.md` (mark Week 4 complete)

### Step 1: Create user guide

Create `docs/USER_GUIDE.md`:

```markdown
# FIRE Forecast Engine - User Guide

## Introduction

The FIRE Forecast Engine helps you understand the probability of achieving Financial Independence through Monte Carlo simulation. This guide walks through using the tool effectively.

## Understanding the Output

### FIRE Success Rate

The percentage of simulations where you achieved FIRE (portfolio ≥ 25× annual expenses) by your target age.

- **≥80%**: High confidence
- **60-80%**: Moderate confidence
- **<60%**: May need to adjust plan

### Percentiles

- **10th percentile**: Pessimistic scenario (1 in 10 worse)
- **50th (median)**: Typical outcome
- **90th percentile**: Optimistic scenario (1 in 10 better)

## Customizing Your Scenario

### Profile Settings

- `age`: Current age
- `target_age`: Desired FIRE age
- `income`: Annual gross income
- `expenses_rate`: Percentage of income spent (0.0-1.0)
- `savings_rate`: Percentage of income saved (0.0-1.0)

**Note**: `expenses_rate + savings_rate` must equal ~1.0

### Portfolio Assets

Each asset needs:

- `name`: Descriptive name
- `allocation`: Percentage of portfolio (must sum to 1.0)
- `expected_return`: Expected annual return (e.g., 0.08 = 8%)
- `volatility`: Standard deviation of returns (higher = more variable)

**Common ETF values**:

- Stock index (VTI): return 0.10, volatility 0.15
- Bond index (BND): return 0.04, volatility 0.04

### Strategy Selection

- **Aggressive**: For younger investors with high risk tolerance
- **Balanced**: For moderate risk tolerance
- **Conservative**: For near-retirement or low risk tolerance

## Interpreting Charts

### Projection Fan Chart

Shows portfolio value over time with percentile bands:

- Blue shaded areas show range of likely outcomes
- Darker band (25th-75th) = 50% of scenarios fall here
- Lighter band (10th-90th) = 80% of scenarios
- Blue line = median outcome

### FIRE Probability Chart

Shows success probability at different ages:

- Green line trends upward over time
- Red star marks your target age
- Percentages show likelihood of FIRE by that age

## Tips for Better Projections

1. **Be conservative**: Use realistic expense and return assumptions
2. **Run many simulations**: 10,000+ for stable results
3. **Test sensitivity**: Try different savings rates to see impact
4. **Account for inflation**: Use real (inflation-adjusted) returns
5. **Plan for contingencies**: 80% success rate leaves margin for error

## Common Questions

**Q: Why does success rate change each run?**
A: Monte Carlo uses randomization. Use `seed` parameter for reproducible results.

**Q: Should I use historical returns?**
A: Historical returns (S&P 500 ~10% nominal, ~7% real) are a reasonable baseline, but past performance doesn't guarantee future results.

**Q: What if my target age shows <50% success?**
A: Consider increasing savings rate, adjusting target age, or using a more aggressive strategy.

**Q: How do I account for Social Security or pension?**
A: Reduce your `expenses_rate` to reflect the portion covered by guaranteed income.

## Next Steps

1. Create your scenario YAML based on `scenarios/default.yaml`
2. Run simulation: `uv run python main.py --scenario your_scenario.yaml`
3. Analyze results and adjust parameters
4. Run sensitivity analysis on key variables
5. Save charts for future reference: `--output results/`

For technical details, see the main [README.md](../README.md).
```

### Step 2: Update ROADMAP to mark Week 4 complete

```bash
# This would update the ROADMAP.md to check off Week 4 items
# We'll do this as the final step after all tasks are complete
```

### Step 3: Test the full application one more time

```bash
uv run python main.py --scenario scenarios/default.yaml --no-charts -v
```

**Expected:** Successful run with verbose output

### Step 4: Commit user guide

```bash
git add docs/USER_GUIDE.md
git commit -m "$(cat <<'EOF'
docs: add comprehensive user guide

- Explain output metrics and percentiles
- Guide for customizing scenarios
- Chart interpretation instructions
- Tips for better projections
- FAQ section

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

### Step 5: Final commit marking Week 4 complete

```bash
git add docs/ROADMAP.md
# Update ROADMAP.md to check Week 4 items
git commit -m "$(cat <<'EOF'
chore: mark Week 4 (Polish & Visualize) as complete

- YAML configuration ✓
- ScenarioFactory ✓
- Matplotlib visualizations ✓
- CLI argument parsing ✓
- README and user guide ✓
- Code cleanup ✓

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements Week 4 of the FIRE Forecast Engine roadmap:

1. **YAML Configuration** - Load scenarios from config files
2. **Factory Pattern** - Create domain objects from configs
3. **Visualizations** - Projection fan charts and FIRE probability curves
4. **CLI Interface** - Professional argument parsing with validation
5. **Documentation** - Comprehensive README and user guide
6. **Testing** - Unit tests for each component plus integration test
7. **Code Quality** - Cleanup, docstrings, type hints

**Total estimated time**: 6-8 hours across 10 tasks

Each task follows TDD: write test → verify failure → implement → verify success → commit

**Testing strategy**: Unit tests for each module, integration test for end-to-end workflow

**Deployment**: Application is now production-ready with configuration, CLI, charts, and docs

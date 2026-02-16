# Cost Modeling Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add realistic cost modeling (TER and trading fees) to FIRE simulations while completely removing the strategy pattern from the codebase.

**Architecture:** Extend Asset model with optional cost structure, modify SimulationEngine to calculate returns directly and apply costs at appropriate points (TER annually on portfolio balance, trading fees on monthly contributions), systematically remove all Strategy pattern code—different scenarios use different YAML files with different allocations.

**Tech Stack:** Python 3.13+, dataclasses, pytest, YAML

---

## Task 1: Add Optional Cost Fields to Asset Model

**Files:**
- Modify: `src/models.py:6-10`
- Test: `test/test_models.py`

**Step 1: Write the failing test**

Add to `test/test_models.py` in TestAsset class:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_models.py::TestAsset::test_asset_creation_with_costs -v`

Expected: FAIL with "Asset.__init__() got an unexpected keyword argument 'costs'"

**Step 3: Update Asset model**

In `src/models.py`, modify the Asset class (lines 6-10):

```python
@dataclass
class Asset:
    name: str
    allocation: float
    expected_return: float
    volatility: float
    costs: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None
```

**Step 4: Run tests**

Run: `uv run pytest test/test_models.py::TestAsset -v`

Expected: All TestAsset tests PASS

**Step 5: Commit**

```bash
git add src/models.py test/test_models.py
git commit -m "feat: add optional costs and risk_metrics to Asset model"
```

---

## Task 2: Add Cost Utilities

**Files:**
- Create: `src/cost_calculator.py`
- Test: `test/test_cost_calculator.py`

**Step 1: Write tests**

Create `test/test_cost_calculator.py`:

```python
import pytest
from src.cost_calculator import apply_ter, apply_trading_fee


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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_cost_calculator.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement cost calculator**

Create `src/cost_calculator.py`:

```python
"""Cost calculation utilities for investment fees and expenses."""


def apply_ter(portfolio_value: float, ter: float) -> float:
    """
    Apply Total Expense Ratio (TER) to portfolio value.

    Args:
        portfolio_value: Current portfolio value
        ter: Total Expense Ratio as decimal (e.g., 0.0019 for 0.19%)

    Returns:
        Portfolio value after TER deduction
    """
    return portfolio_value * (1 - ter)


def apply_trading_fee(contribution: float, trading_fee: float) -> float:
    """
    Apply trading fee to a contribution/purchase amount.

    Args:
        contribution: Amount being invested
        trading_fee: Trading fee as decimal (e.g., 0.0005 for 0.05%)

    Returns:
        Net contribution after trading fee deduction
    """
    return contribution * (1 - trading_fee)
```

**Step 4: Run tests**

Run: `uv run pytest test/test_cost_calculator.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/cost_calculator.py test/test_cost_calculator.py
git commit -m "feat: add cost calculation utilities"
```

---

## Task 3: Add Cost Validation

**Files:**
- Create: `src/validators.py`
- Test: `test/test_validators.py`
- Modify: `src/models.py`

**Step 1: Write tests**

Create `test/test_validators.py`:

```python
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
```

**Step 2: Run test**

Run: `uv run pytest test/test_validators.py -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement validators**

Create `src/validators.py`:

```python
"""Validation utilities for financial models."""

from typing import Dict, Optional


def validate_asset_costs(costs: Optional[Dict[str, float]]) -> None:
    """
    Validate asset cost structure.

    Args:
        costs: Dictionary with "ter" and/or "trading_fee" keys

    Raises:
        ValueError: If costs are invalid (negative or unreasonably high)
    """
    if costs is None:
        return

    if "ter" in costs:
        ter = costs["ter"]
        if not (0 <= ter <= 0.05):
            raise ValueError(f"TER must be between 0 and 0.05 (0-5%), got {ter}")

    if "trading_fee" in costs:
        trading_fee = costs["trading_fee"]
        if not (0 <= trading_fee <= 0.05):
            raise ValueError(
                f"trading_fee must be between 0 and 0.05 (0-5%), got {trading_fee}"
            )
```

**Step 4: Add validation to Asset model**

In `src/models.py`, add to Asset class:

```python
@dataclass
class Asset:
    name: str
    allocation: float
    expected_return: float
    volatility: float
    costs: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        from .validators import validate_asset_costs
        validate_asset_costs(self.costs)
```

**Step 5: Add test for invalid Asset creation**

Add to `test/test_models.py` in TestAsset class:

```python
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
```

**Step 6: Run tests**

Run: `uv run pytest test/test_validators.py test/test_models.py::TestAsset -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add src/validators.py src/models.py test/test_validators.py test/test_models.py
git commit -m "feat: add cost validation to Asset model"
```

---

## Task 4: Remove Strategy from SimulationEngine

**Files:**
- Modify: `src/SimulationEngine.py`
- Test: `test/test_simulationEngine.py`

**Step 1: Write test without strategy**

Add to `test/test_simulationEngine.py`:

```python
import numpy as np
import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine


@pytest.fixture
def simple_asset():
    return Asset(
        name="Test Asset",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.10
    )


@pytest.fixture
def simple_portfolio(simple_asset):
    return Portfolio(
        composition=[simple_asset],
        total_value=100000.0,
        allocation_methods="100% test"
    )


@pytest.fixture
def simple_profile(simple_portfolio):
    return FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=simple_portfolio,
        age=30,
        target_age=31
    )


class TestSimulationEngineWithoutStrategy:
    def test_engine_initializes_without_strategy(self, simple_profile):
        """SimulationEngine can be created without strategy parameter"""
        engine = SimulationEngine(simple_profile)
        assert engine.profile == simple_profile
        assert not hasattr(engine, 'strategy')

    def test_calculate_portfolio_return_uses_asset_volatility(self, simple_profile):
        """Returns are calculated using asset expected_return and volatility"""
        np.random.seed(42)
        engine = SimulationEngine(simple_profile)

        # Generate returns to verify distribution
        returns = [engine._calculate_portfolio_return() for _ in range(1000)]

        import statistics
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        # Mean should be close to expected_return (0.07)
        assert 0.06 < mean_return < 0.08

        # Std dev should be close to volatility (0.10)
        assert 0.09 < std_return < 0.11
```

**Step 2: Run test**

Run: `uv run pytest test/test_simulationEngine.py::TestSimulationEngineWithoutStrategy -v`

Expected: FAIL - SimulationEngine still requires strategy

**Step 3: Rewrite SimulationEngine without strategy**

In `src/SimulationEngine.py`, replace entire file:

```python
from typing import List, Optional
import numpy as np

from .models import FinancialProfile


class SimulationEngine:
    """
    Runs a single financial independence simulation using the Template Method pattern.

    The run() method defines the simulation lifecycle:
    1. setup() - initialize state from profile
    2. simulate_year() - apply returns, contributions (loop until target age)
    3. collect_results() - gather final outcomes
    """

    def __init__(self, profile: FinancialProfile):
        self.profile = profile

        # Simulation state (initialized in setup())
        self.current_portfolio_value: float = 0.0
        self.current_age: int = 0
        self.year_count: int = 0

        self.portfolio_history: List[float] = []
        self.fire_age: Optional[int] = None
        self.fire_target = self.profile.annual_expenses() / 0.04

    def run(self) -> dict:
        """
        Template Method: orchestrates the simulation lifecycle.

        Returns:
            dict with simulation results
        """
        self.setup()

        while self.current_age < self.profile.target_age:
            self.simulate_year()

        return self.collect_results()

    def setup(self) -> None:
        """Initialize simulation state from the financial profile."""
        self.current_portfolio_value = self.profile.portfolio.total_value
        self.current_age = self.profile.age
        self.year_count = 0

    def _calculate_portfolio_return(self) -> float:
        """
        Calculate weighted average return with volatility for the portfolio.

        Generates a randomized return using normal distribution based on
        each asset's expected return and volatility.

        Returns:
            The portfolio return for this year (e.g., 0.08 for 8%)
        """
        total_return = 0.0

        for asset in self.profile.portfolio.composition:
            # Base expected return
            expected = asset.expected_return

            # Add randomness based on asset's natural volatility
            random_shock = np.random.normal(0, asset.volatility)

            # Weighted by allocation
            asset_return = (expected + random_shock) * asset.allocation
            total_return += asset_return

        return total_return

    def simulate_year(self) -> None:
        """
        Simulate one year of the financial journey.

        Steps:
        1. Calculate annual return
        2. Apply return to portfolio
        3. Add annual savings
        4. Increment age and year counter
        """
        annual_return = self._calculate_portfolio_return()
        self.current_portfolio_value *= 1 + annual_return
        self.current_portfolio_value += self.profile.annual_savings()
        self.current_age += 1
        self.year_count += 1

        self.portfolio_history.append(self.current_portfolio_value)

        if self.current_portfolio_value >= self.fire_target and self.fire_age is None:
            self.fire_age = self.current_age

    def collect_results(self) -> dict:
        """
        Gather final simulation outcomes.

        Returns:
            dict containing simulation results
        """
        return {
            "final_portfolio_value": self.current_portfolio_value,
            "fire_target": self.fire_target,
            "fire_achieved": True if self.fire_age else False,
            "years_simulated": self.year_count,
            "final_age": self.current_age,
            "fire_age": self.fire_age,
            "portfolio_history": self.portfolio_history,
        }

    def reset(self):
        """Reset simulation state for next run."""
        self.current_portfolio_value = 0.0
        self.current_age = 0
        self.year_count = 0
        self.portfolio_history = []
        self.fire_age = None
```

**Step 4: Run tests**

Run: `uv run pytest test/test_simulationEngine.py::TestSimulationEngineWithoutStrategy -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
git add src/SimulationEngine.py test/test_simulationEngine.py
git commit -m "refactor: remove strategy pattern from SimulationEngine"
```

---

## Task 5: Add Cost Application to SimulationEngine

**Files:**
- Modify: `src/SimulationEngine.py:simulate_year()`
- Test: `test/test_simulationEngine.py`

**Step 1: Write tests for costs**

Add to `test/test_simulationEngine.py`:

```python
@pytest.fixture
def asset_with_costs():
    """Asset with TER and trading fees"""
    return Asset(
        name="VWRA",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.0862,
        costs={"ter": 0.0019, "trading_fee": 0.0005}
    )


@pytest.fixture
def portfolio_with_costs(asset_with_costs):
    return Portfolio(
        composition=[asset_with_costs],
        total_value=100000.0,
        allocation_methods="100% VWRA"
    )


@pytest.fixture
def profile_with_costs(portfolio_with_costs):
    return FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio_with_costs,
        age=30,
        target_age=31
    )


class TestSimulationEngineWithCosts:
    def test_ter_reduces_portfolio_value(self, profile_with_costs):
        """TER is applied annually to portfolio value"""
        np.random.seed(42)
        engine = SimulationEngine(profile_with_costs)
        engine._calculate_portfolio_return = lambda: 0.0  # Zero return for testing

        result = engine.run()

        # Expected: 100k + net_savings - TER
        annual_savings = 24000.0  # 60k * 0.4
        net_savings = annual_savings * (1 - 0.0005)  # After trading fee
        expected_before_ter = 100000.0 + net_savings
        expected_after_ter = expected_before_ter * (1 - 0.0019)

        assert abs(result["final_portfolio_value"] - expected_after_ter) < 1.0

    def test_trading_fee_reduces_contributions(self, profile_with_costs):
        """Trading fee is applied to contributions"""
        np.random.seed(42)
        engine = SimulationEngine(profile_with_costs)
        engine._calculate_portfolio_return = lambda: 0.0

        result = engine.run()

        # Verify trading fee was applied
        annual_savings = 24000.0
        fee = annual_savings * 0.0005
        assert fee == 12.0  # Sanity check
```

**Step 2: Run test**

Run: `uv run pytest test/test_simulationEngine.py::TestSimulationEngineWithCosts -v`

Expected: FAIL - costs not applied

**Step 3: Modify simulate_year to apply costs**

In `src/SimulationEngine.py`, update `simulate_year` method:

```python
def simulate_year(self) -> None:
    """
    Simulate one year of the financial journey.

    Steps:
    1. Calculate annual savings and apply trading fees
    2. Add net savings to portfolio
    3. Calculate annual return
    4. Apply return to portfolio
    5. Apply TER costs
    6. Record state and check FIRE
    """
    from .cost_calculator import apply_ter, apply_trading_fee

    # Step 1: Calculate net savings after trading fees
    annual_savings = self.profile.annual_savings()
    net_savings = annual_savings

    for asset in self.profile.portfolio.composition:
        if asset.costs and "trading_fee" in asset.costs:
            asset_contribution = annual_savings * asset.allocation
            trading_fee = asset.costs["trading_fee"]
            net_asset_contribution = apply_trading_fee(asset_contribution, trading_fee)
            net_savings -= (asset_contribution - net_asset_contribution)

    # Step 2: Add net savings
    self.current_portfolio_value += net_savings

    # Step 3: Calculate and apply return
    annual_return = self._calculate_portfolio_return()
    self.current_portfolio_value *= (1 + annual_return)

    # Step 4: Apply TER costs
    for asset in self.profile.portfolio.composition:
        if asset.costs and "ter" in asset.costs:
            ter = asset.costs["ter"]
            asset_value = self.current_portfolio_value * asset.allocation
            ter_cost = asset_value * ter
            self.current_portfolio_value -= ter_cost

    # Step 5: Record and check FIRE
    self.current_age += 1
    self.year_count += 1
    self.portfolio_history.append(self.current_portfolio_value)

    if self.current_portfolio_value >= self.fire_target and self.fire_age is None:
        self.fire_age = self.current_age
```

**Step 4: Run tests**

Run: `uv run pytest test/test_simulationEngine.py::TestSimulationEngineWithCosts -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
git add src/SimulationEngine.py test/test_simulationEngine.py
git commit -m "feat: apply TER and trading fees in simulation"
```

---

## Task 6: Remove Strategy from ScenarioConfig

**Files:**
- Modify: `src/ScenarioConfig.py`
- Test: `test/test_scenario_config.py`

**Step 1: Update ScenarioConfig class**

In `src/ScenarioConfig.py`, modify the dataclass:

```python
@dataclass
class ScenarioConfig:
    """Represents a loaded scenario configuration."""

    profile_data: dict[str, Any]
    portfolio_data: dict[str, Any]
    simulation_params: dict[str, Any]  # Removed strategy_name

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
            simulation_params=data.get("simulation", {})
            # Removed strategy_name parsing
        )
```

**Step 2: Update tests**

In `test/test_scenario_config.py`, remove any tests that reference `strategy_name` field.

**Step 3: Run tests**

Run: `uv run pytest test/test_scenario_config.py -v`

Expected: Tests PASS

**Step 4: Commit**

```bash
git add src/ScenarioConfig.py test/test_scenario_config.py
git commit -m "refactor: remove strategy_name from ScenarioConfig"
```

---

## Task 7: Remove Strategy from ScenarioFactory

**Files:**
- Modify: `src/ScenarioFactory.py`
- Test: `test/test_scenario_factory.py`

**Step 1: Update ScenarioFactory to remove strategy imports and methods**

In `src/ScenarioFactory.py`:

```python
"""Factory for creating domain objects from scenario configurations."""
from src.ScenarioConfig import ScenarioConfig
from src.models import Asset, Portfolio, FinancialProfile


class ScenarioFactory:
    """Factory for creating scenarios from configuration."""

    def create_profile(self, config: ScenarioConfig) -> FinancialProfile:
        """Create FinancialProfile from config."""
        # Create assets with optional costs
        assets = []
        for asset_data in config.portfolio_data["assets"]:
            asset = Asset(
                name=asset_data["name"],
                allocation=asset_data["allocation"],
                expected_return=asset_data["expected_return"],
                volatility=asset_data["volatility"],
                costs=asset_data.get("costs"),  # Optional
                risk_metrics=asset_data.get("risk_metrics")  # Optional
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

    # Removed create_strategy method entirely
```

**Step 2: Update tests**

In `test/test_scenario_factory.py`, remove any tests for `create_strategy` method.

**Step 3: Run tests**

Run: `uv run pytest test/test_scenario_factory.py -v`

Expected: Tests PASS

**Step 4: Commit**

```bash
git add src/ScenarioFactory.py test/test_scenario_factory.py
git commit -m "refactor: remove strategy creation from ScenarioFactory"
```

---

## Task 8: Update MonteCarloRunner (No Strategy Changes Needed)

**Files:**
- Verify: `src/MonteCarloRunner.py`
- Test: `test/test_monte_carlo.py`

**Step 1: Verify MonteCarloRunner doesn't reference strategy**

Run: `grep -n "strategy" src/MonteCarloRunner.py`

Expected: No matches (good!)

**Step 2: Update tests to not pass strategy to SimulationEngine**

In `test/test_monte_carlo.py`, find lines like:
```python
engine = SimulationEngine(profile, strategy)
```

Replace with:
```python
engine = SimulationEngine(profile)
```

**Step 3: Run tests**

Run: `uv run pytest test/test_monte_carlo.py -v`

Expected: Tests PASS

**Step 4: Commit**

```bash
git add test/test_monte_carlo.py
git commit -m "test: remove strategy from MonteCarloRunner tests"
```

---

## Task 9: Update SensitivityAnalyzer

**Files:**
- Modify: `src/SensitivityAnalyzer.py`
- Test: `test/test_sensitivity.py`

**Step 1: Check current implementation**

Run: `grep -n "strategy" src/SensitivityAnalyzer.py`

Expected: Lines that reference strategy

**Step 2: Update SensitivityAnalyzer to remove strategy**

In `src/SensitivityAnalyzer.py`, find code that creates/uses strategies and remove those references. Update to create SimulationEngine without strategy:

```python
engine = SimulationEngine(profile)  # Remove strategy parameter
```

**Step 3: Update tests**

In `test/test_sensitivity.py`, remove strategy imports and usage.

**Step 4: Run tests**

Run: `uv run pytest test/test_sensitivity.py -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
git add src/SensitivityAnalyzer.py test/test_sensitivity.py
git commit -m "refactor: remove strategy from SensitivityAnalyzer"
```

---

## Task 10: Delete Strategy Directory and Tests

**Files:**
- Delete: `src/Strategy/` (entire directory)
- Delete: `test/test_strategies.py`

**Step 1: Delete Strategy directory**

Run: `rm -rf src/Strategy/`

**Step 2: Delete strategy tests**

Run: `rm test/test_strategies.py`

**Step 3: Update integration tests**

In `test/test_integration.py`, remove strategy imports and update SimulationEngine calls:

```python
# Remove: from src.Strategy.balanced import BalancedStrategy
# Change: engine = SimulationEngine(profile, BalancedStrategy())
# To: engine = SimulationEngine(profile)
```

**Step 4: Run tests**

Run: `uv run pytest -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete Strategy pattern code entirely"
```

---

## Task 11: Update main.py

**Files:**
- Modify: `main.py`

**Step 1: Remove strategy imports**

In `main.py`, find and remove lines like:
```python
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy
```

**Step 2: Update SimulationEngine calls**

Find:
```python
engine = SimulationEngine(profile, BalancedStrategy())
```

Replace with:
```python
engine = SimulationEngine(profile)
```

**Step 3: Remove strategy comparisons**

If main.py compares multiple strategies, remove that logic entirely. Different scenarios should be in different YAML files.

**Step 4: Run main.py**

Run: `uv run main.py`

Expected: Program runs successfully

**Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: remove strategy usage from main.py"
```

---

## Task 12: Update YAML Scenario Files

**Files:**
- Modify: `scenarios/basic.yaml`
- Modify: `scenarios/default.yaml`
- Modify: `scenarios/example.yaml`
- Create: `scenarios/vwra-with-costs.yaml`

**Step 1: Update basic.yaml**

Modify `scenarios/basic.yaml`:

```yaml
profile:
  age: 25
  target_age: 50
  income: 75000
  expenses_rate: 0.75
  savings_rate: 0.25

portfolio:
  total_value: 0.0
  assets:
    - name: "VWRA"
      allocation: 0.75
      expected_return: 0.07
      volatility: 0.0862
      costs:
        ter: 0.0019
        trading_fee: 0.0005
      risk_metrics:
        semi_deviation: 0.0278
        downside_deviation: 0.0012
        correlation: 0.99
    - name: "IGLN"
      allocation: 0.25
      expected_return: 0.04
      volatility: 0.04

simulation:
  n_simulations: 10000
  seed: 42
```

**Step 2: Remove strategy field from all YAML files**

In `scenarios/basic.yaml`, `scenarios/default.yaml`, `scenarios/example.yaml`:
- Delete line: `strategy: balanced`

**Step 3: Create comprehensive example**

Create `scenarios/vwra-with-costs.yaml`:

```yaml
profile:
  age: 30
  target_age: 55
  income: 100000
  expenses_rate: 0.6
  savings_rate: 0.4

portfolio:
  total_value: 50000.0
  assets:
    - name: "VWRA (Vanguard FTSE All-World)"
      allocation: 0.70
      expected_return: 0.08
      volatility: 0.0862
      costs:
        ter: 0.0019
        trading_fee: 0.0005
      risk_metrics:
        semi_deviation: 0.0278
        downside_deviation: 0.0012
        correlation: 0.99
    - name: "IGLN (iShares Gold)"
      allocation: 0.20
      expected_return: 0.03
      volatility: 0.06
      costs:
        ter: 0.0025
        trading_fee: 0.0005
    - name: "Cash Reserve"
      allocation: 0.10
      expected_return: 0.02
      volatility: 0.01

simulation:
  n_simulations: 10000
  seed: 42
```

**Step 4: Verify YAML files**

Run: `python -c "import yaml; yaml.safe_load(open('scenarios/basic.yaml'))"`

Expected: No errors

**Step 5: Commit**

```bash
git add scenarios/*.yaml
git commit -m "docs: update YAML scenarios with costs, remove strategy field"
```

---

## Task 13: Integration Tests

**Files:**
- Test: `test/test_integration.py`

**Step 1: Add comprehensive integration tests**

Add to `test/test_integration.py`:

```python
import numpy as np
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine


def test_full_simulation_with_costs():
    """End-to-end simulation with cost-bearing assets"""
    np.random.seed(42)

    vwra = Asset(
        name="VWRA",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.0862,
        costs={"ter": 0.0019, "trading_fee": 0.0005}
    )

    portfolio = Portfolio(
        composition=[vwra],
        total_value=10000.0,
        allocation_methods="100% VWRA"
    )

    profile = FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio,
        age=30,
        target_age=35
    )

    engine = SimulationEngine(profile)
    result = engine.run()

    assert result["years_simulated"] == 5
    assert result["final_age"] == 35
    assert result["final_portfolio_value"] > 0
    assert len(result["portfolio_history"]) == 5


def test_cost_impact_comparison():
    """Compare simulations with and without costs"""
    asset_no_cost = Asset(
        name="No Cost Asset",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.0862
    )

    asset_with_cost = Asset(
        name="Cost Asset",
        allocation=1.0,
        expected_return=0.07,
        volatility=0.0862,
        costs={"ter": 0.0019, "trading_fee": 0.0005}
    )

    portfolio_no_cost = Portfolio(
        composition=[asset_no_cost],
        total_value=100000.0,
        allocation_methods="100%"
    )

    portfolio_with_cost = Portfolio(
        composition=[asset_with_cost],
        total_value=100000.0,
        allocation_methods="100%"
    )

    profile_no_cost = FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio_no_cost,
        age=30,
        target_age=60
    )

    profile_with_cost = FinancialProfile(
        income=60000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio_with_cost,
        age=30,
        target_age=60
    )

    np.random.seed(42)
    engine_no_cost = SimulationEngine(profile_no_cost)
    result_no_cost = engine_no_cost.run()

    np.random.seed(42)
    engine_with_cost = SimulationEngine(profile_with_cost)
    result_with_cost = engine_with_cost.run()

    # Verify cost-bearing portfolio has lower final value
    assert result_with_cost["final_portfolio_value"] < result_no_cost["final_portfolio_value"]

    # At least 3% impact over 30 years
    cost_impact_pct = (
        result_no_cost["final_portfolio_value"] - result_with_cost["final_portfolio_value"]
    ) / result_no_cost["final_portfolio_value"]
    assert cost_impact_pct > 0.03
```

**Step 2: Run tests**

Run: `uv run pytest test/test_integration.py -v`

Expected: Tests PASS

**Step 3: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add test/test_integration.py
git commit -m "test: add integration tests for cost modeling"
```

---

## Task 14: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update README**

Add cost modeling section to README.md:

```markdown
### Cost Modeling

The simulation engine models realistic investment costs:

**Total Expense Ratio (TER):**
- Applied annually to portfolio balance
- Example: 0.19% TER on VWRA reduces wealth by ~5.5% over 30 years

**Trading Fees:**
- Applied to each contribution (monthly DCA)
- Example: 0.05% per trade

**YAML Configuration:**
```yaml
assets:
  - name: "VWRA"
    allocation: 0.75
    expected_return: 0.07
    volatility: 0.0862
    costs:
      ter: 0.0019  # 0.19% annual
      trading_fee: 0.0005  # 0.05% per trade
    risk_metrics:  # Optional
      semi_deviation: 0.0278
      correlation: 0.99
```

Costs are optional—assets without costs assume zero fees.

Different risk levels are modeled by creating different YAML scenario files (e.g., `aggressive.yaml` with 90% stocks, `conservative.yaml` with 50% bonds).
```

**Step 2: Update CLAUDE.md**

Update architecture sections in `CLAUDE.md`:

```markdown
**Implemented domain models** in `src/models.py`:
- `Asset` — individual investment (name, allocation, expected return, volatility)
  - Optional `costs` dict: `{"ter": 0.0019, "trading_fee": 0.0005}`
  - Optional `risk_metrics` dict: for advanced analysis
  - Validates costs on creation (0-5% range)
- `Portfolio` — composition of Assets
- `FinancialProfile` — complete financial situation

**Implemented simulation engine** in `src/SimulationEngine.py`:
- `SimulationEngine` — runs single simulation with cost modeling
- Calculates returns directly from asset volatility and expected returns
- **Applies realistic costs:**
  - TER: annual drag on portfolio balance
  - Trading fees: applied to contributions
- Calculates FIRE achievement using 4% rule
```

Remove strategy pattern section:

```markdown
~~**Implemented investment strategies**~~
- Strategy pattern removed for simplicity
- Different risk levels handled via different YAML scenario files
```

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update documentation for cost modeling and strategy removal"
```

---

## Task 15: Final Verification

**Files:**
- All files

**Step 1: Run complete test suite**

Run: `uv run pytest -v`

Expected: All tests PASS

**Step 2: Verify no strategy references remain**

Run: `grep -r "from src.Strategy" src/ test/`

Expected: No matches

Run: `grep -r "Strategy\|strategy" src/ | grep -v "# " | grep -v Binary`

Expected: Only references in comments or docstrings

**Step 3: Run main.py**

Run: `uv run main.py`

Expected: Program runs successfully

**Step 4: Check test coverage**

Run: `uv run pytest --cov=src --cov-report=term-missing`

Expected: Good coverage on new modules

**Step 5: Final commit**

```bash
git add .
git commit -m "chore: final verification and cleanup"
```

---

## Completion Checklist

- [ ] Task 1: Asset model extended with costs
- [ ] Task 2: Cost calculator utilities
- [ ] Task 3: Cost validation
- [ ] Task 4: Strategy removed from SimulationEngine
- [ ] Task 5: Costs applied in simulation
- [ ] Task 6: Strategy removed from ScenarioConfig
- [ ] Task 7: Strategy removed from ScenarioFactory
- [ ] Task 8: MonteCarloRunner tests updated
- [ ] Task 9: SensitivityAnalyzer updated
- [ ] Task 10: Strategy directory deleted
- [ ] Task 11: main.py updated
- [ ] Task 12: YAML scenarios updated
- [ ] Task 13: Integration tests
- [ ] Task 14: Documentation updated
- [ ] Task 15: Final verification

**Next Steps:**
1. Run simulations with real VWRA data
2. Visualize cost impact over 30 years
3. Compare cost vs no-cost scenarios
4. Consider adding cost optimization analysis

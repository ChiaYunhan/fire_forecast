# Phase 3 Monte Carlo Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 3 by implementing statistical aggregation, percentile calculations, FIRE probability analysis, sensitivity testing, and comprehensive test coverage.

**Architecture:** Enhance `MonteCarloRunner.aggregate_results()` to compute percentiles using numpy, create `SensitivityAnalyzer` for parameter sweeps, and build comprehensive test suite covering edge cases and statistical validity.

**Tech Stack:** Python 3.13, numpy (percentile calculations), pytest (testing), dataclasses (results)

---

## Task 1: Implement Portfolio Percentile Calculations

**Files:**
- Modify: `src/MonteCarloRunner.py:25-28`
- Test: `test/test_monte_carlo.py` (create new)

**Step 1: Write the failing test**

Create test file with percentile calculation test:

```python
import pytest
import numpy as np
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.Strategy.balanced import BalancedStrategy


@pytest.fixture
def sample_portfolio():
    """Test portfolio for Monte Carlo runs"""
    stock = Asset(
        name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05)
    return Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="70/30"
    )


@pytest.fixture
def sample_profile(sample_portfolio):
    """Financial profile for testing"""
    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=sample_portfolio,
        age=30,
        target_age=45,
    )


class TestMonteCarloPortfolioPercentiles:
    def test_portfolio_percentiles_structure(self, sample_profile):
        """Percentiles dict contains all required keys (10, 25, 50, 75, 90)"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        expected_percentiles = [10, 25, 50, 75, 90]
        assert all(p in results.portfolio_percentiles for p in expected_percentiles)

    def test_portfolio_percentiles_ordering(self, sample_profile):
        """Percentiles are in ascending order (10th < 25th < 50th < 75th < 90th)"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        p = results.portfolio_percentiles
        assert p[10] < p[25] < p[50] < p[75] < p[90]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloPortfolioPercentiles::test_portfolio_percentiles_structure -v`

Expected: FAIL with AttributeError or incomplete implementation

**Step 3: Implement portfolio percentile calculation**

In `src/MonteCarloRunner.py`, implement the portfolio percentiles calculation:

```python
def aggregate_results(self) -> "MonteCarloSimResults":
    """
    Aggregate results from all simulation runs.

    Returns:
        MonteCarloSimResults with percentile calculations, success rates, and metrics
    """
    # Extract final portfolio values from all runs
    final_values = [result["final_portfolio_value"] for result in self.results]

    # Calculate portfolio percentiles
    portfolio_percentiles = {
        10: float(np.percentile(final_values, 10)),
        25: float(np.percentile(final_values, 25)),
        50: float(np.percentile(final_values, 50)),
        75: float(np.percentile(final_values, 75)),
        90: float(np.percentile(final_values, 90)),
    }

    # Placeholder for remaining fields (will implement in next tasks)
    from src.models import MonteCarloSimResults

    return MonteCarloSimResults(
        success_rate=0.0,  # TODO: Task 2
        median_fire_age=None,  # TODO: Task 2
        average_years_to_fire=None,  # TODO: Task 2
        portfolio_percentiles=portfolio_percentiles,
        fire_age_percentiles={},  # TODO: Task 2
        worst_case_portfolio=min(final_values),
        best_case_portfolio=max(final_values),
        shortfall_amount=None,  # TODO: Task 3
        max_drawdown=0.0,  # TODO: Task 3
        strategy_name=self.engine.strategy.name,
        input_params=self.engine.profile,
        n_simulations=self.n_simulations,
        np_seed=42,  # Will fix in later task
        annual_trajectories=[],  # TODO: Task 4
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloPortfolioPercentiles -v`

Expected: PASS for both tests

**Step 5: Commit**

```bash
git add src/MonteCarloRunner.py test/test_monte_carlo.py
git commit -m "feat: add portfolio percentile calculations to MonteCarloRunner

- Implement aggregate_results() with numpy percentile calculations
- Add test suite for Monte Carlo aggregation
- Calculate 10th, 25th, 50th, 75th, 90th percentiles
- Track worst/best case portfolio values"
```

---

## Task 2: Implement FIRE Success Rate and Age Percentiles

**Files:**
- Modify: `src/MonteCarloRunner.py:25-60`
- Test: `test/test_monte_carlo.py`

**Step 1: Write the failing tests**

Add to `test/test_monte_carlo.py`:

```python
class TestMonteCarloFIREMetrics:
    def test_success_rate_calculation(self, sample_profile):
        """Success rate is between 0 and 1"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        assert 0.0 <= results.success_rate <= 1.0
        assert results.failure_rate == 1.0 - results.success_rate

    def test_fire_age_percentiles_for_successful_runs(self, sample_portfolio):
        """FIRE age percentiles calculated only from successful runs"""
        # Create profile likely to succeed
        success_profile = FinancialProfile(
            income=100000.0,
            expenses_rate=0.3,
            savings_rate=0.7,
            portfolio=sample_portfolio,
            age=25,
            target_age=55,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(success_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # Should have fire_age_percentiles if any runs succeeded
        if results.success_rate > 0:
            expected_percentiles = [10, 25, 50, 75, 90]
            assert all(p in results.fire_age_percentiles for p in expected_percentiles)
            # Median fire age should be set
            assert results.median_fire_age is not None

    def test_median_fire_age_none_when_all_fail(self, sample_portfolio):
        """Median fire age is None when no runs achieve FIRE"""
        # Create profile unlikely to succeed
        fail_profile = FinancialProfile(
            income=50000.0,
            expenses_rate=0.95,
            savings_rate=0.05,
            portfolio=sample_portfolio,
            age=40,
            target_age=45,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(fail_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        if results.success_rate == 0:
            assert results.median_fire_age is None
            assert results.average_years_to_fire is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloFIREMetrics -v`

Expected: FAIL - fields not properly calculated

**Step 3: Implement FIRE metrics calculation**

Update `aggregate_results()` in `src/MonteCarloRunner.py`:

```python
def aggregate_results(self) -> "MonteCarloSimResults":
    """
    Aggregate results from all simulation runs.

    Returns:
        MonteCarloSimResults with percentile calculations, success rates, and metrics
    """
    # Extract final portfolio values from all runs
    final_values = [result["final_portfolio_value"] for result in self.results]

    # Calculate portfolio percentiles
    portfolio_percentiles = {
        10: float(np.percentile(final_values, 10)),
        25: float(np.percentile(final_values, 25)),
        50: float(np.percentile(final_values, 50)),
        75: float(np.percentile(final_values, 75)),
        90: float(np.percentile(final_values, 90)),
    }

    # Calculate FIRE success metrics
    successful_runs = [r for r in self.results if r["fire_achieved"]]
    success_rate = len(successful_runs) / len(self.results)

    # Calculate FIRE age statistics (only for successful runs)
    fire_age_percentiles = {}
    median_fire_age = None
    average_years_to_fire = None

    if successful_runs:
        fire_ages = [r["fire_age"] for r in successful_runs]
        fire_age_percentiles = {
            10: int(np.percentile(fire_ages, 10)),
            25: int(np.percentile(fire_ages, 25)),
            50: int(np.percentile(fire_ages, 50)),
            75: int(np.percentile(fire_ages, 75)),
            90: int(np.percentile(fire_ages, 90)),
        }
        median_fire_age = fire_age_percentiles[50]

        # Average years to FIRE from starting age
        starting_age = self.engine.profile.age
        years_to_fire = [r["fire_age"] - starting_age for r in successful_runs]
        average_years_to_fire = float(np.mean(years_to_fire))

    from src.models import MonteCarloSimResults

    return MonteCarloSimResults(
        success_rate=success_rate,
        median_fire_age=median_fire_age,
        average_years_to_fire=average_years_to_fire,
        portfolio_percentiles=portfolio_percentiles,
        fire_age_percentiles=fire_age_percentiles,
        worst_case_portfolio=min(final_values),
        best_case_portfolio=max(final_values),
        shortfall_amount=None,  # TODO: Task 3
        max_drawdown=0.0,  # TODO: Task 3
        strategy_name=self.engine.strategy.name,
        input_params=self.engine.profile,
        n_simulations=self.n_simulations,
        np_seed=42,  # Will fix in later task
        annual_trajectories=[],  # TODO: Task 4
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloFIREMetrics -v`

Expected: PASS for all tests

**Step 5: Commit**

```bash
git add src/MonteCarloRunner.py test/test_monte_carlo.py
git commit -m "feat: add FIRE success rate and age percentile calculations

- Calculate success_rate as percentage of runs achieving FIRE
- Compute FIRE age percentiles (10th-90th) for successful runs
- Calculate median_fire_age and average_years_to_fire
- Handle edge case when no runs achieve FIRE"
```

---

## Task 3: Implement Risk Metrics (Shortfall and Drawdown)

**Files:**
- Modify: `src/MonteCarloRunner.py:25-70`
- Test: `test/test_monte_carlo.py`

**Step 1: Write the failing tests**

Add to `test/test_monte_carlo.py`:

```python
class TestMonteCarloRiskMetrics:
    def test_shortfall_amount_calculated(self, sample_portfolio):
        """Shortfall amount is calculated for failed runs"""
        # Create profile likely to fail
        fail_profile = FinancialProfile(
            income=50000.0,
            expenses_rate=0.9,
            savings_rate=0.1,
            portfolio=sample_portfolio,
            age=40,
            target_age=50,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(fail_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # If there are failures, shortfall should be calculated
        if results.success_rate < 1.0:
            assert results.shortfall_amount is not None
            assert results.shortfall_amount > 0

    def test_max_drawdown_positive(self, sample_profile):
        """Max drawdown is calculated and non-negative"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        assert results.max_drawdown >= 0.0
        # Max drawdown is a percentage, shouldn't exceed 1.0 typically
        assert results.max_drawdown <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloRiskMetrics -v`

Expected: FAIL - metrics not implemented

**Step 3: Implement risk metrics calculation**

Update `aggregate_results()` in `src/MonteCarloRunner.py`:

```python
def aggregate_results(self) -> "MonteCarloSimResults":
    """
    Aggregate results from all simulation runs.

    Returns:
        MonteCarloSimResults with percentile calculations, success rates, and metrics
    """
    # Extract final portfolio values from all runs
    final_values = [result["final_portfolio_value"] for result in self.results]

    # Calculate portfolio percentiles
    portfolio_percentiles = {
        10: float(np.percentile(final_values, 10)),
        25: float(np.percentile(final_values, 25)),
        50: float(np.percentile(final_values, 50)),
        75: float(np.percentile(final_values, 75)),
        90: float(np.percentile(final_values, 90)),
    }

    # Calculate FIRE success metrics
    successful_runs = [r for r in self.results if r["fire_achieved"]]
    failed_runs = [r for r in self.results if not r["fire_achieved"]]
    success_rate = len(successful_runs) / len(self.results)

    # Calculate FIRE age statistics (only for successful runs)
    fire_age_percentiles = {}
    median_fire_age = None
    average_years_to_fire = None

    if successful_runs:
        fire_ages = [r["fire_age"] for r in successful_runs]
        fire_age_percentiles = {
            10: int(np.percentile(fire_ages, 10)),
            25: int(np.percentile(fire_ages, 25)),
            50: int(np.percentile(fire_ages, 50)),
            75: int(np.percentile(fire_ages, 75)),
            90: int(np.percentile(fire_ages, 90)),
        }
        median_fire_age = fire_age_percentiles[50]

        # Average years to FIRE from starting age
        starting_age = self.engine.profile.age
        years_to_fire = [r["fire_age"] - starting_age for r in successful_runs]
        average_years_to_fire = float(np.mean(years_to_fire))

    # Calculate risk metrics
    # Shortfall: average gap below FIRE target for failed runs
    shortfall_amount = None
    if failed_runs:
        fire_target = self.engine.profile.annual_expenses() / 0.04
        shortfalls = [fire_target - r["final_portfolio_value"] for r in failed_runs]
        shortfall_amount = float(np.mean(shortfalls))

    # Max drawdown: largest peak-to-trough decline across all runs
    max_drawdown = 0.0
    for result in self.results:
        history = result["portfolio_history"]
        if len(history) > 0:
            peak = history[0]
            for value in history:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

    from src.models import MonteCarloSimResults

    return MonteCarloSimResults(
        success_rate=success_rate,
        median_fire_age=median_fire_age,
        average_years_to_fire=average_years_to_fire,
        portfolio_percentiles=portfolio_percentiles,
        fire_age_percentiles=fire_age_percentiles,
        worst_case_portfolio=min(final_values),
        best_case_portfolio=max(final_values),
        shortfall_amount=shortfall_amount,
        max_drawdown=max_drawdown,
        strategy_name=self.engine.strategy.name,
        input_params=self.engine.profile,
        n_simulations=self.n_simulations,
        np_seed=42,  # Will fix in later task
        annual_trajectories=[],  # TODO: Task 4
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloRiskMetrics -v`

Expected: PASS for both tests

**Step 5: Commit**

```bash
git add src/MonteCarloRunner.py test/test_monte_carlo.py
git commit -m "feat: add risk metrics (shortfall and max drawdown)

- Calculate average shortfall amount for failed runs
- Compute max drawdown across all simulation runs
- Track peak-to-trough portfolio declines"
```

---

## Task 4: Capture Annual Trajectories for Charting

**Files:**
- Modify: `src/MonteCarloRunner.py:15-75`
- Test: `test/test_monte_carlo.py`

**Step 1: Write the failing test**

Add to `test/test_monte_carlo.py`:

```python
class TestMonteCarloAnnualTrajectories:
    def test_annual_trajectories_captured(self, sample_profile):
        """Annual trajectories list matches number of simulations"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=10, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        assert len(results.annual_trajectories) == 10
        # Each trajectory should have values for each year
        years_simulated = sample_profile.target_age - sample_profile.age
        assert all(len(traj) == years_simulated for traj in results.annual_trajectories)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloAnnualTrajectories -v`

Expected: FAIL - annual_trajectories empty list

**Step 3: Capture portfolio histories during aggregation**

Update `aggregate_results()` in `src/MonteCarloRunner.py`:

```python
def aggregate_results(self) -> "MonteCarloSimResults":
    """
    Aggregate results from all simulation runs.

    Returns:
        MonteCarloSimResults with percentile calculations, success rates, and metrics
    """
    # Extract final portfolio values from all runs
    final_values = [result["final_portfolio_value"] for result in self.results]

    # Calculate portfolio percentiles
    portfolio_percentiles = {
        10: float(np.percentile(final_values, 10)),
        25: float(np.percentile(final_values, 25)),
        50: float(np.percentile(final_values, 50)),
        75: float(np.percentile(final_values, 75)),
        90: float(np.percentile(final_values, 90)),
    }

    # Calculate FIRE success metrics
    successful_runs = [r for r in self.results if r["fire_achieved"]]
    failed_runs = [r for r in self.results if not r["fire_achieved"]]
    success_rate = len(successful_runs) / len(self.results)

    # Calculate FIRE age statistics (only for successful runs)
    fire_age_percentiles = {}
    median_fire_age = None
    average_years_to_fire = None

    if successful_runs:
        fire_ages = [r["fire_age"] for r in successful_runs]
        fire_age_percentiles = {
            10: int(np.percentile(fire_ages, 10)),
            25: int(np.percentile(fire_ages, 25)),
            50: int(np.percentile(fire_ages, 50)),
            75: int(np.percentile(fire_ages, 75)),
            90: int(np.percentile(fire_ages, 90)),
        }
        median_fire_age = fire_age_percentiles[50]

        # Average years to FIRE from starting age
        starting_age = self.engine.profile.age
        years_to_fire = [r["fire_age"] - starting_age for r in successful_runs]
        average_years_to_fire = float(np.mean(years_to_fire))

    # Calculate risk metrics
    # Shortfall: average gap below FIRE target for failed runs
    shortfall_amount = None
    if failed_runs:
        fire_target = self.engine.profile.annual_expenses() / 0.04
        shortfalls = [fire_target - r["final_portfolio_value"] for r in failed_runs]
        shortfall_amount = float(np.mean(shortfalls))

    # Max drawdown: largest peak-to-trough decline across all runs
    max_drawdown = 0.0
    for result in self.results:
        history = result["portfolio_history"]
        if len(history) > 0:
            peak = history[0]
            for value in history:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)

    # Capture annual trajectories for charting (Phase 4)
    annual_trajectories = [result["portfolio_history"] for result in self.results]

    from src.models import MonteCarloSimResults

    return MonteCarloSimResults(
        success_rate=success_rate,
        median_fire_age=median_fire_age,
        average_years_to_fire=average_years_to_fire,
        portfolio_percentiles=portfolio_percentiles,
        fire_age_percentiles=fire_age_percentiles,
        worst_case_portfolio=min(final_values),
        best_case_portfolio=max(final_values),
        shortfall_amount=shortfall_amount,
        max_drawdown=max_drawdown,
        strategy_name=self.engine.strategy.name,
        input_params=self.engine.profile,
        n_simulations=self.n_simulations,
        np_seed=42,  # Will fix in later task
        annual_trajectories=annual_trajectories,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloAnnualTrajectories -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/MonteCarloRunner.py test/test_monte_carlo.py
git commit -m "feat: capture annual portfolio trajectories for charting

- Store portfolio_history from each simulation run
- Enable future fan chart visualization (Phase 4)
- Trajectories contain year-by-year portfolio values"
```

---

## Task 5: Fix Seed Tracking and Add Integration Test

**Files:**
- Modify: `src/MonteCarloRunner.py:9-12`
- Test: `test/test_monte_carlo.py`

**Step 1: Write the failing test**

Add to `test/test_monte_carlo.py`:

```python
class TestMonteCarloIntegration:
    def test_seed_preservation(self, sample_profile):
        """Seed is preserved in results for reproducibility"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=12345)

        runner.run_simulations()
        results = runner.aggregate_results()

        assert results.np_seed == 12345

    def test_deterministic_results_with_same_seed(self, sample_profile):
        """Same seed produces identical results"""
        strategy = BalancedStrategy()

        # Run 1
        engine1 = SimulationEngine(sample_profile, strategy)
        runner1 = MonteCarloRunner(engine1, n_simulations=50, seed=999)
        runner1.run_simulations()
        results1 = runner1.aggregate_results()

        # Run 2 with same seed
        engine2 = SimulationEngine(sample_profile, strategy)
        runner2 = MonteCarloRunner(engine2, n_simulations=50, seed=999)
        runner2.run_simulations()
        results2 = runner2.aggregate_results()

        # Results should be identical
        assert results1.success_rate == results2.success_rate
        assert results1.portfolio_percentiles[50] == results2.portfolio_percentiles[50]

    def test_complete_monte_carlo_workflow(self, sample_profile):
        """End-to-end Monte Carlo workflow produces valid results"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=100, seed=42)

        # Run simulations
        raw_results = runner.run_simulations()
        assert len(raw_results) == 100

        # Aggregate results
        summary = runner.aggregate_results()

        # Verify all fields are populated correctly
        assert summary.n_simulations == 100
        assert summary.strategy_name == "Balanced"
        assert 0.0 <= summary.success_rate <= 1.0
        assert len(summary.portfolio_percentiles) == 5
        assert summary.worst_case_portfolio <= summary.best_case_portfolio
        assert len(summary.annual_trajectories) == 100
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloIntegration -v`

Expected: FAIL - seed not tracked correctly

**Step 3: Store seed in MonteCarloRunner and use in aggregate_results**

Update `src/MonteCarloRunner.py`:

```python
class MonteCarloRunner:
    def __init__(self, engine: SimulationEngine, n_simulations: int, seed: int):
        np.random.seed(seed)
        self.engine = engine
        self.n_simulations = n_simulations
        self.seed = seed  # Store seed for reproducibility
        self.results = []
```

Then update `aggregate_results()` to use `self.seed`:

```python
    return MonteCarloSimResults(
        success_rate=success_rate,
        median_fire_age=median_fire_age,
        average_years_to_fire=average_years_to_fire,
        portfolio_percentiles=portfolio_percentiles,
        fire_age_percentiles=fire_age_percentiles,
        worst_case_portfolio=min(final_values),
        best_case_portfolio=max(final_values),
        shortfall_amount=shortfall_amount,
        max_drawdown=max_drawdown,
        strategy_name=self.engine.strategy.name,
        input_params=self.engine.profile,
        n_simulations=self.n_simulations,
        np_seed=self.seed,  # Use stored seed
        annual_trajectories=annual_trajectories,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloIntegration -v`

Expected: PASS for all integration tests

**Step 5: Commit**

```bash
git add src/MonteCarloRunner.py test/test_monte_carlo.py
git commit -m "feat: track seed for reproducibility and add integration tests

- Store seed in MonteCarloRunner instance
- Preserve seed in MonteCarloSimResults
- Add integration test for deterministic behavior
- Add end-to-end workflow test"
```

---

## Task 6: Create SensitivityAnalyzer for Parameter Sweeps

**Files:**
- Create: `src/SensitivityAnalyzer.py`
- Test: `test/test_sensitivity.py`

**Step 1: Write the failing test**

Create `test/test_sensitivity.py`:

```python
import pytest
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.SensitivityAnalyzer import SensitivityAnalyzer
from src.Strategy.balanced import BalancedStrategy


@pytest.fixture
def base_profile():
    """Base financial profile for sensitivity analysis"""
    stock = Asset(
        name="Stock ETF", allocation=0.7, expected_return=0.08, volatility=0.15
    )
    bond = Asset(name="Bond ETF", allocation=0.3, expected_return=0.04, volatility=0.05)
    portfolio = Portfolio(
        composition=[stock, bond], total_value=100000.0, allocation_methods="70/30"
    )

    return FinancialProfile(
        income=100000.0,
        expenses_rate=0.6,
        savings_rate=0.4,
        portfolio=portfolio,
        age=30,
        target_age=45,
    )


class TestSensitivityAnalyzer:
    def test_savings_rate_sweep(self, base_profile):
        """Sweeping savings rate produces multiple results"""
        analyzer = SensitivityAnalyzer(base_profile, BalancedStrategy())

        # Sweep savings rates from 0.2 to 0.6 in steps of 0.1
        results = analyzer.sweep_savings_rate(
            rates=[0.2, 0.3, 0.4, 0.5, 0.6],
            n_simulations=50,
            seed=42
        )

        assert len(results) == 5
        # Each result should have a success_rate
        assert all(hasattr(r, "success_rate") for r in results)

    def test_savings_rate_impact_on_success(self, base_profile):
        """Higher savings rates increase FIRE success probability"""
        analyzer = SensitivityAnalyzer(base_profile, BalancedStrategy())

        results = analyzer.sweep_savings_rate(
            rates=[0.2, 0.4, 0.6],
            n_simulations=100,
            seed=42
        )

        # Higher savings should generally lead to higher success
        # (This is probabilistic, but should hold with 100 sims)
        success_rates = [r.success_rate for r in results]
        assert success_rates[1] >= success_rates[0]  # 0.4 >= 0.2
        assert success_rates[2] >= success_rates[1]  # 0.6 >= 0.4
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_sensitivity.py::TestSensitivityAnalyzer::test_savings_rate_sweep -v`

Expected: FAIL with ImportError or ModuleNotFoundError

**Step 3: Implement SensitivityAnalyzer**

Create `src/SensitivityAnalyzer.py`:

```python
from typing import List
from dataclasses import replace

from .models import FinancialProfile, MonteCarloSimResults
from .Strategy.base import InvestmentStrategy
from .SimulationEngine import SimulationEngine
from .MonteCarloRunner import MonteCarloRunner


class SensitivityAnalyzer:
    """
    Analyzes how changes to input parameters affect FIRE outcomes.

    Allows sweeping savings rate, income, or other parameters to understand
    their impact on success probability and FIRE timeline.
    """

    def __init__(self, base_profile: FinancialProfile, strategy: InvestmentStrategy):
        """
        Initialize analyzer with a base financial profile.

        Args:
            base_profile: The baseline financial profile
            strategy: Investment strategy to use for all simulations
        """
        self.base_profile = base_profile
        self.strategy = strategy

    def sweep_savings_rate(
        self, rates: List[float], n_simulations: int, seed: int
    ) -> List[MonteCarloSimResults]:
        """
        Sweep across different savings rates and run Monte Carlo for each.

        Args:
            rates: List of savings rates to test (e.g., [0.2, 0.3, 0.4])
            n_simulations: Number of Monte Carlo runs per rate
            seed: Random seed for reproducibility

        Returns:
            List of MonteCarloSimResults, one for each savings rate
        """
        results = []

        for rate in rates:
            # Create modified profile with new savings rate
            # Expenses rate must adjust to maintain sum = 1.0
            new_expenses_rate = 1.0 - rate

            modified_profile = replace(
                self.base_profile,
                savings_rate=rate,
                expenses_rate=new_expenses_rate
            )

            # Run Monte Carlo with this profile
            engine = SimulationEngine(modified_profile, self.strategy)
            runner = MonteCarloRunner(engine, n_simulations=n_simulations, seed=seed)
            runner.run_simulations()
            result = runner.aggregate_results()

            results.append(result)

        return results
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_sensitivity.py::TestSensitivityAnalyzer -v`

Expected: PASS for both tests

**Step 5: Commit**

```bash
git add src/SensitivityAnalyzer.py test/test_sensitivity.py
git commit -m "feat: add SensitivityAnalyzer for parameter sweep analysis

- Create SensitivityAnalyzer class
- Implement sweep_savings_rate() for savings rate impact analysis
- Automatically adjust expenses_rate to maintain sum = 1.0
- Add tests demonstrating higher savings increases success rate"
```

---

## Task 7: Add Edge Case Tests for Monte Carlo

**Files:**
- Test: `test/test_monte_carlo.py`

**Step 1: Write edge case tests**

Add to `test/test_monte_carlo.py`:

```python
class TestMonteCarloEdgeCases:
    def test_single_simulation_run(self, sample_profile):
        """Monte Carlo works with n_simulations=1"""
        strategy = BalancedStrategy()
        engine = SimulationEngine(sample_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=1, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # Should still compute results without errors
        assert results.n_simulations == 1
        assert results.success_rate in [0.0, 1.0]  # Either succeeded or failed

    def test_all_runs_succeed(self, sample_portfolio):
        """Handles case where all runs achieve FIRE"""
        # Very high savings, long timeline
        success_profile = FinancialProfile(
            income=200000.0,
            expenses_rate=0.2,
            savings_rate=0.8,
            portfolio=sample_portfolio,
            age=25,
            target_age=60,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(success_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # All should succeed
        assert results.success_rate == 1.0
        assert results.shortfall_amount is None
        assert results.median_fire_age is not None

    def test_all_runs_fail(self, sample_portfolio):
        """Handles case where no runs achieve FIRE"""
        # Very low savings, short timeline
        fail_profile = FinancialProfile(
            income=40000.0,
            expenses_rate=0.95,
            savings_rate=0.05,
            portfolio=sample_portfolio,
            age=40,
            target_age=45,
        )

        strategy = BalancedStrategy()
        engine = SimulationEngine(fail_profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=50, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        # All should fail
        assert results.success_rate == 0.0
        assert results.median_fire_age is None
        assert results.average_years_to_fire is None
        assert len(results.fire_age_percentiles) == 0
        assert results.shortfall_amount is not None
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest test/test_monte_carlo.py::TestMonteCarloEdgeCases -v`

Expected: PASS for all edge cases

**Step 3: Commit**

```bash
git add test/test_monte_carlo.py
git commit -m "test: add edge case tests for Monte Carlo

- Test single simulation run (n=1)
- Test 100% success rate scenario
- Test 0% success rate scenario
- Verify graceful handling of boundary conditions"
```

---

## Task 8: Update main.py to Demonstrate Monte Carlo

**Files:**
- Modify: `main.py:70-99`

**Step 1: Write updated main.py demonstration**

Update `main.py` to showcase Monte Carlo:

```python
from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.SensitivityAnalyzer import SensitivityAnalyzer
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


def main():
    # Sample ETF portfolio
    stock_etf = Asset(
        name="VTI (Total Stock Market)",
        allocation=0.80,
        expected_return=0.10,
        volatility=0.15,
    )
    bond_etf = Asset(
        name="BND (Total Bond Market)",
        allocation=0.20,
        expected_return=0.04,
        volatility=0.04,
    )

    portfolio = Portfolio(
        composition=[stock_etf, bond_etf],
        total_value=0.0,
        allocation_methods="aggressive",
    )

    profile = FinancialProfile(
        income=50_056.92,
        expenses_rate=0.58,
        savings_rate=0.42,
        portfolio=portfolio,
        age=25,
        target_age=45,
    )

    print_summary(profile)
    run_monte_carlo_analysis(profile)
    run_sensitivity_analysis(profile)


def print_summary(profile: FinancialProfile):
    print("=" * 50)
    print("  FIRE Forecast — Financial Profile Summary")
    print("=" * 50)

    print(f"\n  Age: {profile.age} → Target FIRE Age: {profile.target_age}")
    print(f"  Years to FIRE: {profile.target_age - profile.age}")

    print(f"\n  Income:       ${profile.income:>12,.2f}")
    print(f"  Expenses Rate: {profile.expenses_rate:>12.0%}")
    print(f"  Savings Rate:  {profile.savings_rate:>12.0%}")
    print(f"  Annual Expenses: ${profile.annual_expenses():>10,.2f}")
    print(f"  Annual Savings:  ${profile.annual_savings():>10,.2f}")

    print(f"\n  Portfolio Value: ${profile.portfolio.total_value:>10,.2f}")
    print(f"  Strategy: {profile.portfolio.allocation_methods}")
    print(f"\n  Assets:")
    for asset in profile.portfolio.composition:
        print(f"    • {asset.name}")
        print(
            f"      Allocation: {asset.allocation:.0%} | "
            f"Return: {asset.expected_return:.0%} | "
            f"Volatility: {asset.volatility:.0%}"
        )

    print("\n" + "=" * 50)


def run_monte_carlo_analysis(profile: FinancialProfile):
    """Run Monte Carlo simulations with all three strategies."""
    print("\n" + "=" * 60)
    print("  Monte Carlo Analysis (1,000 simulations per strategy)")
    print("=" * 60)

    strategies = [
        AggressiveStrategy(),
        BalancedStrategy(),
        ConservativeStrategy(),
    ]

    for strategy in strategies:
        engine = SimulationEngine(profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=1000, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        print(f"\n  Strategy: {strategy.name}")
        print(f"  Risk Multiplier: {strategy.get_risk_multiplier()}x")
        print("-" * 60)
        print(f"  FIRE Success Rate: {results.success_rate:>8.1%}")

        if results.median_fire_age:
            print(f"  Median FIRE Age: {results.median_fire_age:>10}")
            print(f"  Avg Years to FIRE: {results.average_years_to_fire:>8.1f}")

        print(f"\n  Portfolio Value Percentiles:")
        print(f"    10th: ${results.portfolio_percentiles[10]:>15,.2f}")
        print(f"    25th: ${results.portfolio_percentiles[25]:>15,.2f}")
        print(f"    50th: ${results.portfolio_percentiles[50]:>15,.2f}")
        print(f"    75th: ${results.portfolio_percentiles[75]:>15,.2f}")
        print(f"    90th: ${results.portfolio_percentiles[90]:>15,.2f}")

        print(f"\n  Risk Metrics:")
        print(f"    Best Case:  ${results.best_case_portfolio:>15,.2f}")
        print(f"    Worst Case: ${results.worst_case_portfolio:>15,.2f}")
        print(f"    Max Drawdown: {results.max_drawdown:>13.1%}")
        if results.shortfall_amount:
            print(f"    Avg Shortfall: ${results.shortfall_amount:>13,.2f}")

    print("\n" + "=" * 60 + "\n")


def run_sensitivity_analysis(profile: FinancialProfile):
    """Demonstrate sensitivity analysis on savings rate."""
    print("\n" + "=" * 60)
    print("  Sensitivity Analysis: Impact of Savings Rate")
    print("=" * 60)

    analyzer = SensitivityAnalyzer(profile, BalancedStrategy())

    savings_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
    results = analyzer.sweep_savings_rate(
        rates=savings_rates,
        n_simulations=500,
        seed=42
    )

    print("\n  Savings Rate  →  FIRE Success Rate")
    print("-" * 60)
    for rate, result in zip(savings_rates, results):
        print(f"     {rate:>4.0%}                  {result.success_rate:>6.1%}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
```

**Step 2: Run the updated main.py**

Run: `uv run main.py`

Expected: Complete output showing Monte Carlo results and sensitivity analysis

**Step 3: Verify output quality**

Check that:
- All three strategies display percentiles correctly
- FIRE success rates are calculated
- Sensitivity analysis shows increasing success with higher savings

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: update main.py to demonstrate Monte Carlo and sensitivity

- Run 1,000 Monte Carlo simulations per strategy
- Display percentile distributions for portfolio values
- Show FIRE success rates and risk metrics
- Demonstrate sensitivity analysis on savings rate
- Remove note about Phase 3 being future work"
```

---

## Task 9: Update ROADMAP to Mark Phase 3 Complete

**Files:**
- Modify: `docs/ROADMAP.md:33-45`

**Step 1: Mark remaining Phase 3 tasks as complete**

Update the Phase 3 section in `docs/ROADMAP.md`:

```markdown
## Week 3: Monte Carlo & Analysis (~8-10 hours) ✅ COMPLETE

Scale to thousands of runs and extract meaningful insights.

- [x] `MonteCarloRunner` class that executes N simulation runs
- [x] Randomized market returns using historical distributions
- [x] `SimulationResults` class to aggregate outcomes
- [x] Percentile calculations (10th, 25th, 50th, 75th, 90th)
- [x] FIRE probability calculator ("X% chance of retiring by age Y")
- [x] Sensitivity analysis (how does changing savings rate affect outcomes?)
- [x] Tests for Monte Carlo logic and results aggregation

**Key OOP focus:** Inheritance, polymorphism (different market models), encapsulation of results
```

**Step 2: Commit the roadmap update**

```bash
git add docs/ROADMAP.md
git commit -m "docs: mark Phase 3 (Monte Carlo & Analysis) as complete

All Phase 3 deliverables implemented:
- Percentile calculations across portfolio values and FIRE ages
- FIRE probability success rate calculation
- Sensitivity analysis via SensitivityAnalyzer
- Comprehensive test coverage for Monte Carlo logic"
```

---

## Summary

**Phase 3 is now complete with:**
- ✅ Portfolio value percentiles (10th, 25th, 50th, 75th, 90th)
- ✅ FIRE success rate and age distribution calculations
- ✅ Risk metrics (shortfall amount, max drawdown)
- ✅ Annual trajectory capture for future charting
- ✅ Seed tracking for reproducibility
- ✅ SensitivityAnalyzer for parameter sweeps
- ✅ Comprehensive test coverage (edge cases, integration tests)
- ✅ Updated main.py demonstrating all capabilities
- ✅ Documentation updated

**Total tasks:** 9
**Estimated time:** 8-10 hours
**Test files created:** 2 (`test_monte_carlo.py`, `test_sensitivity.py`)
**New classes created:** 1 (`SensitivityAnalyzer`)
**Key patterns demonstrated:** Statistical aggregation, numpy percentiles, dataclass immutability with `replace()`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FIRE Forecast Engine — a Python CLI tool that runs Monte Carlo simulations against a user's financial profile to project the probability of reaching financial independence at various ages. Targets personal use for an ETF investor.

## Development Setup

- Python 3.13+ (managed via pyenv, see `.python-version`)
- Package manager: uv (uses `pyproject.toml`, no external dependencies yet)
- Virtual environment: `.venv/`
- Run the app: `uv run main.py`
- Run tests: `uv run pytest`
- Add a dependency: `uv add <package>`

## Architecture

**Current state:** Phase 3+ complete with cost modeling enhancement. Domain models, simulation engine, Monte Carlo runner, sensitivity analysis, and visualization all implemented. See `docs/ROADMAP.md` for phased build plan.

**Entry point:** `main.py` — CLI that loads YAML scenarios, runs Monte Carlo simulations, and displays results with visualizations

**Implemented domain models** in `src/models.py`:
- `Asset` — individual investment (name, allocation, expected return, volatility, optional costs)
  - Optional `costs` dict with `ter` (Total Expense Ratio) and `trading_fee` fields
  - Validates cost values are between 0-5%
- `Portfolio` — composition of Assets (Portfolio *has* Assets, not inheritance)
- `FinancialProfile` — complete financial situation (income, expenses, savings rate, portfolio, age, target FIRE age)
  - Validates that `expenses_rate + savings_rate ≈ 1.0`
  - Methods: `annual_savings()`, `annual_expenses()`

**Implemented simulation engine** in `src/SimulationEngine.py`:
- `SimulationEngine` — Template Method pattern with run/setup/simulate_year/collect_results lifecycle
- Calculates portfolio returns using asset expected_return and volatility
- Applies investment costs (TER annually, trading fees on contributions)
- Calculates FIRE achievement (portfolio ≥ 25x annual expenses using 4% rule)
- Returns comprehensive results dict with portfolio history

**Implemented components**:
- `MonteCarloRunner` — executes N simulation runs with randomized market returns, aggregates percentile results
- `SensitivityAnalyzer` — sweeps parameter ranges (e.g., savings rates) to analyze impact on FIRE outcomes
- `ScenarioFactory` — creates profiles from YAML config (Factory pattern)
- `visualization` — generates projection fan charts and FIRE age distribution histograms

## Design Patterns

The project deliberately exercises these OOP patterns:
- **Composition over inheritance** — Portfolio contains Assets, Profile contains Portfolio
- **Data-driven configuration** — investment behavior defined in YAML (returns, volatility, costs), not hardcoded strategies
- **Template Method** — simulation lifecycle
- **Factory pattern** — scenario creation from config files

## Tech Stack (Planned)

- `dataclasses` for domain models
- `pytest` for testing
- `matplotlib` for projection fan charts and probability charts
- YAML/JSON for scenario configuration

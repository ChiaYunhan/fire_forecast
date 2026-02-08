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

**Current state:** Phase 3 (Monte Carlo implementation). Weeks 1-2 complete: domain models, strategies, and single simulation engine working. See `docs/ROADMAP.md` for phased build plan.

**Entry point:** `main.py` — CLI that creates a sample profile, runs simulations with all three strategies, and displays results

**Implemented domain models** in `src/models.py`:
- `Asset` — individual investment (name, allocation, expected return, volatility)
- `Portfolio` — composition of Assets (Portfolio *has* Assets, not inheritance)
- `FinancialProfile` — complete financial situation (income, expenses, savings rate, portfolio, age, target FIRE age)
  - Validates that `expenses_rate + savings_rate ≈ 1.0`
  - Methods: `annual_savings()`, `annual_expenses()`

**Implemented investment strategies** in `src/Strategy/`:
- `InvestmentStrategy` ABC — defines interface for all strategies
- `AggressiveStrategy` — 1.5x risk multiplier, higher volatility tolerance
- `BalancedStrategy` — 1.1x risk multiplier, moderate approach
- `ConservativeStrategy` — 0.8x risk multiplier, capital preservation focus

**Implemented simulation engine** in `src/SimulationEngine.py`:
- `SimulationEngine` — Template Method pattern with run/setup/simulate_year/collect_results lifecycle
- Applies annual returns using selected strategy
- Calculates FIRE achievement (portfolio ≥ 25x annual expenses using 4% rule)
- Returns comprehensive results dict

**Planned components**:
- `MonteCarloRunner` — executes N simulation runs with randomized market returns
- `SimulationResults` — aggregates outcomes, percentile calculations
- `ScenarioFactory` — creates profiles/strategies from YAML config (Factory pattern)

## Design Patterns

The project deliberately exercises these OOP patterns:
- **Composition over inheritance** — Portfolio contains Assets, Profile contains Portfolio
- **Strategy pattern** — swappable investment strategies
- **Template Method** — simulation lifecycle
- **Factory pattern** — scenario creation from config files

## Tech Stack (Planned)

- `dataclasses` for domain models
- `pytest` for testing
- `matplotlib` for projection fan charts and probability charts
- YAML/JSON for scenario configuration

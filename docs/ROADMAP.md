# FIRE Forecast Engine — Roadmap

## Week 1: Domain Modeling (~6-8 hours)

Build the core objects that represent your financial world.

- [x] Create project structure with `pyproject.toml` and virtual env
- [x] Implement `Asset` class (name, allocation, expected return, volatility)
- [x] Implement `Portfolio` class (composition of Assets, total value, allocation methods)
- [x] Implement `FinancialProfile` class (income, expenses, savings rate, portfolio, age, target FIRE age)
- [x] Write tests for core domain classes
- [x] Basic CLI entry point that creates a profile and prints a summary

**Key OOP focus:** Classes, encapsulation, composition (Portfolio _has_ Assets, Profile _has_ Portfolio)

## Week 2: Simulation Engine (~8-10 hours)

Build the engine using Strategy and Template Method patterns.

- [ ] Define `InvestmentStrategy` abstract base class
- [ ] Implement `AggressiveStrategy` (high equity, high volatility)
- [ ] Implement `BalancedStrategy` (mixed allocation)
- [ ] Implement `ConservativeStrategy` (bond-heavy, low volatility)
- [ ] Build `SimulationEngine` with Template Method pattern:
  - `setup()` — initialize starting state
  - `simulate_year()` — apply returns, contributions, withdrawals
  - `collect_results()` — gather final outcomes
- [ ] Single simulation run working end-to-end
- [ ] Tests for strategies and simulation engine

**Key OOP focus:** Strategy pattern, Template Method, abstract classes, polymorphism

## Week 3: Monte Carlo & Analysis (~8-10 hours)

Scale to thousands of runs and extract meaningful insights.

- [ ] `MonteCarloRunner` class that executes N simulation runs
- [ ] Randomized market returns using historical distributions
- [ ] `SimulationResults` class to aggregate outcomes
- [ ] Percentile calculations (10th, 25th, 50th, 75th, 90th)
- [ ] FIRE probability calculator ("X% chance of retiring by age Y")
- [ ] Sensitivity analysis (how does changing savings rate affect outcomes?)
- [ ] Tests for Monte Carlo logic and results aggregation

**Key OOP focus:** Inheritance, polymorphism (different market models), encapsulation of results

## Week 4: Polish & Visualize (~6-8 hours)

Make it usable, configurable, and presentable.

- [ ] YAML config file for defining scenarios
- [ ] `ScenarioFactory` to create profiles/strategies from config
- [ ] Matplotlib projection fan chart (percentile bands over time)
- [ ] Matplotlib FIRE probability chart (probability vs age)
- [ ] CLI argument parsing for running different scenarios
- [ ] README with usage examples
- [ ] Code cleanup and final refactor

**Key OOP focus:** Factory pattern, clean separation of concerns

## First 3 Tasks

1. **Create project structure** — `pyproject.toml`, virtual env, `src/fire_forecast/` package layout
2. **Implement `Asset` and `Portfolio`** — with `__init__`, `__repr__`, and basic methods + tests
3. **Build `FinancialProfile`** — composes a Portfolio with income/expense data, calculates annual savings

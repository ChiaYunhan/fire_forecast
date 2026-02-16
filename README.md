# FIRE Forecast Engine

A Python CLI tool that runs Monte Carlo simulations to project the probability of achieving Financial Independence and Retire Early (FIRE) at various ages.

## Features

- **Monte Carlo Simulations**: Run thousands of market scenarios with randomized returns
- **Multiple Investment Strategies**: Aggressive, Balanced, Conservative approaches
- **YAML Configuration**: Define custom scenarios easily
- **Rich Visualizations**: Projection fan charts and FIRE age distribution histograms
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
  assets:
    - name: "Stock Index Fund"
      allocation: 0.70
      expected_return: 0.09
      volatility: 0.15
    - name: "Bond Index Fund"
      allocation: 0.30
      expected_return: 0.04
      volatility: 0.05

strategy: balanced  # Options: aggressive, balanced, conservative

simulation:
  n_simulations: 10000
  seed: 42  # For reproducible results
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

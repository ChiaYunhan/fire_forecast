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

### FIRE Age Distribution

Shows histogram of when people achieve FIRE:

- Green bars show how many simulations achieved FIRE at each age
- Red dashed line marks your target age
- Blue dashed line shows median FIRE age
- Success rate displayed in corner

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

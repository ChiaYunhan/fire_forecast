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

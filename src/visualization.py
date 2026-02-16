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


def create_fire_age_distribution(
    fire_ages: list[int],
    target_age: int,
    median_fire_age: int | None,
    success_rate: float,
    title: str = "FIRE Age Distribution"
) -> Figure:
    """Create a histogram showing distribution of FIRE achievement ages.

    Args:
        fire_ages: List of ages when FIRE was achieved (successful runs only)
        target_age: Target FIRE age for highlighting
        median_fire_age: Median age when FIRE was achieved
        success_rate: Overall FIRE success rate (0.0-1.0)
        title: Chart title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if not fire_ages:
        # Handle case where no one achieved FIRE
        ax.text(0.5, 0.5, 'No FIRE achieved in simulations',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Create histogram
    min_age = min(fire_ages)
    max_age = max(fire_ages)
    bins = range(min_age, max_age + 2)  # +2 to include max_age

    ax.hist(fire_ages, bins=bins, alpha=0.7, color='green',
            edgecolor='black', label=f'FIRE Age Distribution ({success_rate:.1%} achieved)')

    # Mark target age
    ax.axvline(target_age, color='red', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Target Age ({target_age})')

    # Mark median age
    if median_fire_age:
        ax.axvline(median_fire_age, color='blue', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Median Age ({median_fire_age})')

    # Formatting
    ax.set_xlabel("Age When FIRE Achieved")
    ax.set_ylabel("Number of Simulations")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add success rate annotation
    ax.text(0.02, 0.98, f'Success Rate: {success_rate:.1%}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from matplotlib.figure import Figure
from src.visualization import create_projection_fan_chart, create_fire_probability_chart


class MockResults:
    """Mock results object for testing visualization."""
    def __init__(self, all_runs, profile_age, profile_target_age):
        self.all_runs = all_runs
        self.profile_age = profile_age
        self.profile_target_age = profile_target_age


def test_create_projection_fan_chart():
    """Test creating projection fan chart."""
    # Mock results data - create 100 simulation runs with varying portfolio values
    all_runs = []
    for sim in range(100):
        run_data = []
        for year in range(21):
            portfolio_value = 10000 * (1.07 ** year) * (0.9 + 0.2 * (sim / 100))
            run_data.append({"year": year, "portfolio_value": portfolio_value})
        all_runs.append(run_data)

    results = MockResults(
        all_runs=all_runs,
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

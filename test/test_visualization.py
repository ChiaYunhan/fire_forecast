import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
from matplotlib.figure import Figure
from src.visualization import create_projection_fan_chart, create_fire_age_distribution


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


def test_create_fire_age_distribution():
    """Test creating FIRE age distribution histogram."""
    # Mock FIRE ages from successful runs
    fire_ages = [38, 40, 40, 42, 42, 42, 43, 45, 45, 47, 50]

    fig = create_fire_age_distribution(
        fire_ages=fire_ages,
        target_age=45,
        median_fire_age=42,
        success_rate=0.826,
        title="Test FIRE Distribution"
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Age When FIRE Achieved"
    assert ax.get_ylabel() == "Number of Simulations"
    assert "Test FIRE Distribution" in ax.get_title()

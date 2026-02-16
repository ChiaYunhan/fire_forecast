"""FIRE Forecast Engine - Main entry point."""
from pathlib import Path
import sys

from src.cli import parse_and_validate_args
from src.ScenarioConfig import ScenarioConfig
from src.ScenarioFactory import ScenarioFactory
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.visualization import create_projection_fan_chart, create_fire_age_distribution


def main():
    """Main entry point for FIRE Forecast Engine."""
    # Parse command-line arguments
    args = parse_and_validate_args()

    if args.verbose:
        print(f"Loading scenario from: {args.scenario}")

    # Load configuration
    config = ScenarioConfig.from_yaml(args.scenario)
    factory = ScenarioFactory()

    # Create profile
    profile = factory.create_profile(config)

    # Print profile summary
    print_profile_summary(profile)

    # Run Monte Carlo simulation
    n_sims = config.simulation_params.get("n_simulations", 10000)
    seed = config.simulation_params.get("seed", 42)

    if args.verbose:
        print(f"\nRunning {n_sims} Monte Carlo simulations...")

    engine = SimulationEngine(profile)
    runner = MonteCarloRunner(engine, n_simulations=n_sims, seed=seed)
    runner.run_simulations()
    results = runner.aggregate_results()

    # Print results
    print_results_summary(results)

    # Generate charts if requested
    if not args.no_charts:
        generate_charts(runner, results, profile, args.output, args.verbose)

    print("\nAnalysis complete!")


def print_profile_summary(profile):
    """Print financial profile summary."""
    portfolio = profile.portfolio
    expected_return = portfolio.expected_return()
    volatility = portfolio.volatility()

    print("=" * 60)
    print("  FIRE Forecast — Financial Profile")
    print("=" * 60)
    print(f"\n  Age: {profile.age} → Target: {profile.target_age}")
    print(f"  Years to FIRE: {profile.target_age - profile.age}")
    print(f"\n  Income:        ${profile.income:>12,.2f}")
    print(f"  Savings Rate:   {profile.savings_rate:>11.0%}")
    print(f"  Annual Savings: ${profile.annual_savings():>10,.2f}")
    print(f"\n  Portfolio Metrics:")
    print(f"    Expected Return: {expected_return:>9.2%}")
    print(f"    Volatility:      {volatility:>9.2%}")
    print("=" * 60)


def print_results_summary(results):
    """Print Monte Carlo results summary."""
    print("\n" + "=" * 60)
    print("  Monte Carlo Results")
    print("=" * 60)
    print(f"\n  FIRE Success Rate: {results.success_rate:>8.1%}")

    if results.median_fire_age:
        print(f"  Median FIRE Age: {results.median_fire_age:>10}")

    print(f"\n  Portfolio Value Percentiles:")
    print(f"    10th: ${results.portfolio_percentiles[10]:>15,.2f}")
    print(f"    50th: ${results.portfolio_percentiles[50]:>15,.2f}")
    print(f"    90th: ${results.portfolio_percentiles[90]:>15,.2f}")
    print("=" * 60)


def generate_charts(runner, results, profile, output_dir, verbose):
    """Generate and save/show visualization charts."""
    import matplotlib.pyplot as plt

    if verbose:
        print("\nGenerating charts...")

    # Projection fan chart
    fig1 = create_projection_fan_chart(
        runner,
        title=f"Portfolio Projection (Age {profile.age} → {profile.target_age})"
    )

    # FIRE age distribution histogram
    fire_ages = extract_fire_ages(runner)
    fig2 = create_fire_age_distribution(
        fire_ages=fire_ages,
        target_age=profile.target_age,
        median_fire_age=results.median_fire_age,
        success_rate=results.success_rate,
        title="FIRE Age Distribution"
    )

    # Save or show charts
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fig1.savefig(output_path / "projection_fan_chart.png", dpi=150)
        fig2.savefig(output_path / "fire_age_distribution.png", dpi=150)

        print(f"\nCharts saved to: {output_dir}")
    else:
        plt.show()


def extract_fire_ages(runner):
    """Extract list of FIRE ages from successful simulation runs."""
    fire_ages = []
    for result in runner.results:
        if result.get("fire_achieved", False):
            fire_ages.append(result["fire_age"])
    return fire_ages


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

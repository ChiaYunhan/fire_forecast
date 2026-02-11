from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
from src.MonteCarloRunner import MonteCarloRunner
from src.SensitivityAnalyzer import SensitivityAnalyzer
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


def main():
    # Sample ETF portfolio
    stock_etf = Asset(
        name="VTI (Total Stock Market)",
        allocation=0.80,
        expected_return=0.10,
        volatility=0.15,
    )
    bond_etf = Asset(
        name="BND (Total Bond Market)",
        allocation=0.20,
        expected_return=0.04,
        volatility=0.04,
    )

    portfolio = Portfolio(
        composition=[stock_etf, bond_etf],
        total_value=0.0,
        allocation_methods="aggressive",
    )

    profile = FinancialProfile(
        income=50_056.92,
        expenses_rate=0.58,
        savings_rate=0.42,
        portfolio=portfolio,
        age=25,
        target_age=45,
    )

    print_summary(profile)
    run_monte_carlo_analysis(profile)
    run_sensitivity_analysis(profile)


def print_summary(profile: FinancialProfile):
    print("=" * 50)
    print("  FIRE Forecast — Financial Profile Summary")
    print("=" * 50)

    print(f"\n  Age: {profile.age} → Target FIRE Age: {profile.target_age}")
    print(f"  Years to FIRE: {profile.target_age - profile.age}")

    print(f"\n  Income:       ${profile.income:>12,.2f}")
    print(f"  Expenses Rate: {profile.expenses_rate:>12.0%}")
    print(f"  Savings Rate:  {profile.savings_rate:>12.0%}")
    print(f"  Annual Expenses: ${profile.annual_expenses():>10,.2f}")
    print(f"  Annual Savings:  ${profile.annual_savings():>10,.2f}")

    print(f"\n  Portfolio Value: ${profile.portfolio.total_value:>10,.2f}")
    print(f"  Strategy: {profile.portfolio.allocation_methods}")
    print(f"\n  Assets:")
    for asset in profile.portfolio.composition:
        print(f"    • {asset.name}")
        print(
            f"      Allocation: {asset.allocation:.0%} | "
            f"Return: {asset.expected_return:.0%} | "
            f"Volatility: {asset.volatility:.0%}"
        )

    print("\n" + "=" * 50)


def run_monte_carlo_analysis(profile: FinancialProfile):
    """Run Monte Carlo simulations with all three strategies."""
    print("\n" + "=" * 60)
    print("  Monte Carlo Analysis (1,000 simulations per strategy)")
    print("=" * 60)

    strategies = [
        AggressiveStrategy(),
        BalancedStrategy(),
        ConservativeStrategy(),
    ]

    for strategy in strategies:
        engine = SimulationEngine(profile, strategy)
        runner = MonteCarloRunner(engine, n_simulations=1000, seed=42)

        runner.run_simulations()
        results = runner.aggregate_results()

        print(f"\n  Strategy: {strategy.name}")
        print(f"  Risk Multiplier: {strategy.get_risk_multiplier()}x")
        print("-" * 60)
        print(f"  FIRE Success Rate: {results.success_rate:>8.1%}")

        if results.median_fire_age:
            print(f"  Median FIRE Age: {results.median_fire_age:>10}")
            print(f"  Avg Years to FIRE: {results.average_years_to_fire:>8.1f}")

        print(f"\n  Portfolio Value Percentiles:")
        print(f"    10th: ${results.portfolio_percentiles[10]:>15,.2f}")
        print(f"    25th: ${results.portfolio_percentiles[25]:>15,.2f}")
        print(f"    50th: ${results.portfolio_percentiles[50]:>15,.2f}")
        print(f"    75th: ${results.portfolio_percentiles[75]:>15,.2f}")
        print(f"    90th: ${results.portfolio_percentiles[90]:>15,.2f}")

        print(f"\n  Risk Metrics:")
        print(f"    Best Case:  ${results.best_case_portfolio:>15,.2f}")
        print(f"    Worst Case: ${results.worst_case_portfolio:>15,.2f}")
        print(f"    Max Drawdown: {results.max_drawdown:>13.1%}")
        if results.shortfall_amount:
            print(f"    Avg Shortfall: ${results.shortfall_amount:>13,.2f}")

    print("\n" + "=" * 60 + "\n")


def run_sensitivity_analysis(profile: FinancialProfile):
    """Demonstrate sensitivity analysis on savings rate."""
    print("\n" + "=" * 60)
    print("  Sensitivity Analysis: Impact of Savings Rate")
    print("=" * 60)

    analyzer = SensitivityAnalyzer(profile, BalancedStrategy())

    savings_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
    results = analyzer.sweep_savings_rate(
        rates=savings_rates,
        n_simulations=500,
        seed=42
    )

    print("\n  Savings Rate  →  FIRE Success Rate")
    print("-" * 60)
    for rate, result in zip(savings_rates, results):
        print(f"     {rate:>4.0%}                  {result.success_rate:>6.1%}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

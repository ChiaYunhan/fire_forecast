from src.models import Asset, Portfolio, FinancialProfile
from src.SimulationEngine import SimulationEngine
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
    run_simulations(profile)


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


def run_simulations(profile: FinancialProfile):
    """Run simulations with all three strategies and compare results."""
    print("\n" + "=" * 50)
    print("  Running Simulations (Single Run per Strategy)")
    print("=" * 50)

    strategies = [
        AggressiveStrategy(),
        BalancedStrategy(),
        ConservativeStrategy(),
    ]

    for strategy in strategies:
        engine = SimulationEngine(profile, strategy)
        results = engine.run()

        print(f"\n  Strategy: {strategy.name}")
        print(f"  Risk Multiplier: {strategy.get_risk_multiplier()}x")
        print("-" * 50)
        print(f"  Final Portfolio Value: ${results['final_portfolio_value']:>15,.2f}")
        print(f"  FIRE Target (25x expenses): ${results['fire_target']:>12,.2f}")
        print(f"  FIRE Achieved: {results['fire_achieved']}")
        print(f"  Years Simulated: {results['years_simulated']}")
        print(f"  Final Age: {results['final_age']}")

    print("\n" + "=" * 50)
    print("  Note: Single runs show high variance due to randomness.")
    print("  Monte Carlo (Phase 3) will run thousands of simulations.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

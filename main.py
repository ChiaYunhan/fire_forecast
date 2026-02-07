from src.models import Asset, Portfolio, FinancialProfile


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


if __name__ == "__main__":
    main()

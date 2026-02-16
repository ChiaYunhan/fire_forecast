"""Factory for creating domain objects from scenario configurations."""
from src.ScenarioConfig import ScenarioConfig
from src.models import Asset, Portfolio, FinancialProfile
from src.Strategy.base import InvestmentStrategy
from src.Strategy.aggressive import AggressiveStrategy
from src.Strategy.balanced import BalancedStrategy
from src.Strategy.conservative import ConservativeStrategy


class ScenarioFactory:
    """Factory for creating scenarios from configuration."""

    def create_profile(self, config: ScenarioConfig) -> FinancialProfile:
        """Create FinancialProfile from config."""
        # Create assets
        assets = []
        for asset_data in config.portfolio_data["assets"]:
            asset = Asset(
                name=asset_data["name"],
                allocation=asset_data["allocation"],
                expected_return=asset_data["expected_return"],
                volatility=asset_data["volatility"]
            )
            assets.append(asset)

        # Create portfolio
        portfolio = Portfolio(
            composition=assets,
            total_value=config.portfolio_data.get("total_value", 0.0),
            allocation_methods=config.portfolio_data.get("allocation_method", "balanced")
        )

        # Create profile
        profile = FinancialProfile(
            age=config.profile_data["age"],
            target_age=config.profile_data["target_age"],
            income=config.profile_data["income"],
            expenses_rate=config.profile_data["expenses_rate"],
            savings_rate=config.profile_data["savings_rate"],
            portfolio=portfolio
        )

        return profile

    def create_strategy(self, strategy_name: str) -> InvestmentStrategy:
        """Create strategy from name."""
        strategies = {
            "aggressive": AggressiveStrategy,
            "balanced": BalancedStrategy,
            "conservative": ConservativeStrategy
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategies[strategy_name]()

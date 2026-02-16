"""Factory for creating domain objects from scenario configurations."""
from src.ScenarioConfig import ScenarioConfig
from src.models import Asset, Portfolio, FinancialProfile


class ScenarioFactory:
    """Factory for creating scenarios from configuration."""

    def create_profile(self, config: ScenarioConfig) -> FinancialProfile:
        """Create FinancialProfile from config."""
        # Create assets with optional costs
        assets = []
        for asset_data in config.portfolio_data["assets"]:
            asset = Asset(
                name=asset_data["name"],
                allocation=asset_data["allocation"],
                expected_return=asset_data["expected_return"],
                volatility=asset_data["volatility"],
                costs=asset_data.get("costs"),  # Optional
                risk_metrics=asset_data.get("risk_metrics")  # Optional
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

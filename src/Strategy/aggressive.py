from .base import InvestmentStrategy

from ..models import Portfolio


class AggressiveStrategy(InvestmentStrategy):
    """
    Aggressive investment strategy: high equity allocation, higher volatility.

    Suitable for younger investors with long time horizons who can tolerate
    significant market swings for potentially higher returns.
    """

    def calculate_annual_return(self, portfolio: Portfolio, year: int) -> float:
        """Calculate return using high-risk parameters."""
        return self._calculate_portfolio_return(portfolio)

    def get_risk_multiplier(self) -> float:
        """Aggressive strategy accepts 1.5x normal volatility."""
        return 1.5

    @property
    def name(self) -> str:
        return "Aggressive"

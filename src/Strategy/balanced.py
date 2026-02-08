from .base import InvestmentStrategy

from ..models import Portfolio


class BalancedStrategy(InvestmentStrategy):
    """
    Balanced investment strategy: moderate risk and return expectations.

    Suitable for investors seeking growth with moderate volatility tolerance.
    Balances between aggressive growth and capital preservation.
    """

    def calculate_annual_return(self, portfolio: Portfolio, year: int) -> float:
        """Calculate return using moderate-risk parameters."""
        return self._calculate_portfolio_return(portfolio)

    def get_risk_multiplier(self) -> float:
        """Aggressive strategy accepts 1.1x normal volatility."""
        return 1.1

    @property
    def name(self) -> str:
        return "Balanced"

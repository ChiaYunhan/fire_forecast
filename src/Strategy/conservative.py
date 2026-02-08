from .base import InvestmentStrategy

from ..models import Portfolio


class ConservativeStrategy(InvestmentStrategy):
    """
    Conservative investment strategy: low risk and return expectations.

    Suitable for investors nearing retirement or with low risk tolerance.
    Prioritizes capital preservation over aggressive growth.
    """

    def calculate_annual_return(self, portfolio: Portfolio, year: int) -> float:
        """Calculate return using low-risk parameters."""
        return self._calculate_portfolio_return(portfolio)

    def get_risk_multiplier(self) -> float:
        """Conservative strategy accepts 0.8x normal volatility."""
        return 0.8

    @property
    def name(self) -> str:
        return "Conservative"

from abc import ABC, abstractmethod
import numpy as np

from ..models import Portfolio


class InvestmentStrategy(ABC):
    """
    Abstract base class for investment strategies.

    Defines how portfolio returns are calculated during simulation,
    including risk parameters and volatility adjustments.
    """

    @abstractmethod
    def calculate_annual_return(self, portfolio: Portfolio, year: int) -> float:
        """
        Calculate the annual return for the portfolio based on this strategy.

        Args:
            portfolio: The portfolio to calculate returns for
            year: Current simulation year (for age-based adjustments)

        Returns:
            The portfolio return for this year (e.g., 0.08 for 8%)
        """
        pass

    @abstractmethod
    def get_risk_multiplier(self) -> float:
        """
        Return the risk/volatility multiplier for this strategy.

        Returns:
            Multiplier applied to asset volatility (e.g., 1.0 = normal, 1.5 = higher risk)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass

    def _calculate_portfolio_return(self, portfolio: "Portfolio") -> float:
        """
        Helper method: calculate weighted average return with volatility.

        This applies the risk multiplier to each asset's volatility and
        generates a randomized return using normal distribution.
        """
        total_return = 0.0
        risk_mult = self.get_risk_multiplier()

        for asset in portfolio.composition:
            # Base expected return
            expected = asset.expected_return

            # Add randomness based on volatility * risk multiplier
            volatility = asset.volatility * risk_mult
            random_shock = np.random.normal(0, volatility)

            # Weighted by allocation
            asset_return = (expected + random_shock) * asset.allocation
            total_return += asset_return

        return total_return

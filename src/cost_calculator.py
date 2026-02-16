"""Cost calculation utilities for investment fees and expenses."""


def apply_ter(portfolio_value: float, ter: float) -> float:
    """
    Apply Total Expense Ratio (TER) to portfolio value.

    Args:
        portfolio_value: Current portfolio value
        ter: Total Expense Ratio as decimal (e.g., 0.0019 for 0.19%)

    Returns:
        Portfolio value after TER deduction
    """
    return portfolio_value * (1 - ter)


def apply_trading_fee(contribution: float, trading_fee: float) -> float:
    """
    Apply trading fee to a contribution/purchase amount.

    Args:
        contribution: Amount being invested
        trading_fee: Trading fee as decimal (e.g., 0.0005 for 0.05%)

    Returns:
        Net contribution after trading fee deduction
    """
    return contribution * (1 - trading_fee)

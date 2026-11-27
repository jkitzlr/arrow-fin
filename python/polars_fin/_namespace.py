from __future__ import annotations

from typing import TYPE_CHECKING, Self

from polars import Expr
from polars.api import register_expr_namespace

from ._functional import sharpe

if TYPE_CHECKING:
    from polars_fin.typing import IntoExprColumn


@register_expr_namespace("financial")
class FinancialNamespace:
    returns: Expr

    def __init__(self: Self, expr: Expr) -> None:
        self.returns = expr

    def sharpe(self: Self, cash: IntoExprColumn) -> Expr:
        """Calculate the sharpe ratio of the expression over `cash` rate.

        NOTE: The function currently assumes the returns and cash rate(s) are expressed
        with the same compounding frequency. This will be fixed in a future release.

        Args:
            cash: Expression for the cash rate in the calculation.

        Returns:
            Sharpe ratio aggregated over the input return series.

        Examples:
            >>> import polars as pl
            >>> df = pl.DataFrame({
            ...     'portfolio': [*(['a'] * 5), *(['b'] * 5)],
            ...     'returns': [20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4],
            ...     'cash': [4e-2 / 252.0] * 10
            ... })
            >>> df.select(pl.col('returns').financial.sharpe('cash'))
            shape: (1, 1)
            ┌──────────┐
            │ returns  │
            │ ---      │
            │ f64      │
            ╞══════════╡
            │ 0.177067 │
            └──────────┘

            This similarly works over groups:

            >>> df.group_by('portfolio').agg(
            ...     pl.col('returns').financial.sharpe('cash')
            ... ).sort('portfolio')
            shape: (2, 2)
            ┌───────────┬──────────┐
            │ portfolio ┆ returns  │
            │ ---       ┆ ---      │
            │ str       ┆ f64      │
            ╞═══════════╪══════════╡
            │ a         ┆ 0.237947 │
            │ b         ┆ 0.083746 │
            └───────────┴──────────┘
        """
        return sharpe(self.returns, cash)

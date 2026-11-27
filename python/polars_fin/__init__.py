from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from polars import Expr
from polars.api import register_expr_namespace
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars_fin.typing import IntoExprColumn

    Frequency = Literal[
        "annual", "semiannual", "quarterly", "monthly", "weekly", "daily", "busdays"
    ]


LIB = Path(__file__).parent


@register_expr_namespace("financial")
class FinancialNamespace:
    returns: Expr

    def __init__(self: Self, expr: Expr) -> None:
        self.returns = expr

    def sharpe(
        self: Self,
        cash: IntoExprColumn,
        *,
        returns_freq: Frequency,
        cash_freq: Frequency,
    ) -> Expr:
        """Calculate the sharpe ratio of the expression over `cash` rate.

        The resulting Sharpe ratio is stated with the same frequency as the input
        `returns` data. Cash rate(s) are adjusted to the returns freq arithmetically.

        Args:
            cash: Expression for the cash rate in the calculation.
            returns_freq: Frequency per annum of the return periods.
            cash_freq: Frequency per annum of the cash rate periods.

        Returns:
            Sharpe ratio aggregated over the input return series.

        Examples:
            >>> import polars as pl
            >>> df = pl.DataFrame({
            ...     'portfolio': [*(['a'] * 5), *(['b'] * 5)],
            ...     'returns': [20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4],
            ...     'cash': [4e-2 / 252.0] * 10
            ... })
            >>> df.select(
            ...     pl.col('returns')
            ...     .financial
            ...     .sharpe('cash', returns_freq='daily', cash_freq='daily')
            ... )
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
            ...     pl.col('returns')
            ...     .financial
            ...     .sharpe('cash', returns_freq='daily', cash_freq='daily')
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
        return register_plugin_function(
            args=[self.returns, cash],
            plugin_path=LIB,
            function_name="sharpe",
            is_elementwise=False,
            returns_scalar=True,
            kwargs={
                "return_freq": returns_freq,
                "cash_freq": cash_freq,
            },
        )

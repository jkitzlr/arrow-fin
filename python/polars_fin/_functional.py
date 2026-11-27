from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_fin._internal import __version__

if TYPE_CHECKING:
    from polars_fin.typing import IntoExprColumn

LIB = Path(__file__).parent


def sharpe(returns: IntoExprColumn, cash: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[returns, cash],
        plugin_path=LIB,
        function_name="sharpe",
        is_elementwise=False,
        returns_scalar=True,
    )

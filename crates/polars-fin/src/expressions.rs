use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::bridge::ArrowWrapper;

// TODO: validate schema(?)
// TODO: need an accumulating implemenation of the performance functions to avoid the rechnunking
#[polars_expr(output_type=Float64)]
fn sharpe(inputs: &[Series]) -> PolarsResult<Series> {
    let rtn: ArrowWrapper = inputs[0]
        .rechunk()
        .to_arrow(0, CompatLevel::newest())
        .into();
    let bmk: ArrowWrapper = inputs[1]
        .rechunk()
        .to_arrow(0, CompatLevel::newest())
        .into();

    let sharpe = returns::sharpe(&rtn.0, &bmk.0)
        .map_err(|_| polars_err!(ComputeError: "Unable to compute Sharpe ratio."))?;

    Ok(Series::new(PlSmallStr::EMPTY, &[sharpe]))
}

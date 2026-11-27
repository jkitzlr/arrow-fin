use arrow_fin_perf::{Frequency, Sharpe};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::bridge::ArrowWrapper;

#[derive(Deserialize)]
struct SharpeKwargs {
    return_freq: String,
    cash_freq: String,
}

// TODO: validate schema(?)
// TODO: need an accumulating implemenation of the performance functions to avoid the rechnunking
#[polars_expr(output_type=Float64)]
fn sharpe(inputs: &[Series], kwargs: SharpeKwargs) -> PolarsResult<Series> {
    let rtn: ArrowWrapper = inputs[0]
        .rechunk()
        .to_arrow(0, CompatLevel::newest())
        .into();
    let bmk: ArrowWrapper = inputs[1]
        .rechunk()
        .to_arrow(0, CompatLevel::newest())
        .into();

    let return_freq = kwargs
        .return_freq
        .as_str()
        .parse::<Frequency>()
        .map_err(|msg| PolarsError::ComputeError(msg.into()))?;
    let cash_freq = kwargs
        .cash_freq
        .as_str()
        .parse::<Frequency>()
        .map_err(|msg| PolarsError::ComputeError(msg.into()))?;

    let sharpe = Sharpe::from_arrow(&rtn.0, &bmk.0, return_freq, cash_freq);
    let rslt = sharpe
        .map(|val| val.result())
        .map_err(|_| polars_err!(ComputeError: "Unable to compute Sharpe ratio."))?;

    Ok(Series::new(PlSmallStr::EMPTY, &[rslt]))
}

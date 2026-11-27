use std::str::FromStr;

use arrow::array::{ArrayRef, AsArray, Datum};
use arrow::compute::kernels::numeric::sub_wrapping;
use arrow::datatypes::Float64Type;
use arrow::error::ArrowError::ComputeError;
use arrow::error::Result as ArrowResult;
use arrow_fin_stats::PrimitiveArrayExt;

/// Calculate the excess return of a total return series over bmk returns.
///
/// The total and benchmark returns are assumed to be effective rates over consistent periods.
pub fn execess_return(rtn: &dyn Datum, bmk: &dyn Datum) -> ArrowResult<ArrayRef> {
    sub_wrapping(rtn, bmk)
}

// TODO: need to handle compounding frequency logic
pub fn sharpe(rtn: &dyn Datum, cash_rate: &dyn Datum) -> ArrowResult<f64> {
    let xs = execess_return(rtn, cash_rate)?;
    let xs = xs.as_primitive::<Float64Type>();

    let mean = xs.calc_mean().ok_or_else(|| {
        ComputeError(String::from_str("Unable to compute the mean excess return.").unwrap())
    })?;
    let vol = xs.calc_stdev(1).ok_or_else(|| {
        ComputeError(String::from_str("Unable to compute the excess return vol.").unwrap())
    })?;

    Ok(mean / vol)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array};
    use arrow::datatypes::Float64Type;
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand::rngs::StdRng;
    use statrs::distribution::Normal;

    use super::{execess_return, sharpe};

    fn asset_returns() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(0);
        let norm = Normal::new(0.0, 0.05).unwrap();
        norm.sample_iter(&mut rng).take(10).collect()
    }

    fn bmk_returns() -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(42);
        let norm = Normal::new(0.0, 0.05).unwrap();
        norm.sample_iter(&mut rng).take(10).collect()
    }

    #[test]
    fn excess_return_array_array() {
        let expected: Vec<f64> = asset_returns()
            .iter()
            .zip(bmk_returns().iter())
            .map(|(a, b)| a - b)
            .collect();

        let rtn = Float64Array::from(asset_returns());
        let bmk = Float64Array::from(bmk_returns());

        let rslt = execess_return(&rtn, &bmk).unwrap();
        let rslt = rslt.as_primitive::<Float64Type>().values().to_vec();

        assert_eq!(rslt, expected);
    }

    #[test]
    fn excess_return_array_scalar() {
        let cash_rate = 5e-2;

        let expected: Vec<f64> = asset_returns().iter().map(|r| r - cash_rate).collect();

        let rtn = Float64Array::from(asset_returns());
        let bmk = Float64Array::new_scalar(cash_rate);

        let rslt = execess_return(&rtn, &bmk).unwrap();
        let rslt = rslt.as_primitive::<Float64Type>().values().to_vec();

        assert_eq!(rslt, expected);
    }

    #[test]
    fn sharpe_ratio() {
        let returns = Float64Array::from(vec![
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ]);
        let cash_rate = Float64Array::new_scalar(4e-2 / 252.0);

        let rslt = sharpe(&returns, &cash_rate).unwrap();
        let expected = 0.17706677867281076;

        assert!((rslt - expected).abs() < 1e-8)
    }
}

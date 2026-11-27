use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::error::ArrowError;
use arrow_fin_stats::moments::{Mean, Variance};
use arrow_fin_util::ArrowNumericExt;

use crate::Frequency;

#[derive(Clone, Debug)]
pub struct Sharpe {
    var: Variance,
    mean: Mean,
    freq: Frequency,
}

// TODO: does this make sense?
impl Default for Sharpe {
    fn default() -> Self {
        Self::with_freq(Frequency::Annual)
    }
}

impl Sharpe {
    /// Create an empty instance with the given frequency
    pub fn with_freq(freq: Frequency) -> Self {
        Self {
            var: Variance::default(),
            mean: Mean::default(),
            freq,
        }
    }

    // TODO: need to validate that the cash rate can be converted to returns' freq
    // TODO: need proper error type
    pub fn new(
        returns: &[f64],
        cash: &[f64],
        return_freq: Frequency,
        cash_freq: Frequency,
    ) -> Result<Self, String> {
        if returns.len() != cash.len() {
            return Err("Return and cash rate arrays have different sized!".to_owned());
        }

        let factor = cash_freq.multiplier(&return_freq);
        let excess = returns.iter().zip(cash.iter()).map(|(r, c)| r - c * factor);
        let var = Variance::from_iter(excess);

        Ok(Self {
            var,
            mean: var.into(),
            freq: return_freq,
        })
    }

    // TODO: handle proper compounding
    pub fn new_const_cash_rate(
        returns: &[f64],
        cash: f64,
        return_freq: Frequency,
        cash_freq: Frequency,
    ) -> Result<Self, String> {
        // effective cash rate
        let cash = cash * cash_freq.multiplier(&return_freq);
        let excess = returns.iter().map(|r| r - cash);
        let var = Variance::from_iter(excess);

        Ok(Self {
            var,
            mean: var.into(),
            freq: return_freq,
        })
    }

    // TODO: the compute pattern here is going to be used a lot, need to abstract it out
    pub fn from_arrow(
        returns: &dyn Datum,
        cash: &dyn Datum,
        return_freq: Frequency,
        cash_freq: Frequency,
    ) -> Result<Self, ArrowError> {
        let (returns, r_scalar) = returns.get();
        let (cash, c_scalar) = cash.get();

        match (r_scalar, c_scalar) {
            // both arrays
            (false, false) => {
                let returns = returns.f64()?;
                let cash = cash.f64()?;

                let r_values = returns.values();
                let c_values = cash.values();

                // merge null buffers
                let nulls = NullBuffer::union(returns.nulls(), cash.nulls());
                match nulls {
                    // fast path; no nulls
                    None => Self::new(r_values, c_values, return_freq, cash_freq)
                        .map_err(ArrowError::ComputeError),
                    Some(nulls) => {
                        let mut out = Sharpe::with_freq(return_freq);
                        for (start, end) in nulls.inner().set_slices() {
                            let sharpe = Sharpe::new(
                                &r_values[start..end],
                                &c_values[start..end],
                                return_freq,
                                cash_freq,
                            )
                            .map_err(ArrowError::ComputeError)?;
                            out = out.merge(&sharpe).unwrap();
                        }
                        Ok(out)
                    }
                }
            }
            // (false, true) => Err(ArrowError::ComputeError("stub".to_owned())),
            (false, true) => {
                let returns = returns.f64()?;
                let values = returns.values();

                let cash_rate = cash.f64()?.value(0);

                match returns.nulls() {
                    None => Self::new_const_cash_rate(values, cash_rate, return_freq, cash_freq)
                        .map_err(ArrowError::ComputeError),
                    Some(nulls) => {
                        let mut out = Sharpe::with_freq(return_freq);
                        for (start, end) in nulls.inner().set_slices() {
                            let sharpe = Self::new_const_cash_rate(
                                &values[start..end],
                                cash_rate,
                                return_freq,
                                cash_freq,
                            )
                            .map_err(ArrowError::ComputeError)?;
                            out = out.merge(&sharpe).unwrap();
                        }
                        Ok(out)
                    }
                }
            }
            (true, _) => Err(ArrowError::InvalidArgumentError(
                "`returns` must be an array.".to_owned(),
            )),
        }
    }

    pub fn merge(&self, other: &Self) -> Option<Self> {
        // TODO: maybe just convert between them?
        if self.freq != other.freq {
            return None;
        }

        let var = self.var.merge(&other.var);
        let mean = self.mean.merge(&other.mean);
        let freq = self.freq;

        Some(Self { var, mean, freq })
    }

    pub fn result(&self) -> Option<f64> {
        let var = self.var.std(1)?;
        let mean = self.mean.result()?;
        Some(mean / var)
    }

    pub fn result_annualized(&self) -> Option<f64> {
        self.result()
            .map(|sr| sr * self.freq.multiplier(&Frequency::Annual).sqrt())
    }
}

// TODO: proper error handling and need a fallible version
// impl FromIterator<Sharpe> for Sharpe {
//     fn from_iter<I: IntoIterator<Item = Sharpe>>(iter: I) -> Self {
//         iter.into_iter()
//             .fold(Sharpe::default(), |acc, this| acc.merge(&this).unwrap())
//     }
// }

#[cfg(test)]
mod tests {
    use arrow::array::*;

    use crate::Frequency;

    use super::Sharpe;

    #[test]
    fn sharpe() {
        let returns = &[
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ];
        let cash = &[4e-2; 10];
        let sharpe = Sharpe::new(
            returns,
            cash,
            crate::Frequency::BusDays,
            crate::Frequency::Annual,
        )
        .unwrap();

        let rslt = sharpe.result().unwrap();
        let expected = 0.17706677867281076;

        assert!((rslt - expected).abs() < 1e-8)
    }

    #[test]
    fn sharpe_const_cash_rate() {
        let returns = &[
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ];
        let cash = 4e-2;
        let sharpe = Sharpe::new_const_cash_rate(
            returns,
            cash,
            crate::Frequency::BusDays,
            crate::Frequency::Annual,
        )
        .unwrap();

        let rslt = sharpe.result().unwrap();
        let expected = 0.17706677867281076;

        assert!((rslt - expected).abs() < 1e-8)
    }

    #[test]
    fn sharpe_annualized() {
        let returns = &[
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ];
        let cash = 4e-2;
        let sharpe = Sharpe::new_const_cash_rate(
            returns,
            cash,
            crate::Frequency::BusDays,
            crate::Frequency::Annual,
        )
        .unwrap();

        let rslt = sharpe.result_annualized().unwrap();
        let expected = 0.17706677867281076 * 252f64.sqrt();

        assert!((rslt - expected).abs() < 1e-8)
    }

    #[test]
    fn from_arrow() {
        let returns = Float64Array::from(vec![
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ]);
        let cash_rate = Float64Array::from(vec![4e-2; 10]);
        let return_freq = Frequency::BusDays;
        let cash_freq = Frequency::Annual;

        let sharpe = Sharpe::from_arrow(&returns, &cash_rate, return_freq, cash_freq).unwrap();
        let rslt = sharpe.result().unwrap();
        let expected = 0.17706677867281076;

        assert!((rslt - expected).abs() < 1e-8)
    }

    #[test]
    fn from_arrow_cash_scalar() {
        let returns = Float64Array::from(vec![
            20e-4, -40e-4, 50e-4, -10e-4, 30e-4, -20e-4, 40e-4, 10e-4, -30e-4, 20e-4,
        ]);
        let cash_rate = Float64Array::new_scalar(4e-2);
        let return_freq = Frequency::BusDays;
        let cash_freq = Frequency::Annual;

        let sharpe = Sharpe::from_arrow(&returns, &cash_rate, return_freq, cash_freq).unwrap();
        let rslt = sharpe.result().unwrap();
        let expected = 0.17706677867281076;

        assert!((rslt - expected).abs() < 1e-8)
    }
}

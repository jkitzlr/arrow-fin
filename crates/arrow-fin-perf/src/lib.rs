use std::str::FromStr;

use arrow::array::{ArrayRef, Datum};
use arrow::compute::kernels::numeric::sub_wrapping;
use arrow::error::Result as ArrowResult;

pub use sharpe::Sharpe;

mod sharpe;

// TODO: move this to a generic finance functionality crate
// TODO: implement a Period (CompoundingPeriod?) counterpart struct
// TODO: do I need daily? or just let consumers implement own logic on Daily; technically this already wrong, e.g., equity vs fi
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Frequency {
    Annual,
    SemiAnnual,
    Quarterly,
    Monthly,
    Weekly,
    Daily,
    BusDays,
}

impl Frequency {
    /// Tests whether the frequency is meaasured in months (i.e. fixed per year)
    pub fn is_monthly(&self) -> bool {
        use Frequency::*;

        matches!(*self, Annual | SemiAnnual | Quarterly | Monthly)
    }

    /// Get the number of instances of the period in a year.
    pub fn numeric(&self) -> u32 {
        use Frequency::*;

        match *self {
            Annual => 1,
            SemiAnnual => 2,
            Quarterly => 4,
            Monthly => 12,
            Weekly => 52,
            Daily => 356,
            BusDays => 252,
        }
    }

    /// Calculate the mulitplier to convert `self` to `other` Frequency.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// let src = Frequency::SemiAnnual;
    /// let tgt = Frequency::Quarterly;
    /// assert_eq!(src.muliptlier(tgt), 2.0);
    /// ```
    pub fn multiplier(&self, other: &Self) -> f64 {
        (self.numeric() as f64) / (other.numeric() as f64)
    }
}

impl FromStr for Frequency {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use Frequency::*;

        match s.to_lowercase().as_str() {
            "annual" => Ok(Annual),
            "semiannual" => Ok(SemiAnnual),
            "quarterly" => Ok(Quarterly),
            "monthly" => Ok(Monthly),
            "weekly" => Ok(Weekly),
            "daily" => Ok(Daily),
            "busdays" => Ok(BusDays),
            _ => Err(format!("Unsupported frequency {s}.")),
        }
    }
}

macro_rules! freq_try_from_int {
    ($src:ty) => {
        impl TryFrom<$src> for Frequency {
            type Error = String;

            fn try_from(value: $src) -> Result<Self, Self::Error> {
                use Frequency::*;

                match value {
                    1 => Ok(Annual),
                    2 => Ok(SemiAnnual),
                    4 => Ok(Quarterly),
                    12 => Ok(Monthly),
                    52 => Ok(Weekly),
                    365 => Ok(Daily),
                    252 => Ok(BusDays),
                    _ => Err(format!("Unrecognized frequency {value}")),
                }
            }
        }
    };
}

freq_try_from_int!(u16);
freq_try_from_int!(u32);
freq_try_from_int!(u64);
freq_try_from_int!(i16);
freq_try_from_int!(i32);
freq_try_from_int!(i64);

/// Calculate the excess return of a total return series over bmk returns.
///
/// The total and benchmark returns are assumed to be effective rates over consistent periods.
pub fn execess_return(rtn: &dyn Datum, bmk: &dyn Datum) -> ArrowResult<ArrayRef> {
    sub_wrapping(rtn, bmk)
}

#[cfg(test)]
mod tests {
    use arrow::array::{AsArray, Float64Array};
    use arrow::datatypes::Float64Type;
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand::rngs::StdRng;
    use statrs::distribution::Normal;

    use super::execess_return;

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
}

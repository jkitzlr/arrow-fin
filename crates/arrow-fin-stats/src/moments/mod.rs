use arrow::array::*;

pub use mean::Mean;
pub use var::Variance;

mod mean;
mod var;

pub trait Moments {
    /// Get the mean calculation for the data.
    fn mean(&self) -> Option<Mean>;

    /// Get the variance (and standard deviation) calculation for the data.
    fn variance(&self) -> Option<Variance>;
}

impl<T: ArrowNumericType> Moments for PrimitiveArray<T> {
    fn mean(&self) -> Option<Mean> {
        Mean::from_array(self)
    }

    fn variance(&self) -> Option<Variance> {
        Variance::from_array(self)
    }
}

use arrow::array::{Array, ArrowNumericType, AsArray, Float64Array, PrimitiveArray};
use arrow::compute::{cast, sum};
use arrow::datatypes::DataType;

pub use var::Variance;

mod var;

// TODO: seal this extension trait
// TODO: move this into a separate utils crate
/// Extension trait to provide automatic upcasting for numeric arrays
pub trait PrimitiveArrayExt {
    // TODO: return Result?
    fn f64(&self) -> Option<Float64Array>;

    // TODO: implement the struct
    // fn mean(&self) -> Option<ArithmeticMean>;

    fn variance(&self) -> Option<Variance>;

    fn calc_mean(&self) -> Option<f64>;

    fn calc_variance(&self, ddof: u8) -> Option<f64> {
        let var = self.variance()?;
        var.result(ddof)
    }

    fn calc_stdev(&self, ddof: u8) -> Option<f64> {
        self.calc_variance(ddof).map(|v| v.sqrt())
    }
}

impl<T: ArrowNumericType> PrimitiveArrayExt for PrimitiveArray<T> {
    fn f64(&self) -> Option<Float64Array> {
        let data_f64 = cast(self, &DataType::Float64).ok()?;
        Some(data_f64.as_primitive().clone())
    }

    fn calc_mean(&self) -> Option<f64> {
        if self.is_empty() || self.len() == self.null_count() {
            return None;
        }

        let data = self.f64()?;

        let total = sum(&data)?;
        let count = data.len() - data.null_count();

        Some(total / count as f64)
    }

    fn variance(&self) -> Option<Variance> {
        Variance::from_array(self)
    }
}

pub fn mean<T>(data: &PrimitiveArray<T>) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.calc_mean()
}

pub fn variance<T>(data: &PrimitiveArray<T>, ddof: u8) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.calc_variance(ddof)
}

pub fn stdev<T>(data: &PrimitiveArray<T>, ddof: u8) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.calc_stdev(ddof)
}

// TODO: generate test for all numeric types using a macro
#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int32Array};

    use super::mean;

    const VALUES: &[i8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9];

    const VALUES_NULL: &[Option<i8>] = &[
        None,
        None,
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
    ];

    #[test]
    fn int_array() {
        let array = Int32Array::from_iter_values(VALUES.iter().map(|x| *x as i32));
        assert_eq!(mean(&array), Some(5.0));
    }

    #[test]
    fn int_array_nulls() {
        let array = Int32Array::from_iter(VALUES_NULL.iter().map(|x| x.map(|x| x as i32)));
        assert_eq!(mean(&array), Some(6.0))
    }

    #[test]
    fn float_array() {
        let array = Float64Array::from_iter_values(VALUES.iter().map(|x| *x as f64));
        assert_eq!(mean(&array), Some(5.0));
    }

    #[test]
    fn float_array_nulls() {
        let array = Float64Array::from_iter(VALUES_NULL.iter().map(|x| x.map(|x| x as f64)));
        assert_eq!(mean(&array), Some(6.0))
    }
}

use arrow::array::*;

use moments::Moments;

pub mod moments;

// TODO: are these functions really needed...?

pub fn mean<T>(data: &PrimitiveArray<T>) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.mean().and_then(|mean| mean.result())
}

pub fn variance<T>(data: &PrimitiveArray<T>, ddof: usize) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.variance().and_then(|mean| mean.result(ddof))
}

pub fn stdev<T>(data: &PrimitiveArray<T>, ddof: usize) -> Option<f64>
where
    T: ArrowNumericType,
{
    data.variance().and_then(|mean| mean.std(ddof))
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

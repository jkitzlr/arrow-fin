use arrow::array::*;

use arrow_fin_util::ArrowNumericExt;

use super::Variance;

#[derive(Clone, Copy, Debug, Default)]
pub struct Mean {
    pub(super) sum: f64,     // sum of elements
    pub(super) count: usize, // num elements
}

impl Mean {
    /// Create a new instance from a set of data.
    pub fn new(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let count = data.len();
        let sum = data.iter().fold(0.0, |acc, x| acc + x);
        Self { sum, count }
    }

    /// Create a new instance from a PrimitiveArray over a numeric type
    pub fn from_array<T>(array: &PrimitiveArray<T>) -> Option<Self>
    where
        T: ArrowNumericType,
    {
        if array.is_empty() {
            return Some(Self::default());
        }

        let array = &array.f64().ok()?;

        let values = array.values();
        let mean = match array.nulls() {
            None => Self::new(values),
            Some(nulls) => {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;
                for (start, end) in nulls.inner().set_slices() {
                    for x in &values[start..end] {
                        sum += x;
                        count += 1;
                    }
                }
                Self { sum, count }
            }
        };

        Some(mean)
    }

    pub fn result(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            let count = self.count as f64;
            Some(self.sum / count)
        }
    }

    pub fn merge(&self, other: &Self) -> Self {
        let count = self.count + other.count;
        let sum = self.sum + other.sum;
        Self { sum, count }
    }
}

/// Extract the mean from the variance calculation.
///
/// An example usage of this is one-pass Sharpe ratio calculation from returns data.
impl From<Variance> for Mean {
    fn from(value: Variance) -> Self {
        let sum = value.sum;
        let count = value.count;
        Self { sum, count }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::PrimitiveArray;

    use super::Mean;

    const RAW_DATA: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    #[test]
    fn new() {
        let mean = Mean::new(RAW_DATA);
        assert_eq!(mean.sum, 45.0);
        assert_eq!(mean.count, RAW_DATA.len());
    }

    #[test]
    fn from_array() {
        let arr = PrimitiveArray::from(RAW_DATA.to_vec());
        let mean = Mean::from_array(&arr).unwrap();
        assert_eq!(mean.sum, 45.0);
        assert_eq!(mean.count, RAW_DATA.len());
    }

    #[test]
    fn result() {
        let mean = Mean::new(RAW_DATA);
        assert_eq!(mean.result().unwrap(), 5.0);
    }

    #[test]
    fn result_empty() {
        let mean = Mean::new(&[]);
        assert!(mean.result().is_none());
    }
}

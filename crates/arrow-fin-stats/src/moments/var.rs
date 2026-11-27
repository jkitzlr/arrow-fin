use arrow::array::*;

use arrow_fin_util::ArrowNumericExt;

#[derive(Clone, Copy, Debug, Default)]
pub struct Variance {
    pub(super) sum: f64,
    pub(super) sum2: f64,
    pub(super) count: usize,
}

impl Variance {
    /// Create a new instance from a set of data.
    pub fn new(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut sum: f64 = 0.0;
        let mut sum2: f64 = 0.0;

        for x in data {
            sum += x;
            sum2 += x.powi(2);
        }

        let count = data.len();

        Self { sum, sum2, count }
    }

    /// Create a new instance from a PrimitiveArray over a numeric type.
    pub fn from_array<T>(array: &PrimitiveArray<T>) -> Option<Self>
    where
        T: ArrowNumericType,
    {
        if array.is_empty() {
            return Some(Self::default());
        }

        let array = &array.f64().ok()?;

        let values = array.values();
        let var = match array.nulls() {
            None => Self::new(values),
            Some(nulls) => {
                let mut sum: f64 = 0.0;
                let mut sum2: f64 = 0.0;
                let mut count: usize = 0;
                for (start, end) in nulls.inner().set_slices() {
                    for x in &values[start..end] {
                        sum += x;
                        sum2 += x.powi(2);
                        count += 1;
                    }
                }
                Self { sum, sum2, count }
            }
        };

        Some(var)
    }

    /// Merge this instance with another.
    pub fn merge(&self, other: &Self) -> Self {
        let sum = self.sum + other.sum;
        let sum2 = self.sum2 + other.sum2;
        let count = self.count + other.count;
        Self { sum, sum2, count }
    }

    pub fn result(&self, ddof: usize) -> Option<f64> {
        if ddof >= self.count {
            return None;
        }

        let count = self.count as f64;

        let mean = self.sum / count;
        let denom = (self.count - ddof) as f64;

        Some((self.sum2 - count * mean.powi(2)) / denom)
    }

    pub fn std(&self, ddof: usize) -> Option<f64> {
        self.result(ddof).map(|v| v.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::PrimitiveArray;

    use super::Variance;

    const RAW_DATA: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    #[test]
    fn new() {
        let var = Variance::new(RAW_DATA);
        assert_eq!(var.sum, 45.0);
        assert_eq!(var.sum2, 285.0);
        assert_eq!(var.count, RAW_DATA.len());
    }

    #[test]
    fn from_array() {
        let arr = PrimitiveArray::from(RAW_DATA.to_vec());
        let var = Variance::from_array(&arr).unwrap();
        assert_eq!(var.sum, 45.0);
        assert_eq!(var.sum2, 285.0);
        assert_eq!(var.count, RAW_DATA.len());
    }

    #[test]
    fn result() {
        let var = Variance::new(RAW_DATA);
        assert_eq!(var.result(1).unwrap(), 7.5);
    }
}

use arrow::array::{Array, ArrowNumericType, PrimitiveArray};

use crate::PrimitiveArrayExt;

#[derive(Clone, Copy, Debug, Default)]
pub struct Variance {
    n: f64,  // count
    m: f64,  // sum of elements
    m2: f64, // sum of squared elements
}

impl Variance {
    /// Create a new `Variance` instance from an f64 slice.
    pub fn new(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let n = data.len() as f64;

        let mut m: f64 = 0.0;
        let mut m2: f64 = 0.0;

        for x in data {
            m += x;
            m2 += x.powi(2);
        }

        Self { n, m, m2 }
    }

    /// Create a new `Variance` instance from a numeric Arrow PrimitiveArray.
    ///
    /// This function optimizes for the case where `arr` contains no nulls.
    pub fn from_array<T>(arr: &PrimitiveArray<T>) -> Option<Self>
    where
        T: ArrowNumericType,
    {
        let arr = arr.f64()?;

        if arr.nulls().is_none() {
            Some(Self::new(arr.values()))
        } else {
            // TODO: put this in a separate method(?)
            let mut n: usize = 0;
            let mut m: f64 = 0.0;
            let mut m2: f64 = 0.0;

            for x in arr.iter().flatten() {
                n += 1;
                m += x;
                m2 += x.powi(2);
            }

            Some(Self { n: n as f64, m, m2 })
        }
    }

    /// Merge another `Variance` instance with the current instance.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            n: self.n + other.n,
            m: self.m + other.m,
            m2: self.m2 + other.m2,
        }
    }

    pub fn result(&self, ddof: u8) -> Option<f64> {
        let ddof = ddof as f64;

        if ddof >= self.n {
            return None;
        }

        let mean = self.m / self.n;

        Some((self.m2 - self.n * mean.powi(2)) / (self.n - ddof))
    }

    pub fn fold(these: impl Iterator<Item = Self>) -> Self {
        these.fold(Self::default(), |acc, this| acc.merge(&this))
    }
}

#[cfg(test)]
mod tests {
    use super::Variance;

    const RAW_DATA: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    #[test]
    fn new() {
        let var = Variance::new(RAW_DATA);
        assert_eq!(var.n, RAW_DATA.len() as f64);
        assert_eq!(var.m, 45.0);
        assert_eq!(var.m2, 285.0);
    }

    #[test]
    fn result() {
        let var = Variance::new(RAW_DATA);
        assert_eq!(var.result(1).unwrap(), 7.5);
    }
}

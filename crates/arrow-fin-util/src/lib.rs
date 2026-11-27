use arrow::array::*;
use arrow::compute::kernels::cast;
use arrow::datatypes::DataType;

pub type ArrowResult<T> = arrow::error::Result<T>;

/// Generic extension methods for Arrow arrays
pub trait ArrowExt {
    /// Test whether the array contains any logical nulls.
    fn has_nulls(self) -> bool;
}

impl ArrowExt for ArrayRef {
    fn has_nulls(self) -> bool {
        self.nulls().is_some()
    }
}

impl ArrowExt for &ArrayRef {
    fn has_nulls(self) -> bool {
        self.nulls().is_some()
    }
}

impl<T> ArrowExt for PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    fn has_nulls(self) -> bool {
        self.nulls().is_some()
    }
}

impl<T> ArrowExt for &PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    fn has_nulls(self) -> bool {
        self.nulls().is_some()
    }
}

/// Numeric extensions for primitive arrays
pub trait ArrowNumericExt {
    /// Cast this array as a Float64 array (PrimitiveArray<Float64Type>)
    fn f64(&self) -> ArrowResult<Float64Array>;
}

// TODO: implement these with macro
impl<T: ArrowNumericType> ArrowNumericExt for &PrimitiveArray<T> {
    fn f64(&self) -> ArrowResult<Float64Array> {
        let data_f64 = cast(self, &DataType::Float64)?;
        Ok(data_f64.as_primitive().clone())
    }
}

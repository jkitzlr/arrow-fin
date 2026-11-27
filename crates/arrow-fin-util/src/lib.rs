use arrow::array::*;
use arrow::compute::kernels::cast;
use arrow::datatypes::*;
use arrow::error::ArrowError;

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

impl ArrowNumericExt for ArrayRef {
    fn f64(&self) -> ArrowResult<Float64Array> {
        use DataType::*;

        // TODO: is the clone necessary?
        // TODO: implement macro for the arms
        match self.data_type().clone() {
            Int8 => self
                .as_primitive_opt::<Int8Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int16 => self
                .as_primitive_opt::<Int16Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int32 => self
                .as_primitive_opt::<Int32Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int64 => self
                .as_primitive_opt::<Int64Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Float64 => self
                .as_primitive_opt::<Float64Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            datatype => Err(ArrowError::InvalidArgumentError(format!(
                "unsupported input data type {}!",
                datatype
            ))),
        }
    }
}

impl ArrowNumericExt for &dyn Array {
    fn f64(&self) -> ArrowResult<Float64Array> {
        use DataType::*;

        // TODO: is the clone necessary?
        // TODO: implement macro for the arms
        match self.data_type().clone() {
            Int8 => self
                .as_primitive_opt::<Int8Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int16 => self
                .as_primitive_opt::<Int16Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int32 => self
                .as_primitive_opt::<Int32Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Int64 => self
                .as_primitive_opt::<Int64Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            Float64 => self
                .as_primitive_opt::<Float64Type>()
                .ok_or_else(|| {
                    ArrowError::CastError("unable to convert to numeric type.".to_owned())
                })?
                .f64(),
            datatype => Err(ArrowError::InvalidArgumentError(format!(
                "unsupported input data type {}!",
                datatype
            ))),
        }
    }
}

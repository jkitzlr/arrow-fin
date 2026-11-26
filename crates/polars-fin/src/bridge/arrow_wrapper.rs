use std::ops::Deref;

use arrow::array as arrow_array;
use arrow::ffi as arrow_ffi;
use polars_arrow::array as polars_array;
use polars_arrow::datatypes::Field as PolarsField;
use polars_arrow::ffi as polars_ffi;

/// Adapter layer between Polars' Arrow implementation and the canonical `arrow` crate.
pub(crate) struct ArrowWrapper(pub(crate) arrow_array::ArrayRef);

impl ArrowWrapper {
    pub(crate) fn new(array: arrow_array::ArrayRef) -> Self {
        Self(array)
    }
}

impl Deref for ArrowWrapper {
    type Target = arrow_array::ArrayRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// TODO: NEED PROPER ERROR HANDLING!!! (needs to be TryFrom)
impl From<polars_array::ArrayRef> for ArrowWrapper {
    fn from(value: polars_array::ArrayRef) -> Self {
        let field = PolarsField::new("".into(), value.dtype().clone(), value.has_nulls());

        let c_schema = polars_ffi::export_field_to_c(&field);
        let c_array = polars_ffi::export_array_to_c(value);

        let arrow_array: arrow_array::ArrayData = unsafe {
            let sch: arrow_ffi::FFI_ArrowSchema = std::mem::transmute(c_schema);
            let arr: arrow_ffi::FFI_ArrowArray = std::mem::transmute(c_array);
            arrow_ffi::from_ffi(arr, &sch).unwrap()
        };

        Self::new(arrow_array::make_array(arrow_array))
    }
}

// impl From<&dyn polars_array::Array> for ArrowWrapper {
//     fn from(value: &dyn polars_array::Array) -> Self {
//         let field = PolarsField::new("".into(), value.dtype().clone(), value.has_nulls());

//         let t = value.deref().clone();

//         let c_schema = polars_ffi::export_field_to_c(&field);
//         let c_array = polars_ffi::export_array_to_c(Box::new(value.clone()));

//         let arrow_array: arrow_array::ArrayData = unsafe {
//             let sch: arrow_ffi::FFI_ArrowSchema = std::mem::transmute(c_schema);
//             let arr: arrow_ffi::FFI_ArrowArray = std::mem::transmute(c_array);
//             arrow_ffi::from_ffi(arr, &sch).unwrap()
//         };

//         Self::new(arrow_array::make_array(arrow_array))
//     }
// }

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;
    use polars::prelude::*;

    use super::ArrowWrapper;

    #[test]
    fn basic() {
        let s = Series::new("".into(), &[1, 2, 3, 4]);

        let pl_arr = s.to_arrow(0, CompatLevel::newest());
        let arrow_arr: ArrowWrapper = pl_arr.into();

        let output = arrow_arr.as_primitive::<arrow::datatypes::Int32Type>();
        assert_eq!(output.values().to_vec(), vec![1, 2, 3, 4]);
    }
}

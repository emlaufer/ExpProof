use std::collections::{HashMap, HashSet};
use std::ops::{Add, Mul, Range};

use super::errors::{EinsumError, ExplainError};
use crate::tensor::{
    ops::accumulated::{dot, prod},
    ops::add,
    IntoI64, Tensor, TensorError, TensorType, ValTensor,
};
use itertools::Itertools;

// TODO: Move to tensor code...
pub fn matmul<F: TensorType + Mul<Output = F> + Add<Output = F>>(
    inputs: &[Tensor<F>; 2],
) -> Result<Tensor<F>, TensorError> {
    // just do naive for now...
    let left = &inputs[0];
    let right = &inputs[1];

    if left.dims()[1] != right.dims()[0] {
        return Err(TensorError::DimMismatch("matmul".to_string()));
    }

    let mut output = Tensor::<F>::new(None, &[left.dims()[0], right.dims()[1]])?;
    for row in 0..left.dims()[0] {
        for col in 0..right.dims()[1] {
            for i in 0..left.dims()[1] {
                let val = output.get(&[row, col]) + left.get(&[row, i]) * right.get(&[i, col]);
                *output.get_mut(&[row, col]) = val;
            }
        }
    }

    Ok(output)
}

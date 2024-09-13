use crate::graph::errors::GraphError;
use crate::tensor::errors::TensorError;
use thiserror::Error;

use linfa_elasticnet::ElasticNetError;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum ExplainError {
    /// Missing input for a node
    #[error("graph error: {0}")]
    InnerGraphError(#[from] GraphError),

    #[error("elasticnet error: {0}")]
    InnerElasticNetError(#[from] ElasticNetError),
}

#[derive(Debug, Error)]
pub enum EinsumError {
    #[error("invalid einsum")]
    InvalidEinsum,

    #[error("tensor error: {0}")]
    InnerTensorError(#[from] TensorError),

    #[error("invalid einsum")]
    MissingEinsumProduct,
}

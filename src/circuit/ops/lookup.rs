use super::*;
use serde::{Deserialize, Serialize};

use super::super::table::LOOKUP_RANGE;

use crate::{
    circuit::{layouts, table::Range, utils},
    fieldutils::{felt_to_i64, i64_to_felt},
    graph::multiplier_to_scale,
    tensor::{self, IntoI64, Tensor, TensorError, TensorType},
};

use super::Op;
use halo2curves::ff::PrimeField;

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Abs,
    Div {
        denom: utils::F32,
    },
    Cast {
        scale: utils::F32,
    },
    ReLU,
    Max {
        scale: utils::F32,
        a: utils::F32,
    },
    Min {
        scale: utils::F32,
        a: utils::F32,
    },
    Ceil {
        scale: utils::F32,
    },
    Floor {
        scale: utils::F32,
    },
    Round {
        scale: utils::F32,
    },
    RoundHalfToEven {
        scale: utils::F32,
    },
    Sqrt {
        scale: utils::F32,
    },
    Rsqrt {
        scale: utils::F32,
    },
    Recip {
        input_scale: utils::F32,
        output_scale: utils::F32,
    },
    LeakyReLU {
        slope: utils::F32,
    },
    Sigmoid {
        scale: utils::F32,
    },
    Ln {
        scale: utils::F32,
    },
    Exp {
        scale: utils::F32,
    },
    Cos {
        scale: utils::F32,
    },
    ACos {
        scale: utils::F32,
    },
    Cosh {
        scale: utils::F32,
    },
    ACosh {
        scale: utils::F32,
    },
    Sin {
        scale: utils::F32,
    },
    ASin {
        scale: utils::F32,
    },
    Sinh {
        scale: utils::F32,
    },
    ASinh {
        scale: utils::F32,
    },
    Tan {
        scale: utils::F32,
    },
    ATan {
        scale: utils::F32,
    },
    Tanh {
        scale: utils::F32,
    },
    ATanh {
        scale: utils::F32,
    },
    Erf {
        scale: utils::F32,
    },
    GreaterThan {
        a: utils::F32,
    },
    LessThan {
        a: utils::F32,
    },
    GreaterThanEqual {
        a: utils::F32,
    },
    LessThanEqual {
        a: utils::F32,
    },
    Sign,
    IsZero,
    KroneckerDelta,
    Pow {
        scale: utils::F32,
        a: utils::F32,
    },
    HardSwish {
        scale: utils::F32,
    },
    // TODO:
    Norm {
        scale: utils::F32,
        mean: utils::F32,
        std: utils::F32,
    },
    RecipSqrt {
        input_scale: utils::F32,
        output_scale: utils::F32,
    },
    // Exponential kernel for lime weights...
    LimeWeight {
        input_scale: utils::F32,
        sigma: utils::F32,
    },
}

impl LookupOp {
    /// Returns the range of values that can be represented by the table
    pub fn bit_range(max_len: usize) -> Range {
        let range = (max_len - 1) as f64 / 2_f64;
        let range = range as i64;
        (-range, range)
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    pub(crate) fn f<F: PrimeField + TensorType + PartialOrd + std::hash::Hash + IntoI64>(
        &self,
        x: &[Tensor<F>],
    ) -> Result<ForwardResult<F>, TensorError> {
        let x = x[0].clone().map(|x| felt_to_i64(x));
        // ensure the input is in range...otherwise throw an error

        {
            let range = LOOKUP_RANGE.lock().unwrap();
            x.enum_map(|_, v| {
                if v < range.0 || v > range.1 {
                    return Err(TensorError::LookupOutOfRange(format!(
                        "Lookup out of range! Tensor: {:?}, passed to op: {:?}",
                        x, self
                    )));
                }
                Ok(())
            })
        }?;

        let res = match &self {
            LookupOp::Abs => Ok(tensor::ops::abs(&x)?),
            LookupOp::Ceil { scale } => Ok(tensor::ops::nonlinearities::ceil(&x, scale.into())),
            LookupOp::Floor { scale } => Ok(tensor::ops::nonlinearities::floor(&x, scale.into())),
            LookupOp::Round { scale } => Ok(tensor::ops::nonlinearities::round(&x, scale.into())),
            LookupOp::RoundHalfToEven { scale } => Ok(
                tensor::ops::nonlinearities::round_half_to_even(&x, scale.into()),
            ),
            LookupOp::Pow { scale, a } => Ok(tensor::ops::nonlinearities::pow(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::KroneckerDelta => Ok(tensor::ops::nonlinearities::kronecker_delta(&x)),
            LookupOp::Max { scale, a } => Ok(tensor::ops::nonlinearities::max(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::Min { scale, a } => Ok(tensor::ops::nonlinearities::min(
                &x,
                scale.0.into(),
                a.0.into(),
            )),
            LookupOp::Sign => Ok(tensor::ops::nonlinearities::sign(&x)),
            LookupOp::IsZero => Ok(tensor::ops::nonlinearities::is_zero(&x)),
            LookupOp::LessThan { a } => Ok(tensor::ops::nonlinearities::less_than(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::LessThanEqual { a } => Ok(tensor::ops::nonlinearities::less_than_equal(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::GreaterThan { a } => Ok(tensor::ops::nonlinearities::greater_than(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::GreaterThanEqual { a } => Ok(
                tensor::ops::nonlinearities::greater_than_equal(&x, f32::from(*a).into()),
            ),
            LookupOp::Div { denom } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*denom).into(),
            )),
            LookupOp::Cast { scale } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*scale).into(),
            )),
            LookupOp::Recip {
                input_scale,
                output_scale,
            } => Ok(tensor::ops::nonlinearities::recip(
                &x,
                input_scale.into(),
                output_scale.into(),
            )),
            LookupOp::ReLU => Ok(tensor::ops::nonlinearities::leakyrelu(&x, 0_f64)),

            LookupOp::LeakyReLU { slope: a } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, a.0.into()))
            }
            LookupOp::Sigmoid { scale } => {
                Ok(tensor::ops::nonlinearities::sigmoid(&x, scale.into()))
            }
            LookupOp::Sqrt { scale } => Ok(tensor::ops::nonlinearities::sqrt(&x, scale.into())),
            LookupOp::Rsqrt { scale } => Ok(tensor::ops::nonlinearities::rsqrt(&x, scale.into())),
            LookupOp::Erf { scale } => Ok(tensor::ops::nonlinearities::erffunc(&x, scale.into())),
            LookupOp::Exp { scale } => Ok(tensor::ops::nonlinearities::exp(&x, scale.into())),
            LookupOp::Ln { scale } => Ok(tensor::ops::nonlinearities::ln(&x, scale.into())),
            LookupOp::Cos { scale } => Ok(tensor::ops::nonlinearities::cos(&x, scale.into())),
            LookupOp::ACos { scale } => Ok(tensor::ops::nonlinearities::acos(&x, scale.into())),
            LookupOp::Cosh { scale } => Ok(tensor::ops::nonlinearities::cosh(&x, scale.into())),
            LookupOp::ACosh { scale } => Ok(tensor::ops::nonlinearities::acosh(&x, scale.into())),
            LookupOp::Sin { scale } => Ok(tensor::ops::nonlinearities::sin(&x, scale.into())),
            LookupOp::ASin { scale } => Ok(tensor::ops::nonlinearities::asin(&x, scale.into())),
            LookupOp::Sinh { scale } => Ok(tensor::ops::nonlinearities::sinh(&x, scale.into())),
            LookupOp::ASinh { scale } => Ok(tensor::ops::nonlinearities::asinh(&x, scale.into())),
            LookupOp::Tan { scale } => Ok(tensor::ops::nonlinearities::tan(&x, scale.into())),
            LookupOp::ATan { scale } => Ok(tensor::ops::nonlinearities::atan(&x, scale.into())),
            LookupOp::ATanh { scale } => Ok(tensor::ops::nonlinearities::atanh(&x, scale.into())),
            LookupOp::Tanh { scale } => Ok(tensor::ops::nonlinearities::tanh(&x, scale.into())),
            LookupOp::HardSwish { scale } => {
                Ok(tensor::ops::nonlinearities::hardswish(&x, scale.into()))
            }
            LookupOp::Norm { scale, mean, std } => {
                Ok(tensor::ops::nonlinearities::normal_inverse_cdf(
                    &x,
                    scale.into(),
                    mean.into(),
                    std.into(),
                ))
            }
            LookupOp::RecipSqrt {
                input_scale,
                output_scale,
            } => Ok(tensor::ops::nonlinearities::recip_sqrt(
                &x,
                input_scale.into(),
                output_scale.into(),
            )),
            LookupOp::LimeWeight { input_scale, sigma } => Ok(
                tensor::ops::nonlinearities::lime_weight(&x, input_scale.into(), sigma.into()),
            ),
        }?;

        let output = res.map(|x| i64_to_felt(x));

        Ok(ForwardResult { output })
    }

    /// Returns the scale of the output of the operation.
    pub fn in_scale(&self, inputs_scale: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        let scale = match self {
            Self::LimeWeight { input_scale, .. } => input_scale.0.log2().floor() as i32,
            Self::Norm { scale, .. } => scale.0.log2().floor() as i32,
            Self::RecipSqrt { input_scale, .. } => input_scale.0.log2().floor() as i32,
            Self::Recip { input_scale, .. } => input_scale.0.log2().floor() as i32,
            Self::Sqrt { scale, .. } => scale.0.log2().floor() as i32,
            Self::Abs => inputs_scale[0],
            _ => panic!(),
        };
        Ok(scale)
    }
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash + IntoI64> Op<F> for LookupOp {
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns the name of the operation
    fn as_string(&self) -> String {
        match self {
            LookupOp::Abs => "ABS".into(),
            LookupOp::Ceil { scale } => format!("CEIL(scale={})", scale),
            LookupOp::Floor { scale } => format!("FLOOR(scale={})", scale),
            LookupOp::Round { scale } => format!("ROUND(scale={})", scale),
            LookupOp::RoundHalfToEven { scale } => format!("ROUND_HALF_TO_EVEN(scale={})", scale),
            LookupOp::Pow { a, scale } => format!("POW(scale={}, exponent={})", scale, a),
            LookupOp::KroneckerDelta => "K_DELTA".into(),
            LookupOp::Max { scale, a } => format!("MAX(scale={}, a={})", scale, a),
            LookupOp::Min { scale, a } => format!("MIN(scale={}, a={})", scale, a),
            LookupOp::Sign => "SIGN".into(),
            LookupOp::IsZero => "IS_ZERO".into(),
            LookupOp::GreaterThan { a } => format!("GREATER_THAN(a={})", a),
            LookupOp::GreaterThanEqual { a } => format!("GREATER_THAN_EQUAL(a={})", a),
            LookupOp::LessThan { a } => format!("LESS_THAN(a={})", a),
            LookupOp::LessThanEqual { a } => format!("LESS_THAN_EQUAL(a={})", a),
            LookupOp::Recip {
                input_scale,
                output_scale,
            } => format!(
                "RECIP(input_scale={}, output_scale={})",
                input_scale, output_scale
            ),
            LookupOp::Div { denom, .. } => format!("DIV(denom={})", denom),
            LookupOp::Cast { scale } => format!("CAST(scale={})", scale),
            LookupOp::Ln { scale } => format!("LN(scale={})", scale),
            LookupOp::ReLU => "RELU".to_string(),
            LookupOp::LeakyReLU { slope: a } => format!("L_RELU(slope={})", a),
            LookupOp::Sigmoid { scale } => format!("SIGMOID(scale={})", scale),
            LookupOp::Sqrt { scale } => format!("SQRT(scale={})", scale),
            LookupOp::Erf { scale } => format!("ERF(scale={})", scale),
            LookupOp::Rsqrt { scale } => format!("RSQRT(scale={})", scale),
            LookupOp::Exp { scale } => format!("EXP(scale={})", scale),
            LookupOp::Tan { scale } => format!("TAN(scale={})", scale),
            LookupOp::ATan { scale } => format!("ATAN(scale={})", scale),
            LookupOp::Tanh { scale } => format!("TANH(scale={})", scale),
            LookupOp::ATanh { scale } => format!("ATANH(scale={})", scale),
            LookupOp::Cos { scale } => format!("COS(scale={})", scale),
            LookupOp::ACos { scale } => format!("ACOS(scale={})", scale),
            LookupOp::Cosh { scale } => format!("COSH(scale={})", scale),
            LookupOp::ACosh { scale } => format!("ACOSH(scale={})", scale),
            LookupOp::Sin { scale } => format!("SIN(scale={})", scale),
            LookupOp::ASin { scale } => format!("ASIN(scale={})", scale),
            LookupOp::Sinh { scale } => format!("SINH(scale={})", scale),
            LookupOp::ASinh { scale } => format!("ASINH(scale={})", scale),
            LookupOp::HardSwish { scale } => format!("HARDSWISH(scale={})", scale),
            LookupOp::Norm { scale, std, mean } => {
                format!("NORM(scale={}, mean={}, std={})", scale, mean, std)
            }
            LookupOp::RecipSqrt {
                input_scale,
                output_scale,
            } => {
                format!(
                    "RecipSqrt(input_scale={}, output_scale={})",
                    input_scale, output_scale
                )
            }
            LookupOp::LimeWeight { input_scale, sigma } => {
                format!("LIMEWEIGHT(input_scale={}, sigma={})", input_scale, sigma)
            }
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        Ok(Some(layouts::nonlinearity(
            config,
            region,
            values[..].try_into()?,
            self,
        )?))
    }

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, inputs_scale: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        let scale = match self {
            LookupOp::Cast { scale } => {
                let in_scale = inputs_scale[0];
                in_scale + multiplier_to_scale(1. / scale.0 as f64)
            }
            LookupOp::Recip { output_scale, .. } => multiplier_to_scale(output_scale.into()),
            LookupOp::Sign
            | LookupOp::GreaterThan { .. }
            | LookupOp::LessThan { .. }
            | LookupOp::GreaterThanEqual { .. }
            | LookupOp::LessThanEqual { .. }
            | LookupOp::KroneckerDelta => 0,
            _ => inputs_scale[0],
        };
        Ok(scale)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

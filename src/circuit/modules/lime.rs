use super::*;
use crate::graph::lime::*;
use crate::graph::utilities::dequantize_tensor;
use crate::tensor::{create_constant_tensor, create_zero_tensor, IntoI64};
use crate::tensor::{Tensor, TensorType};
use crate::Scale;
use serde::Deserialize;
use serde::Serialize;

use itertools::Itertools;
use maybe_rayon::{
    iter::IntoParallelRefIterator,
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimeConfig {
    pub dim: usize,
    pub alpha: f32,
}

pub struct LimeInputs<F: TensorType> {
    pub inputs: Tensor<F>,
    pub outputs: Tensor<F>,
    pub input_scale: Scale,
    pub output_scale: Scale,
    pub out_scale: Scale,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimeModule {
    config: LimeConfig,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash + IntoI64> GraphModule<F>
    for LimeModule
{
    /// Config
    type Config = LimeConfig;
    /// The return type after an input assignment
    type InputAssignments = ();
    /// The inputs used in the run function
    type RunInputs = LimeInputs<F>;
    /// The params used in configure
    type Params = ();

    /// construct new module from config
    fn new(config: Self::Config) -> Self {
        Self { config }
    }
    /// Configure
    fn configure(params: Self::Params) -> Self::Config {
        // TODO:
        Self::Config {
            dim: 10,
            alpha: 0.3,
        }
    }
    /// Name
    fn name(&self) -> &'static str {
        "Lime"
    }
    /// Run the operation the module represents
    fn run(input: Self::RunInputs) -> Result<Vec<Vec<F>>, errors::ModuleError> {
        // generate the lime model
        //let lime_model = LassoModel::new(
        //    &input.inputs,
        //    &input.outputs,
        //    &input.input_scale,
        //    &input.output_scale,
        //    &input.out_scale,
        //);

        //return Ok(vec![lime_model.coeffs, vec![lime_model.intercept]]);
        Ok(vec![])
    }
    /// Layout
    fn layout(
        &self,
        config: &BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<ValTensor<F>, errors::ModuleError> {
        use crate::circuit::ops::layouts::*;
        use crate::circuit::{ops::base::BaseOp, utils};
        use crate::fieldutils::{felt_to_i64, i64_to_felt};
        use crate::graph::utilities::{dequantize, quantize_float};
        use crate::tensor;
        use halo2_proofs::circuit::Value;
        // TODOS:
        // 1. check_range will panic if value is out of range... this is a problem for first run
        //    without the kkt checks
        //      hacky soln: don't assign coeff_sign for values for first pass, then we can check and
        //          abort
        //      better soln: stub out "model" part and "other subgraph" part... basically module that
        //      has access to the graph stuff...
        //  2. need to scale everything appropriately for ops...
        let inputs = &values[0];
        // hack: need to rescale output... will do...
        let outputs = &values[1];
        let coeffs = &values[2];
        let intercept = &values[3];

        // OK Now we do duality

        // TODOS:
        // 1. better scale down ... use recip? div is bologna

        //log::debug!("input: {}", inputs.show());
        //log::debug!("outputs: {}", outputs.show());
        //log::debug!("coeffs: {}", coeffs.show());
        //log::debug!("intercept: {}", intercept.show());

        //log::debug!(
        //    "scales!: {:?} {:?} {:?} {:?}",
        //    inputs.pshow(7),
        //    outputs.pshow(0),
        //    coeffs.pshow(12),
        //    intercept.pshow(12),
        //);

        //let output_rescaled =
        //    rescale(config, region, &[outputs.clone()], &[(0, 2_u128.pow(12))])?[0].clone();
        //let input_rescaled =
        //    rescale(config, region, &[inputs.clone()], &[(0, 2_u128.pow(5))])?[0].clone();
        //log::debug!("input_rescaled!: {}", input_rescaled.pshow(12));
        //log::debug!("output_rescaled!: {}", output_rescaled.pshow(12));

        //let input_slice = slice(
        //    config,
        //    region,
        //    &[input_rescaled.clone()],
        //    &0,
        //    &1,
        //    &(self.config.dim + 1),
        //)?;
        //log::debug!("input_slice: {}", input_slice.show());
        //let output_slice = slice(
        //    config,
        //    region,
        //    &[output_rescaled.clone()],
        //    &0,
        //    &1,
        //    &(self.config.dim + 1),
        //)?;

        //let inputs_dequant = input_slice.dequantize(12);
        //let outputs_dequant = output_slice.dequantize(12);
        //let coeffs_dequant = coeffs.dequantize(12);
        //let intercept_dequant = intercept.dequantize(12);
        //if inputs_dequant.is_some() {
        //    LassoModel::<F>::test_kkt(
        //        inputs_dequant.unwrap(),
        //        outputs_dequant.unwrap(),
        //        intercept_dequant.unwrap()[0],
        //        coeffs_dequant.unwrap(),
        //        0.3,
        //    );
        //}

        //log::debug!("output_slice: {}", output_slice.show());
        //let expanded_intercept = expand(config, region, &[intercept.clone()], &[self.config.dim])?;
        //log::debug!("expaneded_intercept: {}", expanded_intercept.show());

        //// doubles precision
        //let predictions = einsum(
        //    config,
        //    region,
        //    &[input_slice.clone(), coeffs.clone()],
        //    "ij,jk->ik",
        //)?;
        //log::debug!("predictions: {}", predictions.pshow(24));
        //let intermediate = pairwise(
        //    config,
        //    region,
        //    &[output_slice.clone(), expanded_intercept],
        //    BaseOp::Sub,
        //)?;
        //let intermediate_rescaled = rescale(
        //    config,
        //    region,
        //    &[intermediate.clone()],
        //    &[(0, 2_u128.pow(12))],
        //)?[0]
        //    .clone();
        //log::debug!("sub1: {}", intermediate_rescaled.pshow(24));
        //// need to mult intermediate by scale diff...
        //let intermediate = pairwise(
        //    config,
        //    region,
        //    &[intermediate_rescaled, predictions],
        //    BaseOp::Sub,
        //)?;
        //log::debug!("deltas: {}", intermediate.pshow(24));
        //let input_rescaled = rescale(
        //    config,
        //    region,
        //    &[input_slice.clone()],
        //    &[(0, 2_u128.pow(12))],
        //)?[0]
        //    .clone();
        //log::debug!("input recale: {}", input_rescaled.pshow(24));
        //let intermediate = einsum(
        //    config,
        //    region,
        //    &[input_rescaled.clone(), intermediate.clone()],
        //    "ji,jk->ik",
        //)
        //.unwrap();
        //log::debug!("transpose einsum: {}", intermediate.pshow(48));

        //let mut coeff_sign: ValTensor<F> = if !coeffs.any_unknowns()? {
        //    tensor::ops::nonlinearities::sign(&coeffs.get_int_evals()?)
        //        .par_iter()
        //        .map(|x| Value::known(i64_to_felt(*x)))
        //        .collect::<Tensor<Value<F>>>()
        //        .into()
        //} else {
        //    Tensor::new(
        //        Some(&vec![Value::<F>::unknown(); coeffs.len()]),
        //        &[coeffs.len()],
        //    )?
        //    .into()
        //};
        //let coeff_sign = region.assign(&config.custom_gates.output, &coeff_sign)?;
        //region.increment(coeff_sign.len());
        //log::debug!("coeff_sign: {}", coeff_sign.show());

        //let equal_zero_mask = equals_zero(config, region, &[coeffs.clone()])?;
        //log::debug!("is zero: {}", equal_zero_mask.show());
        //log::debug!("mask len: {}", equal_zero_mask.len());

        //// TODO: need to ensure scale is high enough for good accuracy...
        //let alpha_quant: F = i64_to_felt(quantize_float(&(self.config.alpha as f64), 0.0, 24)?);
        //let test = dequantize(alpha_quant, 24, 0.0);
        //log::debug!("dequant alpha: {} (scale: {})", test, 24);
        //let alphas = create_constant_tensor(alpha_quant, equal_zero_mask.len());
        //log::debug!("alphas: {}", alphas.show());

        //// TODO: can't I just make another ocnst?? no need for rescale...
        //let alphas_rescaled =
        //    rescale(config, region, &[alphas.clone()], &[(0, 2_u128.pow(24))])?[0].clone();
        //let signed_alphas = pairwise(
        //    config,
        //    region,
        //    &[coeff_sign, alphas_rescaled.clone()],
        //    BaseOp::Mult,
        //)?;
        //log::debug!("signed alphas: {}", signed_alphas.pshow(48));

        //// TODO: do div or just multiply other side lmao
        ////let intermediate = loop_div(
        ////    config,
        ////    region,
        ////    &[intermediate],
        ////    i64_to_felt::<F>(self.config.dim as i64),
        ////)?;
        ////log::debug!("div: {}", intermediate.pshow(48));

        //let intermediate = pairwise(config, region, &[intermediate, signed_alphas], BaseOp::Sub)?;
        //log::debug!("intermediate: {}", intermediate.pshow(48));

        //// TODO:
        //// hmmm do I need to rescale down again.............
        ////
        //let input_felt_scale = F::from(24 as u64);
        //let test = loop_div(config, region, &[intermediate.clone()], input_felt_scale).unwrap();
        //let mut intermediate_sign: ValTensor<F> = if !test.any_unknowns()? {
        //    tensor::ops::nonlinearities::sign(&test.get_int_evals()?)
        //        .par_iter()
        //        .map(|x| Value::known(i64_to_felt(*x)))
        //        .collect::<Tensor<Value<F>>>()
        //        .into()
        //} else {
        //    Tensor::new(
        //        Some(&vec![Value::<F>::unknown(); test.len()]),
        //        &[test.len()],
        //    )?
        //    .into()
        //};
        //log::debug!("intermediate_sign: {}", intermediate_sign.show());
        //let intermediate_abs = pairwise(
        //    config,
        //    region,
        //    &[intermediate.clone(), intermediate_sign],
        //    BaseOp::Mult,
        //)?;
        //log::debug!("intermediate: {}", intermediate_abs.pshow(48));

        //let dim_quant: F = i64_to_felt(quantize_float(&(self.config.dim as f64), 0.0, 24)?);
        //let ns = create_constant_tensor(dim_quant, alphas.len());
        //log::debug!("ns: {}", ns.pshow(24));
        //// TODO: this is a constant!! don't compute this...
        //let n_alphas = pairwise(config, region, &[ns, alphas.clone()], BaseOp::Mult)?;
        //log::debug!("n_alphas: {}", n_alphas.pshow(48));

        //let input_felt_scale = F::from(2.pow(40) as u64);
        //let test = loop_div(config, region, &[intermediate.clone()], input_felt_scale).unwrap();
        //log::debug!("intermediate: {}", test.pshow(8));

        // ensure in range...
        //        let range_test = less(config, region, &[intermediate_abs, n_alphas])?;
        //        log::debug!("range test: {}", range_test.show());
        //
        //let range_bracket = felt_to_i64(alpha_quant);
        //let alpha_test = range_check(
        //    config,
        //    region,
        //    &[intermediate],
        //    &(-range_bracket, range_bracket),
        //)?;
        //log::debug!("alpha test: {}", alpha_test.show());

        Ok(intercept.clone())
    }
}

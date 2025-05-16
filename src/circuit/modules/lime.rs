use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use dyn_clone::DynClone;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;

use std::collections::HashSet;

use super::errors::ModuleError;
use crate::circuit::{lookup::LookupOp, BaseConfig as PolyConfig, CheckMode, Op};
use crate::graph::errors::GraphError;
use crate::graph::lime::LassoModel;
use crate::graph::GraphSettings;
use crate::tensor::val::{create_constant_tensor, create_unit_tensor};
use crate::tensor::{Tensor, TensorType, ValTensor, ValType};
use crate::{SamplingStrategy, Scale, SurrogateStrategy};

use crate::circuit::ops::layouts::*;
use crate::circuit::table::Range as CRange;
use crate::circuit::utils::F32;
use crate::circuit::{ops::base::BaseOp, utils};
use crate::fieldutils::{felt_to_i64, i64_to_felt};

use super::perturb::PerturbChip;
use super::sample::SampleChip;
use crate::circuit::ops::chip::BaseConfig;
use crate::circuit::ops::region::RegionCtx;
use crate::LimeWeightStrategy;

use crate::graph::utilities::{dequantize, quantize_float};

use std::ops::{Add, Deref, DerefMut, Div, Mul, Neg, Range, Sub};

use crate::ablate::*;

/// helper to track precision
/// Note: This isn't used in all the lime circuit code, since I created it to debug some precision errors
/// In the future, we should use this instead of manually tracking it through
#[derive(Debug, Clone)]
struct PValTensor {
    inner: ValTensor<Fp>,
    precision: u64,
}

impl PValTensor {
    fn new(inner: ValTensor<Fp>, precision: u64) -> Self {
        Self { inner, precision }
    }

    fn constant(val: Fp, precision: u64) -> Self {
        let val = create_constant_tensor(val, 1);
        Self::new(val, precision)
    }

    fn constant_f64(val: f64, precision: u64) -> Self {
        let val = i64_to_felt(quantize_float(&val, 0.0, precision as i32).unwrap());
        Self::constant(val, precision)
    }

    fn precisionf(&self) -> f32 {
        self.precision as f32
    }

    fn expand(&self, to: &[usize]) -> Self {
        let mut inner = self.inner.clone();
        inner.expand(to).unwrap();
        Self {
            inner,
            precision: self.precision,
        }
    }

    fn reshape(&self, to: &[usize]) -> Self {
        let mut inner = self.inner.clone();
        inner.reshape(to);
        Self {
            inner,
            precision: self.precision,
        }
    }

    fn slice(&self, indices: &[Range<usize>]) -> Self {
        let mut inner = self.inner.get_slice(indices).unwrap();
        Self {
            inner,
            precision: self.precision,
        }
    }

    fn shape(&self) -> &[usize] {
        self.inner.dims()
    }

    fn add(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        rhs: &PValTensor,
    ) -> PValTensor {
        assert_eq!(self.precision, rhs.precision);
        let result = pairwise(
            config,
            region,
            &[self.inner.clone(), rhs.inner.clone()],
            BaseOp::Add,
        )
        .unwrap();
        Self {
            inner: result,
            precision: self.precision,
        }
    }

    fn sub(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        rhs: &PValTensor,
    ) -> PValTensor {
        assert_eq!(self.precision, rhs.precision);
        let result = pairwise(
            config,
            region,
            &[self.inner.clone(), rhs.inner.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        Self {
            inner: result,
            precision: self.precision,
        }
    }

    fn mul(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        rhs: &PValTensor,
    ) -> PValTensor {
        let result = pairwise(
            config,
            region,
            &[self.inner.clone(), rhs.inner.clone()],
            BaseOp::Mult,
        )
        .unwrap();
        Self {
            inner: result,
            precision: self.precision + rhs.precision,
        }
    }

    fn dot(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        rhs: &PValTensor,
    ) -> PValTensor {
        let result = dot(config, region, &[self.inner.clone(), rhs.inner.clone()]).unwrap();
        Self {
            inner: result,
            precision: self.precision + rhs.precision,
        }
    }

    fn einsum(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        rhs: &PValTensor,
        op: &str,
    ) -> PValTensor {
        let mut equation = op.split("->");
        let inputs_eq = equation.next().unwrap();
        let output_eq = equation.next().unwrap();
        let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();
        assert!(inputs_eq.len() == 2);

        let set: HashSet<char> = inputs_eq[0].chars().collect();
        let mut precision = if inputs_eq[1].chars().any(|c| set.contains(&c)) {
            self.precision + rhs.precision
        } else {
            assert_eq!(self.precision, rhs.precision);
            self.precision
        };

        // figure out the precision based on mults

        let result = einsum(config, region, &[self.inner.clone(), rhs.inner.clone()], op).unwrap();
        Self {
            inner: result,
            precision,
        }
    }

    fn nonlinearity(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        op: &LookupOp,
    ) -> PValTensor {
        let in_scale = op.in_scale([self.precision as i32].to_vec()).unwrap();
        assert_eq!(in_scale, self.precision as i32);
        let inner = nonlinearity(config, region, &[self.inner.clone()], &op).unwrap();
        let out_scale =
            <crate::circuit::ops::lookup::LookupOp as crate::circuit::ops::Op<Fp>>::out_scale(
                op,
                [self.precision as i32].to_vec(),
            )
            .unwrap();
        Self {
            inner,
            precision: out_scale as u64,
        }
    }

    pub fn rescale(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        to: u64,
    ) -> PValTensor {
        let from = self.precision;

        if to > from {
            self.rescale_up(config, region, to)
        } else {
            self.rescale_down(config, region, to)
        }
    }

    pub fn rescale_up(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        to: u64,
    ) -> PValTensor {
        let from = self.precision;
        assert!(from % 4 == 0);
        assert!(to % 4 == 0);
        assert!(to >= from);

        if to == from {
            return self.clone();
        }

        let multiplier = PValTensor::constant_f64(1.0, to - from);
        self.mul(config, region, &multiplier)
    }

    pub fn rescale_down(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        to: u64,
    ) -> PValTensor {
        let from = self.precision;
        assert!(from % 4 == 0);
        assert!(to % 4 == 0);
        assert!(to <= from);

        if to == from {
            return self.clone();
        }

        let mut input = self.inner.clone();
        let mut from = from;
        while from - 8 >= to {
            input = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[input.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
            from -= 8;
        }

        if from - 4 == to {
            input = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[input.clone()],
                Fp::from(2u64.pow(4)),
            )
            .unwrap();
            from -= 4;
        }

        assert_eq!(from, to);
        Self {
            inner: input,
            precision: to,
        }
    }

    fn sum(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>) -> PValTensor {
        let inner = sum(config, region, &[self.inner.clone()]).unwrap();
        Self {
            inner,
            precision: self.precision,
        }
    }

    fn range_check(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, range: &(i64, i64)) {
        range_check(config, region, &[self.inner.clone()], range).unwrap();
    }

    fn show(&self) -> String {
        self.inner.pshow(self.precision as i32)
    }
}

trait LimePointSampler:
    std::fmt::Debug + serde_traitobject::Serialize + serde_traitobject::Deserialize + DynClone
{
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>);
    fn num_samples(&self) -> usize;
    fn num_points(&self) -> usize;
    fn generate_witness(&self, samples: &[Fp], std: Fp) -> Vec<Fp>;
    fn layout(
        &self,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        samples: &ValTensor<Fp>,
        std: &ValTensor<Fp>,
    ) -> ValTensor<Fp>;
}

dyn_clone::clone_trait_object!(LimePointSampler);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UniformPointSampleCircuit {
    num_points: usize,
    d: usize,
}

impl UniformPointSampleCircuit {
    fn new(num_points: usize, d: usize) -> Self {
        Self { num_points, d }
    }
}

impl LimePointSampler for UniformPointSampleCircuit {
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {}

    fn num_samples(&self) -> usize {
        self.num_points * self.d
    }

    fn num_points(&self) -> usize {
        self.num_points
    }

    fn generate_witness(&self, samples: &[Fp], std: Fp) -> Vec<Fp> {
        samples
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let value = (v - Fp::from(128)) * std;

                // convert back to float to rescale down to 8 bits...
                let rescaled = dequantize(value, 16, 0.0);
                println!("rescaled: {:?}", rescaled);
                i64_to_felt::<Fp>(quantize_float(&rescaled, 0.0, 8).unwrap())
            })
            .collect::<Vec<_>>()
    }

    fn layout(
        &self,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        samples: &ValTensor<Fp>,
        std: &ValTensor<Fp>, // not really std in this case ... instead is bound of uniform
    ) -> ValTensor<Fp> {
        if ABLATE_INPUTS {
            let samples_fields = samples.get_field_evals().unwrap();
            let std_field = std.get_field_evals().unwrap();
            let samples = self.generate_witness(&samples_fields, std_field[0].clone());
            ValTensor::known_from_vec(&samples)
        } else {
            //let lime_samples = self.get_lime_samples(samples);
            // perturb x_border by lime samples...
            let d = samples.dims()[1];

            // center to 0
            let mut half_sample = create_constant_tensor(Fp::from(128), self.num_points * d);
            half_sample.reshape(&[self.num_points, d]);
            let perturbations = pairwise(
                base_config,
                region,
                &[samples.clone(), half_sample],
                BaseOp::Sub,
            )
            .unwrap();

            // scale points by uniform range
            let mut std = std.clone();
            std.expand(&[self.num_points, self.d]).unwrap();
            let std_scaled =
                pairwise(base_config, region, &[perturbations, std], BaseOp::Mult).unwrap();

            // scale back down to 8 bits of precision
            let points_scaled = crate::circuit::ops::layouts::loop_div(
                base_config,
                region,
                &[std_scaled.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();

            points_scaled
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GaussianPointSampleCircuit {
    num_points: usize,
    d: usize,
}

impl GaussianPointSampleCircuit {
    fn new(num_points: usize, d: usize) -> Self {
        Self { num_points, d }
    }
}

impl LimePointSampler for GaussianPointSampleCircuit {
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {
        required_lookups.push(crate::circuit::ops::lookup::LookupOp::Norm {
            scale: F32(2f32.powf(8.0)),
            mean: F32(0f32),
            std: F32(1f32),
        });
    }

    fn num_samples(&self) -> usize {
        self.num_points * self.d
    }

    fn num_points(&self) -> usize {
        self.num_points
    }

    fn generate_witness(&self, samples: &[Fp], std: Fp) -> Vec<Fp> {
        let samples_tensor = Tensor::new(Some(samples), &[self.num_points, self.d]).unwrap();
        let samples_tensor = samples_tensor.clone().map(|x| felt_to_i64(x));
        let samples_normal = crate::tensor::ops::nonlinearities::normal_inverse_cdf(
            &samples_tensor,
            2f64.powf(8.0),
            0.0,
            1.0,
        );
        let samples_normal = samples_normal
            .enum_map::<_, Fp, ModuleError>(|_, v| Ok(i64_to_felt(v)))
            .unwrap();
        let perts = samples_normal
            .enum_map::<_, Fp, ModuleError>(|_, v| {
                let value = (v - Fp::from(128));

                // convert back to float to rescale down to 8 bits...
                //let rescaled = dequantize(value, 16, 0.0);
                //println!("rescaled: {:?}", rescaled);
                //Ok(i64_to_felt::<Fp>(quantize_float(&value, 0.0, 8).unwrap()))
                Ok(value)
            })
            .unwrap();
        let samples_normal = samples_normal
            .enum_map::<_, Fp, ModuleError>(|_, v| {
                let value = (v - Fp::from(128)) * std;

                // convert back to float to rescale down to 8 bits...
                let rescaled = dequantize(value, 16, 0.0);
                Ok(i64_to_felt::<Fp>(
                    quantize_float(&rescaled, 0.0, 8).unwrap(),
                ))
            })
            .unwrap();

        samples_normal.to_vec()
    }

    fn layout(
        &self,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        samples: &ValTensor<Fp>,
        std: &ValTensor<Fp>, // not really std in this case ... instead is bound of uniform
    ) -> ValTensor<Fp> {
        //let lime_samples = self.get_lime_samples(samples);
        // perturb x_border by lime samples...
        let d = samples.dims()[1];

        let normal_samples = crate::circuit::ops::layouts::nonlinearity(
            base_config,
            region,
            &[samples.clone()],
            &crate::circuit::ops::lookup::LookupOp::Norm {
                scale: F32(2f32.powf(8.0)),
                mean: F32(0f32),
                std: F32(1f32),
            },
        )
        .unwrap();

        // center to 0
        let mut half_sample = create_constant_tensor(Fp::from(128), self.num_points * d);
        half_sample.reshape(&[self.num_points, d]);
        let perturbations = pairwise(
            base_config,
            region,
            &[normal_samples.clone(), half_sample],
            BaseOp::Sub,
        )
        .unwrap();

        // scale points by uniform range
        let mut std = std.clone();
        std.expand(&[self.num_points, self.d]).unwrap();
        let std_scaled =
            pairwise(base_config, region, &[perturbations, std], BaseOp::Mult).unwrap();

        // scale back down to 8 bits of precision
        let points_scaled = crate::circuit::ops::layouts::loop_div(
            base_config,
            region,
            &[std_scaled.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();

        points_scaled
    }
}

// bad name
// Given inputs, returns lime std dev and input point
trait LimeInputConfig: std::fmt::Debug {
    fn num_samples(&self) -> usize;
    fn run(&self, x: Vec<Fp>, d: usize) -> Result<(Vec<Fp>, Fp), ModuleError>;
    fn layout(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError>;
}

#[derive(Debug, Clone)]
pub struct PlainInputPicker {
    std: f64,
}

impl LimeInputConfig for PlainInputPicker {
    fn num_samples(&self) -> usize {
        0
    }

    fn run(&self, x: Vec<Fp>, d: usize) -> Result<(Vec<Fp>, Fp), ModuleError> {
        Ok((x, i64_to_felt(quantize_float(&self.std, 0.0, 8)?)))
    }

    fn layout(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        // TODO:
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
pub struct VectorInputPicker {
    step_size: f64,
    max_steps: f64,
}

impl LimeInputConfig for VectorInputPicker {
    fn num_samples(&self) -> usize {
        0
    }
    fn run(&self, x: Vec<Fp>, d: usize) -> Result<(Vec<Fp>, Fp), ModuleError> {
        unimplemented!();

        // generate random vector...
    }

    fn layout(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let d = x.dims()[0];
        assert_eq!(x.dims(), &[d]);

        unimplemented!();
    }
}

// TODO:
#[derive(Debug, Clone)]
pub struct SpheresInputPicker {}

trait LimeInputCircuit:
    std::fmt::Debug + serde_traitobject::Serialize + serde_traitobject::Deserialize + DynClone
{
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>);

    fn num_samples(&self) -> usize;

    fn num_points(&self) -> usize;

    fn find_surrogate(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Tensor<Fp>;

    fn find_std(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        samples: &[Fp],
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Fp;

    fn generate_model_input(&self, x: &Tensor<Fp>, samples: &[Fp]) -> Tensor<Fp>;

    fn layout_inputs(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        surrogate: &ValTensor<Fp>,
        samples: &ValTensor<Fp>,
    ) -> ValTensor<Fp>;

    fn layout_std(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_label: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
        labels: &ValTensor<Fp>,
    ) -> ValTensor<Fp>;
}

dyn_clone::clone_trait_object!(LimeInputCircuit);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlainInputCircuit {
    d: usize,
}

impl LimeInputCircuit for PlainInputCircuit {
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {}

    fn num_samples(&self) -> usize {
        0
    }

    fn num_points(&self) -> usize {
        0
    }

    fn find_surrogate(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Tensor<Fp> {
        x.clone()
    }

    fn find_std(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        samples: &[Fp],
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Fp {
        i64_to_felt(quantize_float(&1.0, 0.0, 8).unwrap())
    }

    // Empty
    fn generate_model_input(&self, x: &Tensor<Fp>, samples: &[Fp]) -> Tensor<Fp> {
        Tensor::new(None, &[0, self.d]).expect("error creating empty tensor")
    }

    fn layout_inputs(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        surrogate: &ValTensor<Fp>,
        samples: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        ValTensor::known_from_vec(&vec![])
    }

    fn layout_std(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_label: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
        labels: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        ValTensor::known_from_vec(&vec![])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorInputCircuit {
    //num_points: usize,
    n: usize, // max length of each vector
    m: usize, // number of vectors
    step: f64,
    d: usize,
}

impl LimeInputCircuit for VectorInputCircuit {
    fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {
        required_lookups.push(crate::circuit::ops::lookup::LookupOp::RecipSqrt {
            input_scale: F32(2f32.powf(8.0)),
            output_scale: F32(2f32.powf(8.0)),
        });
    }

    fn num_samples(&self) -> usize {
        self.m * self.d
    }

    fn num_points(&self) -> usize {
        self.n * self.m
    }

    fn find_surrogate(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Tensor<Fp> {
        x.clone()
    }

    fn find_std(
        &self,
        classify: &dyn Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        perturb: &dyn Fn(Tensor<Fp>) -> Tensor<Fp>,
        samples: &[Fp],
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Fp {
        let mut inputs = self.generate_model_input(x, samples);
        let mut inputs = inputs.pad_rows(self.num_points() + 1, Fp::zero()).unwrap();
        inputs
            .set_slice(&[self.num_points()..self.num_points() + 1, 0..self.d], x)
            .unwrap();

        let labels = classify(inputs).unwrap();

        let x_label = labels[self.num_points()];

        for i in 1..self.num_points() + 1 {
            if labels[i - 1] != x_label {
                // TODO: Why 10*step?
                return (Fp::from(i as u64)
                    * i64_to_felt::<Fp>(quantize_float(&(10.0 * self.step), 0.0, 8).unwrap()));
            }
        }

        let res = (Fp::from((self.num_points() + 1) as u64)
            * i64_to_felt::<Fp>(quantize_float(&(10.0 * self.step), 0.0, 8).unwrap()));
        return res;
    }

    // Empty
    // get samples, generate model input points...
    fn generate_model_input(&self, input: &Tensor<Fp>, samples: &[Fp]) -> Tensor<Fp> {
        assert_eq!(samples.len(), self.d * self.m);

        // TODO: normalized sampler
        // normalize point to length "step"
        let mut norms = vec![];
        for i in 0..self.m {
            let mut norm = Fp::zero();
            for j in 0..self.d {
                let val = samples[i * self.d + j];
                norm += val * val;
            }
            norms.push((felt_to_i64(norm) as f64 / 2f64.powf(8.0)).round() as i64);
        }

        // scale down to 8 bits...
        let recip_norms = crate::tensor::ops::nonlinearities::recip_sqrt(
            &norms.iter().cloned().into(),
            2f64.powf(8.0),
            2f64.powf(8.0),
        );
        // TODO: FINISH!!!

        let mut normalized = vec![];
        for i in 0..self.num_samples() {
            let val = (felt_to_i64(samples[i]) * recip_norms[(i / self.d)]);
            let val_scaled = (val as f64 / 2f64.powf(8.0)).round() as i64;
            normalized.push(val_scaled);
        }

        // TODO: must normalize sample vector to length "step"
        let mut points = vec![];

        // offset by 1 to never scale by 0
        for i in 1..self.n + 1 {
            let multiplier = Fp::from(i as u64);
            for j in 0..self.m {
                let step = input
                    .iter()
                    .enumerate()
                    .map(|(i, v)| multiplier * i64_to_felt::<Fp>(normalized[j * self.d + i]))
                    .collect::<Vec<_>>();
                let point = input
                    .iter()
                    .enumerate()
                    .map(|(i, v)| multiplier * i64_to_felt::<Fp>(normalized[j * self.d + i]) + v)
                    .collect::<Vec<_>>();
                points.extend(point);
            }
        }

        let res = Tensor::new(Some(&points), &[self.num_points(), self.d])
            .expect("error creating points tensor");
        res
    }

    fn layout_inputs(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        surrogate: &ValTensor<Fp>,
        samples: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        let mut points = vec![];
        let mut samples = samples.clone();
        samples.reshape(&[self.m, self.d]).unwrap();

        let mut surrogate = surrogate.clone();
        surrogate.reshape(&[1, self.d]);

        // compute
        let square_norms = einsum(
            base_config,
            region,
            &[samples.clone(), samples.clone()],
            "ij,ij->ik",
        )
        .unwrap();
        // scale back down to 8 bits..
        let square_norms_scaled = crate::circuit::ops::layouts::loop_div(
            base_config,
            region,
            &[square_norms.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        // scale down to 8 bits...
        let recip_norms = crate::circuit::ops::layouts::nonlinearity(
            base_config,
            region,
            &[square_norms_scaled],
            &crate::circuit::ops::lookup::LookupOp::RecipSqrt {
                input_scale: F32(2f32.powf(8.0)),
                output_scale: F32(2f32.powf(8.0)),
            },
        )
        .unwrap();
        // multiply by recips...
        let normalized = einsum(
            base_config,
            region,
            &[samples.clone(), recip_norms],
            "ij,ik->ij",
        )
        .unwrap();

        let normalized = crate::circuit::ops::layouts::loop_div(
            base_config,
            region,
            &[normalized.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();

        for i in 1..self.n + 1 {
            let mut multiplier = create_constant_tensor(Fp::from(i as u64), self.d);
            let step = pairwise(
                base_config,
                region,
                &[normalized.clone(), multiplier],
                BaseOp::Mult,
            )
            .unwrap();
            for j in 0..self.m {
                let m_step = step.get_slice(&[j..j + 1, 0..self.d]).unwrap();
                let point = pairwise(
                    base_config,
                    region,
                    &[surrogate.clone(), m_step.clone()],
                    BaseOp::Add,
                )
                .unwrap();

                points.push(point);
            }
        }

        let mut result = points[0].clone();
        for point in &points[1..] {
            result = result.concat(point.clone()).unwrap();
        }

        result
    }

    // layout checks for std
    fn layout_std(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_label: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
        labels: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        // get the first std_dev...

        // std_dev should be the first point that is enemy...

        let mut x_label = x_label.clone();
        x_label.expand(&[labels.len()]).unwrap();

        let opposite_labels =
            pairwise(base_config, region, &[labels.clone(), x_label], BaseOp::Sub).unwrap();
        let opposite_labels = crate::circuit::ops::layouts::nonlinearity(
            base_config,
            region,
            &[opposite_labels.clone()],
            &crate::circuit::ops::lookup::LookupOp::Abs,
        )
        .unwrap();

        let one = create_constant_tensor(Fp::one(), labels.len());
        let same_labels = pairwise(
            base_config,
            region,
            &[one.clone(), opposite_labels.clone()],
            BaseOp::Sub,
        )
        .unwrap();

        let step: Fp = i64_to_felt(quantize_float(&(10.0 * self.step), 0.0, 8).unwrap());
        let all_stds: ValTensor<Fp> = (0..self.num_points())
            .map(|i| ValType::Constant(Fp::from(((i / 3) + 1) as u64) * step))
            .collect::<Vec<_>>()
            .into();
        let opposite_stds = pairwise(
            base_config,
            region,
            &[opposite_labels.clone(), all_stds],
            BaseOp::Mult,
        )
        .unwrap();
        let max = create_constant_tensor(Fp::from(self.num_points() as u64) * step, labels.len());
        let same_stds = pairwise(
            base_config,
            region,
            &[same_labels.clone(), max],
            BaseOp::Mult,
        )
        .unwrap();

        let stds = pairwise(
            base_config,
            region,
            &[opposite_stds.clone(), same_stds],
            BaseOp::Add,
        )
        .unwrap();

        let min_std = min(base_config, region, &[stds]).unwrap();

        min_std.clone()
    }
}

pub struct SpheresInputCircuit {
    pub num_points: usize,
}

impl SpheresInputCircuit {
    fn num_points(&self) -> usize {
        self.num_points
    }

    fn find_surrogate<G, H>(
        self,
        classify: G,
        perturb: H,
        x: &Tensor<Fp>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Tensor<Fp>
    where
        G: Fn(Tensor<Fp>) -> Result<Tensor<Fp>, GraphError>,
        H: Fn(Tensor<Fp>) -> Tensor<Fp>,
    {
        let surrogate = LassoModel::find_local_surrogate(classify, perturb, x, 8, 0);
        surrogate.expect("Could not compute spheres surrogate")
    }

    fn generate_model_input(&self) -> Tensor<Fp> {
        unimplemented!()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LimeWitness {
    pub model_input: Tensor<Fp>,
    pub surrogate: Vec<Fp>,
    pub coefficients: Vec<Fp>,
    pub intercept: Fp,
    pub dual: Vec<Fp>,
    pub std: Fp,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LimeCircuit {
    #[serde(with = "serde_traitobject")]
    pub input_circuit: Box<dyn LimeInputCircuit>,
    #[serde(with = "serde_traitobject")]
    pub sample_circuit: Box<dyn LimePointSampler>,
    pub weight_strategy: LimeWeightStrategy,
    pub num_points: usize,
    d: usize,
}

impl Default for LimeCircuit {
    fn default() -> Self {
        Self {
            input_circuit: Box::new(PlainInputCircuit { d: 0 }),
            sample_circuit: Box::new(UniformPointSampleCircuit::new(0, 0)),
            num_points: 0,
            weight_strategy: LimeWeightStrategy::Exponential,
            d: 0,
        }
    }
}

use crate::graph::model::Model;
use crate::RunArgs;
impl LimeCircuit {
    const LIME_MULTIPLIER: u64 = 4;
    const DUAL_GAP_TOLERANCE: f64 = 0.1;

    const INPUT_PRECISION: u64 = 8;
    const DUAL_INPUT_PRECISION: u64 = 16;
    const DUAL_PRECISION: u64 = 16;
    const DUAL_GAP_PRECISION: u64 = 16;

    const ALPHA: f64 = 0.010009765625;

    pub fn new(model: &Model, run_args: &RunArgs) -> Self {
        let d = model
            .graph
            .input_shapes()
            .expect("No shapes for lime circuit!")[0][1];
        let input_circuit = Self::new_input_circuit(run_args, d);
        let num_points = run_args.generate_explanation.unwrap();
        let sample_circuit = Self::new_sampling_circuit(run_args, d);
        let weight_strategy = run_args.lime_weight_strategy.clone().unwrap();

        Self {
            input_circuit,
            sample_circuit,
            num_points,
            weight_strategy,
            d,
        }
    }

    pub fn from_run_args(d: usize, run_args: &RunArgs) -> Self {
        let input_circuit = Self::new_input_circuit(run_args, d);
        let num_points = run_args.generate_explanation.unwrap();
        let sample_circuit = Self::new_sampling_circuit(run_args, d);
        let weight_strategy = run_args.lime_weight_strategy.clone().unwrap();

        Self {
            input_circuit,
            sample_circuit,
            num_points,
            weight_strategy,
            d,
        }
    }

    fn new_input_circuit(run_args: &RunArgs, d: usize) -> Box<dyn LimeInputCircuit> {
        match run_args.surrogate_strategy {
            Some(SurrogateStrategy::Plain) => Box::new(PlainInputCircuit { d }),
            Some(SurrogateStrategy::Vector) => Box::new(VectorInputCircuit {
                d,
                n: run_args.surrogate_n.unwrap(),
                m: run_args.surrogate_m.unwrap(),
                step: run_args.surrogate_step.unwrap(),
            }),
            Some(SurrogateStrategy::Spheres) => unimplemented!(),
            None => Box::new(PlainInputCircuit { d }),
        }
    }

    fn new_sampling_circuit(run_args: &RunArgs, d: usize) -> Box<dyn LimePointSampler> {
        match run_args.lime_sampling {
            Some(SamplingStrategy::Uniform) => Box::new(UniformPointSampleCircuit::new(
                run_args.generate_explanation.unwrap(),
                d,
            )),
            Some(SamplingStrategy::Gaussian) => Box::new(GaussianPointSampleCircuit::new(
                run_args.generate_explanation.unwrap(),
                d,
            )),
            Some(SamplingStrategy::Spherical) => unimplemented!(),
            None => Box::new(UniformPointSampleCircuit::new(
                run_args.generate_explanation.unwrap(),
                d,
            )),
        }
    }

    fn from_settings(settings: &GraphSettings) -> Self {
        let input_circuit = Self::new_input_circuit(&settings.run_args, settings.d);
        let num_points = settings.run_args.generate_explanation.unwrap();
        let d = settings.d;
        let sample_circuit = Self::new_sampling_circuit(&settings.run_args, d);
        let weight_strategy = settings.run_args.lime_weight_strategy.clone().unwrap();

        Self {
            input_circuit,
            sample_circuit,
            num_points,
            weight_strategy,
            d,
        }
    }

    // taken from lime repo
    pub fn kernel_width(d: usize) -> f32 {
        (d as f32).sqrt() * 0.75
    }

    pub fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {
        if !ABLATE_LIME_CHECKS {
            if !matches!(self.weight_strategy, LimeWeightStrategy::Uniform) {
                required_lookups.push(crate::circuit::ops::lookup::LookupOp::RecipSqrt {
                    input_scale: F32(2f32.powf(8.0)),
                    output_scale: F32(2f32.powf(8.0)),
                });
                required_lookups.push(crate::circuit::ops::lookup::LookupOp::LimeWeight {
                    input_scale: F32(2f32.powf(8.0)),
                    sigma: Self::kernel_width(self.d).into(),
                });
                required_lookups.push(crate::circuit::ops::lookup::LookupOp::Sqrt {
                    scale: F32(2f32.powf(8.0)),
                });
            }
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::Div {
                denom: F32(2f32.powf(8.0)),
            });
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::Div {
                denom: F32(2f32.powf(4.0)),
            });
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::Abs);
        }
        if !ABLATE_INPUTS {
            self.sample_circuit.add_lookups(required_lookups);
            self.input_circuit.add_lookups(required_lookups);
        }
    }

    pub fn add_range_checks(&self, required_range_checks: &mut Vec<CRange>) {
        if !ABLATE_INPUTS {
            required_range_checks.push((-128, 128));
        }
        if !ABLATE_LIME_CHECKS {
            if ABLATE_INPUTS {
                required_range_checks.push((-128, 128));
            }
            required_range_checks.push((-8, 8));
            let dual_gap_tolerance =
                (Self::DUAL_GAP_TOLERANCE * 2f64.powf(Self::DUAL_GAP_PRECISION as f64)) as i64;
            required_range_checks.push((-dual_gap_tolerance, dual_gap_tolerance));
            let dual_feasible_tolerance = (1.05
                * (Self::ALPHA * self.num_points as f64)
                * 2f64.powf(Self::DUAL_GAP_PRECISION as f64))
            .ceil() as i64;
            required_range_checks.push((-dual_feasible_tolerance, dual_feasible_tolerance));
        }
    }

    fn create_config(&self, meta: &mut ConstraintSystem<Fp>) -> LimeConfig {
        let num_points = self.num_points;
        let sample_chip = SampleChip::configure(meta, self.num_samples());
        let perturb_chip = PerturbChip::configure(meta, Some(256));
        let d = self.d;

        //let input_config = self.input_circuit.create_config(meta);

        LimeConfig {
            sample_chip,
            perturb_chip,
            samples: None,
        }
    }

    // Need this to compute model dimensions...
    // Gross but its fine
    pub fn num_points_from_run_args(run_args: &RunArgs) -> usize {
        let num_points = run_args.generate_explanation.unwrap();
        let input_circuit = Self::new_input_circuit(run_args, 0);

        num_points + input_circuit.num_points()
    }

    // Number of randomness samples needed for the circuit
    fn num_samples(&self) -> usize {
        self.num_points * self.d + self.input_circuit.num_samples()
    }

    // Number of points needed for the circuit
    pub fn num_points(&self) -> usize {
        self.num_points + self.input_circuit.num_points()
    }

    // split samples into ones for LIME and for input circuit
    // TODO: convert to tensor
    fn split_samples<'a>(&self, samples: &'a [Fp]) -> (&'a [Fp], &'a [Fp]) {
        let lime_end = self.num_points * self.d;
        let lime_samples = &samples[0..lime_end];
        let input_samples = &samples[lime_end..];
        assert_eq!(
            input_samples.len(),
            self.input_circuit.num_samples(),
            "Insufficient input circuit samples"
        );
        (lime_samples, input_samples)
    }

    fn split_samples_tensor(&self, samples: &ValTensor<Fp>) -> (ValTensor<Fp>, ValTensor<Fp>) {
        let lime_end = self.num_points * self.d;
        let mut lime_samples = samples
            .get_slice(&[0..lime_end])
            .expect("Failed to slice samples");
        lime_samples
            .reshape(&[self.num_points, self.d])
            .expect("Failed to reshape samples");

        let mut input_samples = samples
            .get_slice(&[lime_end..samples.len()])
            .expect("Failed to slice input samples");
        (lime_samples, input_samples)
    }

    //fn get_lime_samples(&self, samples: &ValTensor<Fp>) -> ValTensor<Fp> {}

    pub fn generate_lime_inputs(
        &self,
        surrogate: &[Fp],
        samples: &[Fp],
        std: Fp,
    ) -> Result<Vec<Fp>, ModuleError> {
        let multiplier = Fp::from(Self::LIME_MULTIPLIER);
        let samples = self.sample_circuit.generate_witness(samples, std);
        let perturbations = samples
            .iter()
            .enumerate()
            .map(|(i, v)| v + surrogate[i % self.d])
            .collect::<Vec<_>>();

        let mut res = surrogate.to_vec();
        res.extend(perturbations);
        Ok(res)
    }

    // TODO: rename -> gen_witness
    pub fn run(&self, inputs: &mut [Tensor<Fp>], model: &Model, run_args: &RunArgs) -> LimeWitness {
        let d = inputs[0].dims()[1];
        assert_eq!(self.d, d);
        let batch_size = model.graph.input_shapes().unwrap()[0][0];

        let input = inputs[0].get_slice(&[0..1, 0..d]).unwrap();

        use crate::circuit::modules::sample::SampleChip;
        let samples = SampleChip::<8>::run(self.num_samples()).unwrap();
        {
            let (lime_samples, input_samples) = self.split_samples(&samples);

            // This closure accepts batches of any size for inferrence
            let classify = |mut batch: Tensor<Fp>| {
                let mut n = batch.dims()[0];
                let mut i = 0;

                let mut output = vec![];
                while n > 0 {
                    let mini_batch = if n < batch_size {
                        let mini_batch = batch
                            .get_slice(&[i * batch_size..batch.dims()[0], 0..d])
                            .unwrap();
                        mini_batch.pad_rows(batch_size, Fp::zero()).unwrap()
                    } else {
                        batch
                            .get_slice(&[i * batch_size..(i + 1) * batch_size, 0..d])
                            .unwrap()
                    };

                    // TODO pad the batch.... then we can handle it...
                    let mut fresh_inputs = inputs.to_vec();
                    fresh_inputs[0] = mini_batch;

                    let res = model.forward(&fresh_inputs, run_args, true, true)?;
                    output.extend(
                        res.outputs[0]
                            .get_slice(&[0..(std::cmp::min(n, batch_size))])
                            .unwrap()
                            .to_vec()
                            .clone(),
                    );
                    n = if n < batch_size { 0 } else { n - batch_size };
                    i += 1;
                }

                Ok(Tensor::new(Some(&output), &[batch.dims()[0]]).unwrap())
            };

            // TODO: compute samples via our hash function....
            let perturb = |x: Tensor<Fp>| {
                let d = x.dims()[1];
                x.expand(&[self.num_points, d]).unwrap();
                // get first n*d samples
                let y = Tensor::new(Some(&lime_samples), &[self.num_points, d]).unwrap();
                let multiplier_int = 2;
                let multiplier = Fp::from(2);
                let y = y
                    .par_enum_map::<_, _, Error>(|i, v| Ok(multiplier * v))
                    .unwrap();
                let add = (x + y).unwrap();
                let res = add
                    .par_enum_map::<_, _, GraphError>(|i, v| Ok(v - Fp::from(multiplier_int * 128)))
                    .unwrap();
                res
            };

            // add surrogate point to inputs...
            // TODO: fix constants
            let surrogate = self
                .input_circuit
                .find_surrogate(&classify, &perturb, &input, 8, 0);
            let input_points = self
                .input_circuit
                .generate_model_input(&input, input_samples);
            let std =
                self.input_circuit
                    .find_std(&classify, &perturb, &input_samples, &input, 8, 0);

            let lime_inputs = self
                .generate_lime_inputs(&surrogate, lime_samples, std)
                .expect("Error generating lime input points");

            let lime_points = Tensor::new(
                Some(&lime_inputs[self.d..(self.num_points + 1) * self.d]),
                &[self.num_points, self.d],
            )
            .expect("Lime points tensor wrong shape");
            let labels = classify(lime_points.clone()).unwrap();

            let (coeffs, top_k, top_k_idx, intercept, dual) = LassoModel::lasso(
                &surrogate,
                &lime_points,
                &labels,
                Self::INPUT_PRECISION as i32,
                0,
                Self::DUAL_GAP_PRECISION as i32,
                self.weight_strategy.clone(),
                run_args.top_k.unwrap(),
            );

            let mut model_input =
                crate::tensor::ops::concat(&[&input, &lime_points, &input_points], 0)
                    .expect("Failed to concat model input tensor");
            assert_eq!(model_input.dims(), &[1 + self.num_points(), self.d]);

            LimeWitness {
                coefficients: coeffs.to_vec(),
                dual: dual.to_vec(),
                intercept,
                surrogate: surrogate.to_vec(),
                std,

                model_input,
            }
        }
    }

    pub fn layout_samples(
        &self,
        config: &LimeConfig,
        layouter: &mut impl Layouter<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        if ABLATE_INPUTS {
            let samples = SampleChip::<8>::run(config.sample_chip.config.n).unwrap();
            return Ok(ValTensor::known_from_vec(&samples));
        }

        let samples = config.sample_chip.layout(layouter)?;
        let samples = samples
            .iter()
            .map(|v| ValType::PrevAssigned(v.clone()))
            .collect::<Vec<ValType<_>>>();
        Ok(samples.into())
    }

    pub fn layout_perturbations(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        lime_samples: &ValTensor<Fp>,
        std: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        //let lime_samples = self.get_lime_samples(samples);
        // perturb x_border by lime samples...

        let d = lime_samples.dims()[1];
        if ABLATE_INPUTS {
            let samples = self.sample_circuit.generate_witness(
                &lime_samples.get_field_evals().unwrap(),
                std.get_field_evals().unwrap()[0],
            );
            let mut x_border_expanded = x_border.clone();
            x_border_expanded.reshape(&[1, d]);
            x_border_expanded.expand(&[self.num_points, d]).unwrap();
            let x_border_values = x_border_expanded.get_field_evals().unwrap();

            let perturbations = x_border_values
                .clone()
                .iter()
                .zip(samples.clone())
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>();
            return Ok(ValTensor::known_from_vec(&perturbations));
        }

        let samples = self
            .sample_circuit
            .layout(base_config, region, lime_samples, std);

        let mut x_border_expanded = x_border.clone();
        x_border_expanded.reshape(&[1, d]);
        x_border_expanded.expand(&[self.num_points, d]).unwrap();

        let lime_samples = pairwise(
            base_config,
            region,
            &[x_border_expanded.clone(), samples.clone()],
            BaseOp::Add,
        )
        .unwrap();

        Ok(lime_samples)
    }

    // TODO....
    pub fn layout_inputs(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        std: &ValTensor<Fp>,
        surrogate: &ValTensor<Fp>,
        samples: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let d = x.dims()[0];
        assert_eq!(x.dims(), &[d]);

        let (lime_samples, input_samples) = self.split_samples_tensor(samples);

        let lime_perturbations = self
            .layout_perturbations(config, base_config, region, surrogate, &lime_samples, std)
            .expect("Failed to layotu lime perturbations");
        let input_points = self.input_circuit.layout_inputs(
            config,
            base_config,
            region,
            surrogate,
            &input_samples,
        );
        let mut result = x.concat(lime_perturbations).unwrap();
        result = result.concat(input_points).unwrap();
        result.reshape(&[self.num_points() + 1, d]).unwrap();
        Ok(result)
    }

    pub fn layout_exp_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &PValTensor,
        inputs: &PValTensor,
    ) -> PValTensor {
        // expand x
        let d = x_border.shape()[0];
        let mut x_expanded = x_border.clone();
        x_expanded = x_expanded.reshape(&[1, d]);
        x_expanded = x_expanded.expand(&[self.num_points, d]);
        let deltas = x_expanded.sub(config, region, inputs);

        let square_distance = deltas
            .einsum(config, region, &deltas, "ij,ij->i")
            .rescale(config, region, 8);

        // lookup table
        let weights = square_distance.nonlinearity(
            config,
            region,
            &crate::circuit::ops::lookup::LookupOp::LimeWeight {
                input_scale: 2f64.powf(8.0).into(),
                sigma: Self::kernel_width(d).into(),
            },
        );

        weights
    }

    pub fn layout_dist_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &PValTensor,
        inputs: &PValTensor,
    ) -> PValTensor {
        // expand x
        let d = x_border.shape()[0];
        let mut x_expanded = x_border.clone();
        x_expanded = x_expanded.reshape(&[1, d]);
        x_expanded = x_expanded.expand(&[self.num_points, d]);
        let deltas = x_expanded.sub(config, region, inputs);
        let square_distance = deltas
            .einsum(config, region, &deltas, "ij,ij->i")
            .rescale(config, region, 8);

        square_distance
    }

    pub fn layout_lime_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &PValTensor,
        inputs: &PValTensor,
    ) -> PValTensor {
        match self.weight_strategy {
            LimeWeightStrategy::Exponential => {
                self.layout_exp_weights(config, region, x_border, inputs)
            }
            LimeWeightStrategy::Distance => {
                self.layout_dist_weights(config, region, x_border, inputs)
                //unimplemented!();
            }
            // handled outside TODO(Evan): make this more uniform...
            LimeWeightStrategy::Uniform => unimplemented!(),
        }
    }

    pub fn rescale(
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        input: &ValTensor<Fp>,
        from: u64,
        to: u64,
    ) -> ValTensor<Fp> {
        assert!(from % 4 == 0);
        assert!(to % 4 == 0);
        assert!(to <= from);

        if to == from {
            return input.clone();
        }

        let mut input = input.clone();
        let mut from = from;
        while from - 8 >= to {
            input = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[input.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
            from -= 8;
        }

        if from - 4 == to {
            input = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[input.clone()],
                Fp::from(2u64.pow(4)),
            )
            .unwrap();
            from -= 4;
        }

        assert_eq!(from, to);
        input
    }

    pub fn layout_lime_checks(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
        outputs: &ValTensor<Fp>,
        lime_model: &ValTensor<Fp>,
        lime_intercept: &ValTensor<Fp>,
        dual: &ValTensor<Fp>,
    ) {
        if ABLATE_LIME_CHECKS {
            return;
        }
        // ok...
        //
        // STEPS:
        // 1. compute weights
        //   a. compute square distance between x and points
        //   b. pass into lookup table (exp(_ / sigma^2))
        // 2. multiply points and labels by sqrt of weights

        let x_border = PValTensor::new(x_border.clone(), Self::INPUT_PRECISION);
        let inputs = PValTensor::new(inputs.clone(), Self::INPUT_PRECISION);
        let outputs = PValTensor::new(outputs.clone(), 0);
        let lime_model = PValTensor::new(lime_model.clone(), Self::DUAL_GAP_PRECISION);
        let lime_intercept = PValTensor::new(lime_intercept.clone(), Self::DUAL_GAP_PRECISION);
        let dual = PValTensor::new(dual.clone(), Self::DUAL_PRECISION);

        log::debug!("LIME LAYOUT");
        region.debug_report();
        let d = x_border.shape()[0];
        let inputs = inputs.slice(&[1..self.num_points + 1, 0..d]);
        let outputs = outputs.slice(&[1..self.num_points + 1]);

        let (inputs, outputs) = if !matches!(self.weight_strategy, LimeWeightStrategy::Uniform) {
            let weights = self.layout_lime_weights(config, region, &x_border, &inputs);

            // sqrt weights
            let mut sqrt_weights = weights.nonlinearity(
                config,
                region,
                &crate::circuit::ops::lookup::LookupOp::Sqrt {
                    scale: F32(2f32.powf(weights.precisionf())),
                },
            );

            // multiply inputs and outputs by sqrt weights
            // see: https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca01Self::INPUT_PRECISION7c07b82157601b32/lime/lime_base.py#L1Self::DUAL_PRECISION
            let mut input_sqrt_weights = sqrt_weights.clone();
            input_sqrt_weights = input_sqrt_weights.expand(&[self.num_points, d]);
            let inputs = input_sqrt_weights.mul(config, region, &inputs);

            let mut output_sqrt_weights = sqrt_weights.clone();
            output_sqrt_weights = output_sqrt_weights.expand(&[self.num_points]);
            let outputs = output_sqrt_weights.mul(config, region, &outputs).rescale(
                config,
                region,
                Self::INPUT_PRECISION,
            );
            (inputs, outputs)
        } else {
            (inputs, outputs)
        };

        // RESCALE things for the computation...
        let outputsd = outputs.rescale(config, region, Self::DUAL_GAP_PRECISION);
        let inputs = inputs.rescale(config, region, Self::DUAL_GAP_PRECISION);

        // residuals part
        let deltas = inputs.einsum(config, region, &lime_model, "ij,j->ik");

        // rescale intercept to Self::DUAL_PRECISION bits
        let mut lime_intercept = lime_intercept.clone();
        lime_intercept = lime_intercept.expand(&[self.num_points]);
        let centered_outputs =
            outputsd
                .sub(config, region, &lime_intercept)
                .rescale(config, region, deltas.precision);
        let intermediate = centered_outputs.sub(config, region, &deltas);
        let intermediate = intermediate.rescale(config, region, 28);
        let square_residuals = intermediate.dot(config, region, &intermediate);
        let square_residuals = square_residuals.rescale(config, region, Self::DUAL_PRECISION);

        let multiplier = PValTensor::constant_f64(1f64 / (2f64), Self::DUAL_PRECISION)
            .expand(square_residuals.shape());
        let square_residuals_dived = square_residuals.mul(config, region, &multiplier).rescale(
            config,
            region,
            Self::DUAL_GAP_PRECISION,
        );

        let alpha = PValTensor::constant_f64(Self::ALPHA, Self::DUAL_GAP_PRECISION);
        let multiplier = PValTensor::constant_f64(self.num_points as f64, Self::DUAL_GAP_PRECISION);
        let l1_part =
            lime_model.nonlinearity(config, region, &crate::circuit::ops::lookup::LookupOp::Abs);
        let l1_part = l1_part.sum(config, region);
        let l1_part = l1_part.mul(config, region, &alpha);
        let l1_part = l1_part.mul(config, region, &multiplier);
        let l1_part = l1_part.rescale(config, region, Self::DUAL_GAP_PRECISION);
        let l1_part = l1_part.rescale(config, region, Self::DUAL_GAP_PRECISION);

        let mut primal_objective = square_residuals_dived.add(config, region, &l1_part);

        // compute dual objective
        let mut square_dual = dual.dot(config, region, &dual);
        square_dual = square_dual.rescale(config, region, Self::DUAL_PRECISION);

        let multiplier = PValTensor::constant_f64(-1f64 / (2f64), Self::DUAL_PRECISION)
            .expand(square_dual.shape());
        let square_dual_dived = square_dual.mul(config, region, &multiplier).rescale(
            config,
            region,
            Self::DUAL_PRECISION,
        );

        let mut dual_res = dual.dot(config, region, &centered_outputs).rescale(
            config,
            region,
            Self::DUAL_PRECISION,
        );

        let dual_res = square_dual_dived.sub(config, region, &dual_res).rescale(
            config,
            region,
            Self::DUAL_GAP_PRECISION,
        );

        let dual_objective = dual_res;
        let dual_gap = primal_objective.sub(config, region, &dual_objective);

        let range_check_bracket =
            (Self::DUAL_GAP_TOLERANCE * 2f64.powf(Self::DUAL_GAP_PRECISION as f64)) as i64;
        dual_gap.range_check(config, region, &(-range_check_bracket, range_check_bracket));

        // check dual is feasible
        let range_check_bracket = (1.05
            * (Self::ALPHA * self.num_points as f64)
            * 2f64.powf(Self::DUAL_GAP_PRECISION as f64))
        .ceil() as i64;
        let range_check_felt: Fp = i64_to_felt(range_check_bracket);
        let mut dual_feasible = inputs.einsum(config, region, &dual, "ji,j->i");
        dual_feasible
            .rescale(config, region, Self::DUAL_GAP_PRECISION)
            .range_check(config, region, &(-range_check_bracket, range_check_bracket));

        region.debug_report();
    }

    pub fn layout_checks(
        &self,
        config: &LimeConfig,
        base_config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        surrogate: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
        outputs: &ValTensor<Fp>,
        lime_witness: &LimeWitness,
    ) {
        let lime_model = ValTensor::known_from_vec(&lime_witness.coefficients);
        let lime_intercept = ValTensor::known_from_vec(&vec![lime_witness.intercept]);
        let dual = ValTensor::known_from_vec(&lime_witness.dual);
        self.layout_lime_checks(
            base_config,
            region,
            surrogate,
            inputs,
            outputs,
            &lime_model,
            &lime_intercept,
            &dual,
        );
        let x_label = outputs.get_slice(&[0..1]).unwrap();
        let input_inputs = inputs
            .get_slice(&[self.num_points + 1..outputs.len()])
            .unwrap();
        let input_labels = outputs
            .get_slice(&[self.num_points + 1..outputs.len()])
            .unwrap();
        self.input_circuit.layout_std(
            config,
            base_config,
            region,
            &x_label,
            &input_inputs,
            &input_labels,
        );
    }
}

// TODO(Refactor perturb)
#[derive(Clone, Debug)]
pub struct LimeConfig {
    sample_chip: SampleChip<8>,
    perturb_chip: PerturbChip<Fp>,

    samples: Option<Vec<AssignedCell<Fp, Fp>>>,
}

impl LimeConfig {
    pub fn from_settings(cs: &mut ConstraintSystem<Fp>, settings: &GraphSettings) -> Self {
        let c = LimeCircuit::from_settings(settings);
        c.create_config(cs)
    }
}

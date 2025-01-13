/// OKKKKKKKKKKKES:
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use dyn_clone::DynClone;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;

use super::errors::ModuleError;
use crate::circuit::{lookup::LookupOp, BaseConfig as PolyConfig, CheckMode, Op};
use crate::graph::errors::GraphError;
use crate::graph::lime::LassoModel;
use crate::graph::GraphSettings;
use crate::tensor::val::{create_constant_tensor, create_unit_tensor};
use crate::tensor::{Tensor, TensorType, ValTensor, ValType};
use crate::{SamplingStrategy, Scale, SurrogateStrategy};

use crate::circuit::ops::layouts::*;
use crate::circuit::table::Range;
use crate::circuit::utils::F32;
use crate::circuit::{ops::base::BaseOp, utils};
use crate::fieldutils::{felt_to_i64, i64_to_felt};

use super::perturb::PerturbChip;
use super::sample::SampleChip;
use crate::circuit::ops::chip::BaseConfig;
use crate::circuit::ops::region::RegionCtx;
use crate::LimeWeightStrategy;

use crate::graph::utilities::{dequantize, quantize_float};
// helper to assign tensors to column of cells

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
        println!("inputs: {:?}", inputs);
        let mut inputs = inputs.pad_rows(self.num_points() + 1, Fp::zero()).unwrap();
        inputs
            .set_slice(&[self.num_points()..self.num_points() + 1, 0..self.d], x)
            .unwrap();

        let labels = classify(inputs).unwrap();
        println!("labels: {:?}", labels);

        let x_label = labels[self.num_points()];

        for i in 1..self.num_points() + 1 {
            if labels[i - 1] != x_label {
                println!("OP LABEL: {:?}", i);
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

        //println!("square norms scaled!: {:?}", square_norms);
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
        //println!("square_norms_scaled: {:?}", square_norms_scaled.show());
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

        println!("inputs: {:?}", inputs);
        println!("labels: {:?}", labels);
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
        println!("opp-labels: {:?}", opposite_labels);
        println!("all stds: {:?}", all_stds);
        let opposite_stds = pairwise(
            base_config,
            region,
            &[opposite_labels.clone(), all_stds],
            BaseOp::Mult,
        )
        .unwrap();
        println!("opp-stds: {:?}", opposite_stds);
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
        println!("stds: {:?}", stds);

        let min_std = min(base_config, region, &[stds]).unwrap();
        println!("min_std: {:?}", min_std);

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

    fn run() {
        //let ball_samples = &samples[n_lime * d..];
        ////println!("perturbations: {:?}", perturbations);

        ////println!("x: {:?}", x);
        ////println!("x_border: {:?}", x_border);
        //let mut radius = Fp::from(0);
        //let mut vals = vec![];
        //for i in 0..d {
        //    let val = x[i] - x_border[i];
        //    vals.push(val);
        //    radius += val * val;
        //}
        ////println!("diffs: {:?}", vals);
        ////println!("square RADIUS: {:?}", radius);
        //let radius = (felt_to_i64(radius) as f64 / 2f64.powf(8.0)).round() as i64;
        ////println!("scaled RADIUS: {:?}", radius);
        //let mut radius = crate::tensor::ops::nonlinearities::sqrt(
        //    &Tensor::new(Some(&[radius]), &[1]).unwrap(),
        //    2f64.powf(8.0),
        //);
        ////println!("RUN RADIUS: {:?}", radius);

        //let ball_tensor = Tensor::new(Some(ball_samples), &[n_ball, d + 2]).unwrap();
        //let ball_tensor = ball_tensor.clone().map(|x| felt_to_i64(x));
        //let ball_samples_normal = crate::tensor::ops::nonlinearities::normal_inverse_cdf(
        //    &ball_tensor,
        //    2f64.powf(8.0),
        //    0.0,
        //    1.0,
        //);
        ////println!("ball_samples_normal: {:?}", ball_samples_normal);
        ////println!(
        ////    "ball_samples_normal float: {:?}",
        ////    ball_samples_normal
        ////        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
        ////        .unwrap()
        ////);
        //let mut square_norms = vec![];
        //for i in 0..n_ball {
        //    let mut norm = 0;

        //    for j in 0..d + 2 {
        //        let val = ball_samples_normal[i * (d + 2) + j];
        //        norm += val * val;
        //    }
        //    square_norms.push(((norm as f64) / 2f64.powf(8.0)).round() as i64);
        //}
        ////println!("square norms scaled!: {:?}", square_norms);
        //// scale down to 8 bits...
        //let recip_norms = crate::tensor::ops::nonlinearities::recip_sqrt(
        //    &square_norms.iter().cloned().into(),
        //    2f64.powf(8.0),
        //    2f64.powf(8.0),
        //);
        ////println!("norms!: {:?}", recip_norms.show());

        //let mut normalized = vec![];
        //for i in 0..(n_ball) * (d + 2) {
        //    let val = (ball_samples_normal[i] * recip_norms[(i / (d + 2))]);
        //    let val_scaled = (val as f64 / 2f64.powf(8.0)).round() as i64;
        //    normalized.push(val_scaled);
        //}
        ////println!("normalized: {:?}", normalized);

        //let normalized = Tensor::new(Some(&normalized), &[n_ball, d + 2]).unwrap();
        ////println!(
        ////    "normalized float: {:?}",
        ////    normalized
        ////        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
        ////        .unwrap()
        ////);
        //let sphere_samples = normalized.get_slice(&[0..n_ball, 0..d]).unwrap();
        ////println!("sphere_samples_scaled: {:?}", sphere_samples.show());
        ////println!(
        ////    "sphere_samples_scaled float: {:?}",
        ////    sphere_samples
        ////        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
        ////        .unwrap()
        ////);
        ////let ball_norms =

        //// scale to radius...
        //let radius_expanded = radius.expand(&[n_ball, d]).unwrap();
        //let sphere_samples_radius = mult(&[sphere_samples.clone(), radius_expanded])?;
        ////println!("sphere_samples_radius: {:?}", sphere_samples_radius.show());
        //let sphere_samples_radius_scaled = sphere_samples_radius
        //    .enum_map::<_, _, ModuleError>(|i, v| Ok((v as f64 / 2f64.powf(8.0)).round() as i64))?;
        ////println!(
        ////    "sphere_samples_radius_scaled: {:?}",
        ////    sphere_samples_radius_scaled.show()
        ////);
        ////println!(
        ////    "sphere_samples_radius_scaled float: {:?}",
        ////    sphere_samples_radius_scaled
        ////        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
        ////        .unwrap()
        ////);

        ////println!("X: {:?}", x);
        //let mut perturbations2 = sphere_samples_radius_scaled
        //    .enum_map::<_, _, ModuleError>(|i, v| Ok(x[i % d] + i64_to_felt::<Fp>(v)))
        //    .unwrap();
        ////println!("perturbations2: {:?}", perturbations2.show());

        //// print out the floating point...
        //let x_test = Tensor::new(Some(&x), &[d]).unwrap();
        ////println!("x: {:?}", x_test.dequantize(8));
        ////println!("perturbations2: {:?}", perturbations2.dequantize(8));

        //// Must pass through normal etc...
        //// FIXME(EVAN)
        ////perturbations.extend(perturbations2);

        //let mut res = x.clone();
        //res.extend(perturbations);
        //res.extend(x_border);
        //res.extend(perturbations2);
        //res
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LimeWitness {
    pub model_input: Tensor<Fp>,
    pub surrogate: Vec<Fp>,
    pub coefficients: Vec<Fp>,
    // TODO: topk...
    // TODO
    pub intercept: Fp,
    pub dual: Vec<Fp>,
    pub std: Fp,
    // TODO: surrogate?
    //Some((coeffs, top_k, top_k_idx, intercept, dual, surrogate))
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

    pub fn new(model: &Model, run_args: &RunArgs) -> Self {
        let d = model
            .graph
            .input_shapes()
            .expect("No shapes for lime circuit!")[0][1];
        let input_circuit = Self::new_input_circuit(run_args, d);
        let num_points = run_args.generate_explanation.unwrap();
        let sample_circuit = Self::new_sampling_circuit(run_args, d);
        let weight_strategy = run_args.lime_weight_strategy.clone().unwrap();
        println!("model shapes: {:?}", model.graph.input_shapes());

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

    pub fn add_lookups(&self, required_lookups: &mut Vec<LookupOp>) {
        if !matches!(self.weight_strategy, LimeWeightStrategy::Uniform) {
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::RecipSqrt {
                input_scale: F32(2f32.powf(8.0)),
                output_scale: F32(2f32.powf(8.0)),
            });
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::LimeWeight {
                input_scale: F32(2f32.powf(8.0)),
                sigma: Lime2Chip::kernel_width(self.d).into(),
            });
            required_lookups.push(crate::circuit::ops::lookup::LookupOp::Sqrt {
                scale: F32(2f32.powf(8.0)),
            });
        }
        required_lookups.push(crate::circuit::ops::lookup::LookupOp::Div {
            denom: F32(2f32.powf(8.0)),
        });
        required_lookups.push(crate::circuit::ops::lookup::LookupOp::Abs);
        self.sample_circuit.add_lookups(required_lookups);
    }

    pub fn add_range_checks(&self, required_range_checks: &mut Vec<Range>) {
        required_range_checks.push((-128, 128));
        let dual_gap_tolerance = (0.1 * 3.0 * 2f64.powf(8.0)) as i64;
        required_range_checks.push((-dual_gap_tolerance, dual_gap_tolerance));
        let dual_feasible_tolerance = ((0.01 * 3.0) * 2f64.powf(16.0)).ceil() as i64;
        required_range_checks.push((-dual_feasible_tolerance, dual_feasible_tolerance));
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
    // taken from lime repo
    pub fn kernel_width(d: usize) -> f32 {
        (d as f32).sqrt() * 0.75
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

        println!("GOT PERTS: {:?}", perturbations);

        let mut res = surrogate.to_vec();
        res.extend(perturbations);
        Ok(res)
    }

    // TODO: rename -> gen_witness
    pub fn run(&self, inputs: &mut [Tensor<Fp>], model: &Model, run_args: &RunArgs) -> LimeWitness {
        let d = inputs[0].dims()[1];
        assert_eq!(self.d, d);
        println!("d: {}", d);
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

                println!("OUTPUT: {:?}", output);
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
            println!("GOT STD: {:?}", std);

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
                8,
                0,
                8,
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

        println!("GOT SAMPLES: {:?}", lime_samples);

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
        //self.generate_samples(layouter)?;
        //
        //println!("SAMPLES: {}", samples.show());
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
        //println!("lime perts: {:?}", lime_perturbations.show());
        let mut result = x.concat(lime_perturbations).unwrap();
        result = result.concat(input_points).unwrap();
        result.reshape(&[self.num_points() + 1, d]).unwrap();
        Ok(result)
        //unimplemented!();
    }

    pub fn layout_exp_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        // expand x
        let d = x_border.dims()[0];
        println!("X_BORDER IS: {:?}", x_border);
        let mut x_expanded = x_border.clone();
        x_expanded.reshape(&[1, d]);
        x_expanded.expand(&[self.num_points, d]).unwrap();
        let deltas = pairwise(
            config,
            region,
            &[x_expanded.clone(), inputs.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        println!("DELTAS IS: {:?}", x_border);
        let square_distance = einsum(
            config,
            region,
            &[deltas.clone(), deltas.clone()],
            "ij,ij->i",
        )
        .unwrap();
        let square_distance = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_distance.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("SQUARE DISTANCE: {:?}", square_distance.show());

        // lookup table
        let weights = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[square_distance.clone()],
            &crate::circuit::ops::lookup::LookupOp::LimeWeight {
                input_scale: 2f64.powf(8.0).into(),
                sigma: Self::kernel_width(d).into(),
            },
        )
        .unwrap();
        //println!("WEIGHTS: {:?}", weights.show());

        weights
    }

    pub fn layout_dist_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        // expand x
        let d = x_border.dims()[0];
        println!("X_BORDER IS: {:?}", x_border);
        let mut x_expanded = x_border.clone();
        x_expanded.reshape(&[1, d]);
        x_expanded.expand(&[self.num_points, d]).unwrap();
        let deltas = pairwise(
            config,
            region,
            &[x_expanded.clone(), inputs.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        println!("DELTAS IS: {:?}", x_border);
        let square_distance = einsum(
            config,
            region,
            &[deltas.clone(), deltas.clone()],
            "ij,ij->i",
        )
        .unwrap();
        let square_distance = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_distance.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();

        square_distance
    }

    pub fn layout_lime_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        match self.weight_strategy {
            LimeWeightStrategy::Exponential => {
                self.layout_exp_weights(config, region, x_border, inputs)
            }
            LimeWeightStrategy::Distance => {
                self.layout_dist_weights(config, region, x_border, inputs)
            }
            // handled outside TODO(Evan): make this more uniform...
            LimeWeightStrategy::Uniform => unimplemented!(),
        }
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
        // ok...
        //
        // STEPS:
        // 1. compute weights
        //   a. compute square distance between x and points
        //   b. pass into lookup table (exp(_ / sigma^2))
        // 2. multiply points and labels by sqrt of weights

        log::debug!("LIME LAYOUT");
        region.debug_report();
        let d = x_border.dims()[0];
        let inputs = inputs.get_slice(&[1..self.num_points + 1, 0..d]).unwrap();
        let outputs = outputs.get_slice(&[1..self.num_points + 1]).unwrap();

        let (inputs, outputs) = if !matches!(self.weight_strategy, LimeWeightStrategy::Uniform) {
            println!("HI THERE");
            let weights = self.layout_lime_weights(config, region, x_border, &inputs);

            // sqrt weights
            let mut sqrt_weights = crate::circuit::ops::layouts::nonlinearity(
                config,
                region,
                &[weights],
                &crate::circuit::ops::lookup::LookupOp::Sqrt {
                    scale: F32(2f32.powf(8.0)),
                },
            )
            .unwrap();
            //println!("SQRT WEIGHTS: {:?}", sqrt_weights.show());

            // multiply inputs and outputs by sqrt weights
            // see: https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L116
            let mut input_sqrt_weights = sqrt_weights.clone();
            input_sqrt_weights.expand(&[self.num_points, d]);
            let inputs = pairwise(
                config,
                region,
                &[input_sqrt_weights.clone(), inputs],
                BaseOp::Mult,
            )
            .unwrap();
            //println!("GOT INPUTS: {:?}", inputs.pshow(16));

            let mut output_sqrt_weights = sqrt_weights.clone();
            output_sqrt_weights.expand(&[self.num_points]);
            let outputs = pairwise(
                config,
                region,
                &[output_sqrt_weights.clone(), outputs],
                BaseOp::Mult,
            )
            .unwrap();
            (inputs, outputs)
        } else {
            (inputs, outputs)
        };

        // residuals part
        let deltas = einsum(
            config,
            region,
            &[inputs.clone(), lime_model.clone()],
            "ij,j->ik",
        )
        .unwrap();
        let deltas = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[deltas.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("deltas: {:?}", deltas.pshow(16));

        // rescale intercept to 16 bits
        let mut lime_intercept = lime_intercept.clone();
        lime_intercept.expand(&[self.num_points]).unwrap();
        let intermediate = pairwise(
            config,
            region,
            &[outputs.clone(), lime_intercept],
            BaseOp::Sub,
        )
        .unwrap();
        //println!("y - int: {:?}", intermediate.pshow(8));

        // scale to 16 bits
        let multiplier = 1f64;
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 8).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let intermediate = pairwise(
            config,
            region,
            &[intermediate.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();

        let intermediate = pairwise(config, region, &[intermediate, deltas], BaseOp::Sub).unwrap();
        //println!("intermediate: {:?}", intermediate.pshow(16));
        let mut square_residuals = dot(
            config,
            region,
            &[intermediate.clone(), intermediate.clone()],
        )
        .unwrap();

        // scale back down to 16 bits...
        for i in 0..2 {
            square_residuals = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_residuals.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }

        //let square_residuals = crate::circuit::ops::layouts::loop_div(
        //    config,
        //    region,
        //    &[square_residuals.clone()],
        //    Fp::from(2u64.pow(8)),
        //)
        //.unwrap();
        //println!("square_residuals: {:?}", square_residuals.pshow(16));

        let multiplier = 1f64 / (2f64 * self.num_points as f64);
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 16).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let mut square_residuals_dived = pairwise(
            config,
            region,
            &[square_residuals.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();
        //println!(
        //    "square_residuals_dived: {:?}",
        //    square_residuals_dived.pshow(32)
        //);
        // scale back down to 8 bits by 2**8 division...there might be a better / more efficient
        // way to do this
        for i in 0..3 {
            square_residuals_dived = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_residuals_dived.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!(
        //    "square_residuals_dived: {:?}",
        //    square_residuals_dived.pshow(8)
        //);

        let alpha = 0.01f64;
        let alpha = i64_to_felt(quantize_float(&alpha, 0.0, 8).unwrap());
        let alpha = create_constant_tensor(alpha, 1);
        // TODO: fix alpha ....
        let l1_part = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[lime_model.clone()],
            &crate::circuit::ops::lookup::LookupOp::Abs,
        )
        .unwrap();
        // compute sum...
        let l1_part = sum(config, region, &[l1_part]).unwrap();
        let mut l1_part =
            pairwise(config, region, &[alpha.clone(), l1_part], BaseOp::Mult).unwrap();
        let mut l1_part = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[l1_part.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("l1 part: {:?}", l1_part.pshow(8));

        let mut primal_objective = pairwise(
            config,
            region,
            &[square_residuals_dived.clone(), l1_part],
            BaseOp::Add,
        )
        .unwrap();

        //println!("primal objective: {}", primal_objective.pshow(8));

        // compute dual objective
        let mut square_dual = dot(config, region, &[dual.clone(), dual.clone()]).unwrap();
        //let square_residuals = crate::circuit::ops::layouts::loop_div(
        //    config,
        //    region,
        //    &[square_residuals.clone()],
        //    Fp::from(2u64.pow(8)),
        //)
        //.unwrap();
        for i in 0..2 {
            square_dual = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_dual.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("square dual: {:?}", square_dual.pshow(16));

        let multiplier = -1f64 * self.num_points as f64 / (2f64);
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 16).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let mut square_dual_dived = pairwise(
            config,
            region,
            &[square_dual.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();
        //println!("square_dual_dived: {:?}", square_dual_dived.pshow(32));
        // scale back down to 8 bits by 2**8 division...there might be a better / more efficient
        // way to do this
        for i in 0..3 {
            square_dual_dived = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_dual_dived.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("square_dual_dived: {:?}", square_dual_dived.pshow(8));

        let mut dual_res = dot(config, region, &[dual.clone(), outputs]).unwrap();
        //println!("dual_dot: {:?}", dual_res.pshow(24));
        for i in 0..2 {
            dual_res = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[dual_res.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("dual_dot: {:?}", dual_res.pshow(8));
        let dual_res = pairwise(
            config,
            region,
            &[dual_res.clone(), square_dual_dived.clone()],
            BaseOp::Add,
        )
        .unwrap();
        //println!("dual_res: {:?}", dual_res.pshow(8));

        let dual_objective = dual_res;
        let dual_gap = pairwise(
            config,
            region,
            &[primal_objective.clone(), dual_objective.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        println!("dual gap: {:?}", dual_gap.pshow(8));

        // HUGE TODO(EVAN): ensure dual is feasible.........

        // ensure within 0.1?
        let range_check_bracket = (0.1 * 3.0 * 2f64.powf(8.0)) as i64;
        range_check(
            config,
            region,
            &[dual_gap],
            &(-range_check_bracket, range_check_bracket),
        )
        .unwrap();

        // check dual is feasible
        let mut dual_feasible =
            einsum(config, region, &[inputs.clone(), dual.clone()], "ji,j->i").unwrap();
        println!("dual feasible: {:?}", dual_feasible.pshow(32));
        for i in 0..2 {
            dual_feasible = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[dual_feasible.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        // add some slack due to quantization error...
        let range_check_bracket = ((0.01 * 3.0) * 2f64.powf(16.0)).ceil() as i64;
        let range_check_felt: Fp = i64_to_felt(range_check_bracket);
        //println!("GOT {:?}", range_check_felt);
        //println!("bracket: {:?}", dual_feasible);
        range_check(
            config,
            region,
            &[dual_feasible],
            &(-range_check_bracket, range_check_bracket),
        )
        .unwrap();

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

// Generate rng ahead of time ... makes things easier...
#[derive(Debug)]
pub struct Lime2Chip {
    sample_chip: SampleChip<8>,
    perturb_chip: PerturbChip<Fp>,

    samples: Option<Vec<AssignedCell<Fp, Fp>>>,

    // ok...
    pub input_picker: Box<dyn LimeInputConfig>,

    pub n_ball: usize,
    pub n_lime: usize,
    d: usize,
}

impl Lime2Chip {
    const LIME_MULTIPLIER: u64 = 4;

    //pub fn input_size(n_lime: usize, n_ball: usize) -> usize {
    //    if (crate::USE_SURROGATE) {
    //        n_lime + n_ball + 2
    //    } else {
    //        n_lime + 1
    //    }
    //}
    pub fn sample_size(n_lime: usize, n_input: usize, d: usize) -> usize {
        n_lime * d + n_input
    }

    // taken from lime repo
    pub fn kernel_width(d: usize) -> f32 {
        (d as f32).sqrt() * 0.75
    }

    pub fn configure_nonlinearities() {
        unimplemented!();
    }

    pub fn configure_range_checks() {
        unimplemented!();
    }

    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        n_lime: usize,
        input_picker: Box<dyn LimeInputConfig>,
        d: usize,
    ) -> Self {
        //println!("CONFIG: {} {} {}", n_lime, n_ball, d);
        let sample_chip = SampleChip::configure(
            meta,
            Self::sample_size(n_lime, input_picker.num_samples(), d),
        );
        let perturb_chip = PerturbChip::configure(meta, Some(256));

        Self {
            sample_chip,
            perturb_chip,

            input_picker,

            samples: None,
            n_lime,
            n_ball: 0,
            d,
        }
    }

    pub fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: ValTensor<Fp>,
    ) -> Result<Vec<AssignedCell<Fp, Fp>>, ModuleError> {
        unimplemented!();
    }

    /*
     * Takes in:
     * x: original point
     * y: lime point
     * n_lime: number of lime samples
     * TODO: how to handle run for input picker...
     */
    pub fn run(
        x: Vec<Fp>,
        x_border: Vec<Fp>,
        n_lime: usize,
        n_ball: usize,
        d: usize,
    ) -> Result<Vec<Fp>, ModuleError> {
        use crate::tensor::ops::*;

        let samples = SampleChip::<8>::run(Self::sample_size(n_lime, n_ball, d))?;
        //println!("samples: {:?}", samples);

        // if no surrogate, then just normal perturbations...
        let lime_samples = &samples[0..n_lime * d];

        // TODO(EVAN): could clean this up
        let res = if (crate::USE_SURROGATE) {
            let ball_samples = &samples[n_lime * d..];
            let multiplier = Fp::from(Self::LIME_MULTIPLIER);
            let perturbations = lime_samples
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    multiplier * v + x_border[i % d] - Fp::from(Self::LIME_MULTIPLIER * 128)
                })
                .collect::<Vec<_>>();
            //println!("perturbations: {:?}", perturbations);

            //println!("x: {:?}", x);
            //println!("x_border: {:?}", x_border);
            let mut radius = Fp::from(0);
            let mut vals = vec![];
            for i in 0..d {
                let val = x[i] - x_border[i];
                vals.push(val);
                radius += val * val;
            }
            //println!("diffs: {:?}", vals);
            //println!("square RADIUS: {:?}", radius);
            let radius = (felt_to_i64(radius) as f64 / 2f64.powf(8.0)).round() as i64;
            //println!("scaled RADIUS: {:?}", radius);
            let mut radius = crate::tensor::ops::nonlinearities::sqrt(
                &Tensor::new(Some(&[radius]), &[1]).unwrap(),
                2f64.powf(8.0),
            );
            //println!("RUN RADIUS: {:?}", radius);

            let ball_tensor = Tensor::new(Some(ball_samples), &[n_ball, d + 2]).unwrap();
            let ball_tensor = ball_tensor.clone().map(|x| felt_to_i64(x));
            let ball_samples_normal = crate::tensor::ops::nonlinearities::normal_inverse_cdf(
                &ball_tensor,
                2f64.powf(8.0),
                0.0,
                1.0,
            );
            //println!("ball_samples_normal: {:?}", ball_samples_normal);
            //println!(
            //    "ball_samples_normal float: {:?}",
            //    ball_samples_normal
            //        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
            //        .unwrap()
            //);
            let mut square_norms = vec![];
            for i in 0..n_ball {
                let mut norm = 0;

                for j in 0..d + 2 {
                    let val = ball_samples_normal[i * (d + 2) + j];
                    norm += val * val;
                }
                square_norms.push(((norm as f64) / 2f64.powf(8.0)).round() as i64);
            }
            //println!("square norms scaled!: {:?}", square_norms);
            // scale down to 8 bits...
            let recip_norms = crate::tensor::ops::nonlinearities::recip_sqrt(
                &square_norms.iter().cloned().into(),
                2f64.powf(8.0),
                2f64.powf(8.0),
            );
            //println!("norms!: {:?}", recip_norms.show());

            let mut normalized = vec![];
            for i in 0..(n_ball) * (d + 2) {
                let val = (ball_samples_normal[i] * recip_norms[(i / (d + 2))]);
                let val_scaled = (val as f64 / 2f64.powf(8.0)).round() as i64;
                normalized.push(val_scaled);
            }
            //println!("normalized: {:?}", normalized);

            let normalized = Tensor::new(Some(&normalized), &[n_ball, d + 2]).unwrap();
            //println!(
            //    "normalized float: {:?}",
            //    normalized
            //        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
            //        .unwrap()
            //);
            let sphere_samples = normalized.get_slice(&[0..n_ball, 0..d]).unwrap();
            //println!("sphere_samples_scaled: {:?}", sphere_samples.show());
            //println!(
            //    "sphere_samples_scaled float: {:?}",
            //    sphere_samples
            //        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
            //        .unwrap()
            //);
            //let ball_norms =

            // scale to radius...
            let radius_expanded = radius.expand(&[n_ball, d]).unwrap();
            let sphere_samples_radius = mult(&[sphere_samples.clone(), radius_expanded])?;
            //println!("sphere_samples_radius: {:?}", sphere_samples_radius.show());
            let sphere_samples_radius_scaled = sphere_samples_radius
                .enum_map::<_, _, ModuleError>(|i, v| {
                    Ok((v as f64 / 2f64.powf(8.0)).round() as i64)
                })?;
            //println!(
            //    "sphere_samples_radius_scaled: {:?}",
            //    sphere_samples_radius_scaled.show()
            //);
            //println!(
            //    "sphere_samples_radius_scaled float: {:?}",
            //    sphere_samples_radius_scaled
            //        .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
            //        .unwrap()
            //);

            //println!("X: {:?}", x);
            let mut perturbations2 = sphere_samples_radius_scaled
                .enum_map::<_, _, ModuleError>(|i, v| Ok(x[i % d] + i64_to_felt::<Fp>(v)))
                .unwrap();
            //println!("perturbations2: {:?}", perturbations2.show());

            // print out the floating point...
            let x_test = Tensor::new(Some(&x), &[d]).unwrap();
            //println!("x: {:?}", x_test.dequantize(8));
            //println!("perturbations2: {:?}", perturbations2.dequantize(8));

            // Must pass through normal etc...
            // FIXME(EVAN)
            //perturbations.extend(perturbations2);

            let mut res = x.clone();
            res.extend(perturbations);
            res.extend(x_border);
            res.extend(perturbations2);
            res
        } else {
            let multiplier = Fp::from(Self::LIME_MULTIPLIER);
            let perturbations = lime_samples
                .iter()
                .enumerate()
                .map(|(i, v)| multiplier * v + x[i % d] - Fp::from(Self::LIME_MULTIPLIER * 128))
                .collect::<Vec<_>>();
            //println!("perturbations: {:?}", perturbations);
            let mut res = x.clone();
            res.extend(perturbations);
            res
        };

        //println!("RES: {:?}", res);
        Ok(res)
    }

    pub fn layout_samples(
        &self,
        layouter: &mut impl Layouter<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let samples = self.sample_chip.layout(layouter)?;
        let samples = samples
            .iter()
            .map(|v| ValType::PrevAssigned(v.clone()))
            .collect::<Vec<ValType<_>>>();
        Ok(samples.into())
    }

    pub fn run_samples(&self) -> Result<ValTensor<Fp>, ModuleError> {
        let samples =
            SampleChip::<8>::run((self.n_lime * self.d) + self.input_picker.num_samples())?;
        let samples = samples
            .iter()
            .map(|v| ValType::Value(Value::known(v.clone())))
            .collect::<Vec<ValType<_>>>();
        Ok(samples.into())
    }

    pub fn layout_lime_samples(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        lime_samples: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        unimplemented!();
    }

    pub fn layout_ball_radius(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        x_border: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        //println!("x: {:?}", x.show());
        //println!("x_border: {:?}", x_border.show());
        let difference =
            pairwise(config, region, &[x.clone(), x_border.clone()], BaseOp::Sub).unwrap();
        //println!("diffs!: {:?}", difference.show());
        let square_norms = dot(config, region, &[difference.clone(), difference.clone()])?;
        //println!("square diffs!: {:?}", square_norms.show());
        // scale back down to 8 bits..
        let square_norms_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_norms.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("square_diffs_scaled: {:?}", square_norms_scaled.show());

        // sqrt to get norm...
        let norm = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[square_norms_scaled],
            &crate::circuit::ops::lookup::LookupOp::Sqrt {
                scale: F32(2f32.powf(8.0)),
            },
        )
        .unwrap();
        //println!("radius: {:?}", norm.show());

        Ok(norm)
    }

    /*pub fn layout_ball_samples(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        ball_samples: &ValTensor<Fp>,
        radius: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let d = x.dims()[0];

        let ball_samples_normal = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[ball_samples.clone()],
            &crate::circuit::ops::lookup::LookupOp::Norm {
                scale: F32(2f32.powf(8.0)),
                mean: F32(0f32),
                std: F32(1f32),
            },
        )
        .unwrap();
        //println!("BALL SAMPLES!: {:?}", ball_samples_normal.show());

        // compute
        let square_norms = einsum(
            config,
            region,
            &[ball_samples_normal.clone(), ball_samples_normal.clone()],
            "ij,ij->ik",
        )?;
        //println!("square norms!: {:?}", square_norms.show());
        // scale back down to 8 bits..
        let square_norms_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_norms.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("square_norms_scaled: {:?}", square_norms_scaled.show());
        // scale down to 8 bits...
        let recip_norms = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[square_norms_scaled],
            &crate::circuit::ops::lookup::LookupOp::RecipSqrt {
                input_scale: F32(2f32.powf(8.0)),
                output_scale: F32(2f32.powf(8.0)),
            },
        )
        .unwrap();
        //println!("norms!: {:?}", recip_norms.show());
        // multiply by recips...
        let normalized = einsum(
            config,
            region,
            &[ball_samples_normal.clone(), recip_norms],
            "ij,ik->ij",
        )?;
        //println!("normalized: {:?}", normalized.show());

        // TODO: rescale by distance to point....
        let sphere_samples = normalized.get_slice(&[0..self.n_ball, 0..d]).unwrap();
        //println!("sphere samples: {:?}", sphere_samples.show());
        // scale back down to 8 bits..
        let sphere_samples_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[sphere_samples.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("sphere_samples_scaled: {:?}", sphere_samples_scaled.show());

        // scale sphere_samples to radius
        let mut radius_expanded = radius.clone();
        radius_expanded.expand(&[self.n_ball, d]).unwrap();
        let sphere_samples_radius = pairwise(
            config,
            region,
            &[sphere_samples_scaled.clone(), radius_expanded],
            BaseOp::Mult,
        )
        .unwrap();
        //println!("sphere_samples_radius: {:?}", sphere_samples_radius.show());
        // scale back down to 8 bits..
        let sphere_samples_radius_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[sphere_samples_radius.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!(
        //    "sphere_samples_radius_scaled: {:?}",
        //    sphere_samples_radius_scaled.show()
        //);

        let mut x_expanded = x.clone();
        x_expanded.reshape(&[1, d]);
        x_expanded.expand(&[self.n_ball, d]).unwrap();

        let ball_perturbations = pairwise(
            config,
            region,
            &[x_expanded.clone(), sphere_samples_radius_scaled.clone()],
            BaseOp::Add,
        )
        .unwrap();
        //println!("X: {:?}", x_expanded.show());
        //println!("perturbations2: {:?}", ball_perturbations.show());

        Ok(ball_perturbations)
    }*/

    // layout all Lime operations...
    /*pub fn layout(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        x_border: &Option<ValTensor<Fp>>,
        samples: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        //self.generate_samples(layouter)?;
        //
        //println!("SAMPLES: {}", samples.show());
        let d = x.dims()[0];
        assert_eq!(x.dims(), &[d]);
        //assert_eq!(x_border.dims(), &[d]);

        //let lime_samples = self.layout_perturb_uniform(layouter, x_border)?;
        //assert_eq!(
        //    samples.dims(),
        //    &[(self.n_lime * d) + (self.n_ball * (d + 2))]
        //);

        let result = if (crate::USE_SURROGATE) {
            let x_border = x_border.clone().unwrap();
            let mut lime_samples = samples.get_slice(&[0..(self.n_lime * d)]).unwrap();
            lime_samples.reshape(&[self.n_lime, d]).unwrap();
            let mut ball_samples = samples
                .get_slice(&[(self.n_lime * d)..samples.dims()[0]])
                .unwrap();
            ball_samples.reshape(&[self.n_ball, d + 2]).unwrap();

            let lime_perturbations =
                self.layout_lime_samples(config, region, &x_border, &lime_samples);
            let radius = self.layout_ball_radius(config, region, x, &x_border)?;
            let ball_perturbations =
                self.layout_ball_samples(config, region, x, &ball_samples, &radius)?;

            // concat all the points together...
            let result = x.concat(lime_perturbations).unwrap();
            let result = result.concat(x_border.clone()).unwrap();
            let mut result = result.concat(ball_perturbations).unwrap();
            result
                .reshape(&[(self.n_lime + self.n_ball + 2), d])
                .unwrap();
            result
        } else {
            let mut lime_samples = samples.get_slice(&[0..(self.n_lime * d)]).unwrap();
            lime_samples.reshape(&[self.n_lime, d]).unwrap();
            let lime_perturbations = self.layout_lime_samples(config, region, x, &lime_samples);
            //println!("lime perts: {:?}", lime_perturbations.show());
            let mut result = x.concat(lime_perturbations).unwrap();
            result
                .reshape(&[Lime2Chip::input_size(self.n_lime, self.n_ball), d])
                .unwrap();
            result
        };
        //println!("RES: {:?}", result.show());
        Ok(result)
        //unimplemented!();
    }*/

    pub fn layout_ball_checks(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        outputs: &ValTensor<Fp>,
    ) {
        // get ball outputs
        let len = outputs.dims()[0];
        let mut x_output = outputs.get_single_elem(0).unwrap();
        x_output.expand(&[self.n_ball]);
        let ball_outputs = outputs.get_slice(&[self.n_lime + 2..len]).unwrap();

        enforce_equality(config, region, &[x_output, ball_outputs]);
    }

    pub fn layout_lime_weights(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x_border: &ValTensor<Fp>,
        inputs: &ValTensor<Fp>,
    ) -> ValTensor<Fp> {
        // expand x
        let d = x_border.dims()[0];
        let mut x_expanded = x_border.clone();
        x_expanded.reshape(&[1, d]);
        x_expanded.expand(&[self.n_lime, d]).unwrap();
        let deltas = pairwise(
            config,
            region,
            &[x_expanded.clone(), inputs.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        let square_distance = einsum(
            config,
            region,
            &[deltas.clone(), deltas.clone()],
            "ij,ij->i",
        )
        .unwrap();
        let square_distance = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_distance.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("SQUARE DISTANCE: {:?}", square_distance.show());

        // lookup table
        let weights = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[square_distance.clone()],
            &crate::circuit::ops::lookup::LookupOp::LimeWeight {
                input_scale: 2f64.powf(8.0).into(),
                sigma: Self::kernel_width(d).into(),
            },
        )
        .unwrap();
        //println!("WEIGHTS: {:?}", weights.show());

        weights
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
        // ok...
        //
        // STEPS:
        // 1. compute weights
        //   a. compute square distance between x and points
        //   b. pass into lookup table (exp(_ / sigma^2))
        // 2. multiply points and labels by sqrt of weights

        log::debug!("LIME LAYOUT");
        region.debug_report();
        let d = x_border.dims()[0];
        let inputs = inputs.get_slice(&[1..self.n_lime + 1, 0..d]).unwrap();
        let outputs = outputs.get_slice(&[1..self.n_lime + 1]).unwrap();

        let weights = self.layout_lime_weights(config, region, x_border, &inputs);

        // sqrt weights
        let mut sqrt_weights = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[weights],
            &crate::circuit::ops::lookup::LookupOp::Sqrt {
                scale: F32(2f32.powf(8.0)),
            },
        )
        .unwrap();
        //println!("SQRT WEIGHTS: {:?}", sqrt_weights.show());

        // multiply inputs and outputs by sqrt weights
        // see: https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_base.py#L116
        let mut input_sqrt_weights = sqrt_weights.clone();
        input_sqrt_weights.expand(&[self.n_lime, d]);
        let inputs = pairwise(
            config,
            region,
            &[input_sqrt_weights.clone(), inputs],
            BaseOp::Mult,
        )
        .unwrap();
        //println!("GOT INPUTS: {:?}", inputs.pshow(16));

        let mut output_sqrt_weights = sqrt_weights.clone();
        output_sqrt_weights.expand(&[self.n_lime]);
        let outputs = pairwise(
            config,
            region,
            &[output_sqrt_weights.clone(), outputs],
            BaseOp::Mult,
        )
        .unwrap();
        //println!("GOT OUTPUTS: {:?}", outputs.pshow(8));

        // residuals part
        let deltas = einsum(
            config,
            region,
            &[inputs.clone(), lime_model.clone()],
            "ij,j->ik",
        )
        .unwrap();
        let deltas = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[deltas.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("deltas: {:?}", deltas.pshow(16));

        // rescale intercept to 16 bits
        let mut lime_intercept = lime_intercept.clone();
        lime_intercept.expand(&[self.n_lime]).unwrap();
        let intermediate = pairwise(
            config,
            region,
            &[outputs.clone(), lime_intercept],
            BaseOp::Sub,
        )
        .unwrap();
        //println!("y - int: {:?}", intermediate.pshow(8));

        // scale to 16 bits
        let multiplier = 1f64;
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 8).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let intermediate = pairwise(
            config,
            region,
            &[intermediate.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();

        let intermediate = pairwise(config, region, &[intermediate, deltas], BaseOp::Sub).unwrap();
        //println!("intermediate: {:?}", intermediate.pshow(16));
        let mut square_residuals = dot(
            config,
            region,
            &[intermediate.clone(), intermediate.clone()],
        )
        .unwrap();

        // scale back down to 16 bits...
        for i in 0..2 {
            square_residuals = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_residuals.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }

        //let square_residuals = crate::circuit::ops::layouts::loop_div(
        //    config,
        //    region,
        //    &[square_residuals.clone()],
        //    Fp::from(2u64.pow(8)),
        //)
        //.unwrap();
        //println!("square_residuals: {:?}", square_residuals.pshow(16));

        let multiplier = 1f64 / (2f64 * self.n_lime as f64);
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 16).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let mut square_residuals_dived = pairwise(
            config,
            region,
            &[square_residuals.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();
        //println!(
        //    "square_residuals_dived: {:?}",
        //    square_residuals_dived.pshow(32)
        //);
        // scale back down to 8 bits by 2**8 division...there might be a better / more efficient
        // way to do this
        for i in 0..3 {
            square_residuals_dived = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_residuals_dived.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!(
        //    "square_residuals_dived: {:?}",
        //    square_residuals_dived.pshow(8)
        //);

        let alpha = 0.01f64;
        let alpha = i64_to_felt(quantize_float(&alpha, 0.0, 8).unwrap());
        let alpha = create_constant_tensor(alpha, 1);
        // TODO: fix alpha ....
        let l1_part = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[lime_model.clone()],
            &crate::circuit::ops::lookup::LookupOp::Abs,
        )
        .unwrap();
        // compute sum...
        let l1_part = sum(config, region, &[l1_part]).unwrap();
        let mut l1_part =
            pairwise(config, region, &[alpha.clone(), l1_part], BaseOp::Mult).unwrap();
        let mut l1_part = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[l1_part.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        //println!("l1 part: {:?}", l1_part.pshow(8));

        let mut primal_objective = pairwise(
            config,
            region,
            &[square_residuals_dived.clone(), l1_part],
            BaseOp::Add,
        )
        .unwrap();

        //println!("primal objective: {}", primal_objective.pshow(8));

        // compute dual objective
        let mut square_dual = dot(config, region, &[dual.clone(), dual.clone()]).unwrap();
        //let square_residuals = crate::circuit::ops::layouts::loop_div(
        //    config,
        //    region,
        //    &[square_residuals.clone()],
        //    Fp::from(2u64.pow(8)),
        //)
        //.unwrap();
        for i in 0..2 {
            square_dual = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_dual.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("square dual: {:?}", square_dual.pshow(16));

        let multiplier = -1f64 * self.n_lime as f64 / (2f64);
        let multiplier = i64_to_felt(quantize_float(&multiplier, 0.0, 16).unwrap());
        let multiplier = create_constant_tensor(multiplier, 1);
        let mut square_dual_dived = pairwise(
            config,
            region,
            &[square_dual.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();
        //println!("square_dual_dived: {:?}", square_dual_dived.pshow(32));
        // scale back down to 8 bits by 2**8 division...there might be a better / more efficient
        // way to do this
        for i in 0..3 {
            square_dual_dived = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[square_dual_dived.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("square_dual_dived: {:?}", square_dual_dived.pshow(8));

        let mut dual_res = dot(config, region, &[dual.clone(), outputs]).unwrap();
        //println!("dual_dot: {:?}", dual_res.pshow(24));
        for i in 0..2 {
            dual_res = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[dual_res.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        //println!("dual_dot: {:?}", dual_res.pshow(8));
        let dual_res = pairwise(
            config,
            region,
            &[dual_res.clone(), square_dual_dived.clone()],
            BaseOp::Add,
        )
        .unwrap();
        //println!("dual_res: {:?}", dual_res.pshow(8));

        let dual_objective = dual_res;
        let dual_gap = pairwise(
            config,
            region,
            &[primal_objective.clone(), dual_objective.clone()],
            BaseOp::Sub,
        )
        .unwrap();
        println!("dual gap: {:?}", dual_gap.pshow(8));

        // HUGE TODO(EVAN): ensure dual is feasible.........

        // ensure within 0.1?
        let range_check_bracket = (0.1 * 2f64.powf(8.0)) as i64;
        range_check(
            config,
            region,
            &[dual_gap],
            &(-range_check_bracket, range_check_bracket),
        )
        .unwrap();

        // check dual is feasible
        let mut dual_feasible =
            einsum(config, region, &[inputs.clone(), dual.clone()], "ji,j->i").unwrap();
        //println!("dual feasible: {:?}", dual_feasible.pshow(32));
        for i in 0..2 {
            dual_feasible = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[dual_feasible.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        println!("dual feasible: {:?}", dual_feasible.pshow(16));
        // add some slack due to quantization error...
        let range_check_bracket = ((0.01 * 1.5) * 2f64.powf(16.0)).ceil() as i64;
        let range_check_felt: Fp = i64_to_felt(range_check_bracket);
        //println!("GOT {:?}", range_check_felt);
        //println!("bracket: {:?}", dual_feasible);
        range_check(
            config,
            region,
            &[dual_feasible],
            &(-range_check_bracket, range_check_bracket),
        )
        .unwrap();

        region.debug_report();
    }

    // TOOD(Evan): Top k function exists in tensor ops...
    pub fn layout_top_k_checks(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        lime_model: &ValTensor<Fp>,
        lime_model_topk: &ValTensor<Fp>,
        lime_model_topk_idxs: &ValTensor<Fp>,
        k: usize,
    ) {
        //
        //
        // test gather...
        let lime_model_abs = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[lime_model.clone()],
            &crate::circuit::ops::lookup::LookupOp::Abs,
        )
        .unwrap();
        let top_k_abs = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[lime_model_topk.clone()],
            &crate::circuit::ops::lookup::LookupOp::Abs,
        )
        .unwrap();
        let sorted = _sort_ascending(config, region, &[lime_model_abs.clone()]).unwrap();
        //println!("GOT model: {:?}", lime_model_abs.pshow(8));
        //println!("GOT SORTED: {:?}", sorted.pshow(8));

        // ensure topk is really topk
        let checking_topk = sorted.get_slice(&[sorted.len() - k..sorted.len()]).unwrap();
        //println!("topk check: {:?}", checking_topk.pshow(8));
        enforce_equality(config, region, &[top_k_abs.clone(), checking_topk.clone()]);

        //println!("model: {:?}", lime_model.dims());
        //println!("topk_idx: {:?}", lime_model_topk_idxs.dims());
        // ensure idxs really idxs
        let topk_from_idx = gather(
            config,
            region,
            &[lime_model.clone(), lime_model_topk_idxs.clone()],
            0,
        )
        .unwrap();
        //println!("TOPK_IDXS: {:?}", topk_from_idx.pshow(8));
        enforce_equality(config, region, &[lime_model_topk.clone(), topk_from_idx]);
    }

    pub fn layout_perturb_ball(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        // Input should just be values...copy to assigned cells
        let values = match input {
            ValTensor::Value { inner: v, .. } => v
                .iter()
                .map(|v| match v {
                    ValType::Value(v) => Ok(v.clone()),
                    _ => {
                        return Err(ModuleError::WrongInputType(
                            "Input".to_string(),
                            "Tensor of values".to_string(),
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>(),
            _ => Err(ModuleError::WrongInputType(
                "Input".to_string(),
                "Tensor of values".to_string(),
            )),
        }?;

        //println!("got values: {:?}", values);

        let samples = self.sample_chip.layout(layouter)?;
        //println!("got samples: {:?}", samples.len());
        // pass values into perturbchip

        let perturbations = self
            .perturb_chip
            .layout(layouter, values.clone(), samples)?;
        //println!("got perts: {:?}", perturbations);

        let pert_values = perturbations
            .iter()
            .map(|v| ValType::PrevAssigned(v.clone()))
            .collect::<Vec<ValType<_>>>();
        let input_values = values
            .iter()
            .map(|v| ValType::Value(v.clone()))
            .collect::<Vec<_>>();
        //.into()

        // convert perturbations back into tensor
        Ok(vec![input_values, pert_values].into_iter().concat().into())
    }

    // lets just do this first...
    pub fn layout_perturb_uniform(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        // Input should just be values...copy to assigned cells
        let values = match input {
            ValTensor::Value { inner: v, .. } => v
                .iter()
                .map(|v| match v {
                    ValType::Value(v) => Ok(v.clone()),
                    _ => {
                        return Err(ModuleError::WrongInputType(
                            "Input".to_string(),
                            "Tensor of values".to_string(),
                        ))
                    }
                })
                .collect::<Result<Vec<_>, _>>(),
            _ => Err(ModuleError::WrongInputType(
                "Input".to_string(),
                "Tensor of values".to_string(),
            )),
        }?;

        //println!("got values: {:?}", values);

        let samples = self.sample_chip.layout(layouter)?;
        //println!("got samples: {:?}", samples.len());
        // pass values into perturbchip

        let perturbations = self
            .perturb_chip
            .layout(layouter, values.clone(), samples)?;
        //println!("got perts: {:?}", perturbations);

        let pert_values = perturbations
            .iter()
            .map(|v| ValType::PrevAssigned(v.clone()))
            .collect::<Vec<ValType<_>>>();
        let input_values = values
            .iter()
            .map(|v| ValType::Value(v.clone()))
            .collect::<Vec<_>>();
        //.into()

        // convert perturbations back into tensor
        Ok(vec![input_values, pert_values].into_iter().concat().into())
    }
}

#[cfg(test)]
mod test {

    fn test_lime_perturb() {
        //LimeChip {}
    }
}

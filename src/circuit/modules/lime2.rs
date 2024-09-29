/// OKKKKKKKKKKKES:
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use itertools::Itertools;

use super::errors::ModuleError;
use crate::tensor::val::{create_constant_tensor, create_unit_tensor};
use crate::tensor::{Tensor, ValTensor, ValType};

use crate::circuit::ops::layouts::*;
use crate::circuit::utils::F32;
use crate::circuit::{ops::base::BaseOp, utils};
use crate::fieldutils::{felt_to_i64, i64_to_felt};

use super::perturb::PerturbChip;
use super::sample::SampleChip;
use crate::circuit::ops::chip::BaseConfig;
use crate::circuit::ops::region::RegionCtx;

use crate::graph::utilities::{dequantize, quantize_float};
// helper to assign tensors to column of cells

// Generate rng ahead of time ... makes things easier...
#[derive(Debug, Clone)]
pub struct Lime2Chip {
    sample_chip: SampleChip<8>,
    perturb_chip: PerturbChip<Fp>,

    samples: Option<Vec<AssignedCell<Fp, Fp>>>,

    pub n_lime: usize,
    pub n_ball: usize,
    d: usize,
}

impl Lime2Chip {
    const LIME_MULTIPLIER: u64 = 4;

    pub fn input_size(n_lime: usize, n_ball: usize) -> usize {
        if (crate::USE_SURROGATE) {
            n_lime + n_ball + 2
        } else {
            n_lime + 1
        }
    }

    pub fn sample_size(n_lime: usize, n_ball: usize, d: usize) -> usize {
        if (crate::USE_SURROGATE) {
            (n_lime * d) + (n_ball * (d + 2))
        } else {
            n_lime * d
        }
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
        n_ball: usize,
        d: usize,
    ) -> Self {
        println!("CONFIG: {} {} {}", n_lime, n_ball, d);
        let sample_chip = SampleChip::configure(meta, Self::sample_size(n_lime, n_ball, d));
        let perturb_chip = PerturbChip::configure(meta, Some(256));

        Self {
            sample_chip,
            perturb_chip,

            samples: None,
            n_lime,
            n_ball,
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

    pub fn run(
        x: Vec<Fp>,
        x_border: Vec<Fp>,
        n_lime: usize,
        n_ball: usize,
        d: usize,
    ) -> Result<Vec<Fp>, ModuleError> {
        use crate::tensor::ops::*;

        let samples = SampleChip::<8>::run(Self::sample_size(n_lime, n_ball, d))?;
        println!("samples: {:?}", samples);

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
            println!("perturbations: {:?}", perturbations);

            println!("x: {:?}", x);
            println!("x_border: {:?}", x_border);
            let mut radius = Fp::from(0);
            let mut vals = vec![];
            for i in 0..d {
                let val = x[i] - x_border[i];
                vals.push(val);
                radius += val * val;
            }
            println!("diffs: {:?}", vals);
            println!("square RADIUS: {:?}", radius);
            let radius = (felt_to_i64(radius) as f64 / 2f64.powf(8.0)).round() as i64;
            println!("scaled RADIUS: {:?}", radius);
            let mut radius = crate::tensor::ops::nonlinearities::sqrt(
                &Tensor::new(Some(&[radius]), &[1]).unwrap(),
                2f64.powf(8.0),
            );
            println!("RUN RADIUS: {:?}", radius);

            let ball_tensor = Tensor::new(Some(ball_samples), &[n_ball, d + 2]).unwrap();
            let ball_tensor = ball_tensor.clone().map(|x| felt_to_i64(x));
            let ball_samples_normal = crate::tensor::ops::nonlinearities::normal_inverse_cdf(
                &ball_tensor,
                2f64.powf(8.0),
                0.0,
                1.0,
            );
            println!("ball_samples_normal: {:?}", ball_samples_normal);
            println!(
                "ball_samples_normal float: {:?}",
                ball_samples_normal
                    .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
                    .unwrap()
            );
            let mut square_norms = vec![];
            for i in 0..n_ball {
                let mut norm = 0;

                for j in 0..d + 2 {
                    let val = ball_samples_normal[i * (d + 2) + j];
                    norm += val * val;
                }
                square_norms.push(((norm as f64) / 2f64.powf(8.0)).round() as i64);
            }
            println!("square norms scaled!: {:?}", square_norms);
            // scale down to 8 bits...
            let recip_norms = crate::tensor::ops::nonlinearities::recip_sqrt(
                &square_norms.iter().cloned().into(),
                2f64.powf(8.0),
                2f64.powf(8.0),
            );
            println!("norms!: {:?}", recip_norms.show());

            let mut normalized = vec![];
            for i in 0..(n_ball) * (d + 2) {
                let val = (ball_samples_normal[i] * recip_norms[(i / (d + 2))]);
                let val_scaled = (val as f64 / 2f64.powf(8.0)).round() as i64;
                normalized.push(val_scaled);
            }
            println!("normalized: {:?}", normalized);

            let normalized = Tensor::new(Some(&normalized), &[n_ball, d + 2]).unwrap();
            println!(
                "normalized float: {:?}",
                normalized
                    .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
                    .unwrap()
            );
            let sphere_samples = normalized.get_slice(&[0..n_ball, 0..d]).unwrap();
            println!("sphere_samples_scaled: {:?}", sphere_samples.show());
            println!(
                "sphere_samples_scaled float: {:?}",
                sphere_samples
                    .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
                    .unwrap()
            );
            //let ball_norms =

            // scale to radius...
            let radius_expanded = radius.expand(&[n_ball, d]).unwrap();
            let sphere_samples_radius = mult(&[sphere_samples.clone(), radius_expanded])?;
            println!("sphere_samples_radius: {:?}", sphere_samples_radius.show());
            let sphere_samples_radius_scaled = sphere_samples_radius
                .enum_map::<_, _, ModuleError>(|i, v| {
                    Ok((v as f64 / 2f64.powf(8.0)).round() as i64)
                })?;
            println!(
                "sphere_samples_radius_scaled: {:?}",
                sphere_samples_radius_scaled.show()
            );
            println!(
                "sphere_samples_radius_scaled float: {:?}",
                sphere_samples_radius_scaled
                    .enum_map::<_, _, ModuleError>(|i, s| Ok(s as f64 / 2f64.powf(8.0)))
                    .unwrap()
            );

            println!("X: {:?}", x);
            let mut perturbations2 = sphere_samples_radius_scaled
                .enum_map::<_, _, ModuleError>(|i, v| Ok(x[i % d] + i64_to_felt::<Fp>(v)))
                .unwrap();
            println!("perturbations2: {:?}", perturbations2.show());

            // print out the floating point...
            let x_test = Tensor::new(Some(&x), &[d]).unwrap();
            println!("x: {:?}", x_test.dequantize(8));
            println!("perturbations2: {:?}", perturbations2.dequantize(8));

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
            println!("perturbations: {:?}", perturbations);
            let mut res = x.clone();
            res.extend(perturbations);
            res
        };

        println!("RES: {:?}", res);
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
        let samples = SampleChip::<8>::run((self.n_lime * self.d) + (self.n_ball * (self.d + 2)))?;
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
        // perturb x_border by lime samples...
        let d = lime_samples.dims()[1];

        // scale the noise by 2x...
        // TODO(EVAN): make this scale by our stddev

        let mut multiplier =
            create_constant_tensor(Fp::from(Self::LIME_MULTIPLIER), self.n_lime * d);
        multiplier.reshape(&[self.n_lime, d]);
        let lime_samples = pairwise(
            config,
            region,
            &[lime_samples.clone(), multiplier],
            BaseOp::Mult,
        )
        .unwrap();

        let mut x_border_expanded = x_border.clone();
        x_border_expanded.reshape(&[1, d]);
        x_border_expanded.expand(&[self.n_lime, d]).unwrap();
        println!("X BORDER EXPANDED: {}", x_border_expanded.show());
        let perturbations = pairwise(
            config,
            region,
            &[x_border_expanded.clone(), lime_samples.clone()],
            BaseOp::Add,
        )
        .unwrap();

        // center to 0
        let mut half_sample =
            create_constant_tensor(Fp::from(128 * Self::LIME_MULTIPLIER), self.n_lime * d);
        half_sample.reshape(&[self.n_lime, d]);
        let perturbations =
            pairwise(config, region, &[perturbations, half_sample], BaseOp::Sub).unwrap();

        println!("PERTURBATIONS BORDER: {}", perturbations.show());

        perturbations
    }

    pub fn layout_ball_radius(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        x_border: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        println!("x: {:?}", x.show());
        println!("x_border: {:?}", x_border.show());
        let difference =
            pairwise(config, region, &[x.clone(), x_border.clone()], BaseOp::Sub).unwrap();
        println!("diffs!: {:?}", difference.show());
        let square_norms = dot(config, region, &[difference.clone(), difference.clone()])?;
        println!("square diffs!: {:?}", square_norms.show());
        // scale back down to 8 bits..
        let square_norms_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_norms.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        println!("square_diffs_scaled: {:?}", square_norms_scaled.show());

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
        println!("radius: {:?}", norm.show());

        Ok(norm)
    }

    pub fn layout_ball_samples(
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
        println!("BALL SAMPLES!: {:?}", ball_samples_normal.show());

        // compute
        let square_norms = einsum(
            config,
            region,
            &[ball_samples_normal.clone(), ball_samples_normal.clone()],
            "ij,ij->ik",
        )?;
        println!("square norms!: {:?}", square_norms.show());
        // scale back down to 8 bits..
        let square_norms_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[square_norms.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        println!("square_norms_scaled: {:?}", square_norms_scaled.show());
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
        println!("norms!: {:?}", recip_norms.show());
        // multiply by recips...
        let normalized = einsum(
            config,
            region,
            &[ball_samples_normal.clone(), recip_norms],
            "ij,ik->ij",
        )?;
        println!("normalized: {:?}", normalized.show());

        // TODO: rescale by distance to point....
        let sphere_samples = normalized.get_slice(&[0..self.n_ball, 0..d]).unwrap();
        println!("sphere samples: {:?}", sphere_samples.show());
        // scale back down to 8 bits..
        let sphere_samples_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[sphere_samples.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        println!("sphere_samples_scaled: {:?}", sphere_samples_scaled.show());

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
        println!("sphere_samples_radius: {:?}", sphere_samples_radius.show());
        // scale back down to 8 bits..
        let sphere_samples_radius_scaled = crate::circuit::ops::layouts::loop_div(
            config,
            region,
            &[sphere_samples_radius.clone()],
            Fp::from(2u64.pow(8)),
        )
        .unwrap();
        println!(
            "sphere_samples_radius_scaled: {:?}",
            sphere_samples_radius_scaled.show()
        );

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
        println!("X: {:?}", x_expanded.show());
        println!("perturbations2: {:?}", ball_perturbations.show());

        Ok(ball_perturbations)
    }

    // layout all Lime operations...
    pub fn layout(
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
        println!("SAMPLES: {}", samples.show());
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
            println!("lime perts: {:?}", lime_perturbations.show());
            let mut result = x.concat(lime_perturbations).unwrap();
            result
                .reshape(&[Lime2Chip::input_size(self.n_lime, self.n_ball), d])
                .unwrap();
            result
        };
        println!("RES: {:?}", result.show());
        Ok(result)
        //unimplemented!();
    }

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
        println!("SQUARE DISTANCE: {:?}", square_distance.show());

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
        println!("WEIGHTS: {:?}", weights.show());

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
        println!("SQRT WEIGHTS: {:?}", sqrt_weights.show());

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
        println!("GOT INPUTS: {:?}", inputs.pshow(16));

        let mut output_sqrt_weights = sqrt_weights.clone();
        output_sqrt_weights.expand(&[self.n_lime]);
        let outputs = pairwise(
            config,
            region,
            &[output_sqrt_weights.clone(), outputs],
            BaseOp::Mult,
        )
        .unwrap();
        println!("GOT OUTPUTS: {:?}", outputs.pshow(8));

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
        println!("deltas: {:?}", deltas.pshow(16));

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
        println!("y - int: {:?}", intermediate.pshow(8));

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
        println!("intermediate: {:?}", intermediate.pshow(16));
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
        println!("square_residuals: {:?}", square_residuals.pshow(16));

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
        println!(
            "square_residuals_dived: {:?}",
            square_residuals_dived.pshow(32)
        );
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
        println!(
            "square_residuals_dived: {:?}",
            square_residuals_dived.pshow(8)
        );

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
        println!("l1 part: {:?}", l1_part.pshow(8));

        let mut primal_objective = pairwise(
            config,
            region,
            &[square_residuals_dived.clone(), l1_part],
            BaseOp::Add,
        )
        .unwrap();

        println!("primal objective: {}", primal_objective.pshow(8));

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
        println!("square dual: {:?}", square_dual.pshow(16));

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
        println!("square_dual_dived: {:?}", square_dual_dived.pshow(32));
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
        println!("square_dual_dived: {:?}", square_dual_dived.pshow(8));

        let mut dual_res = dot(config, region, &[dual.clone(), outputs]).unwrap();
        println!("dual_dot: {:?}", dual_res.pshow(24));
        for i in 0..2 {
            dual_res = crate::circuit::ops::layouts::loop_div(
                config,
                region,
                &[dual_res.clone()],
                Fp::from(2u64.pow(8)),
            )
            .unwrap();
        }
        println!("dual_dot: {:?}", dual_res.pshow(8));
        let dual_res = pairwise(
            config,
            region,
            &[dual_res.clone(), square_dual_dived.clone()],
            BaseOp::Add,
        )
        .unwrap();
        println!("dual_res: {:?}", dual_res.pshow(8));

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
        println!("dual feasible: {:?}", dual_feasible.pshow(16));
        // add some slack due to quantization error...
        let range_check_bracket = ((0.01 * 1.5) * 2f64.powf(16.0)).ceil() as i64;
        let range_check_felt: Fp = i64_to_felt(range_check_bracket);
        println!("GOT {:?}", range_check_felt);
        println!("bracket: {:?}", dual_feasible);
        range_check(
            config,
            region,
            &[dual_feasible],
            &(-range_check_bracket, range_check_bracket),
        )
        .unwrap();

        region.debug_report();
    }

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
        println!("GOT model: {:?}", lime_model_abs.pshow(8));
        println!("GOT SORTED: {:?}", sorted.pshow(8));

        // ensure topk is really topk
        let checking_topk = sorted.get_slice(&[sorted.len() - k..sorted.len()]).unwrap();
        println!("topk check: {:?}", checking_topk.pshow(8));
        enforce_equality(config, region, &[top_k_abs.clone(), checking_topk.clone()]);

        println!("model: {:?}", lime_model.dims());
        println!("topk_idx: {:?}", lime_model_topk_idxs.dims());
        // ensure idxs really idxs
        let topk_from_idx = gather(
            config,
            region,
            &[lime_model.clone(), lime_model_topk_idxs.clone()],
            0,
        )
        .unwrap();
        println!("TOPK_IDXS: {:?}", topk_from_idx.pshow(8));
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

        println!("got values: {:?}", values);

        let samples = self.sample_chip.layout(layouter)?;
        println!("got samples: {:?}", samples.len());
        // pass values into perturbchip

        let perturbations = self
            .perturb_chip
            .layout(layouter, values.clone(), samples)?;
        println!("got perts: {:?}", perturbations);

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

        println!("got values: {:?}", values);

        let samples = self.sample_chip.layout(layouter)?;
        println!("got samples: {:?}", samples.len());
        // pass values into perturbchip

        let perturbations = self
            .perturb_chip
            .layout(layouter, values.clone(), samples)?;
        println!("got perts: {:?}", perturbations);

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

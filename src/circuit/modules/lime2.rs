/// OKKKKKKKKKKKK
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use itertools::Itertools;

use super::errors::ModuleError;
use crate::tensor::{Tensor, ValTensor, ValType};

use crate::circuit::ops::layouts::*;
use crate::circuit::{ops::base::BaseOp, utils};
use crate::fieldutils::{felt_to_i64, i64_to_felt};

use super::perturb::PerturbChip;
use super::sample::SampleChip;
use crate::circuit::ops::chip::BaseConfig;
use crate::circuit::ops::region::RegionCtx;

// helper to assign tensors to column of cells

// Generate rng ahead of time ... makes things easier...
#[derive(Debug, Clone)]
pub struct Lime2Chip {
    sample_chip: SampleChip<8>,
    perturb_chip: PerturbChip<Fp>,

    samples: Option<Vec<AssignedCell<Fp, Fp>>>,

    n_lime: usize,
    n_ball: usize,
    d: usize,
}

impl Lime2Chip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        n_lime: usize,
        n_ball: usize,
        d: usize,
    ) -> Self {
        println!("CONFIG: {} {} {}", n_lime, n_ball, d);
        let sample_chip = SampleChip::configure(meta, (n_lime * d) + (n_ball * (d + 2)));
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
        let samples = SampleChip::<8>::run((n_lime * d) + (n_ball * (d + 2)))?;

        let lime_samples = &samples[0..n_lime * d];
        let ball_samples = &samples[0..n_ball * (d + 2)];

        let lime_inputs = lime_samples.iter().enumerate().map(|(i, v)| v + x[i % d]);

        let ball_tensor = Tensor::new(Some(ball_samples), &[n_ball, d + 2]).unwrap();
        let ball_tensor = ball_tensor.clone().map(|x| felt_to_i64(x));
        let ball_samples_normal = crate::tensor::ops::nonlinearities::normal_inverse_cdf(
            &ball_tensor,
            2f64.powf(8.0),
            0.0,
            1.0,
        );
        println!("ball_samples_normal: {:?}", ball_samples_normal);
        //let ball_norms =

        // Must pass through normal etc...
        // FIXME(EVAN)
        //perturbations.extend(perturbations2);

        //Ok(perturbations)
        let perturbations = vec![Fp::zero(); n_lime * d];
        let perturbations2 = vec![Fp::zero(); n_ball * d];

        let mut res = x.clone();
        res.extend(x_border);
        res.extend(perturbations);
        res.extend(perturbations2);
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

    // layout all Lime operations...
    pub fn layout(
        &self,
        //layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: &ValTensor<Fp>,
        x_border: &ValTensor<Fp>,
        samples: &ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        //self.generate_samples(layouter)?;
        //
        let d = x.dims()[0];
        assert_eq!(x.dims(), &[d]);
        assert_eq!(x_border.dims(), &[d]);

        //let lime_samples = self.layout_perturb_uniform(layouter, x_border)?;
        println!("SAMPLE LEN: {:?}", samples.dims());
        println!("N_LIME {} N_BALL {} d {}", self.n_lime, self.n_ball, d);
        assert_eq!(
            samples.dims(),
            &[(self.n_lime * d) + (self.n_ball * (d + 2))]
        );

        let mut lime_samples = samples.get_slice(&[0..(self.n_lime * d)]).unwrap();
        lime_samples.reshape(&[self.n_lime, d]).unwrap();
        let mut ball_samples = samples
            .get_slice(&[(self.n_lime * d)..samples.dims()[0]])
            .unwrap();
        ball_samples.reshape(&[self.n_ball, d + 2]).unwrap();

        // perturb x_border by lime samples...
        let mut x_border_expanded = x_border.clone();
        x_border_expanded.expand(&[self.n_lime, d]).unwrap();
        let perturbations = pairwise(
            config,
            region,
            &[x_border_expanded.clone(), lime_samples.clone()],
            BaseOp::Add,
        )
        .unwrap();
        println!("X_BORDER: {:?}", x_border);
        println!("SAMPLES: {:?}", lime_samples);
        println!("perturbations: {:?}", perturbations);

        use crate::circuit::utils::F32;
        let ball_samples_normal = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[ball_samples],
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
        // scale down to 8 bits...
        let recip_norms = crate::circuit::ops::layouts::nonlinearity(
            config,
            region,
            &[square_norms],
            &crate::circuit::ops::lookup::LookupOp::RecipSqrt {
                input_scale: F32(2f32.powf(16.0)),
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

        // TODO: rescale by distance to point....
        let sphere_samples = normalized.get_slice(&[0..self.n_ball, 0..d]).unwrap();
        let mut x_expanded = x.clone();
        x_expanded.expand(&[self.n_ball, d]).unwrap();
        let perturbations2 = pairwise(
            config,
            region,
            &[x_expanded.clone(), sphere_samples.clone()],
            BaseOp::Add,
        )
        .unwrap();

        // concat all the points together...
        let result = x.concat(x_border.clone()).unwrap();
        let result = result.concat(perturbations).unwrap();
        let mut result = result.concat(perturbations2).unwrap();
        result
            .reshape(&[(self.n_lime + self.n_ball + 2), d])
            .unwrap();
        Ok(result)
        //unimplemented!();
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

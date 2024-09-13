/// OKKKKKKKKKKKK
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};

use itertools::Itertools;

use super::errors::ModuleError;
use crate::tensor::{Tensor, ValTensor, ValType};

use crate::circuit::ops::layouts::*;
use crate::circuit::{ops::base::BaseOp, utils};

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
}

impl Lime2Chip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        n_lime: usize,
        n_ball: usize,
        d: usize,
    ) -> Self {
        let sample_chip = SampleChip::configure(meta, (n_lime + n_ball) * d);
        let perturb_chip = PerturbChip::configure(meta, Some(256));

        Self {
            sample_chip,
            perturb_chip,

            samples: None,
            n_lime,
            n_ball,
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
        input: Vec<Fp>,
        x_border: Vec<Fp>,
        n: usize,
        n_surrogate: usize,
        d: usize,
    ) -> Result<Vec<Fp>, ModuleError> {
        let samples = SampleChip::<8>::run((n + n_surrogate) * d)?;
        // Must pass through normal etc...
        // FIXME(EVAN)
        let mut perturbations = PerturbChip::run(input, samples[0..n_surrogate * d].to_vec(), 256)?;
        let mut perturbations2 =
            PerturbChip::run(x_border, samples[n_surrogate * d..].to_vec(), 256)?;
        perturbations.extend(perturbations2);

        Ok(perturbations)
    }

    pub fn generate_samples(
        &mut self,
        layouter: &mut impl Layouter<Fp>,
    ) -> Result<(), ModuleError> {
        self.samples = Some(self.sample_chip.layout(layouter)?);
        Ok(())
    }

    pub fn distance(
        &mut self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        a: ValTensor<Fp>,
        b: ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let delta = pairwise(config, region, &[a.clone(), b.clone()], BaseOp::Sub)?;
        let square = dot(config, region, &[delta.clone(), delta.clone()]);
        println!("Got {:?}", square);

        unimplemented!();
    }

    // layout all Lime operations...
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        x: ValTensor<Fp>,
        x_border: ValTensor<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        //self.generate_samples(layouter)?;

        let lime_samples = self.layout_perturb_uniform(layouter, x_border)?;

        // some checks that dims are ccorrect
        assert!(false);

        unimplemented!();
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

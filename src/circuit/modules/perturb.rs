/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use super::utilities::DecomposeConfig;
use halo2_gadgets::poseidon::{primitives::*, Hash, Pow5Chip, Pow5Config};
use halo2_gadgets::utilities::decompose_running_sum;
use halo2_gadgets::utilities::{
    decompose_running_sum::{RunningSum, RunningSumConfig},
    decompose_word,
};
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2curves::ff::{Field, PrimeField};
// use maybe_rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use maybe_rayon::prelude::ParallelIterator;
use maybe_rayon::slice::ParallelSlice;

use std::marker::PhantomData;

use crate::circuit::region::ConstantsMap;
use crate::graph::modules::ModulePoseidon;
use crate::tensor::{Tensor, ValTensor, ValType};

use super::errors::ModuleError;
use super::Module;

use super::poseidon::spec::PoseidonSpec;

use rand::{thread_rng, Rng};

/// The number of instance columns used by the Poseidon hash function
pub const POSEIDON_WIDTH: usize = 2;
pub const NUM_INSTANCE_COLUMNS: usize = 1;

// TODO:: add randomness strategy...allow for public randomness.
#[derive(Debug, Clone)]
pub struct PerturbConfig {
    ///// Input to be perturbed...
    ///// Seed to the prg
    //pub seed: Option<Column<Instance>>,
    pub input: Column<Advice>,
    pub perturbation: Column<Advice>,
    pub output: Column<Advice>,
    pub selector: Selector,

    pub shift: Option<usize>,
}

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct PerturbChip<F: PrimeField> {
    config: PerturbConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> PerturbChip<F> {
    /// Configure
    pub fn configure(meta: &mut ConstraintSystem<F>, shift: Option<usize>) -> Self {
        let input = meta.advice_column();
        let perturbation = meta.advice_column();
        meta.enable_equality(perturbation);
        let output = meta.advice_column();
        meta.enable_equality(output);
        let selector = meta.selector();

        meta.create_gate("perturb", |meta| {
            let input = meta.query_advice(input, Rotation::cur());
            let perturbation = meta.query_advice(perturbation, Rotation::cur());
            let output = meta.query_advice(output, Rotation::cur());
            let selector = meta.query_selector(selector);

            match shift {
                Some(shift_value) => {
                    vec![
                        selector
                            * (input + perturbation
                                - Expression::Constant(F::from(shift_value as u64))
                                - output),
                    ]
                }
                None => vec![selector * (input + perturbation - output)],
            }
        });

        let config = PerturbConfig {
            input,
            perturbation,
            output,
            selector,
            shift,
        };

        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Name
    fn name(&self) -> &'static str {
        "Perturb"
    }

    /// Run the operation the module represents
    pub fn run(
        mut input: Vec<F>,
        perturbation: Vec<F>,
        shift: usize,
    ) -> Result<Vec<F>, ModuleError> {
        let res = perturbation
            .iter()
            .enumerate()
            .map(|(i, pert)| input[i % input.len()] + pert - F::from(shift as u64))
            .collect::<Vec<_>>();
        input.extend(res);
        Ok(input)
    }

    /// Layout
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Value<F>>,
        pert: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, ModuleError> {
        let shift = self.config.shift.unwrap_or(0);
        let output = pert
            .iter()
            .enumerate()
            .map(|(i, pert)| {
                input[i % input.len()] + pert.value() - Value::known(F::from(shift as u64))
            })
            .collect::<Vec<_>>();

        let res = layouter.assign_region(
            || "pert add region",
            |mut region| {
                let mut output_cells = vec![];
                for i in 0..pert.len() {
                    self.config.selector.enable(&mut region, i).unwrap();
                    region.assign_advice(
                        || "input",
                        self.config.input,
                        i,
                        || input[i % input.len()].clone(),
                    )?;
                    pert[i].copy_advice(|| "pert", &mut region, self.config.perturbation, i)?;
                    let output_cell = region.assign_advice(
                        || "output",
                        self.config.output,
                        i,
                        || output[i].clone(),
                    )?;
                    output_cells.push(output_cell);
                }
                Ok(output_cells)
            },
        )?;

        Ok(res)
    }
}

#[allow(unused)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor, ValTensor, ValType};

    use crate::circuit::modules::ModulePlanner;

    use std::{collections::HashMap, marker::PhantomData};

    use halo2_gadgets::poseidon::primitives::Spec;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem},
    };
    use halo2curves::ff::Field;

    struct PerturbCircuit {
        message: Vec<Value<Fp>>,
        pert: Vec<Value<Fp>>,
    }

    impl Circuit<Fp> for PerturbCircuit {
        type Config = PerturbChip<Fp>;
        type FloorPlanner = ModulePlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            unimplemented!();
            //let empty_val: Vec<ValType<Fp>> = vec![Value::<Fp>::unknown().into()];
            //let message: Tensor<ValType<Fp>> = empty_val.into_iter().into();

            //Self {
            //    message: message.into(),
            //    _spec: PhantomData,
            //}
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> PerturbChip<Fp> {
            PerturbChip::configure(meta, Some(128))
        }

        fn synthesize(
            &self,
            config: PerturbChip<Fp>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            unimplemented!();
            //let output = config.layout(&mut layouter, self.message.clone(), self.pert.clone())?;
            //println!("got output: {:?}", output);

            Ok(())
        }
    }

    #[test]
    fn perturb() {
        let rng = rand::rngs::OsRng;

        let message = (0..23).map(|i| Fp::from(i)).collect::<Vec<_>>();
        let pert = (0..23).map(|i| Fp::from(i)).collect::<Vec<_>>();
        let output = PerturbChip::run(message.clone(), pert.clone(), 128).unwrap();
        println!("got dryrun output: {:?}", output);

        let k = 9;
        let message = message.iter().map(|m| Value::known(*m)).collect::<Vec<_>>();
        let pert = pert.iter().map(|m| Value::known(*m)).collect::<Vec<_>>();
        let circuit = PerturbCircuit { message, pert };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

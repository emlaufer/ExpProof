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

// Samples n random values using
// TODO: support a seed...
#[derive(Debug, Clone)]
pub struct SampleConfig<const BITS: usize> {
    ///// Input to be perturbed...
    ///// Seed to the prg
    //pub seed: Option<Column<Instance>>,
    pub n: usize,
    pub pow5_config: Pow5Config<Fp, POSEIDON_WIDTH, 1>,
    pub decompose_config: DecomposeConfig<Fp, BITS, 8>,
    // TODO: shouldn't this just be consts?
    pub hash_inputs: Vec<Column<Advice>>,
    pub decompose_test: Column<Advice>,
}

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct SampleChip<const BITS: usize> {
    config: SampleConfig<BITS>,
}

impl<const BITS: usize> SampleChip<BITS> {
    /// Creates a new PerturbChip
    pub fn configure_with_cols(
        meta: &mut ConstraintSystem<Fp>,
        n: usize, // num outputs
        partial_sbox: Column<Advice>,
        rc_a: [Column<Fixed>; POSEIDON_WIDTH],
        rc_b: [Column<Fixed>; POSEIDON_WIDTH],
        hash_inputs: Vec<Column<Advice>>,
        decompose_input: Column<Advice>,
        bit_input: Column<Advice>,
        decompose_selector: Selector,
    ) -> SampleConfig<BITS> {
        let pow5_config = Pow5Chip::configure::<PoseidonSpec>(
            meta,
            hash_inputs.clone().try_into().unwrap(),
            partial_sbox,
            rc_a,
            rc_b,
        );

        // TODO: there should be a lookup table used elsewhere we can use...
        let range_table = meta.lookup_table_column();
        let decompose_config = DecomposeConfig::configure(
            meta,
            decompose_selector,
            decompose_input,
            bit_input,
            range_table,
        );

        // TODO: compute num output columns needed for bits...

        SampleConfig {
            pow5_config,
            n,
            hash_inputs,
            decompose_config,
            decompose_test: decompose_input,
        }
    }
}

impl<const BITS: usize> SampleChip<BITS> {
    /// Configure
    pub fn configure(meta: &mut ConstraintSystem<Fp>, n: usize) -> Self {
        //  instantiate the required columns
        println!("MAKING N SAMPLES: {:?}", n);
        let hash_inputs = (0..POSEIDON_WIDTH)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        for input in &hash_inputs {
            meta.enable_equality(*input);
        }

        let partial_sbox = meta.advice_column();
        let rc_a = (0..POSEIDON_WIDTH)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>();
        let rc_b = (0..POSEIDON_WIDTH)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>();

        for input in hash_inputs.iter().take(POSEIDON_WIDTH) {
            meta.enable_equality(*input);
        }
        meta.enable_constant(rc_b[0]);

        let decompose_input = meta.advice_column();
        let bit_input = meta.advice_column();
        meta.enable_equality(bit_input);
        let decompose_selector = meta.selector();

        let config = Self::configure_with_cols(
            meta,
            n,
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
            hash_inputs,
            decompose_input,
            bit_input,
            decompose_selector,
        );

        Self { config }
    }

    /// Name
    fn name(&self) -> &'static str {
        "Sample"
    }

    /// Run the operation the module represents
    pub fn run(n: usize) -> Result<Vec<Fp>, ModuleError> {
        // TODO: move as config
        let perts_per_hash = (Fp::NUM_BITS as usize) / BITS;
        let n_hashes = ((n as f64) / (perts_per_hash as f64)).ceil() as usize;

        // ignore remainer
        let ignore_hash_remainder = (Fp::NUM_BITS as usize) % BITS != 0;
        println!("N HASHES: {:?}", n_hashes);

        let hasher = halo2_gadgets::poseidon::primitives::Hash::<
            _,
            PoseidonSpec,
            ConstantLength<1>,
            POSEIDON_WIDTH,
            1,
        >::init();

        let mut hashes = vec![];
        for i in 0..n_hashes {
            let hash = hasher.clone().hash([Fp::from(i as u64)]);
            hashes.push(hash);
        }

        println!("hashes: {:?}", hashes);

        let mut rng = thread_rng();

        // bit-split hashes
        let hash_bits = hashes
            .iter()
            .map(|h| {
                let mut words = decompose_word(h, Fp::NUM_BITS as usize, BITS);
                if ignore_hash_remainder {
                    words.pop();
                }
                words
            })
            .flatten()
            .map(|v| Fp::from(v as u64))
            .collect::<Vec<_>>();

        let hash_words = hash_bits[0..n].to_owned();

        return Ok(hash_words);
    }

    /// Layout
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
    ) -> Result<Vec<AssignedCell<Fp, Fp>>, ModuleError> {
        // TODO MAJOR: need to sign extend...or else the rng only goes positive direction...
        // WAIT easy...just subtract by half range.........
        let perts_per_hash = (Fp::NUM_BITS as usize) / BITS;
        let n_hashes = ((self.config.n as f64) / (perts_per_hash as f64)).ceil() as usize;

        // ignore remainer
        let ignore_hash_remainder = (Fp::NUM_BITS as usize) % BITS != 0;

        // TODO: remove later...
        self.config.decompose_config.load(layouter)?;

        let mut hash_inputs = vec![];
        for i in 0..n_hashes {
            let res = layouter.assign_region(
                || "load perturb hash input",
                |mut region| {
                    region.assign_advice_from_constant(
                        || format!("load message_pert const"),
                        self.config.hash_inputs[0],
                        i,
                        Fp::from(i as u64),
                    )
                },
            )?;
            hash_inputs.push(res);
        }
        println!("HASHE inputs: {:?}", hash_inputs);

        let mut hashes = vec![];
        for i in 0..n_hashes {
            // test run....
            let pow5_chip = Pow5Chip::construct(self.config.pow5_config.clone());
            // TODO: length is 1?
            let hasher = Hash::<_, _, PoseidonSpec, ConstantLength<1>, POSEIDON_WIDTH, 1>::init(
                pow5_chip,
                layouter.namespace(|| "block_hasher"),
            )?;

            let hash = hasher.hash(layouter.namespace(|| "hash"), [hash_inputs[i].clone()])?;
            println!("hash: {:?}", hash);
            println!("n-bits: {:?}", Fp::NUM_BITS);
            hashes.push(hash);
        }

        let samples = layouter.assign_region(
            || "decompose",
            |mut region| {
                let mut rngs = vec![];
                for (i, hash) in hashes.iter().enumerate() {
                    let offset = i * ((Fp::NUM_BITS as usize / BITS) + 2);
                    let mut block_rngs = self.config.decompose_config.copy_decompose(
                        &mut region,
                        offset,
                        hash.clone(),
                        true,
                        Fp::NUM_BITS as usize,
                        // TODO: wrong if a even split...
                        (Fp::NUM_BITS as usize / BITS) + 1,
                    )?;
                    if ignore_hash_remainder {
                        block_rngs.pop();
                    }
                    rngs.append(&mut block_rngs);
                }
                Ok(rngs)
            },
        )?;

        Ok(samples[0..self.config.n].to_vec())
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

    struct PerturbCircuit {}

    impl Circuit<Fp> for PerturbCircuit {
        type Config = SampleChip<8>;
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

        fn configure(meta: &mut ConstraintSystem<Fp>) -> SampleChip<8> {
            SampleChip::configure(meta, 100)
        }

        fn synthesize(
            &self,
            config: SampleChip<8>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let samples = config.layout(&mut layouter)?;

            Ok(())
        }
    }

    #[test]
    fn sample() {
        let rng = rand::rngs::OsRng;

        let output = SampleChip::<8>::run(100).unwrap();
        println!("got dryrun output: {:?}", output);

        let k = 9;
        let circuit = PerturbCircuit {};
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

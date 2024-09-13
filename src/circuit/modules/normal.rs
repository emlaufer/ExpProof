use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use halo2curves::ff::{Field, PrimeField};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::Distribution;

use std::marker::PhantomData;

use super::errors::ModuleError;
use crate::fieldutils::i64_to_felt;
use crate::graph::utilities::{dequantize, quantize_float};

// TODO: Table op should be responsible for quantizing. ..not the table...
pub trait TableOp {
    fn name(&self) -> String;
    fn f(&self, input: f64) -> f64;
}

#[derive(Clone, Debug)]
pub struct NormalInverseCDF {
    dist: Normal,
}

impl NormalInverseCDF {
    pub fn new(mean: f64, std: f64) -> NormalInverseCDF {
        let dist = Normal::new(mean, std).unwrap();
        Self { dist }
    }
}

impl TableOp for NormalInverseCDF {
    fn name(&self) -> String {
        format!(
            "NormalInverseCDF({}, {})",
            self.dist.mean().unwrap(),
            self.dist.std_dev().unwrap()
        )
    }

    fn f(&self, input: f64) -> f64 {
        // add small constant to prevent 0.0
        self.dist.inverse_cdf(input + 0.001)
    }
}

#[derive(Debug, Clone)]
struct LookupTable<F: PrimeField + PartialOrd + Field, OP: TableOp, const BITS: usize> {
    input: TableColumn,
    output: TableColumn,
    op: OP,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + PartialOrd + Field, OP: TableOp, const BITS: usize> LookupTable<F, OP, BITS> {
    pub(super) fn configure(meta: &mut ConstraintSystem<F>, op: OP) -> Self {
        let input = meta.lookup_table_column();
        let output = meta.lookup_table_column();

        Self {
            input,
            output,
            op,
            _marker: PhantomData,
        }
    }

    pub(super) fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        let range = 2usize.pow(BITS as u32);

        layouter.assign_table(
            || "load normal inverse CDF table",
            |mut table| {
                let mut offset = 0;
                for value in 0..range {
                    let float_value = value as f64 / range as f64;
                    let float_output = self.op.f(float_value);
                    // quantize to 8-bit
                    let normal: F =
                        i64_to_felt(quantize_float(&float_output, 0.0, BITS as i32).unwrap());

                    table.assign_cell(
                        || format!("{} input", self.op.name()),
                        self.input,
                        offset,
                        || Value::known(F::from(value as u64)),
                    )?;
                    table.assign_cell(
                        || format!("{} output", self.op.name()),
                        self.output,
                        offset,
                        || Value::known(normal),
                    );
                    offset += 1;
                }

                Ok(())
            },
        )
    }
}

#[derive(Debug, Clone)]
struct TableChip<F: PrimeField + PartialOrd + Field, OP: TableOp, const BITS: usize> {
    q_lookup: Selector,
    // TODO: construct from assigned cell instead? or even from tensor?
    input: Column<Advice>,
    output: Column<Advice>,

    table: LookupTable<F, OP, BITS>,
}

impl<F: PrimeField + PartialOrd + Field, OP: TableOp, const BITS: usize> TableChip<F, OP, BITS> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        op: OP,
        input: Column<Advice>,
        output: Column<Advice>,
    ) -> Self {
        let table = LookupTable::configure(meta, op);
        let q_lookup = meta.complex_selector();

        // get default value for unconstrained values
        let default_index = F::from(0);
        let default_value_float = table.op.f(0.0);
        let default_value: F =
            i64_to_felt(quantize_float(&default_value_float, 0.0, BITS as i32).unwrap());

        meta.lookup("uniform to normal lookup", |meta| {
            let lookup = meta.query_selector(q_lookup);
            let input = meta.query_advice(input, Rotation::cur());
            let output = meta.query_advice(output, Rotation::cur());

            // TODO: defaults...
            vec![
                (
                    lookup.clone() * input
                        + (Expression::Constant(F::from(1)) - lookup.clone()) * default_index,
                    table.input,
                ),
                (
                    lookup.clone() * output
                        + (Expression::Constant(F::from(1)) - lookup.clone()) * default_value,
                    table.output,
                ),
            ]
        });

        Self {
            q_lookup,
            input,
            output,
            table,
        }
    }

    pub fn max_input(&self) -> u64 {
        2u64.pow(BITS as u32)
    }

    pub fn witness_layout(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        unimplemented!();
    }

    pub fn layout(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        // compute outputs from values...
        layouter.assign_region(
            || format!("{} assign", self.table.op.name()),
            |mut region| {
                let mut res = vec![];
                for (i, input) in inputs.iter().enumerate() {
                    self.q_lookup.enable(&mut region, i)?;

                    let output = input.value().map(|f| {
                        let float = dequantize(*f, self.max_input() as i32, 0.0);
                        let value = self.table.op.f(float);
                        i64_to_felt(quantize_float(&value, 0.0, self.max_input() as i32).unwrap())
                    });
                    //let output = self.table.op.f(value_float);

                    let a = input.copy_advice(|| "i_0 = input", &mut region, self.input, i)?;
                    let b = region.assign_advice(
                        || "o_0 = output",
                        self.output,
                        i,
                        || output.clone(),
                    )?;
                    res.push(b)
                }
                Ok(res)
            },
        )
    }

    pub fn assign_input_output(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<AssignedCell<F, F>>,
        outputs: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        // compute outputs from values...
        layouter.assign_region(
            || format!("{} assign", self.table.op.name()),
            |mut region| {
                let mut res = vec![];
                for (i, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                    self.q_lookup.enable(&mut region, i)?;

                    let a = input.copy_advice(|| "i_0 = input", &mut region, self.input, i)?;
                    let b = output.copy_advice(|| "o_0 = output", &mut region, self.output, i)?;
                    res.push(b)
                }
                Ok(res)
            },
        )
    }

    // TODO: can we dedup some of this grossness...
    pub fn assign_input_output_values(
        &self,
        mut layouter: impl Layouter<F>,
        inputs: Vec<Value<F>>,
        outputs: Vec<Value<F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        // compute outputs from values...
        layouter.assign_region(
            || format!("{} assign", self.table.op.name()),
            |mut region| {
                let mut res = vec![];
                for (i, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                    self.q_lookup.enable(&mut region, i)?;

                    let a =
                        region.assign_advice(|| "i_0 = input", self.input, i, || input.clone())?;
                    let b = region.assign_advice(
                        || "o_0 = output",
                        self.output,
                        i,
                        || output.clone(),
                    )?;
                    res.push(b)
                }
                Ok(res)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use halo2_proofs::{
        circuit::floor_planner::V1,
        dev::{FailureLocation, MockProver, VerifyFailure},
        plonk::{Any, Circuit},
    };
    use halo2curves::pasta::Fp;
    use rand::prelude::*;

    use super::*;

    #[derive(Default)]
    struct MyCircuit<F: PrimeField + PartialOrd + Field, const BITS: usize> {
        input: Value<F>,
        output: Value<F>,
    }

    impl<F: PrimeField + PartialOrd + Field, const BITS: usize> Circuit<F> for MyCircuit<F, BITS> {
        type Config = TableChip<F, NormalInverseCDF, BITS>;
        type Params = ();
        type FloorPlanner = V1;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let input = meta.advice_column();
            let output = meta.advice_column();
            let op = NormalInverseCDF::new(0.0, 1.0);
            TableChip::configure(meta, op, input, output)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.table.load(&mut layouter)?;

            config.assign_input_output_values(
                layouter.namespace(|| "Assign simple value"),
                vec![self.input],
                vec![self.output],
            )?;

            Ok(())
        }
    }

    #[test]
    fn test_normal_lookup_1() {
        //let k = 9;
        //const RANGE: usize = 8; // 3-bit value
        //const LOOKUP_RANGE: usize = 256; // 8-bit value
        let k = 9;
        const BITS: usize = 8;
        let mut rng = thread_rng();

        let range = (2u64.pow(BITS as u32));
        let dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..10 {
            let input_value: u64 = rng.gen::<u64>() % range;
            let input: Fp = i64_to_felt(input_value as i64);
            let output_float = dist.inverse_cdf((input_value as f64 / range as f64) + 0.001);
            let output: Fp = i64_to_felt(quantize_float(&output_float, 0.0, BITS as i32).unwrap());

            let circuit = MyCircuit::<Fp, BITS> {
                input: Value::known(input),
                output: Value::known(output),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }

        for i in 0..10 {
            let input_value: u64 = rng.gen::<u64>() % range;
            let input: Fp = i64_to_felt(input_value as i64);
            // wrong value
            let output_float = dist.inverse_cdf((input_value as f64 / range as f64) + 0.001) + 1.0;
            let output: Fp = i64_to_felt(quantize_float(&output_float, 0.0, BITS as i32).unwrap());

            let circuit = MyCircuit::<Fp, BITS> {
                input: Value::known(input),
                output: Value::known(output),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            assert_eq!(
                prover.verify(),
                Err(vec![VerifyFailure::Lookup {
                    lookup_index: 0,
                    location: FailureLocation::InRegion {
                        region: (1, "NormalInverseCDF(0, 1) assign").into(),
                        offset: 0
                    }
                }])
            );
        }
    }
}

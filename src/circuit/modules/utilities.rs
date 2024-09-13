use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Column, ConstraintSystem, Constraints, Error, Selector, TableColumn},
    poly::Rotation,
};
use halo2curves::ff::PrimeFieldBits;

use super::errors::ModuleError;
use crate::tensor::{TensorType, ValTensor, ValType};
use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_gadgets::utilities::range_check;
use halo2curves::ff::PrimeField;
use std::marker::PhantomData;

// TODO: this will copy ... but thats fine...
// TODO: constant map
/*pub fn assign_value_to_advice_column<F: PrimeField + TensorType + PartialOrd, A, AR>(
    annotation: A,
    region: &mut Region<F>,
    column: Column<Advice>,
    offset: usize,
    value: &ValType<F>,
) -> Result<AssignedCell<F, F>, ModuleError>
where
    A: Fn() -> AR,
    AR: Into<String>,
{
    match value {
        ValType::Value(v) => region
            .assign_advice(annotation, column, offset, || *v)
            .map_err(|e| e.into()),
        ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => v
            .copy_advice(annotation, region, column, offset)
            .map_err(|e| e.into()),
        ValType::Constant(f) => region
            .assign_advice_from_constant(annotation, column, offset, *f)
            .map_err(|e| e.into()),
        e => Err(ModuleError::WrongInputType(
            format!("{:?}", e),
            "PrevAssigned".to_string(),
        )),
    }
}

// Assigns a tensor to an advice column
// TODO: could add offset here...
pub fn assign_tensor_to_advice_column<F: PrimeField + TensorType + PartialOrd, A, AR>(
    annotation: A,
    region: &mut Region<F>,
    column: Column<Advice>,
    value: &ValTensor<F>,
) -> Result<Vec<AssignedCell<F, F>>, ModuleError>
where
    A: Fn() -> AR,
    AR: Into<String>,
{
    match value {
        ValTensor::Value { inner: v, .. } => v
            .iter()
            .enumerate()
            .map(|(i, value)| assign_value_to_advice_column(annotation, region, column, i, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.into()),
        ValTensor::Instance {
            dims,
            inner: col,
            idx,
            initial_offset,
            ..
        } => {
            let num_elems = dims[*idx].iter().product::<usize>();
            (0..num_elems)
                .map(|i| region.assign_advice_from_instance(annotation, *col, i, column, i))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.into())
        }
    }
}*/

/// Split a value into K-bit values
/// If K is small, use a lookup to ensure in range
/// Else, split K into limbs and check those instead

/// Configuration that provides methods for bit decomposition.
/// TODO: make transparent...allow for not just bit decomp
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecomposeConfig<F: PrimeFieldBits, const WINDOW_BITS: usize, const TABLE_BITS: usize> {
    q_range_check: Selector,
    running_sum: Column<Advice>,                 // running sum column
    decomposition: Column<Advice>,               // WINDOW_BIT decomposition of the value
    lookup_columns: Option<Vec<Column<Advice>>>, // columns to check each window is in range...None
    range_checks: Option<Vec<LookupRangeCheckConfig<F, TABLE_BITS>>>, // lookup decomp args if values are too big
    pub table_column: TableColumn,                                    // lookup table for TABLE bits
    // if TABLE_BITS >= WINDOW_BITS
    _marker: PhantomData<F>,
}

impl<F: PrimeFieldBits, const WINDOW_NUM_BITS: usize, const TABLE_BITS: usize>
    DecomposeConfig<F, WINDOW_NUM_BITS, TABLE_BITS>
{
    /// Returns the q_range_check selector of this [`RunningSumConfig`].
    pub(crate) fn q_range_check(&self) -> Selector {
        self.q_range_check
    }

    /// `perm` MUST include the advice column `z`.
    ///
    /// # Panics
    ///
    /// Panics if WINDOW_NUM_BITS > 3.
    ///
    /// # Side-effects
    ///
    /// `z` will be equality-enabled.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        q_range_check: Selector,
        running_sum: Column<Advice>,
        decomposition: Column<Advice>,
        table_column: TableColumn,
    ) -> Self {
        meta.enable_equality(running_sum);

        let small_windows = WINDOW_NUM_BITS <= TABLE_BITS;

        let config = Self {
            q_range_check,
            running_sum,
            decomposition,
            lookup_columns: None,
            range_checks: None,
            table_column,
            _marker: PhantomData,
        };

        // https://p.z.cash/halo2-0.1:decompose-short-range
        if small_windows {
            // decompose value...
            meta.create_gate("decompose", |meta| {
                let q_range_check = meta.query_selector(config.q_range_check);
                let z_cur = meta.query_advice(config.running_sum, Rotation::cur());
                let z_next = meta.query_advice(config.running_sum, Rotation::next());
                let b_cur = meta.query_advice(config.decomposition, Rotation::cur());
                //    z_i = 2^{K}⋅z_{i + 1} + k_i
                // => k_i = z_i - 2^{K}⋅z_{i + 1}
                let word = z_cur - z_next * F::from(1 << WINDOW_NUM_BITS);

                Constraints::with_selector(
                    q_range_check,
                    vec![
                        b_cur - word, // decomp equals
                    ],
                )
            });
            // ensure values are within range
            meta.lookup("decompose lookup", |meta| {
                // TODO: selector
                let b_cur = meta.query_advice(config.decomposition, Rotation::cur());
                vec![(b_cur, table_column)]
            });
        } else {
            unimplemented!();
        }

        config
    }

    // Loads the values [0..2^K) into `table_idx`. This is only used in testing
    // for now, since the Sinsemilla chip provides a pre-loaded table in the
    // Orchard context.
    pub fn load(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_table(
            || "table_idx",
            |mut table| {
                // We generate the row values lazily (we only need them during keygen).
                for index in 0..(1 << TABLE_BITS) {
                    table.assign_cell(
                        || "table_idx",
                        self.table_column,
                        index,
                        || Value::known(F::from(index as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Decompose a field element alpha that is witnessed in this helper.
    ///
    /// `strict` = true constrains the final running sum to be zero, i.e.
    /// constrains alpha to be within WINDOW_NUM_BITS * num_windows bits.
    pub fn witness_decompose(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        alpha: Value<F>,
        strict: bool,
        word_num_bits: usize,
        num_windows: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let z_0 = region.assign_advice(|| "z_0 = alpha", self.running_sum, offset, || alpha)?;
        self.decompose(region, offset, z_0, strict, word_num_bits, num_windows)
    }

    /// Decompose an existing variable alpha that is copied into this helper.
    ///
    /// `strict` = true constrains the final running sum to be zero, i.e.
    /// constrains alpha to be within WINDOW_NUM_BITS * num_windows bits.
    pub fn copy_decompose(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        alpha: AssignedCell<F, F>,
        strict: bool,
        word_num_bits: usize,
        num_windows: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        let z_0 = alpha.copy_advice(|| "copy z_0 = alpha", region, self.running_sum, offset)?;
        self.decompose(region, offset, z_0, strict, word_num_bits, num_windows)
    }

    /// `z_0` must be the cell at `(self.z, offset)` in `region`.
    ///
    /// # Panics
    ///
    /// Panics if there are too many windows for the given word size.
    fn decompose(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        z_0: AssignedCell<F, F>,
        strict: bool,
        word_num_bits: usize,
        num_windows: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        // Make sure that we do not have more windows than required for the number
        // of bits in the word. In other words, every window must contain at least
        // one bit of the word (no empty windows).
        //
        // For example, let:
        //      - word_num_bits = 64
        //      - WINDOW_NUM_BITS = 3
        // In this case, the maximum allowed num_windows is 22:
        //                    3 * 22 < 64 + 3
        //
        assert!(WINDOW_NUM_BITS * num_windows < word_num_bits + WINDOW_NUM_BITS);

        // Enable selectors
        for idx in 0..num_windows {
            self.q_range_check.enable(region, offset + idx)?;
        }

        // Decompose base field element into K-bit words.
        let words = z_0
            .value()
            .map(|word| {
                halo2_gadgets::utilities::decompose_word::<F>(word, word_num_bits, WINDOW_NUM_BITS)
            })
            .transpose_vec(num_windows);

        // Initialize empty vector to store running sum values [z_0, ..., z_W].
        let mut zs: Vec<AssignedCell<F, F>> = vec![z_0.clone()];
        let mut bs: Vec<AssignedCell<F, F>> = vec![];
        let mut z = z_0;

        // Assign running sum `z_{i+1}` = (z_i - k_i) / (2^K) for i = 0..=n-1.
        // Outside of this helper, z_0 = alpha must have already been loaded into the
        // `z` column at `offset`.
        let two_pow_k_inv = Value::known(F::from(1 << WINDOW_NUM_BITS as u64).invert().unwrap());
        for (i, word) in words.iter().enumerate() {
            // z_next = (z_cur - word) / (2^K)
            let (z_next, b) = {
                let z_cur_val = z.value().copied();
                let word = word.map(|word| F::from(word as u64));
                let z_next_val = (z_cur_val - word) * two_pow_k_inv;
                (
                    region.assign_advice(
                        || format!("z_{:?}", i + 1),
                        self.running_sum,
                        offset + i + 1,
                        || z_next_val,
                    )?,
                    region.assign_advice(
                        || format!("b_{:?}", i),
                        self.decomposition,
                        offset + i,
                        || word,
                    )?,
                )
            };

            // Update `z`.
            z = z_next;
            zs.push(z.clone());
            bs.push(b.clone());
        }
        assert_eq!(zs.len(), num_windows + 1);

        if strict {
            // Constrain the final running sum output to be zero.
            region.constrain_constant(zs.last().unwrap().cell(), F::ZERO)?;
        }

        Ok(bs)
    }
}

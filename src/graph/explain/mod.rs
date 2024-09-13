pub mod errors;
pub mod utilities;

use errors::ExplainError;
use num::abs;
use rand::{thread_rng, Rng};

use super::utilities::{dequantize, quantize_float};
use super::vars::VarVisibility;
use super::{GraphCircuit, GraphData, GraphError, GraphWitness};

use utilities::matmul;

use crate::fieldutils::i64_to_felt;
use crate::tensor::{ops::accumulated::dot, Tensor, ValTensor};
use crate::{RunArgs, EZKL_BUF_CAPACITY};

use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2_proofs::{
    circuit::Layouter,
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::{self, Fr as Fp, G1Affine};
use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use ndarray::Array;
use serde::{Deserialize, Serialize};

type GraphPoint = Vec<Tensor<Fp>>;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExplainInput {
    /// Actual inferrence input point
    pub point: GraphPoint,
    /// Sampled points around the input for LIME
    pub samples: Vec<GraphPoint>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExplainOutput {
    /// Actual inferrence output point
    point: GraphPoint,
    /// Inferrence output for LIME samples
    samples: Vec<GraphPoint>,
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ExplainWitness {
    // TODO: should I combine these, use the same lookup ranges etc?
    // witness for the first inferrence
    pub input_witness: GraphWitness,

    // witness for the perturbed inferrences
    pub perturbed_inferrences: Vec<GraphWitness>,

    // intercept for linear model
    pub intercept: Fp,

    // coeffs for linear model
    pub coeffs: Vec<Fp>,
}

impl ExplainWitness {
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => return Err(e.into()),
        };
        Ok(serialized)
    }

    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, GraphError> {
        let file = std::fs::File::open(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;

        let reader = std::io::BufReader::with_capacity(*EZKL_BUF_CAPACITY, file);
        serde_json::from_reader(reader).map_err(|e| e.into())
    }

    /// Save the model input to a file
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let file = std::fs::File::create(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        // use buf writer
        let writer = std::io::BufWriter::with_capacity(*EZKL_BUF_CAPACITY, file);

        serde_json::to_writer(writer, &self).map_err(|e| e.into())
    }
}

pub struct ExplainCircuit {
    pub graph: GraphCircuit,
}

impl ExplainCircuit {
    pub fn new(graph: GraphCircuit) -> Result<ExplainCircuit, ExplainError> {
        Ok(ExplainCircuit { graph })
    }

    pub async fn load_graph_input(&self, data: &GraphData) -> Result<Vec<Tensor<Fp>>, GraphError> {
        // TODO: get rng passed in

        let graph_input = self.graph.load_graph_input(data).await?;

        Ok(graph_input)
    }

    pub fn generate_lime_samples(
        &self,
        graph_input: &Vec<Tensor<Fp>>,
    ) -> Result<Vec<GraphPoint>, ExplainError> {
        let mut rng = thread_rng();

        let shapes = self.graph.model().graph.input_shapes()?;
        let scales = self.graph.model().graph.get_input_scales();
        let input_types = self.graph.model().graph.get_input_types()?;

        // 1000 samples for now...
        let mut lime_samples = vec![];
        for i in 0..1000 {
            // TODO: should realy just perturb the quantized versions...
            // TODO: perturb using normal dist
            let perturbed_input =
                graph_input
                    .iter()
                    .zip(&scales)
                    .zip(&shapes)
                    .map(|((fp, scale), shape)| {
                        let t = fp
                            .iter()
                            .map(|f| dequantize(*f, *scale, 0.0))
                            .map(|f| f + rng.gen_range(-20..20) as f64)
                            .map(|f| i64_to_felt(quantize_float(&f, 0.0, *scale).unwrap()))
                            .collect::<Vec<_>>();

                        let mut t: Tensor<Fp> = t.into_iter().into();
                        t.reshape(shape).unwrap();
                        t
                    });
            lime_samples.push(perturbed_input.collect::<Vec<_>>());
        }
        Ok(lime_samples)
    }

    /// Runs the forward pass of the model / graph of computations and any associated hashing.
    pub fn forward<Scheme: CommitmentScheme<Scalar = Fp, Curve = G1Affine>>(
        &self,
        inputs: &mut ExplainInput,
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&Scheme::ParamsProver>,
        witness_gen: bool,
        check_lookup: bool,
    ) -> Result<(), ExplainError> {
        let visibility = VarVisibility::from_args(&self.graph.settings().run_args)?;
        //let mut processed_inputs = None;
        //let mut processed_params = None;
        //let mut processed_outputs = None;

        //if visibility.input.requires_processing() {
        //    let module_outlets = visibility.input.overwrites_inputs();
        //    if !module_outlets.is_empty() {
        //        let mut module_inputs = vec![];
        //        for outlet in &module_outlets {
        //            module_inputs.push(inputs[*outlet].clone());
        //        }
        //        let res =
        //            GraphModules::forward::<Scheme>(&module_inputs, &visibility.input, vk, srs)?;
        //        processed_inputs = Some(res.clone());
        //        let module_results = res.get_result(visibility.input.clone());

        //        for (i, outlet) in module_outlets.iter().enumerate() {
        //            inputs[*outlet] = Tensor::from(module_results[i].clone().into_iter());
        //        }
        //    } else {
        //        processed_inputs = Some(GraphModules::forward::<Scheme>(
        //            inputs,
        //            &visibility.input,
        //            vk,
        //            srs,
        //        )?);
        //    }
        //}

        //if visibility.params.requires_processing() {
        //    let params = self.model().get_all_params();
        //    if !params.is_empty() {
        //        let flattened_params = Tensor::new(Some(&params), &[params.len()])?.combine()?;
        //        processed_params = Some(GraphModules::forward::<Scheme>(
        //            &[flattened_params],
        //            &visibility.params,
        //            vk,
        //            srs,
        //        )?);
        //    }
        //}

        let mut model_results = self.graph.model().forward(
            &inputs.point,
            &self.graph.settings().run_args,
            witness_gen,
            check_lookup,
        )?;

        // TODO: batch sample points?
        let sample_outputs = inputs
            .samples
            .iter()
            .map(|sample| {
                self.graph
                    .model()
                    .forward(
                        &sample,
                        &self.graph.settings().run_args,
                        witness_gen,
                        check_lookup,
                    )
                    .unwrap()
                    .outputs
            })
            .collect::<Vec<_>>();

        let (intercept, coeffs) = self.compute_lasso(&inputs.samples, &sample_outputs)?;

        //ExplainWitness {

        //}

        //println!("GOT MODEL: {:?}", model);
        Ok(())

        //if visibility.output.requires_processing() {
        //    let module_outlets = visibility.output.overwrites_inputs();
        //    if !module_outlets.is_empty() {
        //        let mut module_inputs = vec![];
        //        for outlet in &module_outlets {
        //            module_inputs.push(model_results.outputs[*outlet].clone());
        //        }
        //        let res =
        //            GraphModules::forward::<Scheme>(&module_inputs, &visibility.output, vk, srs)?;
        //        processed_outputs = Some(res.clone());
        //        let module_results = res.get_result(visibility.output.clone());

        //        for (i, outlet) in module_outlets.iter().enumerate() {
        //            model_results.outputs[*outlet] =
        //                Tensor::from(module_results[i].clone().into_iter());
        //        }
        //    } else {
        //        processed_outputs = Some(GraphModules::forward::<Scheme>(
        //            &model_results.outputs,
        //            &visibility.output,
        //            vk,
        //            srs,
        //        )?);
        //    }
        //}

        //let mut witness = GraphWitness {
        //    inputs: original_inputs
        //        .iter()
        //        .map(|t| t.deref().to_vec())
        //        .collect_vec(),
        //    pretty_elements: None,
        //    outputs: model_results
        //        .outputs
        //        .iter()
        //        .map(|t| t.deref().to_vec())
        //        .collect_vec(),
        //    processed_inputs,
        //    processed_params,
        //    processed_outputs,
        //    max_lookup_inputs: model_results.max_lookup_inputs,
        //    min_lookup_inputs: model_results.min_lookup_inputs,
        //    max_range_size: model_results.max_range_size,
        //};

        //witness.generate_rescaled_elements(
        //    self.model().graph.get_input_scales(),
        //    self.model().graph.get_output_scales()?,
        //    visibility,
        //);

        //#[cfg(not(target_arch = "wasm32"))]
        //log::trace!(
        //    "witness: \n {}",
        //    &witness.as_json()?.to_colored_json_auto()?
        //);

        //Ok(witness)
    }

    // TODO: this function assumes the length of inputs is 1
    fn compute_lasso(
        &self,
        inputs: &[GraphPoint],
        outputs: &[GraphPoint],
    ) -> Result<(Fp, Tensor<Fp>), ExplainError> {
        let dequantize = |point: &GraphPoint, scales: &[crate::Scale]| {
            point
                .iter()
                .zip(scales)
                .map(|(t, scale)| {
                    t.iter()
                        .map(|f| dequantize(*f, *scale, 0.0))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()[0]
                .clone()
        };

        // dequantize inputs and outputs
        let input_scales = self.graph.model().graph.get_input_scales();
        let output_scales = self.graph.model().graph.get_output_scales()?;

        let inputs_float = inputs.iter().map(|point| dequantize(point, &input_scales));
        let outputs_float = outputs
            .iter()
            .map(|output| dequantize(output, &output_scales));

        // TODO: how to deal with multiple input shapes??
        let input_shapes = self.graph.model().graph.input_shapes()?;

        let inputs_linfa = Array::from_shape_vec(
            (inputs_float.len(), input_shapes[0][1]),
            inputs_float.flatten().collect::<Vec<_>>(),
        )
        .unwrap();
        let outputs_linfa = Array::from_shape_vec(
            outputs_float.len(),
            outputs_float.flatten().collect::<Vec<_>>(),
        )
        .unwrap();

        let data = Dataset::new(inputs_linfa, outputs_linfa);
        // train pure LASSO model with 0.3 penalty
        // TODO: customize params
        let model = ElasticNet::params()
            .penalty(0.3)
            .l1_ratio(1.0)
            .fit(&data)
            .unwrap();

        // Ensure kkt holds
        assert!(self.test_kkt(
            Tensor::from_ndarray(&data.records).unwrap(),
            Tensor::from_ndarray(&data.targets).unwrap(),
            model.intercept(),
            Tensor::from_ndarray(model.hyperplane()).unwrap(),
            0.3,
        ));

        // quantize the model back into Fp
        // Should be quantized into the input scale
        // outputs are always 0 or 1 anyway...
        let intercept =
            i64_to_felt(quantize_float(&model.intercept(), 0.0, input_scales[0]).unwrap());
        let hyperplane = model
            .hyperplane()
            .iter()
            .map(|c| i64_to_felt(quantize_float(c, 0.0, input_scales[0]).unwrap()));

        return Ok((intercept, hyperplane.into()));
    }

    fn test_kkt(
        &self,
        mut input: Tensor<f64>,
        label: Tensor<f64>,
        intercept: f64,
        mut coeffs: Tensor<f64>,
        alpha: f64,
    ) -> bool {
        let zero_threshold = 0.00001;
        let n = label.len() as f64;
        let ns = vec![n; coeffs.len()].into_iter().into();
        //println!("n: {}", n);
        let intercepts = vec![intercept; label.dims()[0]].into_iter().into();
        //println!("{:?}, {:?}", input.dims(), coeffs.dims());
        coeffs.reshape(&[coeffs.dims()[0], 1]).unwrap();
        let predictions = matmul(&[input.clone(), coeffs.clone()].try_into().unwrap()).unwrap();
        let deltas = ((label - intercepts).unwrap() - predictions).unwrap();
        let input_transpose = input.swap_axes(0, 1).unwrap();

        let intermediate = matmul(&[input_transpose, deltas].try_into().unwrap()).unwrap();
        // TODO: 0 case
        let coeff_sign = coeffs.map(|x| {
            if x > 0.0 {
                1.0
            } else {
                if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        });
        let zero_mask = coeffs.map(|x| if x == 0.0 { 0.0 } else { 1.0 });

        //let res = res - alpha * sign.map;
        let alphas: Tensor<f64> = vec![alpha; coeff_sign.dims()[0]].into_iter().into();

        let res = (intermediate.clone() / ns).unwrap() - (alphas * coeff_sign).unwrap();

        let test = res.clone().unwrap().map(|x| x < alpha);
        let test2 = (res.clone().unwrap() * zero_mask.clone())
            .unwrap()
            .map(|x| abs(x) < zero_threshold);

        //println!("res: {:?}, mask: {:?}", res, zero_mask);
        //println!("test: {:?}, test2: {:?}", test, test2);

        return test.iter().all(|x| *x) && test2.iter().all(|x| *x);
        //let test = (label - intercepts).unwrap() - ;
    }
}

//impl Circuit<Fp> for ExplainCircuit {
//    type Config = GraphConfig;
//    type FloorPlanner = ModulePlanner;
//    type Params = GraphSettings;
//}

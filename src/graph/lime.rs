use super::utilities::{dequantize, quantize_float};
use crate::circuit::modules::lime2::Lime2Chip;
use crate::fieldutils::{felt_to_i64, i64_to_felt};
use crate::graph::errors::GraphError;
use crate::tensor::ops::*;
use crate::tensor::{
    ops::accumulated::{dot, prod},
    ops::add,
    IntoI64, Tensor, TensorError, TensorType, ValTensor,
};
use crate::Scale;

use std::cmp::PartialOrd;
use std::ops::{Add, Mul, Range};

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

use halo2curves::bn256::{self, Fr as Fp, G1Affine};
use halo2curves::ff::PrimeField;
use serde::{Deserialize, Serialize};

use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use ndarray::{s, Array};
use num::abs;

// todo: maybe add some checks, ensure that original input not
//       included here
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LassoModel<F: TensorType> {
    pub local_surrogate: Tensor<F>,
    pub surrogate_samples: Tensor<F>,
    pub lasso_samples: Tensor<F>,
    pub coeffs: Tensor<F>,
    // TODO: do we give this publicly?
    pub intercept: F,
    pub dual: Tensor<F>,
}

impl<F: TensorType + PrimeField + PartialOrd + IntoI64> LassoModel<F> {
    //pub fn new(
    //    inputs: &Tensor<F>,
    //    outputs: &Tensor<F>,
    //    input_scale: &Scale,
    //    output_scale: &Scale,
    //    out_scale: &Scale,
    //) -> Self {
    //    // TODO: use dequantize tensor method
    //    let input_shape = inputs.dims();
    //    let output_shape = outputs.dims();

    //    let dequantize_tensor = |point: &Tensor<F>, scale: &Scale| {
    //        point
    //            .iter()
    //            .map(|t| dequantize(*t, *scale, 0.0))
    //            .collect::<Vec<_>>()
    //            .clone()
    //    };

    //    // dequantize inputs and outputs
    //    let inputs_float = dequantize_tensor(inputs, input_scale);
    //    let outputs_float = dequantize_tensor(outputs, output_scale);

    //    let inputs_linfa =
    //        Array::from_shape_vec((input_shape[0], input_shape[1]), inputs_float).unwrap();
    //    let outputs_linfa = Array::from_shape_vec(output_shape[0], outputs_float).unwrap();

    //    let data = Dataset::new(inputs_linfa, outputs_linfa);
    //    // train pure LASSO model with 0.3 penalty
    //    // TODO: customize params
    //    let model = ElasticNet::params()
    //        .penalty(0.3)
    //        .l1_ratio(1.0)
    //        .fit(&data)
    //        .unwrap();

    //    // Ensure kkt holds
    //    assert!(Self::test_kkt(
    //        Tensor::from_ndarray(&data.records).unwrap(),
    //        Tensor::from_ndarray(&data.targets).unwrap(),
    //        model.intercept(),
    //        Tensor::from_ndarray(model.hyperplane()).unwrap(),
    //        0.3,
    //    ));

    //    println!(
    //        "lime float is: {:?}, int: {:?}",
    //        model.hyperplane(),
    //        model.intercept()
    //    );

    //    // quantize the model back into Fp
    //    // Should be quantized into the input scale
    //    // outputs are always 0 or 1 anyway...
    //    let intercept = i64_to_felt(quantize_float(&model.intercept(), 0.0, *out_scale).unwrap());
    //    let hyperplane = model
    //        .hyperplane()
    //        .iter()
    //        .map(|c| i64_to_felt(quantize_float(c, 0.0, *out_scale).unwrap()));

    //    // dequantize inputs and outputs
    //    let hyperplane2 = dequantize_tensor(
    //        &Tensor::new(
    //            Some(&hyperplane.clone().collect::<Vec<_>>()),
    //            &[model.hyperplane().len()],
    //        )
    //        .unwrap(),
    //        out_scale,
    //    );
    //    let intercept2 = dequantize(intercept, *out_scale, 0.0);

    //    // Ensure kkt still holds
    //    assert!(Self::test_kkt(
    //        Tensor::from_ndarray(&data.records).unwrap(),
    //        Tensor::from_ndarray(&data.targets).unwrap(),
    //        intercept2,
    //        Tensor::new(Some(&hyperplane2), &[hyperplane2.len()]).unwrap(),
    //        0.3,
    //    ));

    //    Self::test_dual(
    //        Tensor::from_ndarray(&data.records).unwrap(),
    //        Tensor::from_ndarray(&data.targets).unwrap(),
    //        intercept2,
    //        Tensor::new(Some(&hyperplane2), &[hyperplane2.len()]).unwrap(),
    //        0.3,
    //    );

    //    return LassoModel {
    //        local_surrogate: vec![],
    //        samples: inputs.clone(),
    //        coeffs: hyperplane.collect::<Vec<_>>(),
    //        intercept,
    //    };
    //}

    pub fn find_local_surrogate<G, H>(
        classify: G,
        perturb: H,
        x: &Tensor<F>,
        input_scale: Scale,
        output_scale: Scale,
    ) -> Option<Tensor<F>>
    where
        G: Fn(Tensor<F>) -> Result<Tensor<F>, GraphError>,
        H: Fn(Tensor<F>) -> Tensor<F>,
    {
        if (!crate::USE_SURROGATE) {
            return None;
        }

        let classify_wrapper = |x: Tensor<f64>| {
            // quantize x
            let x = x.quantize(input_scale).unwrap();

            // TODO: handle different shapes...
            let label = classify(x).unwrap();

            //println!("LABEL: {:?}", label);
            // dequantize result
            let dequant = label.dequantize(output_scale);
            //println!("dequant: {:?}", dequant);
            dequant
        };

        let x_dequant = x.dequantize(input_scale);
        let target_class = 1.0 - classify_wrapper(x_dequant.clone())[0];
        //println!("GOT TARGET CLASS: {:?}", target_class);

        let local_surrogate = Self::find_closest_enemy(classify_wrapper, &x_dequant, target_class);
        //println!("LOCAL SURROGATE float: {:?}", local_surrogate);
        let local_surrogate = local_surrogate.quantize(input_scale).unwrap();
        //println!("LOCAL SURROGATE: {}", local_surrogate.show());

        //println!(
        //    "LOCAL SURROGATE INT: {:?}",
        //    local_surrogate
        //        .iter()
        //        .map(|v| felt_to_i64(*v))
        //        .collect::<Vec<_>>()
        //);
        //println!("x: {:?}", x);

        return Some(local_surrogate);
    }

    fn compute_weights(x_border: &Tensor<F>, inputs: &Tensor<F>) -> Tensor<F> {
        let d = x_border.dims()[1];
        let n = inputs.dims()[0];

        let test = inputs.clone();
        let x_border = x_border
            .enum_map::<_, _, TensorError>(|i, v| Ok(felt_to_i64(v)))
            .unwrap();
        let inputs = inputs
            .enum_map::<_, _, TensorError>(|i, v| Ok(felt_to_i64(v)))
            .unwrap();

        let mut x_expanded = x_border.clone();
        x_expanded.reshape(&[1, d]);
        x_expanded.expand(&[n, d]).unwrap();
        let deltas = (x_expanded.clone() - inputs.clone()).unwrap();
        // TODO: is this right??? do I need einsum?
        //
        let mut square_distance = vec![];
        for i in 0..n {
            let mut res = 0i64;
            for j in 0..d {
                let val = deltas.get(&[i, j]);
                res += val * val;
            }
            square_distance.push(res);
        }
        let square_distance = Tensor::new(Some(&square_distance), &[n]).unwrap();
        let square_distance = square_distance
            .enum_map::<_, _, Error>(|i, v| Ok(v / 2i64.pow(8)))
            .unwrap();
        //println!("SQUARE DIST: {:?}", square_distance);
        //println!("SQUARE DIST SCALED: {:?}", square_distance);

        let weights = crate::tensor::ops::nonlinearities::lime_weight(
            &square_distance,
            2f64.powf(8.0).into(),
            Lime2Chip::kernel_width(d).into(),
        );

        let sqrt_weights =
            crate::tensor::ops::nonlinearities::sqrt(&weights, 2f64.powf(8.0).into());

        sqrt_weights
            .enum_map::<_, _, Error>(|i, v| Ok(i64_to_felt(v)))
            .unwrap()
    }

    pub fn lasso(
        x: &Tensor<F>,
        inputs: &Tensor<F>,
        outputs: &Tensor<F>,
        input_scale: Scale,
        output_scale: Scale,
        model_scale: Scale,
        k: usize,
    ) -> (Tensor<F>, Tensor<F>, Tensor<F>, F, Tensor<F>) {
        let sqrt_weights = Self::compute_weights(x, inputs);
        //println!("weights: {:?}", sqrt_weights);

        let inputs = mult(&[inputs.clone(), sqrt_weights.clone()]).unwrap();
        let outputs = mult(&[outputs.clone(), sqrt_weights.clone()]).unwrap();

        let inputs_float = inputs.dequantize(input_scale + 8);
        let outputs_float = outputs.dequantize(output_scale + 8);
        //println!("INPUTS ARE: {:?}", inputs_float);
        //println!("OUTPUTS ARE: {:?}", outputs_float);

        let input_shape = inputs.dims();
        let output_shape = outputs.dims();

        // TODO: compute the kernel here... and use modified lime algorithm
        //println!("Inputs: {:?}", inputs_float.to_vec());
        //println!("outputs: {:?}", outputs_float.to_vec());
        let inputs_linfa =
            Array::from_shape_vec((input_shape[0], input_shape[1]), inputs_float.to_vec()).unwrap();
        let outputs_linfa = Array::from_shape_vec(output_shape[0], outputs_float.to_vec()).unwrap();

        let data = Dataset::new(inputs_linfa.clone(), outputs_linfa.clone());
        // train pure LASSO model with 0.3 penalty
        // TODO: customize params
        // TODO: ensure penalty is quantizable...
        // TODO(EVAN): what should the penalty be?
        let model = ElasticNet::params()
            .penalty(0.01)
            .l1_ratio(1.0)
            .max_iterations(10000)
            .tolerance(1e-8)
            .fit(&data)
            .unwrap();

        // TODO: wackyness...
        let dual = if model.hyperplane().iter().all(|x| *x == 0.0) {
            inputs_linfa.dot(model.hyperplane())
        } else {
            let dual: Array<f64, _> =
                (outputs_linfa.clone() - model.intercept() - inputs_linfa.dot(model.hyperplane()))
                    / (inputs.dims()[0] as f64);

            // TODO: other dual alg?
            //println!("DUAL1: {:?}", dual);
            let test = inputs_linfa.t().dot(&inputs_linfa);
            let test2 = test.dot(model.hyperplane());
            let mut ss = (0..test.shape()[0])
                .map(|i| 0.01 / (2.0 * (test2[i] - 2.0 * (outputs_linfa[i])).abs()))
                .fold(f64::INFINITY, |a, b| a.min(b));
            if ss == f64::INFINITY {
                ss = 0.0;
            }
            2.0 * ss * (inputs_linfa.dot(model.hyperplane()) - (outputs_linfa))
            //dual
        };

        let hyperplane = model.hyperplane().to_vec();
        let dual: Vec<f64> = dual.to_vec();
        println!("hyperplane: {:?}", hyperplane);
        println!("intercept: {:?}", model.intercept());
        println!("dual: {:?}", dual);

        // TODO(EVAN): could there be issues with rounding? No right?
        let mut top_k = hyperplane.clone();
        top_k.sort_by(|a, b| abs(*a).partial_cmp(&abs(*b)).unwrap());
        let top_k = top_k[hyperplane.len() - k..].to_vec();
        let mut top_k_idx = (0..hyperplane.len()).collect::<Vec<_>>();
        top_k_idx.sort_by(|a, b| {
            abs(hyperplane[*a])
                .partial_cmp(&abs(hyperplane[*b]))
                .unwrap()
        });
        let top_k_idx = top_k_idx[hyperplane.len() - k..].to_vec();
        //println!("top_k: {:?}", top_k);
        //println!("top_k_idx: {:?}", top_k_idx);
        let coeffs = Tensor::new(Some(&hyperplane), &[hyperplane.len()])
            .unwrap()
            .quantize(model_scale)
            .unwrap();
        let top_k = Tensor::new(Some(&top_k), &[top_k.len()])
            .unwrap()
            .quantize(model_scale)
            .unwrap();
        //println!("COEFFS: {:?}", coeffs.show());
        //println!("topk: {:?}", top_k.show());
        let top_k_idx = Tensor::new(Some(&top_k_idx), &[top_k_idx.len()])
            .unwrap()
            .enum_map::<_, _, Error>(|i, v| Ok(i64_to_felt(v as i64)))
            .unwrap();
        let intercept =
            i64_to_felt::<F>(quantize_float(&model.intercept(), 0.0, model_scale).unwrap());
        let dual = Tensor::new(Some(&dual), &[dual.len()])
            .unwrap()
            .quantize(16)
            .unwrap();
        //.quantize(model_scale)
        //.unwrap();

        //println!(
        //    "Inputs int: {:?}",
        //    inputs.iter().map(|v| felt_to_i64(*v)).collect::<Vec<_>>()
        //);
        //println!(
        //    "Outputs int: {:?}",
        //    outputs.iter().map(|v| felt_to_i64(*v)).collect::<Vec<_>>()
        //);
        //println!(
        //    "hyperplane int: {:?}",
        //    coeffs.iter().map(|v| felt_to_i64(*v)).collect::<Vec<_>>()
        //);
        //println!("intercept int: {:?}", felt_to_i64(intercept));
        //println!(
        //    "dual int: {:?}",
        //    dual.iter().map(|v| felt_to_i64(*v)).collect::<Vec<_>>()
        //);
        //println!(
        //    "topk int: {:?}",
        //    top_k.iter().map(|v| felt_to_i64(*v)).collect::<Vec<_>>()
        //);
        println!("coeffs: {:?}", coeffs);
        println!("intercept: {:?}", model.intercept());
        println!("dual: {:?}", dual);

        (coeffs, top_k, top_k_idx, intercept, dual)
    }

    pub fn find_closest_enemy<G>(classify: G, x: &Tensor<f64>, target_class: f64) -> Tensor<f64>
    where
        G: Fn(Tensor<f64>) -> Tensor<f64>,
    {
        // TODO: set these reasonably...
        let MU = 5.0;
        let N = 1000;

        let mut n_enemies = 1000;
        let mut closest_enemy = None;
        let mut radius = MU;

        //println!("shrinking...");
        while n_enemies > 0 {
            (n_enemies, _) =
                Self::find_enemies_in_layer(&classify, x, target_class, radius, 0.0, N, true);
            radius = radius / 2.0; // TODO: change 2 to be tunable?
                                   // TODO: logging
            println!("radius: {:?} (enemies: {})", radius, n_enemies);
        }

        println!("growing...");
        while n_enemies <= 0 {
            let step = radius / 2.0;

            let (n, enemy) =
                Self::find_enemies_in_layer(&classify, x, target_class, radius, step, N, false);
            radius = radius + step;
            println!("radius: {:?}-{:?} (enemies: {})", radius, radius + step, n);

            n_enemies = n;
            closest_enemy = enemy;
        }

        return closest_enemy.unwrap();

        //let z = sample_spherical_layer(x, 0, NAMBLA);
    }

    // TODO: add max and min to prevent growing forever?
    // Returns a tuple (n, enemy) where n is the number of enemies found in the layer, and enemy is
    // the shortest one
    fn find_enemies_in_layer<G>(
        classify: G,
        x: &Tensor<f64>,
        target_class: f64,
        r: f64,
        step: f64,
        n: usize,
        first_layer: bool,
    ) -> (usize, Option<Tensor<f64>>)
    where
        G: Fn(Tensor<f64>) -> Tensor<f64>,
    {
        let d = x.dims()[1];

        // TODO: cannot use sphere... should reallyl try to get the ring version working...
        //       if not...sphere is better.
        //       try test_sample ...
        let layer = if first_layer {
            Self::sample_ball(x, r, n)
        } else {
            Self::sample_ring(x, r, r + step, n)
        };

        let labels: Vec<f64> = classify(layer.clone()).to_vec();

        // TODO: can I just return the min emenm
        let n_enemies = labels
            .to_vec()
            .into_iter()
            .filter(|v| *v == target_class)
            .count();

        let mut shortest_enemy = None;
        let mut shortest_distance = 0.0;
        for i in 0..n {
            let enemy = layer.get_slice(&[i..i + 1, 0..d]).unwrap();
            if labels[i] != target_class {
                continue;
            }
            let distance = l2(x, &enemy)[0];

            if shortest_enemy.is_none() || distance < shortest_distance {
                shortest_enemy = Some(enemy);
                shortest_distance = distance;
            }
        }

        return (n_enemies, shortest_enemy);
    }

    // Code adapted from
    // https://github.com/thibaultlaugel/growingspheres/blob/894ce969f39f4f927f1732e569fb5e7ac6f770a9/growingspheres/utils/gs_utils.py#L23
    fn sample_ball(center: &Tensor<f64>, r: f64, n: usize) -> Tensor<f64> {
        let d = center.dims()[1];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let u = normal
            .sample_iter(&mut thread_rng())
            .take(n * (d + 2))
            .collect::<Vec<f64>>();
        let u = Tensor::new(Some(&u), &[n, d + 2]).unwrap();
        let norms = norm(&u);

        // TODO: won't work cause shapes...
        let u = (u / norms).unwrap();

        let x = (u.get_slice(&[0..n, 0..d]).unwrap() * r).unwrap();
        let x = x + center.clone();

        x.unwrap()
    }

    fn sample_sphere(center: &Tensor<f64>, r: f64, n: usize) -> Tensor<f64> {
        let d = center.dims()[1];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let u = normal
            .sample_iter(&mut thread_rng())
            .take(n * (d))
            .collect::<Vec<f64>>();
        let u = Tensor::new(Some(&u), &[n, d]).unwrap();
        let norms = norm(&u);
        let u = (u / norms).unwrap();
        let z = (u * r).unwrap() + center.clone();

        z.unwrap()
    }

    fn sample_ring(center: &Tensor<f64>, min: f64, max: f64, n: usize) -> Tensor<f64> {
        let d = center.dims()[1];
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::from(min.powf(d as f64)..max.powf(d as f64));
        let z = normal
            .sample_iter(&mut thread_rng())
            .take(n * d)
            .collect::<Vec<f64>>();
        let z = Tensor::new(Some(&z), &[n, d]).unwrap();
        let u = uniform
            .sample_iter(&mut thread_rng())
            .take(n)
            .map(|v| v.powf(1.0 / (d as f64)))
            .collect::<Vec<f64>>();
        let u = Tensor::new(Some(&u), &[n]).unwrap();
        let z = (z.clone() * u).unwrap() / norm(&z);
        let z = z.unwrap() + center.clone();

        z.unwrap()
    }

    pub fn test_dual(
        mut input: Tensor<f64>,
        label: Tensor<f64>,
        intercept: f64,
        mut coeffs: Tensor<f64>,
        alpha: f64,
    ) -> bool {
        println!("DUAL----------------------------");
        let n = label.len() as f64;
        let ns = vec![n; label.len()].into_iter().into();
        //println!("n: {}", n);
        let intercepts = vec![intercept; label.dims()[0]].into_iter().into();
        //println!("{:?}, {:?}", input.dims(), coeffs.dims());
        coeffs.reshape(&[coeffs.dims()[0], 1]).unwrap();
        let predictions = matmul(&[input.clone(), coeffs.clone()].try_into().unwrap()).unwrap();

        println!("preds is: {:?}", predictions);
        let deltas = ((label.clone() - intercepts).unwrap() - predictions).unwrap();

        println!("deltas is: {:?}", deltas);
        println!("ns is: {:?}", ns);
        let dual = (deltas.clone() / ns).unwrap();

        let delta_squared = dot(&[deltas.clone(), deltas.clone()], 1).unwrap();
        let delta_squared = delta_squared[delta_squared.len() - 1] / (2f64 * n);

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

        let primal_objective = delta_squared
            + alpha * dot(&[coeffs.clone(), coeff_sign], 1).unwrap()[coeffs.len() - 1];
        println!("GOT PRIMAL OBJ: {:?}", primal_objective);

        let dual_objective = dot(&[dual.clone(), dual.clone()], 1).unwrap()[dual.len() - 1]
            / (2f64 * n)
            + dot(&[dual.clone(), (label.clone() - dual.clone()).unwrap()], 1).unwrap()
                [dual.len() - 1];
        println!("GOT DUAL OBJ: {:?}", dual_objective);

        let gap = primal_objective - dual_objective;
        let abs_gap = abs(gap);
        println!("GOT GAP: {:?}", gap);

        // check dual is feasible
        let n = label.len() as f64;
        let ns: Tensor<f64> = vec![n; coeffs.len()].into_iter().into();
        //println!("n: {}", n);
        let intercepts = vec![intercept; label.dims()[0]].into_iter().into();
        //println!("{:?}, {:?}", input.dims(), coeffs.dims());
        coeffs.reshape(&[coeffs.dims()[0], 1]).unwrap();
        let predictions = matmul(&[input.clone(), coeffs.clone()].try_into().unwrap()).unwrap();

        println!("preds is: {:?}", predictions);
        let deltas = ((label - intercepts).unwrap() - predictions).unwrap();
        let input_transpose = input.swap_axes(0, 1).unwrap();
        let tests = matmul(&[input_transpose, dual].try_into().unwrap()).unwrap();
        let abs_tests = tests.map(|t| abs(t));
        let test = abs_tests.clone().map(|x| x <= alpha);

        println!("test constraints: {:?}, got: {:?}", abs_tests, test);

        // TODO: reasonable gap?
        abs_gap <= 0.1 && test.iter().all(|x| *x)
    }

    pub fn test_kkt(
        mut input: Tensor<f64>,
        label: Tensor<f64>,
        intercept: f64,
        mut coeffs: Tensor<f64>,
        alpha: f64,
    ) -> bool {
        println!("TEST_KKT ---------------");
        println!("inputs: {:?}", input);
        println!("label: {:?}", label);
        println!("coeffs: {:?}", coeffs);
        println!("intercept: {:?}", intercept);
        // TODO: tolerance...
        let zero_threshold = 0.1;
        let n = label.len() as f64;
        let ns = vec![n; coeffs.len()].into_iter().into();
        //println!("n: {}", n);
        let intercepts = vec![intercept; label.dims()[0]].into_iter().into();
        //println!("{:?}, {:?}", input.dims(), coeffs.dims());
        coeffs.reshape(&[coeffs.dims()[0], 1]).unwrap();
        let predictions = matmul(&[input.clone(), coeffs.clone()].try_into().unwrap()).unwrap();

        println!("preds is: {:?}", predictions);
        let deltas = ((label - intercepts).unwrap() - predictions).unwrap();
        let input_transpose = input.swap_axes(0, 1).unwrap();

        println!("deltas is: {:?}", deltas);
        let intermediate = matmul(&[input_transpose, deltas].try_into().unwrap()).unwrap();
        println!("einsum transose is: {:?}", intermediate);
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

        let res = (intermediate.clone() / ns).unwrap();
        println!("div is: {:?}", res);
        let res = res - (alphas * coeff_sign).unwrap();

        let test = res.clone().unwrap().map(|x| abs(x) <= alpha);
        let test2 = (res.clone().unwrap() * zero_mask.clone())
            .unwrap()
            .map(|x| abs(x) < zero_threshold);

        println!("res: {:?}, mask: {:?}", res, zero_mask);
        println!("test: {:?}, test2: {:?}", test, test2);

        return test.iter().all(|x| *x) && test2.iter().all(|x| *x);
        //let test = (label - intercepts).unwrap() - ;
    }
}

pub fn matmul<F: TensorType + Mul<Output = F> + Add<Output = F>>(
    inputs: &[Tensor<F>; 2],
) -> Result<Tensor<F>, TensorError> {
    // just do naive for now...
    let left = &inputs[0];
    let right = &inputs[1];

    if left.dims()[1] != right.dims()[0] {
        return Err(TensorError::DimMismatch("matmul".to_string()));
    }

    let mut output = Tensor::<F>::new(None, &[left.dims()[0], right.dims()[1]])?;
    for row in 0..left.dims()[0] {
        for col in 0..right.dims()[1] {
            for i in 0..left.dims()[1] {
                let val = output.get(&[row, col]) + left.get(&[row, i]) * right.get(&[i, col]);
                *output.get_mut(&[row, col]) = val;
            }
        }
    }

    Ok(output)
}

// Compute norm across rows of x...
fn norm(x: &Tensor<f64>) -> Tensor<f64> {
    let mut res = vec![];
    for r in 0..x.dims()[0] {
        let mut magnitude = 0.0;
        for c in 0..x.dims()[1] {
            let val = x.get(&[r, c]);
            magnitude += val * val;
        }
        res.push(magnitude.sqrt());
    }

    Tensor::new(Some(&res), &[x.dims()[0]]).unwrap()
}

// L2 distance between rows of x and rows of y
fn l2(x: &Tensor<f64>, y: &Tensor<f64>) -> Tensor<f64> {
    norm(&(x.clone() - y.clone()).unwrap())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_sample_ball() {
        let center = Tensor::new(Some(&vec![1.0, 2.0, 3.0]), &[1, 3]).unwrap();
        let samples = LassoModel::<Fp>::sample_ball(&center, 5.0, 1000);

        assert_eq!(samples.dims(), &[1000, 3]);

        for i in 0..samples.dims()[0] {
            let test = samples.get_slice(&[i..i + 1, 0..3]).unwrap() - center.clone();
            let n = norm(&test.unwrap());
            assert_eq!(n.dims(), &[1]);
            assert!(n[0] < 5.0);
        }
        println!("GOT SAMPLES: {:?}", samples.show());
    }

    #[test]
    pub fn test_sample_sphere() {
        let center = Tensor::new(Some(&vec![1.0, 2.0, 3.0]), &[1, 3]).unwrap();
        let samples = LassoModel::<Fp>::sample_sphere(&center, 5.0, 1000);

        assert_eq!(samples.dims(), &[1000, 3]);

        for i in 0..samples.dims()[0] {
            let test = samples.get_slice(&[i..i + 1, 0..3]).unwrap() - center.clone();
            let n = norm(&test.unwrap());
            assert_eq!(n.dims(), &[1]);
            // check with some tolerance
            assert!(n[0] >= 4.999 && n[0] <= 5.0001);
        }
        println!("GOT SAMPLES: {:?}", samples.show());
    }

    #[test]
    pub fn test_sample_ring() {
        let center = Tensor::new(Some(&vec![1.0, 2.0, 3.0]), &[1, 3]).unwrap();
        let samples = LassoModel::<Fp>::sample_ring(&center, 5.0, 7.0, 1000);

        assert_eq!(samples.dims(), &[1000, 3]);

        for i in 0..samples.dims()[0] {
            let test = samples.get_slice(&[i..i + 1, 0..3]).unwrap() - center.clone();
            let n = norm(&test.unwrap());
            assert_eq!(n.dims(), &[1]);
            //// check with some tolerance
            assert!(n[0] >= 4.999 && n[0] <= 7.0001);
        }
        println!("GOT SAMPLES: {:?}", samples.show());
    }

    #[test]
    pub fn test_find_closest_enemy() {
        let center = Tensor::new(Some(&vec![1.0, 2.0, 3.0]), &[1, 3]).unwrap();
        let classifier = |x: Tensor<f64>| {
            let n = x.dims()[0];
            let d = x.dims()[1];

            let mut res = vec![];
            for i in 0..n {
                let slice = x.get_slice(&[i..i + 1, 0..d]).unwrap();
                if l2(&slice, &center)[0] > 80.0 {
                    res.push(1.0);
                } else {
                    res.push(0.0);
                }
            }

            Tensor::new(Some(&res), &[n]).unwrap()
        };

        let closest_enemy = LassoModel::<Fp>::find_closest_enemy(classifier, &center, 0.0);
        println!("GOT ENEMY: {:?}", closest_enemy);
        println!("dist: {:?}", l2(&closest_enemy, &center)[0]);
    }
}

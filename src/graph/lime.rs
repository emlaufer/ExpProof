use super::utilities::{dequantize, quantize_float};
use crate::fieldutils::{felt_to_i64, i64_to_felt};
use crate::graph::errors::GraphError;
use crate::graph::LimeCircuit;
use crate::tensor::ops::*;
use crate::tensor::{
    ops::accumulated::{dot, prod},
    ops::add,
    IntoI64, Tensor, TensorError, TensorType, ValTensor,
};
use crate::Scale;

use crate::LimeWeightStrategy;
use std::cmp::PartialOrd;
use std::ops::{Add, Mul, Range};

use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};

use halo2curves::bn256::{self, Fr as Fp, G1Affine};
use halo2curves::ff::PrimeField;
use serde::{Deserialize, Serialize};

use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;
use linfa_linalg::norm::Norm;
use ndarray::{
    s, Array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, CowArray, Data,
    Dimension, Ix2, RemoveAxis,
};
use num::abs;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LassoModel<F: TensorType> {
    pub local_surrogate: Tensor<F>,
    pub surrogate_samples: Tensor<F>,
    pub lasso_samples: Tensor<F>,
    pub coeffs: Tensor<F>,
    pub intercept: F,
    pub dual: Tensor<F>,
}

impl<F: TensorType + PrimeField + PartialOrd + IntoI64> LassoModel<F> {
    const ALPHA: f64 = 0.010009765625;

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
        let classify_wrapper = |x: Tensor<f64>| {
            // quantize x
            let x = x.quantize(input_scale).unwrap();

            let label = classify(x).unwrap();

            let dequant = label.dequantize(output_scale);
            dequant
        };

        let x_dequant = x.dequantize(input_scale);
        let target_class = 1.0 - classify_wrapper(x_dequant.clone())[0];

        let local_surrogate = Self::find_closest_enemy(classify_wrapper, &x_dequant, target_class);
        let local_surrogate = local_surrogate.quantize(input_scale).unwrap();

        return Some(local_surrogate);
    }

    fn compute_weights(
        x_border: &Tensor<F>,
        inputs: &Tensor<F>,
        weight_strat: LimeWeightStrategy,
    ) -> Tensor<F> {
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

        let mut weights = square_distance.clone();
        if matches!(weight_strat, LimeWeightStrategy::Exponential) {
            weights = crate::tensor::ops::nonlinearities::lime_weight(
                &square_distance,
                2f64.powf(8.0).into(),
                LimeCircuit::kernel_width(d).into(),
            );
        }

        let sqrt_weights =
            crate::tensor::ops::nonlinearities::sqrt(&weights, 2f64.powf(8.0).into());

        sqrt_weights
            .enum_map::<_, _, Error>(|i, v| Ok(i64_to_felt(v)))
            .unwrap()
    }

    pub fn primal_objective<
        T: TensorType
            + Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + std::cmp::PartialOrd
            + Neg<Output = T>
            + std::marker::Send
            + std::marker::Sync
            + IntoI64,
    >(
        intercept: &Tensor<T>,
        y: &Tensor<T>,
        x: &Tensor<T>,
        lime_model: &Tensor<T>,
        alpha: &T,
        multiplier: &T, // TODO: remove this param
        n_samples: &T,
        scale: bool,
    ) -> T {
        use crate::tensor::ops::*;

        let n = y.len() as f64;
        let y_moved = (y.clone() - intercept.expand(&[y.len()]).unwrap()).unwrap();
        let mut lime_model = lime_model.clone();
        lime_model.reshape(&[lime_model.len(), 1]);
        //let lime_model_t = lime_model.reshape(&[);
        let m = matmul(&[x.clone(), lime_model.clone()]).unwrap();

        let residuals = (y_moved - m).unwrap();
        let residuals_square = dot(&[residuals.clone(), residuals.clone()], 1).unwrap();
        let residuals_square = residuals_square[residuals_square.len() - 1].clone();
        let mut residual_mult = multiplier.clone() * residuals_square.clone();

        let mut abs_model = abs(&lime_model.clone()).unwrap();
        let l1_model = accumulated::sum(&abs_model, 1).unwrap();
        let l1_model = l1_model[l1_model.len() - 1].clone();
        let mut l1_alpha = alpha.clone() * l1_model;
        let mut l1_alpha = n_samples.clone() * l1_alpha;

        let primal_obj = residual_mult + l1_alpha;

        primal_obj
    }

    pub fn dual_objective<
        T: TensorType
            + Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + std::marker::Send
            + std::marker::Sync
            + IntoI64,
    >(
        dual: &Tensor<T>,
        intercept: &Tensor<T>,
        y: &Tensor<T>,
        multiplier: &T, // TODO: remove this param
        scale: bool,
    ) -> T {
        use crate::tensor::ops::*;

        let n = y.len() as f64;
        let dual_squared = dot(&[dual.clone(), dual.clone()], 1).unwrap();
        let mut dual_squared_val = dual_squared[dual_squared.len() - 1].clone();
        if scale {
            dual_squared_val = T::from_i64(
                nonlinearities::const_div(
                    &Tensor::new(Some(&[dual_squared_val.into_i64()]), &[1]).unwrap(),
                    2f64.powf(16.0),
                )[0],
            );
        }

        let y_moved = (y.clone() - intercept.expand(&[y.len()]).unwrap()).unwrap();
        let dual_dot_y = dot(&[dual.clone(), y_moved.clone()], 1).unwrap();
        let mut dual_dot_y_val = dual_dot_y[dual_dot_y.len() - 1].clone();
        if scale {
            dual_dot_y_val = T::from_i64(
                nonlinearities::const_div(
                    &Tensor::new(Some(&[dual_dot_y_val.into_i64()]), &[1]).unwrap(),
                    2f64.powf(16.0),
                )[0],
            );
        }

        let mut dual_mult = multiplier.clone() * dual_squared_val.clone();
        if scale {
            dual_mult = T::from_i64(
                nonlinearities::const_div(
                    &Tensor::new(Some(&[dual_mult.into_i64()]), &[1]).unwrap(),
                    2f64.powf(24.0),
                )[0],
            );
        }
        let dual_gap = dual_mult - dual_dot_y_val.clone();
        dual_gap
    }

    pub fn dual_feasible(x: &Tensor<f64>, dual: &Tensor<f64>) -> bool {
        let mut dual = dual.clone();
        dual.reshape(&[dual.len(), 1]);
        let m = matmul(&[x.clone(), dual.clone()]).unwrap();
        let range_check_bracket = (Self::ALPHA);
        // less than
        let lt_range = m
            .par_enum_map(|_, a_i| {
                Ok::<_, TensorError>(if (a_i as f64 - range_check_bracket) < 0_f64 {
                    1.0f64
                } else {
                    0.0f64
                })
            })
            .unwrap();
        let lt_range = prod(&lt_range, 1).unwrap();
        let lt_range = lt_range[lt_range.len() - 1].clone();
        let gt_range = m
            .par_enum_map(|_, a_i| {
                Ok::<_, TensorError>(if (a_i as f64 + range_check_bracket) > 0_f64 {
                    1.0f64
                } else {
                    0.0f64
                })
            })
            .unwrap();
        let gt_range = prod(&gt_range, 1).unwrap();
        let gt_range = gt_range[gt_range.len() - 1].clone();
        if gt_range * lt_range == 1.0 {
            true
        } else {
            false
        }
    }

    pub fn dual_grad(
        dual: &Tensor<f64>,
        intercept: &Tensor<f64>,
        y: &Tensor<f64>,
        multiplier: &f64, // TODO: remove this param
        scale: bool,
    ) -> Tensor<f64> {
        let test = Tensor::new(Some(&[(multiplier * 2.0)]), &[1])
            .unwrap()
            .expand(&[dual.len()])
            .unwrap()
            * dual.clone();
        let y_moved = (y.clone() - intercept.expand(&[y.len()]).unwrap()).unwrap();
        let grad = test.unwrap() + y_moved;

        grad.unwrap()
    }

    fn duality_gap<'a>(
        x: ArrayView2<'a, f64>,

        y: ArrayView1<'a, f64>,

        w: ArrayView1<'a, f64>,

        l1_ratio: f64,

        penalty: f64,
    ) -> f64 {
        let half = 0.5;

        let r = (&y - x.dot(&w));

        let n_samples = x.nrows();

        let l1_reg = l1_ratio * penalty * (n_samples as f64);

        let l2_reg = (1.0 - l1_ratio) * penalty * (n_samples as f64);

        let xta = x.t().dot(&r) - &w * l2_reg;

        let dual_norm_xta = xta.norm_max();

        let r_norm2 = r.dot(&r);

        let w_norm2 = w.dot(&w);

        let (const_, mut gap) = if dual_norm_xta > l1_reg {
            let const_ = (l1_reg / dual_norm_xta);

            let a_norm2 = r_norm2 * const_ * const_;

            (const_, half * (r_norm2 + a_norm2))
        } else {
            (1.0, r_norm2)
        };

        let l1_norm = w.norm_l1();

        gap += l1_reg * l1_norm - const_ * r.dot(&y)
            + half * l2_reg * (1.0 + const_ * const_) * w_norm2;

        let r_test = r.clone() * const_;
        let r_norm2_test = r_test.dot(&r_test);
        let primal_obj = half * (r_norm2) + l1_reg * l1_norm;
        let dual_obj = -half * (r_norm2) - (-r).dot(&y);
        let dual_obj2 = -half * (r_norm2_test) - (-r_test).dot(&y);

        gap
    }

    pub fn lasso(
        x: &Tensor<F>,
        inputs: &Tensor<F>,
        outputs: &Tensor<F>,
        input_scale: Scale,
        output_scale: Scale,
        model_scale: Scale,
        weight_strat: LimeWeightStrategy,
        k: usize,
    ) -> (Tensor<F>, Tensor<F>, Tensor<F>, F, Tensor<F>) {
        let mut input_scale = input_scale;
        let mut output_scale = output_scale;
        let mut inputs = inputs.clone();
        let mut outputs = outputs.clone();

        if !matches!(weight_strat, LimeWeightStrategy::Uniform) {
            let sqrt_weights = Self::compute_weights(x, &inputs, weight_strat);

            inputs = mult(&[inputs.clone(), sqrt_weights.clone()]).unwrap();
            outputs = mult(&[outputs.clone(), sqrt_weights.clone()]).unwrap();
            input_scale += 8;
            output_scale += 8;
        }

        let inputs_float = inputs.dequantize(input_scale);
        let outputs_float = outputs.dequantize(output_scale);

        let input_shape = inputs.dims();
        let output_shape = outputs.dims();

        let inputs_linfa =
            Array::from_shape_vec((input_shape[0], input_shape[1]), inputs_float.to_vec()).unwrap();
        let outputs_linfa = Array::from_shape_vec(output_shape[0], outputs_float.to_vec()).unwrap();

        let data = Dataset::new(inputs_linfa.clone(), outputs_linfa.clone());
        let model = ElasticNet::params()
            .penalty(Self::ALPHA)
            .l1_ratio(1.0)
            .max_iterations(100000)
            .tolerance(1e-12)
            .fit(&data)
            .unwrap();
        let intercept_arr = Array::from_shape_vec((1), vec![model.intercept()]).unwrap();

        let dual = if model.hyperplane().iter().all(|x| *x == 0.0) {
            inputs_linfa.dot(model.hyperplane())
        } else {
            let mut dual: Array<f64, _> =
                (outputs_linfa.clone() - model.intercept() - inputs_linfa.dot(model.hyperplane()));
            dual = dual.map(|v| {
                dequantize(
                    i64_to_felt::<Fp>(quantize_float(v, 0.0, 16).unwrap()),
                    16,
                    0.0,
                )
            });
            let dual_norm_xta = (inputs_linfa.t().dot(&dual)).norm_max();
            // add some slack for precision issues
            let l1_reg = 1.0 * Self::ALPHA * (outputs_linfa.len() as f64);
            if dual_norm_xta > l1_reg {
                let const_ = (l1_reg / (dual_norm_xta));
                dual *= -const_;
            } else {
                dual *= -1.0;
            }
            dual = dual.map(|v| {
                dequantize(
                    i64_to_felt::<Fp>(quantize_float(v, 0.0, 16).unwrap()),
                    16,
                    0.0,
                )
            });
            dual
        };

        let hyperplane = model.hyperplane().to_vec();
        let dual: Vec<f64> = dual.to_vec();

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
        let coeffs = Tensor::new(Some(&hyperplane), &[hyperplane.len()])
            .unwrap()
            .quantize(model_scale)
            .unwrap();
        let top_k = Tensor::new(Some(&top_k), &[top_k.len()])
            .unwrap()
            .quantize(model_scale)
            .unwrap();
        let top_k_idx = Tensor::new(Some(&top_k_idx), &[top_k_idx.len()])
            .unwrap()
            .enum_map::<_, _, Error>(|i, v| Ok(i64_to_felt(v as i64)))
            .unwrap();
        let intercept =
            i64_to_felt::<F>(quantize_float(&model.intercept(), 0.0, model_scale).unwrap());

        let mult = -1.0 / 2.0;
        let mut dualf = Tensor::new(Some(&dual), &[dual.len()]).unwrap();
        let interceptf = Tensor::new(Some(&[model.intercept()]), &[1]).unwrap();
        let alphaf = Tensor::new(Some(&[model.intercept()]), &[1]).unwrap();
        let hyperplanef = coeffs.dequantize(model_scale);
        let mut obj1 = Self::dual_objective(&dualf, &interceptf, &outputs_float, &mult, false);
        let mult_primal = 1.0 / (2.0);
        let mult_primal_nsamples = outputs_float.len() as f64;
        let mut pobj = Self::primal_objective(
            &interceptf,
            &outputs_float,
            &inputs_float,
            &hyperplanef,
            &Self::ALPHA,
            &mult_primal,
            &mult_primal_nsamples,
            false,
        );

        let dualf = dualf.quantize_f64(16).unwrap();
        let interceptf = interceptf.quantize_f64(12).unwrap();
        let outputs_floatf = outputs_float.quantize_f64(12).unwrap();
        Self::dual_objective(&dualf, &interceptf, &outputs_floatf, &mult, false);

        let dual = dualf.quantize(16).unwrap();
        let mult2 = i64_to_felt::<F>(
            quantize_float(&(-(outputs_float.len() as f64) / 2.0), 0.0, 16).unwrap(),
        );
        let dg = Self::dual_objective(
            &dual,
            &Tensor::new(Some(&[intercept]), &[1]).unwrap(),
            &outputs,
            &mult2,
            true,
        );
        (coeffs, top_k, top_k_idx, intercept, dual)
    }

    /// Old function used for spheres
    /// We don't use this anymore...
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

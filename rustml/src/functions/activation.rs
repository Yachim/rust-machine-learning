use crate::functions::weight_init::{InitFunc, HE, XAVIER};
use std::f32::consts;

pub struct IndependentFunc<'a> {
    pub func: &'a dyn Fn(f32) -> f32,
    pub derivative: &'a dyn Fn(f32) -> f32,
}

pub struct DependentFunc<'a> {
    pub func: &'a dyn Fn(&Vec<f32>) -> Vec<f32>,
    pub derivative: &'a dyn Fn(&Vec<f32>) -> Vec<Vec<f32>>,
}

/// element-wise independent functions and element-wise dependent functions
///  - https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html
///
/// independent functions and their derivatives take a float and return a float
///
/// dependent functions take a vector of floats and returs a vector of floats
/// their derivatives take a vector of floats and return a two dimensional vector - jacobian matrix
///
/// the first value is the function, the second is a the derivative
pub enum FuncElementWiseDependency<'a> {
    Independent(IndependentFunc<'a>),
    Dependent(DependentFunc<'a>),
}

pub struct ActivationFunc<'a> {
    pub funcs: FuncElementWiseDependency<'a>,

    pub description: &'a str,

    /// latex formula
    pub formula: &'a str,
    /// latex formula
    pub formula_derivative: &'a str,

    pub init_func: &'a InitFunc<'a>,
}

/// Sigmoid activation function
/// https://en.wikipedia.org/wiki/Sigmoid_function
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + consts::E.powf(-z))
}

/// Derivative of the sigmoid activation function
fn sigmoid_deriv(z: f32) -> f32 {
    let sig = sigmoid(z);
    sig * (1.0 - sig)
}
pub const SIGMOID: ActivationFunc = ActivationFunc {
    funcs: FuncElementWiseDependency::Independent(IndependentFunc {
        func: &sigmoid,
        derivative: &sigmoid_deriv,
    }),

    description: "",

    formula: "",
    formula_derivative: "",

    init_func: &XAVIER,
};

/// ReLU activation function
/// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
fn relu(z: f32) -> f32 {
    if z > 0.0 {
        z
    } else {
        0.0
    }
}

/// Derivative of the ReLU activation function
fn relu_deriv(z: f32) -> f32 {
    if z > 0.0 {
        1.0
    } else {
        0.0
    }
}
pub const RELU: ActivationFunc = ActivationFunc {
    funcs: FuncElementWiseDependency::Independent(IndependentFunc {
        func: &relu,
        derivative: &relu_deriv,
    }),

    description: "",

    formula: "",
    formula_derivative: "",

    init_func: &HE,
};

/// (stable) softmax activation function
fn softmax(zs: &Vec<f32>) -> Vec<f32> {
    let max = zs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let exps = zs.iter().map(|zj| consts::E.powf(*zj - max));
    let sum: f32 = exps.to_owned().sum();
    exps.map(|e| e / sum).collect()
}

/// derivative of the softmax activation functions
/// returns: [
///     [da_1/dz_1, ..., da_1/dz_j, ..., da_1/dz_n],
///     ...,
///     [da_i/dz_1, ..., da_i/dz_j, ..., da_i/dz_n],
///     ...,
///     [da_n/dz_1, ..., da_n/dz_j, ..., da_n/dz_n]
/// ]
fn softmax_deriv(zs: &Vec<f32>) -> Vec<Vec<f32>> {
    let softmax = softmax(zs);

    softmax
        .iter()
        .enumerate()
        .map(|(i, &si)| {
            let row: Vec<f32> = softmax
                .iter()
                .enumerate()
                .map(|(j, &sj)| {
                    let delta = if i == j { 1.0 } else { 0.0 };

                    si * (delta - sj)
                })
                .collect();

            row
        })
        .collect()
}
pub const SOFTMAX: ActivationFunc = ActivationFunc {
    funcs: FuncElementWiseDependency::Dependent(DependentFunc {
        func: &softmax,
        derivative: &softmax_deriv,
    }),

    description: "",

    formula: "",
    formula_derivative: "",

    init_func: &XAVIER,
};

#[cfg(test)]
mod tests {
    use super::softmax;

    #[test]
    fn test_softmax() {
        let res = softmax(&vec![8.0, 5.0, 0.0]);
        let expected = vec![0.9523, 0.0474, 0.0003];
        assert!(res
            .iter()
            .zip(expected)
            .all(|(val, expect)| (val - expect).abs() < 0.001));
    }
}

use std::iter::zip;

pub struct CostFunc<'a> {
    pub function: &'a dyn Fn(&Vec<f32>, &Vec<f32>) -> f32,
    pub derivative: &'a dyn Fn(f32, f32) -> f32,

    pub description: &'a str,

    /// latex formula
    pub formula: &'a str,
    /// latex formula
    pub formula_derivative: &'a str,
}

fn mse(predicted: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    assert_eq!(predicted.len(), expected.len());

    let sum: f32 = zip(predicted, expected)
        .map(|(a, y)| (a - y).powf(2.0))
        .sum();

    sum / (expected.len() as f32)
}

fn mse_deriv(predicted: f32, expected: f32) -> f32 {
    2.0 * (predicted - expected)
}

pub const MSE: CostFunc = CostFunc {
    function: &mse,
    derivative: &mse_deriv,

    description: "",

    formula: r"\frac{1}{n_L}
  \sum_{j = 0}^{n_L-1}(a^{(L)}_j - y_j)^2",
    formula_derivative: r"\frac{\partial C}{\partial a^{(L)}_j} = 2(a^{(L)}_j - y_j)",
};

fn cross_entropy(predicted: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    assert_eq!(predicted.len(), expected.len());

    let sum = zip(predicted, expected).map(|(a, y)| y * a.ln()).sum();

    sum
}

fn cross_entropy_deriv(predicted: f32, expected: f32) -> f32 {
    expected / predicted
}

pub const CROSS_ENTROPY: CostFunc = CostFunc {
    function: &cross_entropy,
    derivative: &cross_entropy_deriv,

    description: "",

    formula: r"-\sum_{i=0}^{n_L - 1} y_i \ln a^{(L)}_i",
    formula_derivative: r"\frac{y_i}{a^{(L)}_i}",
};

fn binary_cross_entropy(predicted: &Vec<f32>, expected: &Vec<f32>) -> f32 {
    assert_eq!(predicted.len(), expected.len());
    assert_eq!(predicted.len(), 1);

    let a = predicted[0];
    let y = expected[0];

    let sum = -(y * a.ln() + (1.0 - y) * (1.0 - a).ln());

    sum
}

fn binary_cross_entropy_deriv(predicted: f32, expected: f32) -> f32 {
    expected / predicted + (1.0 - expected) / (1.0 - predicted)
}

pub const BINARY_CROSS_ENTROPY: CostFunc = CostFunc {
    function: &binary_cross_entropy,
    derivative: &binary_cross_entropy_deriv,

    description: "",

    formula: r"-[y \ln a + (1 - y) \ln (1 - a)]",
    formula_derivative: r"\frac{y}{a} +
  \frac{1 - y}{1 - a}",
};

#[cfg(test)]
mod tests {
    use super::mse;

    #[test]
    fn test_mse() {
        assert_eq!(mse(&vec![1.0], &vec![1.0]), 0.0);
        assert_eq!(mse(&vec![1.0], &vec![0.5]), 0.25);
        assert_eq!(mse(&vec![1.0], &vec![0.0]), 1.0);
    }
}

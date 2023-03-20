// https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
pub struct NormalizationFn<'a> {
    pub function: &'a dyn Fn(&Vec<f32>) -> Vec<f32>,
    pub description: &'a str,
    /// latex formula
    pub formula: &'a str,
}

fn no_func(input: &Vec<f32>) -> Vec<f32> {
    input.to_vec()
}
pub const NO_NORMALIZATION: NormalizationFn = NormalizationFn {
    function: &no_func,
    description: "",
    formula: "",
};

/// mutates the network so that the input is normalized
fn normalization(input: &Vec<f32>) -> Vec<f32> {
    let max = input
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let min = input
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    if max - min > f32::EPSILON {
        let mut new_input: Vec<f32> = vec![];
        for i in 0..input.len() {
            new_input.push((input[i] - min) / (max - min));
        }

        new_input
    } else {
        input.to_vec()
    }
}
pub const NORMALIZATION: NormalizationFn = NormalizationFn {
    function: &normalization,
    description: "",
    formula: "",
};

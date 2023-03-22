use crate::{
    functions::{
        activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
    },
    supervised::network::{
        multi_layer_perceptron::MultiLayerPerceptron, Classifiable, Predictable, Shape,
    },
};

pub struct MultiLayerPerceptronClassification<'a> {
    network: MultiLayerPerceptron<'a>,
    labels: Vec<&'a str>,
}

impl<'a> MultiLayerPerceptronClassification<'a> {
    fn new(
        shape: Shape,
        activation_funcs: Vec<&'a ActivationFunc>,
        cost_func: &'a CostFunc,
        normalization_func: &'a NormalizationFunc,
        labels: Vec<&'a str>,
    ) -> Self {
        assert_eq!(*shape.last().unwrap(), labels.len());

        Self {
            network: MultiLayerPerceptron::new(
                shape,
                activation_funcs,
                cost_func,
                normalization_func,
            ),
            labels,
        }
    }
}

impl Classifiable for MultiLayerPerceptronClassification<'_> {
    fn get_label(&mut self) -> &str {
        let highest_i = self.network.get_highest_output().1;

        self.labels[highest_i]
    }

    fn one_hot_encode(&mut self, label: &str) -> Vec<f32> {
        let i = self.labels.iter().position(|&val| val == label).unwrap();
        let mut out = vec![0.0; self.labels.len()];

        out[i] = 1.0;

        out
    }
}

#[cfg(test)]
mod tests {
    use super::{Classifiable, MultiLayerPerceptronClassification};
    use crate::functions::{
        activation::SIGMOID, cost::MSE, input_normalizations::NO_NORMALIZATION,
    };

    #[test]
    fn test_one_hot_encode() {
        let mut net = MultiLayerPerceptronClassification::new(
            vec![2, 4],
            vec![&SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
            vec!["1", "2", "3", "4"],
        );

        let one_hot_encoded = net.one_hot_encode("2");

        assert_eq!(one_hot_encoded, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn text_get_label() {
        let mut net = MultiLayerPerceptronClassification::new(
            vec![2, 4],
            vec![&SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
            vec!["1", "2", "3", "4"],
        );

        net.network.activated_layers[0] = vec![0.0, 0.0, 2.0, 0.0];

        let label = net.get_label();

        assert_eq!(label, "3");
    }
}

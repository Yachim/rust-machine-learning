use crate::{
    functions::{
        activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
    },
    supervised::network::{
        multi_layer_perceptron::MultiLayerPerceptron, CSVTrainable, Classifiable, LayerNeurons,
        Predictable, Resetable, Shape, Trainable,
    },
};

pub struct MultiLayerPerceptronClassification<'a> {
    network: MultiLayerPerceptron<'a>,
    labels: Vec<&'a str>,
}

impl<'a> MultiLayerPerceptronClassification<'a> {
    pub fn new(
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

impl Resetable for MultiLayerPerceptronClassification<'_> {
    fn reset_params(&mut self) {
        self.network.reset_params();
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

impl Predictable for MultiLayerPerceptronClassification<'_> {
    fn predict(&mut self, inputs: &LayerNeurons) {
        self.network.predict(inputs);
    }

    fn get_highest_output(&self) -> (f32, usize) {
        self.network.get_highest_output()
    }
}

impl Trainable for MultiLayerPerceptronClassification<'_> {
    /// requires &Vec<(inputs, expected)> where expected is a vector of floats
    fn train(
        &mut self,
        iteration_cnt: usize,
        batch: &Vec<(LayerNeurons, LayerNeurons)>,
        batch_size: usize,
    ) {
        self.network.train(iteration_cnt, batch, batch_size);
    }
}

impl CSVTrainable for MultiLayerPerceptronClassification<'_> {
    fn train_from_csv(&mut self, file_path: &str, batch_size: usize) {
        todo!()
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

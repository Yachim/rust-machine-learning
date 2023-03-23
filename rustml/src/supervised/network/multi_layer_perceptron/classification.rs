// TODO: fix usage of String vs &str

use crate::{
    functions::{
        activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
    },
    supervised::network::{
        multi_layer_perceptron::MultiLayerPerceptron, CSVPredictable, CSVTestable, CSVTrainable,
        Classifiable, LayerNeurons, Predictable, Resetable, Shape, Testable, Trainable,
    },
    utils::{
        csv::{load_labeled_data, load_labeled_data_ohc, load_unlabeled_data, write_data},
        misc::one_hot_encode,
    },
};
use rand::{seq::SliceRandom, thread_rng};
use std::path::Path;

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
    fn get_label(&self) -> &str {
        let highest_i = self.network.get_highest_output().1;

        self.labels[highest_i]
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
        learning_rate: f32,
    ) {
        self.network
            .train(iteration_cnt, batch, batch_size, learning_rate);
    }
}

impl Testable for MultiLayerPerceptronClassification<'_> {
    fn test(&mut self, data: &Vec<(LayerNeurons, &str)>) -> f32 {
        let total_cnt = data.len() as f32;
        let mut total_correct: f32 = 0.0;

        for (inputs, label) in data {
            self.predict(&inputs);

            let predicted = self.get_label();
            if predicted == *label {
                total_correct += 1.0;
            } else {
                println!(
                    "Incorrect, expected: {label}, predicted: {predicted}, outputs: {:?}",
                    self.network.activated_layers.last().unwrap()
                );
            }
        }

        total_correct / total_cnt
    }
}

impl CSVTrainable for MultiLayerPerceptronClassification<'_> {
    fn train_from_csv(
        &mut self,
        file_path: &Path,
        label_col: usize,
        data_cols: &Vec<usize>,
        batch_size: usize,
        iteration_cnt: usize,
        learning_rate: f32,
    ) {
        let data = load_labeled_data_ohc(file_path, label_col, data_cols, &self.labels).unwrap();
        self.train(iteration_cnt, &data, batch_size, learning_rate);
    }
}

impl CSVPredictable for MultiLayerPerceptronClassification<'_> {
    fn predict_from_into_csv(
        &mut self,
        data_file_path: &Path,
        output_file_path: &Path,
        id_header: &str,
        label_header: &str,
        data_cols: &Vec<usize>,
        id_start_at: usize,
    ) {
        let mut predictions: Vec<(String, String)> = vec![];

        let data = load_unlabeled_data(data_file_path, data_cols).unwrap();
        for (i, inputs) in data.iter().enumerate() {
            self.predict(&inputs);
            let label = self.get_label().to_owned();

            let id = (i + id_start_at).to_string();
            predictions.push((id, label));
        }

        write_data(output_file_path, id_header, label_header, &predictions).unwrap();
    }
}

impl CSVTestable for MultiLayerPerceptronClassification<'_> {
    fn train_and_test_from_csv(
        &mut self,
        file_path: &Path,
        label_col: usize,
        data_cols: &Vec<usize>,
        training_part: f32,
        batch_size: usize,
        iteration_cnt: usize,
        shuffle: bool,
        learning_rate: f32,
    ) -> f32 {
        let mut full_data = load_labeled_data(file_path, label_col, data_cols).unwrap();
        if shuffle {
            full_data.shuffle(&mut thread_rng());
        }

        let training_len = full_data.len() * training_part as usize;

        let training_data: Vec<(LayerNeurons, LayerNeurons)> = full_data[0..training_len]
            .iter()
            .map(|(inputs, label)| (inputs.to_vec(), one_hot_encode(&self.labels, label)))
            .collect();
        let testing_data: Vec<(LayerNeurons, &str)> = full_data[training_len..]
            .iter()
            .map(|(inputs, label)| (inputs.to_vec(), label.as_str()))
            .collect();

        self.train(iteration_cnt, &training_data, batch_size, learning_rate);

        self.test(&testing_data)
    }

    /// TODO: not tested
    fn test_from_csv(&mut self, file_path: &Path, label_col: usize, data_cols: &Vec<usize>) -> f32 {
        let full_data = load_labeled_data(file_path, label_col, data_cols).unwrap();

        let data: Vec<(LayerNeurons, &str)> = full_data
            .iter()
            .map(|(inputs, label)| (inputs.to_vec(), label.as_str()))
            .collect();

        self.test(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::{Classifiable, MultiLayerPerceptronClassification};
    use crate::functions::{
        activation::SIGMOID, cost::MSE, input_normalizations::NO_NORMALIZATION,
    };

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

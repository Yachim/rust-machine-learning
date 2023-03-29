use std::path::Path;

pub mod multi_layer_perceptron;

/// weights connected from one neuron to all neuron in the previous layer
/// Vec<f32>
type NeuronWeights = Vec<f32>;

/// Vec<Vec<f32>>
type LayerWeights = Vec<NeuronWeights>;

/// Vec<Vec<Vec<f32>>>
type NetworkWeights = Vec<LayerWeights>;

/// biases in one layer, layer, activated_layer
/// Vec<f32>
type LayerNeurons = Vec<f32>;

/// Vec<Vec<f32>>
type NetworkNeurons = Vec<LayerNeurons>;

/// each element represents number of neurons in the layer
type Shape = Vec<usize>;

/// or initialize
trait Resetable {
    fn reset_params(&mut self);
}

/// testing/predicting
pub trait Predictable {
    /// returns the value and index of the output neuron
    fn get_highest_output(&self) -> (f32, usize);

    fn predict(&mut self, inputs: &LayerNeurons);
}

pub trait Trainable {
    fn train(
        &mut self,
        iteration_cnt: usize,
        batch: &Vec<(LayerNeurons, LayerNeurons)>,
        batch_size: usize,
        learning_rate: f32,
    );
}

pub trait Testable {
    /// returns accuracy as number between 0 and 1
    fn test(&mut self, data: &Vec<(LayerNeurons, &str)>) -> f32;
}

pub trait Classifiable {
    fn get_label(&self) -> &str;
}

pub trait Debuggable {
    fn get_weights(&self) -> &NetworkWeights;
    fn get_biases(&self) -> &NetworkNeurons;
    fn get_inputs(&self) -> &LayerNeurons;
    fn get_normalized_inputs(&self) -> &LayerNeurons;
    fn get_layers(&self) -> &NetworkNeurons;
    fn get_activated_layers(&self) -> &NetworkNeurons;
}

pub trait CSVTrainable {
    fn train_from_csv(
        &mut self,
        file_path: &Path,
        label_col: usize,
        data_cols: &Vec<usize>,
        batch_size: usize,
        iteration_cnt: usize,
        learning_rate: f32,
    );
}

pub trait CSVTestable {
    /// trains from a csv and then tests from the same data splitting between training and testing
    /// data
    /// training_part: number between 0 and 1 specifying how much of the dataset is training
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
    ) -> f32;

    /// loads labeled data and returns accuracy
    fn test_from_csv(&mut self, file_path: &Path, label_col: usize, data_cols: &Vec<usize>) -> f32;
}

pub trait CSVPredictable {
    /// loads a file with data and writes the predictions into another file
    fn predict_from_into_csv(
        &mut self,
        data_file_path: &Path,
        output_file_path: &Path,
        id_header: &str,
        label_header: &str,
        data_cols: &Vec<usize>,
        id_start_at: usize,
    );
}

pub trait CostComputable {
    // calculate average cost across the whole data set
    fn avg_cost_from_vec(&mut self, batch: &Vec<(Vec<f32>, Vec<f32>)>) -> f32;
}

pub trait CSVCostComputable {
    // calculate average cost across the whole data set
    fn avg_cost_from_csv(
        &mut self,
        file_path: &Path,
        label_col: usize,
        data_cols: &Vec<usize>,
    ) -> f32;
}

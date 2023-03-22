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
trait Predictable {
    /// returns the value and index of the output neuron
    fn get_highest_output(&self) -> (f32, usize);

    fn predict(&mut self, inputs: &LayerNeurons);
}

trait Trainable {
    fn train(
        &mut self,
        iteration_cnt: usize,
        batch: &Vec<(LayerNeurons, LayerNeurons)>,
        batch_size: usize,
    );
}

trait Classifiable {
    fn one_hot_encode(&mut self, label: &str) -> Vec<f32>;

    fn get_label(&mut self) -> &str;
}

trait CSVTrainable {
    fn train_from_csv(&mut self, file_path: &str, batch_size: usize);
}

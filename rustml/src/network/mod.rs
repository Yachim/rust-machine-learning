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
pub trait Resetable {
    fn reset_params(&mut self);
}

/// testing/predicting
pub trait Predictable {
    fn normalize_input(&mut self);

    fn feedforward_layer(&mut self, layer_i: usize);

    fn feedforward(&mut self);
}

pub trait Trainable: Predictable {
    /// returns derivatives in order: dC/dw, dC/db, dC/da
    fn backprop_layer(&self) -> (LayerWeights, LayerNeurons, LayerNeurons);

    /// returns derivatives in order: dC/dw, dC/db
    fn backprop(&mut self) -> (LayerWeights, LayerNeurons);

    fn gradient_descent(&mut self);

    fn batch_gradient_descent(&mut self);

    fn train(&mut self);
}

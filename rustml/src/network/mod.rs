pub mod multi_layer_perceptron;

use crate::{
    functions::{activation::ActivationFunc, input_normalizations::NormalizationFunc},
    utils::math::dot_product,
};

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

/// weights, biases, activation functions have length of l - 1
/// where l = length of shape (the number of total layers)
pub trait BaseNetwork {
    fn get_shape(&self) -> &Shape;

    fn get_activation_funcs(&self) -> &Vec<&ActivationFunc>;

    fn get_activation_func(&self, layer_i: usize) -> &ActivationFunc;

    fn get_normalization_func(&self) -> &NormalizationFunc;
}

pub trait Readable {
    fn get_weights(&self) -> &NetworkWeights;
    fn get_layer_weights(&self, layer_i: usize) -> &LayerWeights;

    fn get_biases(&self) -> &NetworkNeurons;
    fn get_layer_biases(&self, layer_i: usize) -> &LayerNeurons;

    fn get_layers(&self) -> &NetworkNeurons;
    fn get_layer(&self, layer_i: usize) -> &LayerNeurons;

    fn get_activated_layers(&self) -> &NetworkNeurons;
    fn get_activated_layer(&self, layer_i: usize) -> &LayerNeurons;

    fn get_inputs(&self) -> &LayerNeurons;
    fn get_normalized_inputs(&self) -> &LayerNeurons;
}

pub trait Writable {
    fn set_weights(&mut self, weights: NetworkWeights);

    fn set_biases(&mut self, biases: NetworkNeurons);

    fn set_layers(&mut self, layers: NetworkNeurons);
    fn set_layer(&mut self, layer: LayerNeurons, layer_i: usize);

    fn set_activated_layers(&mut self, activated_layers: NetworkNeurons);
    fn set_activated_layer(&mut self, activated_layer: LayerNeurons, layer_i: usize);

    fn set_inputs(&mut self, inputs: LayerNeurons);
    fn set_normalized_inputs(&mut self, normalized_inputs: LayerNeurons);
}

/// or initialize
pub trait Resetable: Writable + BaseNetwork {
    fn reset_params(&mut self) {
        let shape = self.get_shape();

        let mut new_weights: NetworkWeights = vec![];
        let mut new_biases: NetworkNeurons = vec![];
        let mut new_layers: NetworkNeurons = vec![];
        let mut new_activated_layers: NetworkNeurons = vec![];

        let new_inputs = vec![0.0; shape[0]];
        let new_normalized_inputs = vec![0.0; shape[0]];

        let weights_init_funcs = self.get_activation_funcs();

        for (layer_index, &layer_neuron_cnt) in shape.iter().enumerate().skip(1) {
            let layer: LayerNeurons = vec![0.0; layer_neuron_cnt];
            new_layers.push(layer);

            let activated_layer: LayerNeurons = vec![0.0; layer_neuron_cnt];
            new_activated_layers.push(activated_layer);

            let layer_biases: Vec<f32> = vec![0.0; layer_neuron_cnt];
            new_biases.push(layer_biases);

            let mut layer_weights: LayerWeights = vec![];

            let prev_layer_neuron_cnt = shape[layer_index - 1];

            let layer_weights_init_func = weights_init_funcs[layer_index - 1].init_fn.function;

            for _ in 0..layer_neuron_cnt {
                let mut neuron_weights: NeuronWeights = vec![];

                for _ in 0..prev_layer_neuron_cnt {
                    neuron_weights.push(layer_weights_init_func(prev_layer_neuron_cnt));
                }

                layer_weights.push(neuron_weights);
            }

            new_weights.push(layer_weights);
        }

        self.set_weights(new_weights);
        self.set_biases(new_biases);
        self.set_layers(new_layers);
        self.set_activated_layers(new_activated_layers);

        self.set_inputs(new_inputs);
        self.set_normalized_inputs(new_normalized_inputs);
    }
}

/// testing/predicting
pub trait Predictable: BaseNetwork + Writable + Readable {
    fn normalize_input(&mut self) {
        let normalization_func = self.get_normalization_func().function;

        let inputs = self.get_inputs();
        let normalized_inputs = normalization_func(inputs);
        self.set_normalized_inputs(normalized_inputs);
    }

    fn feedforward_layer(&mut self, layer_i: usize) {
        let prev_layer = if layer_i == 0 {
            self.get_normalized_inputs()
        } else {
            self.get_activated_layer(layer_i - 1)
        };
        let layer_weights = self.get_layer_weights(layer_i);
        let layer_biases = self.get_layer_biases(layer_i);

        let layer_activation_func = self.get_activation_func(layer_i).function;

        let mut new_layer: LayerNeurons = vec![];
        let mut new_activated_layer: LayerNeurons = vec![];
        for neuron_i in 0..layer_weights.len() {
            let neuron_weights = &layer_weights[neuron_i];
            let neuron_bias = layer_biases[neuron_i];

            let new_neuron = dot_product(prev_layer, neuron_weights) + neuron_bias;
            new_layer.push(new_neuron);

            let new_activated_neuron = layer_activation_func(new_neuron);
            new_activated_layer.push(new_activated_neuron);
        }

        self.set_layer(new_layer, layer_i);
        self.set_activated_layer(new_activated_layer, layer_i);
    }

    fn feedforward(&mut self) {
        let layer_cnt = self.get_layers().len();

        self.normalize_input();

        for layer_i in 0..layer_cnt {
            self.feedforward_layer(layer_i);
        }
    }
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

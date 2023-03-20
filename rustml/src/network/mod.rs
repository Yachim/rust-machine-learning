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
}

pub trait Writable {
    fn set_weights(&mut self, weights: NetworkWeights);

    fn set_biases(&mut self, biases: NetworkNeurons);

    fn set_layers(&mut self, layers: NetworkNeurons);

    fn set_layer(&mut self, layer: LayerNeurons, layer_i: usize);

    fn set_activated_layers(&mut self, activated_layers: NetworkNeurons);

    fn set_activated_layer(&mut self, activated_layer: LayerNeurons, layer_i: usize);
}

pub trait Resetable: Writable + BaseNetwork {
    /// or initialize
    fn reset_params(&mut self) {
        let shape = self.get_shape();

        let mut new_weights: NetworkWeights = vec![];
        let mut new_biases: NetworkNeurons = vec![];
        let mut new_layers: NetworkNeurons = vec![];
        let mut new_activated_layers: NetworkNeurons = vec![];

        let weights_init_funcs = self.get_activation_funcs();

        for (layer_index, &layer_neuron_cnt) in shape.iter().enumerate() {
            let layer_biases: Vec<f32> = vec![0.0; layer_neuron_cnt];
            new_biases.push(layer_biases);

            let layer: LayerNeurons = vec![0.0; layer_neuron_cnt];
            new_layers.push(layer);

            let activated_layer: LayerNeurons = vec![0.0; layer_neuron_cnt];
            new_activated_layers.push(activated_layer);

            if layer_index == 0 {
                continue;
            } else {
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
        }

        self.set_weights(new_weights);
        self.set_biases(new_biases);
        self.set_layers(new_layers);
        self.set_activated_layers(new_activated_layers);
    }
}

/// testing/predicting
pub trait Predictable: BaseNetwork + Writable + Readable {
    fn activate_layer(&mut self, layer_i: usize) {
        let layer = self.get_layer(layer_i);

        if layer_i == 0 {
            let normalization_func = self.get_normalization_func().function;
            let activated_layer = normalization_func(layer);
            self.set_activated_layer(activated_layer, layer_i);
        } else {
            let activation_func = self.get_activation_func(layer_i - 1).function;

            let mut activated_layer: LayerNeurons = vec![];
            for &neuron in layer {
                let activated_neuron = activation_func(neuron);
                activated_layer.push(activated_neuron);
            }

            self.set_activated_layer(activated_layer, layer_i);
        }
    }

    fn feedforward_layer(&mut self, layer_i: usize) {
        self.activate_layer(layer_i);

        if layer_i != 0 {
            let prev_layer = self.get_activated_layer(layer_i - 1);
            let layer_weights = self.get_layer_weights(layer_i - 1);
            let layer_biases = self.get_layer_biases(layer_i - 1);

            let new_layer: LayerNeurons = (0..prev_layer.len())
                .map(|neuron_i| {
                    let neuron_weights = &layer_weights[neuron_i];
                    let neuron_bias = layer_biases[neuron_i];

                    let new_neuron = dot_product(prev_layer, neuron_weights) + neuron_bias;
                    new_neuron
                })
                .collect();

            self.set_layer(new_layer, layer_i);
        }
    }

    fn feedforward(&mut self) {
        let layer_cnt = self.get_shape().len();
        for layer_i in 0..layer_cnt {
            self.feedforward_layer(layer_i);
        }
    }
}

pub trait Trainable: Predictable {
    fn backprop(&mut self);

    fn gradient_descent(&mut self);

    fn batch_gradient_descent(&mut self);

    fn train(&mut self);
}

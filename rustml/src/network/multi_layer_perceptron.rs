use super::{
    Backpropable, LayerNeurons, LayerWeights, NetworkNeurons, NetworkWeights, NeuronWeights,
    Predictable, Resetable, Shape,
};
use crate::{
    functions::{
        activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
    },
    utils::math::dot_product,
};

pub struct MultiLayerPerceptron<'a> {
    shape: Shape,
    activation_funcs: Vec<&'a ActivationFunc<'a>>,
    cost_func: &'a CostFunc<'a>,
    normalization_func: &'a NormalizationFunc<'a>,

    inputs: LayerNeurons,
    normalized_inputs: LayerNeurons,
    weights: NetworkWeights,
    biases: NetworkNeurons,
    layers: NetworkNeurons,
    activated_layers: NetworkNeurons,
}

impl<'a> MultiLayerPerceptron<'a> {
    fn new(
        shape: Shape,
        activation_funcs: Vec<&'a ActivationFunc>,
        cost_func: &'a CostFunc,
        normalization_func: &'a NormalizationFunc,
    ) -> Self {
        assert_eq!(activation_funcs.len(), shape.len() - 1);
        let mut net = Self {
            shape,
            activation_funcs,
            cost_func,
            normalization_func,

            inputs: vec![],
            normalized_inputs: vec![],
            weights: vec![],
            biases: vec![],
            layers: vec![],
            activated_layers: vec![],
        };

        net.reset_params();
        net
    }
}

impl Resetable for MultiLayerPerceptron<'_> {
    fn reset_params(&mut self) {
        let shape = &self.shape;

        let mut new_weights: NetworkWeights = vec![];
        let mut new_biases: NetworkNeurons = vec![];
        let mut new_layers: NetworkNeurons = vec![];
        let mut new_activated_layers: NetworkNeurons = vec![];

        let new_inputs = vec![0.0; shape[0]];
        let new_normalized_inputs = vec![0.0; shape[0]];

        let weights_init_funcs = &self.activation_funcs;

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

        self.weights = new_weights;
        self.biases = new_biases;
        self.layers = new_layers;
        self.activated_layers = new_activated_layers;

        self.inputs = new_inputs;
        self.normalized_inputs = new_normalized_inputs;
    }
}

impl Predictable for MultiLayerPerceptron<'_> {
    fn normalize_input(&mut self) {
        let normalization_func = self.normalization_func.function;

        let inputs = &self.inputs;
        let normalized_inputs = normalization_func(inputs);
        self.normalized_inputs = normalized_inputs;
    }

    fn feedforward_layer(&mut self, layer_i: usize) {
        let prev_layer = if layer_i == 0 {
            &self.normalized_inputs
        } else {
            &self.activated_layers[layer_i - 1]
        };
        let layer_weights = &self.weights[layer_i];
        let layer_biases = &self.biases[layer_i];

        let layer_activation_func = self.activation_funcs[layer_i].function;

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

        self.layers[layer_i] = new_layer;
        self.activated_layers[layer_i] = new_activated_layer;
    }

    fn feedforward(&mut self) {
        let layer_cnt = self.layers.len();

        self.normalize_input();

        for layer_i in 0..layer_cnt {
            self.feedforward_layer(layer_i);
        }
    }
}

impl Backpropable for MultiLayerPerceptron<'_> {
    /// again, prev as in previous iteration ((i + 1)th layer)
    fn backprop_layer(
        &self,
        layer_i: usize,
        prev_cost_activation_derivatives: Option<&LayerNeurons>,
        expected: Option<&NeuronWeights>,
    ) -> (LayerWeights, LayerNeurons, LayerNeurons) {
        // f'[l]
        let activation_func_derivative = self.activation_funcs[layer_i].derivative;

        // dC/dw[l]
        let mut cost_weight_derivatives: LayerWeights = vec![];
        // dC/db[l]
        let mut cost_bias_derivatives: LayerNeurons = vec![];
        // dC/da[l]
        let mut cost_activation_derivatives: LayerNeurons = vec![];

        // z[l]
        let layer = &self.layers[layer_i];
        // a[l]
        let activated_layer = &self.activated_layers[layer_i];
        // w[l]
        let weights = &self.weights[layer_i];

        let mut tmp = vec![];
        let prev_cost_activation_derivatives_unwrapped = prev_cost_activation_derivatives
            .unwrap_or_else(|| {
                tmp = vec![];
                &tmp
            });

        for neuron_i in 0..layer.len() {
            // z[l][j]
            let neuron = layer[neuron_i];
            // a[l][j]
            let activated_neuron = activated_layer[neuron_i];
            // w[l][j]
            let neuron_weights = &weights[neuron_i];

            // dC/da[l][j]
            let cost_activation_derivative = if layer_i == self.layers.len() - 1 {
                // y[j]
                let expected_neuron = expected.unwrap()[neuron_i];

                let cost_func_derivative = self.cost_func.derivative;
                cost_func_derivative(activated_neuron, expected_neuron)
            } else {
                // f'[l + 1]
                let prev_activation_func_derivative = self.activation_funcs[layer_i + 1].derivative;

                // z[l + 1]
                let prev_layer = &self.layers[layer_i + 1];

                // w[l + 1]
                let prev_weights = &self.weights[layer_i + 1];

                prev_cost_activation_derivatives_unwrapped
                    .iter()
                    .enumerate()
                    .map(|(prev_neuron_i, a)| {
                        a * prev_activation_func_derivative(prev_layer[prev_neuron_i])
                            * prev_weights[prev_neuron_i][neuron_i]
                    })
                    .sum()
            };
            cost_activation_derivatives.push(cost_activation_derivative);

            // da[l][j]/dz[l][j]
            let activation_neuron_derivative = activation_func_derivative(neuron);

            // dC/db[l][j]
            let cost_bias_derivative = cost_activation_derivative * activation_neuron_derivative;
            cost_bias_derivatives.push(cost_bias_derivative);

            let mut new_neuron_weights: NeuronWeights = vec![];
            for prev_neuron_i in 0..neuron_weights.len() {
                // a[l - 1][k]
                let connected_neuron = if layer_i == 0 {
                    self.normalized_inputs[prev_neuron_i]
                } else {
                    self.activated_layers[layer_i - 1][prev_neuron_i]
                };

                // dC/dw[l][j][k]
                let cost_weight_derivative =
                    cost_activation_derivative * activation_neuron_derivative * connected_neuron;

                new_neuron_weights.push(cost_weight_derivative);
            }
            cost_weight_derivatives.push(new_neuron_weights)
        }

        (
            cost_weight_derivatives,
            cost_bias_derivatives,
            cost_activation_derivatives,
        )
    }

    fn backprop(&mut self) -> (LayerWeights, LayerNeurons) {
        (vec![], vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::{Backpropable, MultiLayerPerceptron, Predictable};
    use crate::functions::{
        activation::{RELU, SIGMOID},
        cost::MSE,
        input_normalizations::NO_NORMALIZATION,
    };

    #[test]
    fn test_ws_cnt() {
        let net = MultiLayerPerceptron::new(
            vec![3, 2, 3, 2],
            vec![&SIGMOID, &SIGMOID, &SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );

        let mut total_ws = 0;
        for i in net.weights {
            for j in i {
                for _ in j {
                    total_ws += 1;
                }
            }
        }

        assert_eq!(total_ws, 18);
    }

    #[test]
    fn test_bias_cnt() {
        let net = MultiLayerPerceptron::new(
            vec![3, 2, 3, 2],
            vec![&SIGMOID, &SIGMOID, &SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );

        let mut total_biases = 0;
        for i in net.biases {
            for _ in i {
                total_biases += 1;
            }
        }

        assert_eq!(total_biases, 7);
    }

    #[test]
    fn test_neurons_cnt() {
        let net = MultiLayerPerceptron::new(
            vec![3, 2, 3, 2],
            vec![&SIGMOID, &SIGMOID, &SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );

        let mut total_neurons = 0;
        for i in net.layers {
            for _ in i {
                total_neurons += 1;
            }
        }
        total_neurons += net.inputs.len();

        assert_eq!(total_neurons, 10);
    }

    #[test]
    fn test_feedforward_layer1() {
        let mut net =
            MultiLayerPerceptron::new(vec![2, 3], vec![&SIGMOID], &MSE, &NO_NORMALIZATION);
        // test initialization
        assert_eq!(net.inputs.len(), 2);
        assert_eq!(net.normalized_inputs.len(), 2);

        assert_eq!(net.layers.len(), 1);
        assert_eq!(net.activated_layers.len(), 1);
        assert_eq!(net.weights.len(), 1);
        assert_eq!(net.biases.len(), 1);

        assert_eq!(net.layers[0].len(), 3);
        assert_eq!(net.activated_layers[0].len(), 3);
        assert_eq!(net.weights[0].len(), 3);
        assert_eq!(net.biases[0].len(), 3);

        assert_eq!(net.weights[0][0].len(), 2);
        assert_eq!(net.weights[0][1].len(), 2);
        assert_eq!(net.weights[0][2].len(), 2);

        net.inputs = vec![3.0, 2.0];

        net.normalize_input();
        assert_eq!(net.normalized_inputs, vec![3.0, 2.0]);

        net.weights[0] = vec![vec![3.0, 4.0], vec![2.0, 4.0], vec![3.0, 5.0]];
        net.biases[0] = vec![1.0, 1.0, 3.0];

        net.feedforward_layer(0);

        assert_eq!(net.layers[0], vec![18.0, 15.0, 22.0]);
    }

    #[test]
    fn test_feedforward_layer2() {
        let mut net = MultiLayerPerceptron::new(
            vec![2, 3, 3],
            vec![&SIGMOID, &SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );
        // test initialization
        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.activated_layers.len(), 2);
        assert_eq!(net.weights.len(), 2);
        assert_eq!(net.biases.len(), 2);

        assert_eq!(net.inputs.len(), 2);
        assert_eq!(net.normalized_inputs.len(), 2);
        // weights connected to inputs
        assert_eq!(net.weights[0][0].len(), 2);
        assert_eq!(net.weights[0][1].len(), 2);
        assert_eq!(net.weights[0][2].len(), 2);

        assert_eq!(net.layers[0].len(), 3);
        assert_eq!(net.activated_layers[0].len(), 3);
        assert_eq!(net.weights[0].len(), 3);
        assert_eq!(net.biases[0].len(), 3);
        // weights connected to 0th layer
        assert_eq!(net.weights[1][0].len(), 3);
        assert_eq!(net.weights[1][1].len(), 3);
        assert_eq!(net.weights[1][2].len(), 3);

        assert_eq!(net.layers[1].len(), 3);
        assert_eq!(net.activated_layers[1].len(), 3);
        assert_eq!(net.weights[1].len(), 3);
        assert_eq!(net.biases[1].len(), 3);

        net.activated_layers[0] = vec![3.0, 3.0, 2.0];
        net.weights[1] = vec![
            vec![3.0, 4.0, 3.0],
            vec![2.0, 4.0, 2.0],
            vec![3.0, 5.0, 2.0],
        ];
        net.biases[1] = vec![1.0, 1.0, 3.0];

        net.feedforward_layer(1);

        assert_eq!(net.layers[1], vec![28.0, 23.0, 31.0]);
    }

    #[test]
    fn test_feedforward() {
        let mut net =
            MultiLayerPerceptron::new(vec![4, 3, 5], vec![&RELU, &RELU], &MSE, &NO_NORMALIZATION);
        // test initialization
        assert_eq!(net.inputs.len(), 4);
        assert_eq!(net.normalized_inputs.len(), 4);
        // weights connected to inputs
        assert_eq!(net.weights[0][0].len(), 4);
        assert_eq!(net.weights[0][1].len(), 4);
        assert_eq!(net.weights[0][2].len(), 4);

        assert_eq!(net.layers.len(), 2);
        assert_eq!(net.activated_layers.len(), 2);
        assert_eq!(net.weights.len(), 2);
        assert_eq!(net.biases.len(), 2);

        assert_eq!(net.layers[0].len(), 3);
        assert_eq!(net.activated_layers[0].len(), 3);
        assert_eq!(net.weights[0].len(), 3);
        assert_eq!(net.biases[0].len(), 3);
        // weights connected to 0th layer
        assert_eq!(net.weights[1][0].len(), 3);
        assert_eq!(net.weights[1][1].len(), 3);
        assert_eq!(net.weights[1][2].len(), 3);
        assert_eq!(net.weights[1][3].len(), 3);
        assert_eq!(net.weights[1][4].len(), 3);

        assert_eq!(net.layers[1].len(), 5);
        assert_eq!(net.activated_layers[1].len(), 5);
        assert_eq!(net.weights[1].len(), 5);
        assert_eq!(net.biases[1].len(), 5);

        net.inputs = vec![2.0, 1.0, 3.0, 4.0];
        net.weights = vec![
            vec![
                vec![3.0, 2.0, 1.0, 4.0],
                vec![5.0, 1.0, 2.0, 3.0],
                vec![4.0, 1.0, 2.0, 1.0],
            ],
            vec![
                vec![1.0, 2.0, 5.0],
                vec![3.0, 2.0, 1.0],
                vec![2.0, 3.0, 5.0],
                vec![1.0, 4.0, 1.0],
                vec![4.0, 1.0, 2.0],
            ],
        ];
        net.biases = vec![vec![3.0, 1.0, 2.0], vec![3.0, 2.0, 1.0, 2.0, 4.0]];

        net.feedforward();

        assert_eq!(net.normalized_inputs, vec![2.0, 1.0, 3.0, 4.0]);

        assert_eq!(net.layers[0], vec![30.0, 30.0, 21.0]);
        assert_eq!(net.activated_layers[0], vec![30.0, 30.0, 21.0]);

        assert_eq!(net.layers[1], vec![198.0, 173.0, 256.0, 173.0, 196.0]);
        assert_eq!(
            net.activated_layers[1],
            vec![198.0, 173.0, 256.0, 173.0, 196.0]
        );
    }

    // same as test_feedforward_layer1
    #[test]
    fn test_backprop_layer() {
        let mut net = MultiLayerPerceptron::new(vec![2, 3], vec![&RELU], &MSE, &NO_NORMALIZATION);

        net.normalized_inputs = vec![3.0, 2.0];

        net.weights[0] = vec![vec![3.0, 4.0], vec![2.0, 4.0], vec![3.0, 5.0]];
        net.biases[0] = vec![1.0, 1.0, 3.0];

        net.feedforward_layer(0);

        assert_eq!(net.activated_layers[0], vec![18.0, 15.0, 22.0]);

        let (dws, dbs, da) = net.backprop_layer(0, None, Some(&vec![15.0, 12.0, 20.0]));
        assert_eq!(da, vec![6.0, 6.0, 4.0]);
        assert_eq!(
            dws,
            vec![vec![18.0, 12.0], vec![18.0, 12.0], vec![12.0, 8.0]]
        );
        assert_eq!(dbs, vec![6.0, 6.0, 4.0]);
    }

    /*#[test]
    fn test_output() {
        let net = MultiLayerPerceptron::new(vec![3, 3], vec![&SIGMOID], &MSE, &NO_NORMALIZATION);

        net.activated_layers[0] = vec![3.0, 2.0, 1.0];
        let out = net.get_output_str();
        assert_eq!(out, "1");
    }*/
}

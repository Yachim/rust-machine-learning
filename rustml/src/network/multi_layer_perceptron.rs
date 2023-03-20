use super::{
    BaseNetwork, LayerNeurons, LayerWeights, NetworkNeurons, NetworkWeights, Predictable, Readable,
    Resetable, Shape, Writable,
};
use crate::functions::{
    activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
};

pub struct MultiLayerPerceptron<'a> {
    shape: Shape,
    activation_funcs: Vec<&'a ActivationFunc<'a>>,
    cost_func: &'a CostFunc<'a>,
    normalization_func: &'a NormalizationFunc<'a>,

    weights: NetworkWeights,
    biases: NetworkNeurons,
    layers: NetworkNeurons,
    activated_layers: NetworkNeurons,
}

impl Resetable for MultiLayerPerceptron<'_> {}

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

            weights: vec![],
            biases: vec![],
            layers: vec![],
            activated_layers: vec![],
        };

        net.reset_params();
        net
    }
}

impl BaseNetwork for MultiLayerPerceptron<'_> {
    fn get_shape(&self) -> &Shape {
        &self.shape
    }

    fn get_normalization_func(&self) -> &NormalizationFunc {
        &self.normalization_func
    }

    fn get_activation_funcs(&self) -> &Vec<&ActivationFunc> {
        &self.activation_funcs
    }

    fn get_activation_func(&self, layer_i: usize) -> &ActivationFunc {
        self.activation_funcs[layer_i]
    }
}

impl Readable for MultiLayerPerceptron<'_> {
    fn get_weights(&self) -> &NetworkWeights {
        &self.weights
    }
    fn get_layer_weights(&self, layer_i: usize) -> &LayerWeights {
        &self.weights[layer_i]
    }

    fn get_biases(&self) -> &NetworkNeurons {
        &self.biases
    }
    fn get_layer_biases(&self, layer_i: usize) -> &LayerNeurons {
        &self.biases[layer_i]
    }

    fn get_layers(&self) -> &NetworkNeurons {
        &self.layers
    }
    fn get_layer(&self, layer_i: usize) -> &LayerNeurons {
        &self.layers[layer_i]
    }

    fn get_activated_layers(&self) -> &NetworkNeurons {
        &self.layers
    }
    fn get_activated_layer(&self, layer_i: usize) -> &LayerNeurons {
        &self.activated_layers[layer_i]
    }
}

impl Writable for MultiLayerPerceptron<'_> {
    fn set_weights(&mut self, weights: NetworkWeights) {
        self.weights = weights
    }

    fn set_biases(&mut self, biases: NetworkNeurons) {
        self.biases = biases
    }

    fn set_layers(&mut self, layers: NetworkNeurons) {
        self.layers = layers
    }
    fn set_layer(&mut self, layer: LayerNeurons, layer_i: usize) {
        self.layers[layer_i] = layer
    }

    fn set_activated_layers(&mut self, activated_layers: NetworkNeurons) {
        self.activated_layers = activated_layers
    }
    fn set_activated_layer(&mut self, activated_layer: LayerNeurons, layer_i: usize) {
        self.activated_layers[layer_i] = activated_layer
    }
}

impl Predictable for MultiLayerPerceptron<'_> {}

#[cfg(test)]
mod tests {
    use super::{MultiLayerPerceptron, Predictable};
    use crate::functions::{
        activation::SIGMOID, cost::MSE, input_normalizations::NO_NORMALIZATION,
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

        assert_eq!(total_neurons, 10);
    }

    #[test]
    fn test_feedforward_layer() {
        let mut net =
            MultiLayerPerceptron::new(vec![2, 3], vec![&SIGMOID], &MSE, &NO_NORMALIZATION);

        println!("{:?}", net.weights);
        println!("{:?}", net.biases);
        println!();

        println!("{:?}", net.layers);
        println!("{:?}", net.activated_layers);
        println!();

        net.activated_layers[0] = vec![3.0, 2.0];
        net.weights[0] = vec![vec![3.0, 4.0], vec![2.0, 4.0], vec![3.0, 5.0]];
        net.biases[0] = vec![1.0, 1.0, 3.0];

        net.feedforward_layer(1);

        println!("{:?}", net.weights);
        println!("{:?}", net.biases);
        println!();

        println!("{:?}", net.layers);
        println!("{:?}", net.activated_layers);
        println!();

        assert_eq!(net.layers[1], vec![18.0, 15.0, 22.0]);
    }

    /*#[test]
    fn test_output() {
        let net = MultiLayerPerceptron::new(vec![3, 3], vec![&SIGMOID], &MSE, &NO_NORMALIZATION);

        net.activated_layers[0] = vec![3.0, 2.0, 1.0];
        let out = net.get_output_str();
        assert_eq!(out, "1");
    }*/
}

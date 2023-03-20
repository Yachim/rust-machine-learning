use super::{BaseNetwork, NetworkNeurons, NetworkWeights, Readable, Shape};
use crate::functions::{activation::ActivationFunc, input_normalizations::NormalizationFunc};

pub struct MultiLayerPerceptron<'a> {
    shape: Shape,
    normalization_func: &'a NormalizationFunc<'a>,
    activation_funcs: Vec<&'a ActivationFunc<'a>>,

    weights: NetworkWeights,
    biases: NetworkNeurons,
    layers: NetworkNeurons,
    activated_layers: NetworkNeurons,
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
    fn get_layer_weights(&self, layer_i: usize) -> &super::LayerWeights {
        &self.weights[layer_i]
    }

    fn get_biases(&self) -> &NetworkNeurons {
        &self.biases
    }
    fn get_layer_biases(&self, layer_i: usize) -> &super::LayerNeurons {
        &self.biases[layer_i]
    }

    fn get_layers(&self) -> &NetworkNeurons {
        &self.layers
    }
    fn get_layer(&self, layer_i: usize) -> &super::LayerNeurons {
        &self.layers[layer_i]
    }

    fn get_activated_layers(&self) -> &NetworkNeurons {
        &self.layers
    }
    fn get_activated_layer(&self, layer_i: usize) -> &super::LayerNeurons {
        &self.activated_layers[layer_i]
    }
}

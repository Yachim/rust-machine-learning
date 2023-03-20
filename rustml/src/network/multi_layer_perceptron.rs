use super::{BaseNetwork, Shape};
use crate::functions::{activation::ActivationFunc, input_normalizations::NormalizationFunc};

pub struct MultiLayerPerceptron<'a> {
    shape: Shape,
    normalization_func: &'a NormalizationFunc<'a>,
    activation_funcs: Vec<&'a ActivationFunc<'a>>,
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

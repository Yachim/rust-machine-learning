use crate::{
    functions::{
        activation::ActivationFunc, cost::CostFunc, input_normalizations::NormalizationFunc,
    },
    supervised::network::{multi_layer_perceptron::MultiLayerPerceptron, Shape},
};

pub struct MultiLayerPerceptronClassification<'a> {
    network: MultiLayerPerceptron<'a>,
    labels: Vec<&'a str>,
}

impl<'a> MultiLayerPerceptronClassification<'a> {
    fn new(
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

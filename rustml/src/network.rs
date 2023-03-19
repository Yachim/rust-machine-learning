use crate::functions::{
    activation::ActivationFunction, cost::CostFunc, input_normalizations::NormalizationFn,
};
use crate::utils::math::dot_product;
use chrono::offset::Local;
use rand::seq::SliceRandom;
use std::iter::zip;

pub enum NetworkType {
    /// single output regression
    Regression { expected: f32, predicted: f32 },
    /// single label regression (can have multiple classes)
    Classification {
        expected: String,
        predicted: String,
        classes: Vec<String>,
    },
}

pub enum TrainingData {
    Regression(Vec<(Vec<f32>, f32)>),
    Classification(Vec<(Vec<f32>, String)>),
}

pub enum NetworkConstructorType {
    /// single output regression
    Regression,
    /// single label classification (can have multiple classes) - takes labels as parameter
    Classification(Vec<String>),
}

enum NetworkOutputType {
    Regression(f32),
    Classification(String),
}

/// L = number of layers (except the first (input) layer)
/// l = current layer
/// N = number of neurons in the lth layer
/// n = current neuron from the layer N
/// M = number of neurons in the (l - 1)th layer
/// m = current neuron from the layer M
pub struct Network<'a> {
    network_type: NetworkType,

    /// the input fields
    pub input: Vec<f32>,

    /// outer array has len of L
    /// each element represents a layer (l)
    /// does not inluclude the first (input) layer
    ///
    /// inner arrays can have different len of N
    /// each element represents a neuron's (n) value before activation
    pub layers: Vec<Vec<f32>>,

    /// outer array has len of L
    /// each element represents a layer (l)
    /// does not inluclude the first (input) layer
    ///
    /// inner arrays can have different len of N
    /// each element represents a neuron's (n) value after activation
    pub activated_layers: Vec<Vec<f32>>,

    /// outer array has len of (L - 1)
    /// each element represents all the weights between lth layer (N) and (l - 1)th layer (M)
    ///
    /// middle arrays can have different len of N
    /// each element represents all weights connected to neuron n
    ///
    /// inner array can have different len of M
    /// each element represents a weight conected from neuron m to neuron n
    pub weights: Vec<Vec<Vec<f32>>>,

    /// outer array has len of L
    /// each element represents a layer (l)
    /// does not inluclude the first (input) layer
    ///
    /// inner arrays can have different len of N
    /// each element represents a bias of a neuron (n)
    biases: Vec<Vec<f32>>,

    /// array has len of (L - 1)
    /// contains activation functions for all layers except the first (input) layer
    pub activation_functions: Vec<&'a ActivationFunction<'a>>,

    cost_func: &'a CostFunc<'a>,

    /// log epochs when training?
    pub log_epochs: bool,

    /// log costs at the beginning and at the end of training?
    pub log_costs: bool,

    normalization_fn: &'a NormalizationFn<'a>,
}

impl<'a> Network<'a> {
    /// Creates a new network
    ///  - shape: the shape of the network, each element of the array is equal to the number of
    ///  neurons in the layer
    ///  - network_type: classification or regression
    ///  - activation_functions: activation functions for each layer, including output, excluding
    ///  input
    ///  - cost_func: cost function
    ///  - normalization_func: normalization function
    pub fn new(
        shape: Vec<usize>,
        network_type: NetworkConstructorType,
        activation_functions: Vec<&'a ActivationFunction>,
        cost_func: &'a CostFunc,
        normalization_fn: &'a NormalizationFn,
    ) -> Self {
        assert_eq!(activation_functions.len(), shape.len() - 1);
        let network_t = match network_type {
            NetworkConstructorType::Classification(classes) => {
                assert_eq!(classes.len(), *shape.last().unwrap());
                NetworkType::Classification {
                    expected: "".to_string(),
                    predicted: "".to_string(),
                    classes,
                }
            }
            NetworkConstructorType::Regression => {
                assert_eq!(*shape.last().unwrap(), 1);
                NetworkType::Regression {
                    expected: 0.0,
                    predicted: 0.0,
                }
            }
        };

        let input = vec![0.0; shape[0]];
        let ls = &shape[1..];

        let mut layers: Vec<Vec<f32>> = vec![];
        let mut activated_layers: Vec<Vec<f32>> = vec![];

        for cnt in ls {
            layers.push(vec![0.0; *cnt].into());
            activated_layers.push(vec![0.0; *cnt].into());
        }

        let mut s = Self {
            input,
            layers,
            activated_layers,
            weights: vec![],
            biases: vec![],
            activation_functions,
            cost_func,
            log_epochs: false,
            log_costs: false,
            normalization_fn,
            network_type: network_t,
        };

        s.initialize_params();

        s
    }

    /// randomly sets weights and biases
    fn initialize_params(&mut self) {
        let mut biases: Vec<Vec<f32>> = vec![];
        let mut weights: Vec<Vec<Vec<f32>>> = vec![];

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_biases: Vec<f32> = vec![0.0; layer.len()];
            let mut layer_weights: Vec<Vec<f32>> = vec![];

            let layer_weights_init_fn = self.activation_functions[i].init_fn.function;

            for _ in 0..layer.len() {
                let mut neuron_weights: Vec<f32> = vec![];
                // count of neurons in prevous layer
                let neuron_cnt = if i == 0 {
                    self.input.len()
                } else {
                    self.layers[i - 1].len()
                };

                for _ in 0..neuron_cnt {
                    neuron_weights.push(layer_weights_init_fn(neuron_cnt));
                }

                layer_weights.push(neuron_weights.into());
            }

            biases.push(layer_biases.into());
            weights.push(layer_weights.into());
        }

        self.weights = weights.into();
        self.biases = biases.into();
    }

    /// feedforwards (sets values) to the layer (in layers and activated_layers) with the given index
    fn feedforward_layer(&mut self, layer_index: usize) {
        let tmp;
        let prev_layer = if layer_index == 0 {
            &self.input
        } else {
            tmp = self.activated_layers[layer_index - 1].clone();
            &tmp
        };

        for i in 0..self.layers[layer_index].len() {
            let weights = &self.weights[layer_index][i];
            let bias = self.biases[layer_index][i];

            let activation_function = self.activation_functions[layer_index].function;

            let val = dot_product(prev_layer, weights) + bias;
            self.layers[layer_index][i] = val;
            self.activated_layers[layer_index][i] = activation_function(val)
        }
    }

    fn feedforward(&mut self) {
        for i in 0..self.layers.len() {
            self.feedforward_layer(i);
        }
    }

    // returns change in weights and biases
    fn backprop(&self) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        // next = l - 1 layer (as in next iteration - range is backwards)
        // all throughout this function

        let expected: Vec<f32> = match &self.network_type {
            NetworkType::Classification {
                expected, classes, ..
            } => {
                let index = classes.iter().position(|class| class == expected).unwrap();
                let mut _expected: Vec<f32> = vec![0.0; classes.len()];
                _expected[index] = 1.0;

                _expected
            }
            NetworkType::Regression { expected, .. } => vec![*expected],
        };

        // dC/da(l) for any layer l in the network
        let mut layer_cost_activation_derivatives: Vec<f32> =
            (self.cost_func.derivative)(self.activated_layers.last().unwrap(), &expected);

        let mut dws: Vec<Vec<Vec<f32>>> = vec![];
        let mut dbs: Vec<Vec<f32>> = vec![];

        for l in (0..self.activated_layers.len()).rev() {
            // z[l]
            let layer = &self.layers[l];
            // a[l]
            let activated_layer = &self.activated_layers[l];
            // a[l - 1]
            let next_activated_layer = if l == 0 {
                &self.input
            } else {
                &self.activated_layers[l - 1]
            };
            // w[l]
            let layer_weights = &self.weights[l];

            // f'[l]
            let layer_activation_deriv_func = self.activation_functions[l].derivative;

            let mut layer_dws: Vec<Vec<f32>> = vec![];
            let mut layer_dbs: Vec<f32> = vec![];

            // dC/da[l - 1]
            let mut next_layer_cost_activation_derivatives: Vec<f32> =
                vec![0.0; next_activated_layer.len()];

            for j in 0..activated_layer.len() {
                let mut neuron_dws: Vec<f32> = vec![];

                // z[l][j]
                let z = layer[j];
                // w[l][j][k]
                let neuron_weights = &layer_weights[j];

                // dC/da[l][j]
                let cost_activation_deriv = layer_cost_activation_derivatives[j];

                // da[l][j]/dz[l][j]
                let activation_value_deriv = layer_activation_deriv_func(z);

                // dC/dz[l][j] = root[l][j]
                let root = cost_activation_deriv * activation_value_deriv;

                // dC/db[l][j]
                let cost_bias_deriv = root;
                layer_dbs.push(cost_bias_deriv);

                for k in 0..neuron_weights.len() {
                    // w[l][j][k]
                    let w = neuron_weights[k];

                    // dC/dw[l][j][k]
                    let cost_weight_deriv = root * next_activated_layer[k];
                    neuron_dws.push(cost_weight_deriv);

                    // dC/da[l - 1][k]
                    let cost_activation_next_deriv = root * w;
                    next_layer_cost_activation_derivatives[k] += cost_activation_next_deriv;
                }
                layer_dws.push(neuron_dws);
            }
            layer_cost_activation_derivatives = next_layer_cost_activation_derivatives;

            dws.push(layer_dws);
            dbs.push(layer_dbs);
        }

        (dws, dbs)
    }

    /// does gradient descent over a mini batch
    fn gradient_descent(&mut self, batch: TrainingData, learning_rate: f32) {
        let batch_len = match batch {
            TrainingData::Classification(value) => value.len(),
            TrainingData::Regression(value) => value.len(),
        };

        let mut total_dws: Vec<Vec<Vec<f32>>> = vec![];
        let mut total_dbs: Vec<Vec<f32>> = vec![];

        for l in 0..self.weights.len() {
            let mut layer_dws: Vec<Vec<f32>> = vec![];
            let layer_dbs: Vec<f32> = vec![0.0; self.weights[l].len()];

            for j in 0..self.weights[l].len() {
                let neuron_dws = vec![0.0; self.weights[l][j].len()];

                layer_dws.push(neuron_dws);
            }

            total_dws.push(layer_dws);
            total_dbs.push(layer_dbs);
        }

        // for (input, expected) in batch.iter() {
        for i in 0..batch_len {
            let input: Vec<f32> = match batch {
                TrainingData::Regression(value) => value[i].0,
                TrainingData::Classification(value) => value[i].0,
            };

            assert_eq!(input.len(), self.input.len());

            let new_network_type: NetworkType = match &batch {
                TrainingData::Regression(value) => NetworkType::Regression {
                    expected: value[i].1.clone(),
                    predicted: 0.0,
                },
                TrainingData::Classification(value) => match &self.network_type {
                    NetworkType::Classification { classes, .. } => NetworkType::Classification {
                        expected: value[i].1.clone(),
                        predicted: "".to_string(),
                        classes: classes.clone(),
                    },
                    _ => {
                        unreachable!()
                    }
                },
            };
            self.network_type = new_network_type;

            self.predict(input);

            let (dws, dbs) = self.backprop();

            for l in 0..dws.len() {
                for j in 0..dws[l].len() {
                    for k in 0..dws[l][j].len() {
                        total_dws[l][j][k] += dws[l][j][k];
                    }
                    total_dbs[l][j] += dbs[l][j];
                }
            }
        }

        for l in 0..total_dws.len() {
            for j in 0..total_dws[l].len() {
                for k in 0..total_dws[l][j].len() {
                    self.weights[l][j][k] -=
                        total_dws[l][j][k] / (batch_len as f32) * learning_rate;
                }

                self.biases[l][j] -= total_dbs[l][j] / (batch_len as f32) * learning_rate;
            }
        }
    }

    /// splits training data into mini batches
    fn batch_gradient_descent(
        &mut self,
        training_data: &'a TrainingData,
        learning_rate: f32,
        batch_size: usize,
    ) {
        let training_data_len = match training_data {
            TrainingData::Regression(value) => value.len(),
            TrainingData::Classification(value) => value.len(),
        };

        for batch_start_index in (0..training_data_len).step_by(batch_size) {
            let batch = match training_data {
                TrainingData::Regression(value) => TrainingData::Regression(
                    value[batch_start_index..batch_start_index + batch_size].to_vec(),
                ),
                TrainingData::Classification(value) => TrainingData::Classification(
                    value[batch_start_index..batch_start_index + batch_size].to_owned(),
                ),
            };

            self.gradient_descent(batch, learning_rate);
        }
    }

    pub fn train(
        &mut self,
        training_data: TrainingData,
        iteration_cnt: usize,
        learning_rate: f32,
        batch_size: usize,
    ) {
        let mut rng = rand::thread_rng();

        let time_start = Local::now();
        println!("beginning training at {time_start}");

        let training_data_shuffled = match training_data {
            TrainingData::Classification(mut value) => {
                value.shuffle(&mut rng);
                TrainingData::Classification(value)
            }
            TrainingData::Regression(mut value) => {
                value.shuffle(&mut rng);
                TrainingData::Regression(value)
            }
        };

        if self.log_costs {
            println!(
                "average cost: {}",
                self.average_cost(&training_data_shuffled)
            );
        }

        for i in 0..iteration_cnt {
            if self.log_epochs {
                let epoch = i + 1;
                let time_epoch = Local::now();

                println!("beginning training epoch {epoch} out of {iteration_cnt} at {time_epoch}");
            }

            self.batch_gradient_descent(&training_data_shuffled, learning_rate, batch_size);
        }

        if self.log_costs {
            println!(
                "average cost: {}",
                self.average_cost(&training_data_shuffled)
            );
        }

        let time_end = Local::now();
        println!("finishing training at {time_end}");
    }

    pub fn train_from_csv(&mut self) {}

    /// predicts the output from the given data
    pub fn predict(&mut self, data: Vec<f32>) {
        self.input = data.into();
        (self.normalization_fn.function)(self);

        self.feedforward();

        match &self.network_type {
            NetworkType::Classification {
                expected, classes, ..
            } => {
                let predicted = zip(classes.iter(), self.activated_layers.last().unwrap().iter())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;

                self.network_type = NetworkType::Classification {
                    predicted: *predicted,
                    expected: *expected,
                    classes: classes.to_vec(),
                }
            }
            NetworkType::Regression { expected, .. } => {
                let predicted = self.activated_layers.last().unwrap()[0];
                self.network_type = NetworkType::Regression {
                    expected: *expected,
                    predicted,
                }
            }
        }
    }

    /// returns a hasmap with keys from out_labels and their corresponding values
    pub fn get_output(&self) -> NetworkOutputType {
        match self.network_type {
            NetworkType::Classification { predicted, .. } => {
                NetworkOutputType::Classification(predicted)
            }
            NetworkType::Regression { predicted, .. } => NetworkOutputType::Regression(predicted),
        }
    }

    pub fn get_output_str(&self) -> String {
        match self.network_type {
            NetworkType::Classification { predicted, .. } => predicted,
            NetworkType::Regression { predicted, .. } => predicted.to_string(),
        }
    }

    /// predicts values
    /// computes the average cost against the training data
    pub fn average_cost(&mut self, data: &TrainingData) -> f32 {
        let mut cost_sum = 0.0;
        let data_len = match data {
            TrainingData::Regression(value) => value.len(),
            TrainingData::Classification(value) => value.len(),
        };

        for i in 0..data_len {
            let inputs = match data {
                TrainingData::Regression(value) => value[i].0,
                TrainingData::Classification(value) => value[i].0,
            };

            let expected: Vec<f32> = match self.network_type {
                NetworkType::Classification {
                    expected, classes, ..
                } => {
                    let index = classes.iter().position(|&class| class == expected).unwrap();
                    let _expected: Vec<f32> = vec![0.0; classes.len()];
                    _expected[index] = 1.0;

                    _expected
                }
                NetworkType::Regression { expected, .. } => vec![expected],
            };

            self.predict(inputs);
            cost_sum += (self.cost_func.function)(
                self.activated_layers.last().unwrap(),
                &expected.to_vec(),
            );
        }

        cost_sum / (data_len as f32)
    }

    pub fn debug(&self) {
        println!("inputs: {:?}", self.input);
        println!("layers: {:?}", self.layers);
        println!("activations: {:?}", self.activated_layers);
        println!("weights: {:?}", self.weights);
        println!("biases: {:?}", self.biases);
    }
}

#[cfg(test)]
mod tests {
    use super::{Network, NetworkConstructorType};
    use crate::functions::{
        activation::SIGMOID, cost::MSE, input_normalizations::NO_NORMALIZATION,
    };

    #[test]
    fn test_ws_cnt() {
        let net = Network::new(
            vec![3, 2, 3, 2],
            NetworkConstructorType::Classification(vec!["1".to_string(), "2".to_string()]),
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
        let net = Network::new(
            vec![3, 2, 3, 2],
            NetworkConstructorType::Classification(vec!["1".to_string(), "2".to_string()]),
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
        let net = Network::new(
            vec![3, 2, 3, 2],
            NetworkConstructorType::Classification(vec!["1".to_string(), "2".to_string()]),
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
        total_neurons += net.input.len();

        assert_eq!(total_neurons, 10);
    }

    #[test]
    fn test_feedforward_layer() {
        let mut net = Network::new(
            vec![3, 3],
            NetworkConstructorType::Classification(vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
            ]),
            vec![&SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );

        net.input = vec![3.0, 2.0];
        net.weights[0][0] = vec![3.0, 4.0];
        net.weights[0][1] = vec![2.0, 4.0];
        net.weights[0][2] = vec![3.0, 5.0];
        net.biases[0] = vec![1.0, 1.0, 3.0];

        net.feedforward_layer(0);

        assert_eq!(net.layers[0][0], 18.0);
        assert_eq!(net.layers[0][1], 15.0);
        assert_eq!(net.layers[0][2], 22.0);
    }

    #[test]
    fn test_output() {
        let mut net = Network::new(
            vec![3, 3],
            NetworkConstructorType::Classification(vec![
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
            ]),
            vec![&SIGMOID],
            &MSE,
            &NO_NORMALIZATION,
        );

        net.activated_layers[0] = vec![3.0, 2.0, 1.0];
        let out = net.get_output_str();
        assert_eq!(out, "1");
    }
}

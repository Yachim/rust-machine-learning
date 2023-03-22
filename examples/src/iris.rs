use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::MSE,
        input_normalizations::NORMALIZATION,
    },
    supervised::network::{
        multi_layer_perceptron::classification::MultiLayerPerceptronClassification, CSVTestable,
    },
};
use std::path::PathBuf;

// https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv
// FIXME: low accuracy
fn main() {
    let mut net = MultiLayerPerceptronClassification::new(
        vec![4, 6, 3],
        vec![&RELU, &SIGMOID],
        &MSE,
        &NORMALIZATION,
        vec!["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    );

    let path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/iris/Iris.csv");

    let accuracy =
        net.train_and_test_from_csv(path, 5, &vec![1, 2, 3, 4], 0.67, 32, 1_000_000, true, 0.001);

    println!("Iris accuracy: {}%", accuracy * 100.0);
}

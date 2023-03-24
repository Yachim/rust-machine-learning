use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::CROSS_ENTROPY,
        input_normalizations::NORMALIZATION,
    },
    supervised::network::{
        multi_layer_perceptron::classification::MultiLayerPerceptronClassification,
        CSVCostComputable, CSVTestable,
    },
};
use std::path::PathBuf;

// https://www.kaggle.com/datasets/uciml/iris?select=Iris.csv
// FIXME: low accuracy
fn main() {
    let mut net = MultiLayerPerceptronClassification::new(
        vec![4, 6, 3],
        vec![&RELU, &SIGMOID],
        &CROSS_ENTROPY,
        &NORMALIZATION,
        vec!["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
    );

    let path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/iris/Iris.csv");

    let label_col: usize = 5;
    let data_cols: Vec<usize> = vec![1, 2, 3, 4];

    let before_training_cost = net.avg_cost_from_csv(path, label_col, &data_cols);

    let accuracy =
        net.train_and_test_from_csv(path, label_col, &data_cols, 0.67, 32, 10_000, true, 0.001);

    let after_training_cost = net.avg_cost_from_csv(path, label_col, &data_cols);

    println!();
    println!(
        "Cost before training: {}, cost after: {}",
        before_training_cost, after_training_cost
    );
    println!("Iris accuracy: {}%", accuracy * 100.0);
}

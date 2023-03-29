use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::CROSS_ENTROPY,
        input_normalizations::NORMALIZATION,
    },
    supervised::network::{
        multi_layer_perceptron::classification::MultiLayerPerceptronClassification,
        CSVCostComputable, CSVPredictable, CSVTrainable, Debuggable,
    },
};
use std::path::PathBuf;

// https://www.kaggle.com/competitions/digit-recognizer
fn main() {
    let mut net = MultiLayerPerceptronClassification::new(
        vec![784, 16, 16, 10],
        vec![&RELU, &RELU, &SIGMOID],
        &CROSS_ENTROPY,
        &NORMALIZATION,
        vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    );

    let training_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/train.csv");

    let label_col: usize = 0;
    let data_cols: Vec<usize> = (1..785).collect();

    let before_training_cost = net.avg_cost_from_csv(training_path, label_col, &data_cols);
    net.train_from_csv(training_path, label_col, &data_cols, 256, 100, 0.001);
    let after_training_cost = net.avg_cost_from_csv(training_path, label_col, &data_cols);

    println!(
        "Cost before training: {}, cost after: {}",
        before_training_cost, after_training_cost
    );

    let predicting_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/test.csv");
    let out_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/out.csv");

    let weights = net.get_weights();
    let biases = net.get_biases();
    let inputs = net.get_inputs();
    let normalized_inputs = net.get_normalized_inputs();
    let layers = net.get_layers();
    let activated_layers = net.get_activated_layers();

    println!();
    println!("weights: {weights:#?}");
    println!("biases: {biases:#?}");
    println!("inputs: {inputs:#?}");
    println!("normalized inputs: {normalized_inputs:#?}");
    println!("layers: {layers:#?}");
    println!("activated layers: {activated_layers:#?}");
    println!();

    net.predict_from_into_csv(
        predicting_path,
        out_path,
        "ImageId",
        "Label",
        &(0..784).collect(),
        1,
    );
}

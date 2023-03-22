use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::MSE,
        input_normalizations::NORMALIZATION,
    },
    supervised::network::{
        multi_layer_perceptron::classification::MultiLayerPerceptronClassification, CSVPredictable,
        CSVTrainable,
    },
};
use std::path::PathBuf;

fn main() {
    let mut net = MultiLayerPerceptronClassification::new(
        vec![784, 16, 16, 10],
        vec![&RELU, &RELU, &SIGMOID],
        &MSE,
        &NORMALIZATION,
        vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    );

    let training_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/train.csv");

    net.train_from_csv(training_path, 0, &(1..785).collect(), 256, 100);

    let predicting_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/test.csv");
    let out_path = &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/digits/out.csv");

    net.predict_from_into_csv(
        predicting_path,
        out_path,
        "ImageId",
        "Label",
        &(0..784).collect(),
        1,
    );
}

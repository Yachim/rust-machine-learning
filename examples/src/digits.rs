use csv::Error;
use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::MSE,
        input_normalizations::NORMALIZATION,
    },
    network::{Network, NetworkConstructorType, NetworkType},
    utils::csv::{load_labeled_data, load_unlabeled_data},
};

fn main() -> Result<(), Error> {
    // https://www.kaggle.com/competitions/digit-recognizer/data
    let csv_loader = CsvDataLoader::new();

    let labels = vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

    let mut net = Network::new(
        vec![784, 16, 16, 10],
        NetworkConstructorType::Classification(labels),
        vec![&RELU, &RELU, &SIGMOID],
        &MSE,
        &NORMALIZATION,
    );
    net.log_costs = true;
    net.log_epochs = true;

    let training_set = load_labeled_data("examples/data/digits/train.csv", 0, 1..785, NetworkType)?;

    net.train(training_set, 100, 0.001, 10);

    let test_data = load_unlabeled_data("examples/data/digits/test.csv", 0..784)?;

    let mut test_wtr = csv::Writer::from_path("examples/data/digits/out_relu.csv")?;
    test_wtr.write_record(&["ImageId", "Label"])?;

    for (i, test_input) in test_data.iter().enumerate() {
        net.predict(test_input.to_vec());
        let val = net.get_best_output().0;

        test_wtr.write_record(&[(i + 1).to_string(), val.to_string()])?;
    }

    test_wtr.flush()?;

    Ok(())
}

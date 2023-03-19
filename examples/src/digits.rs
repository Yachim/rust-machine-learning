use csv::Error;
use rustml::{
    functions::{
        activation::{RELU, SIGMOID},
        cost::MSE,
        input_normalizations::NORMALIZATION,
    },
    network::Network,
    utils::csv_data::{CsvDataLoader, Label},
};

fn main() -> Result<(), Error> {
    // https://www.kaggle.com/competitions/digit-recognizer/data
    let csv_loader = CsvDataLoader::new();

    let labels = vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
    let training_set = csv_loader.load_labeled_data(
        "examples/data/digits/train.csv",
        Label::SingleLabelClassification(0, &labels),
    )?;

    let mut net = Network::new(
        784,
        vec![16, 16],
        labels,
        vec![&RELU, &RELU, &SIGMOID],
        &MSE,
        &NORMALIZATION,
    );
    net.log_costs = true;
    net.log_epochs = true;

    net.train(training_set, 100, 0.001, 10);

    let test_data = csv_loader.load_unlabeled_data("examples/data/digits/test.csv")?;

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

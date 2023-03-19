use crate::network::{NetworkType, TrainingData};
use csv::{Error, Reader};
use std::iter::zip;

/// data for training and testing
pub fn load_labeled_data<'a>(
    file_path: &str,
    label_col: usize,
    data_cols: Vec<usize>,
    network_type: NetworkType,
) -> Result<TrainingData, Error> {
    let mut rdr = Reader::from_path(file_path)?;

    let mut inputs: Vec<Vec<f32>> = vec![];
    let mut expected_floats: Vec<f32> = vec![];
    let mut expected_strs: Vec<String> = vec![];

    for result in rdr.records() {
        let record = result?;

        let expected = record[label_col].to_owned();
        match network_type {
            NetworkType::Regression { .. } => {
                expected_floats.push(expected.parse::<f32>().expect("label is not a number"))
            }
            NetworkType::Classification { .. } => expected_strs.push(expected),
        }

        let inputs_for_sample: Vec<f32> = data_cols
            .iter()
            .map(|&i| {
                (&record)[i]
                    .parse::<f32>()
                    .expect("value in file is not a number")
            })
            .collect();
        inputs.push(inputs_for_sample);
    }

    match network_type {
        NetworkType::Regression { .. } => Ok(TrainingData::Regression(
            zip(inputs, expected_floats).collect(),
        )),
        NetworkType::Classification { .. } => Ok(TrainingData::Classification(
            zip(inputs, expected_strs).collect(),
        )),
    }
}

/// data for predicting
pub fn load_unlabeled_data(file_path: &str, data_cols: Vec<usize>) -> Result<Vec<Vec<f32>>, Error> {
    let mut rdr = Reader::from_path(file_path)?;

    let mut inputs: Vec<Vec<f32>> = vec![];

    for result in rdr.records() {
        let record = result?;

        let inputs_for_sample: Vec<f32> = data_cols
            .iter()
            .map(|&i| {
                (&record)[i]
                    .parse::<f32>()
                    .expect("value in file is not a number")
            })
            .collect();
        inputs.push(inputs_for_sample);
    }

    Ok(inputs)
}

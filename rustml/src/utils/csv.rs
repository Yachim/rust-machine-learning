use csv::{Error, Reader};
use std::iter::zip;
use std::path::Path;

type TrainingData<'a> = Vec<(Vec<f32>, String)>;

/// data for training and testing
pub fn load_labeled_data<'a>(
    file_path: &Path,
    label_col: usize,
    data_cols: Vec<usize>,
) -> Result<TrainingData, Error> {
    let mut rdr = Reader::from_path(file_path)?;

    let mut inputs: Vec<Vec<f32>> = vec![];
    let mut expected: Vec<String> = vec![];

    for result in rdr.records() {
        let record = result?;

        let expected_for_sample = record[label_col].to_owned();
        expected.push(expected_for_sample);

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

    Ok(zip(inputs, expected).collect())
}

/// data for predicting
pub fn load_unlabeled_data(
    file_path: &Path,
    data_cols: Vec<usize>,
) -> Result<Vec<Vec<f32>>, Error> {
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

#[cfg(test)]
mod tests {
    use super::{load_labeled_data, load_unlabeled_data};
    use std::path::PathBuf;

    #[test]
    fn test_labeled_data() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/csv_testing/file1.csv");

        let res = load_labeled_data(&path, 0, vec![1, 3, 4]).unwrap();
        assert_eq!(
            res,
            vec![
                (vec![2.0, 4.0, 5.0], String::from("1")),
                (vec![3.0, 0.0, 2.0], String::from("3")),
                (vec![0.0, 4.0, 2.0], String::from("1")),
            ]
        );
    }

    #[test]
    fn test_unlabeled_data() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/csv_testing/file1.csv");

        let res = load_unlabeled_data(&path, vec![3, 4]).unwrap();
        assert_eq!(
            res,
            vec![(vec![4.0, 5.0]), (vec![0.0, 2.0]), (vec![4.0, 2.0]),]
        );
    }
}

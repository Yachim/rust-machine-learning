use super::misc::one_hot_encode;
use csv::{Error, Reader, Writer};
use std::path::Path;

type TrainingData = Vec<(Vec<f32>, String)>;
type TrainingDataOHC = Vec<(Vec<f32>, Vec<f32>)>;

/// data for training and testing
pub fn load_labeled_data(
    file_path: &Path,
    label_col: usize,
    data_cols: &Vec<usize>,
) -> Result<TrainingData, Error> {
    let mut rdr = Reader::from_path(file_path)?;

    let mut out: TrainingData = vec![];

    for result in rdr.records() {
        let record = result?;

        let expected = record[label_col].to_owned();

        let inputs: Vec<f32> = data_cols
            .iter()
            .map(|&i| {
                (&record)[i]
                    .parse::<f32>()
                    .expect("value in file is not a number")
            })
            .collect();

        out.push((inputs, expected));
    }

    Ok(out)
}

/// data for training and testing
/// one hot encoded
/// FIXME: violates DRY with load_labeled_data
pub fn load_labeled_data_ohc(
    file_path: &Path,
    label_col: usize,
    data_cols: &Vec<usize>,
    labels: &Vec<&str>,
) -> Result<TrainingDataOHC, Error> {
    let mut rdr = Reader::from_path(file_path)?;

    let mut out: TrainingDataOHC = vec![];

    for result in rdr.records() {
        let record = result?;

        let expected = record[label_col].to_owned();

        let inputs: Vec<f32> = data_cols
            .iter()
            .map(|&i| {
                (&record)[i]
                    .parse::<f32>()
                    .expect("value in file is not a number")
            })
            .collect();

        out.push((inputs, one_hot_encode(labels, &expected)));
    }

    Ok(out)
}

/// data for predicting
pub fn load_unlabeled_data(
    file_path: &Path,
    data_cols: &Vec<usize>,
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

/// data: (id, label)
pub fn write_data(
    file_path: &Path,
    id_header: &str,
    label_header: &str,
    data: &Vec<(String, String)>,
) -> Result<(), Error> {
    let mut wtr = Writer::from_path(file_path)?;
    wtr.write_record(&[id_header, label_header])?;

    for (id, label) in data {
        wtr.write_record(&[id, label])?;
    }

    wtr.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{load_labeled_data, load_labeled_data_ohc, load_unlabeled_data};
    use std::path::PathBuf;

    #[test]
    fn test_labeled_data() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/csv_testing/file1.csv");

        let res = load_labeled_data(&path, 0, &vec![1, 3, 4]).unwrap();
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
    fn test_labeled_data_ohc() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/csv_testing/file1.csv");

        let res = load_labeled_data_ohc(&path, 0, &vec![1, 3, 4], &vec!["1", "2", "3"]).unwrap();
        assert_eq!(
            res,
            vec![
                (vec![2.0, 4.0, 5.0], vec![1.0, 0.0, 0.0]),
                (vec![3.0, 0.0, 2.0], vec![0.0, 0.0, 1.0]),
                (vec![0.0, 4.0, 2.0], vec![1.0, 0.0, 0.0]),
            ]
        );
    }

    #[test]
    fn test_unlabeled_data() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/utils/csv_testing/file1.csv");

        let res = load_unlabeled_data(&path, &vec![3, 4]).unwrap();
        assert_eq!(
            res,
            vec![(vec![4.0, 5.0]), (vec![0.0, 2.0]), (vec![4.0, 2.0]),]
        );
    }
}

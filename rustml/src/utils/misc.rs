pub fn one_hot_encode(labels: &Vec<&str>, label: &str) -> Vec<f32> {
    let i = labels.iter().position(|&val| val == label).unwrap();
    let mut out = vec![0.0; labels.len()];

    out[i] = 1.0;

    out
}

#[cfg(test)]
mod tests {
    use super::one_hot_encode;

    #[test]
    fn test_one_hot_encode() {
        let one_hot_encoded = one_hot_encode(&vec!["1", "2", "3", "4"], "2");

        assert_eq!(one_hot_encoded, vec![0.0, 1.0, 0.0, 0.0]);
    }
}

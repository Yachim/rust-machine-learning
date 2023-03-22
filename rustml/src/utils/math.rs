use std::iter::zip;

pub fn _hadamard_product(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|x| x.0 * x.1).collect()
}

/// divide each vector by float
pub fn divide_vector_float(v: &Vec<f32>, denominator: f32) -> Vec<f32> {
    v.iter().map(|x| x / denominator).collect()
}

/// divide each vector by float
pub fn divide_vector_float_2d(v: &Vec<Vec<f32>>, denominator: f32) -> Vec<Vec<f32>> {
    v.iter()
        .map(|x| divide_vector_float(x, denominator))
        .collect()
}

/// divide each vector by float
pub fn divide_vector_float_3d(v: &Vec<Vec<Vec<f32>>>, denominator: f32) -> Vec<Vec<Vec<f32>>> {
    v.iter()
        .map(|x| divide_vector_float_2d(x, denominator))
        .collect()
}

pub fn dot_product(v1: &Vec<f32>, v2: &Vec<f32>) -> f32 {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|x| x.0 * x.1).sum::<f32>()
}

pub fn add_vecs(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| a + b).collect()
}

pub fn add_2d_vecs(v1: &Vec<Vec<f32>>, v2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| add_vecs(a, b)).collect()
}

pub fn add_3d_vecs(v1: &Vec<Vec<Vec<f32>>>, v2: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| add_2d_vecs(a, b)).collect()
}

pub fn subtract_vecs(v1: &Vec<f32>, v2: &Vec<f32>) -> Vec<f32> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| a - b).collect()
}

pub fn subtract_2d_vecs(v1: &Vec<Vec<f32>>, v2: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| subtract_vecs(a, b)).collect()
}

pub fn subtract_3d_vecs(v1: &Vec<Vec<Vec<f32>>>, v2: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(v1.len(), v2.len());

    zip(v1, v2).map(|(a, b)| subtract_2d_vecs(a, b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let res = dot_product(&vec![4.0, 5.0], &vec![10.0, 20.0]);
        assert_eq!(res, 140.0);
    }

    #[test]
    fn test_hadamard_product() {
        let res = _hadamard_product(&vec![2.0, 3.0], &vec![4.0, 5.0]);
        assert_eq!(res, vec![8.0, 15.0]);
    }

    #[test]
    fn test_add_subtract() {
        let v1 = vec![5.0, 6.0];
        let v2 = vec![2.0, 4.0];

        let sum = add_vecs(&v1, &v2);
        let diff = subtract_vecs(&v1, &v2);

        assert_eq!(sum, vec![7.0, 10.0]);
        assert_eq!(diff, vec![3.0, 2.0]);
    }

    #[test]
    fn test_add_subtract_2d() {
        let v1 = vec![vec![1.0, 2.0], vec![3.0, 3.0]];
        let v2 = vec![vec![2.0, 1.0], vec![3.0, 1.0]];

        let sum = add_2d_vecs(&v1, &v2);
        let diff = subtract_2d_vecs(&v1, &v2);

        assert_eq!(sum, vec![vec![3.0, 3.0], vec![6.0, 4.0]]);
        assert_eq!(diff, vec![vec![-1.0, 1.0], vec![0.0, 2.0]])
    }

    #[test]
    fn test_add_subtract_3d() {
        let v1 = vec![
            vec![vec![3.0, 2.0], vec![1.0, 4.0]],
            vec![vec![1.0, 4.0], vec![3.0, 2.0]],
        ];
        let v2 = vec![
            vec![vec![1.0, 1.0], vec![2.0, 3.0]],
            vec![vec![2.0, 3.0], vec![2.0, 1.0]],
        ];

        let sum = add_3d_vecs(&v1, &v2);
        let diff = subtract_3d_vecs(&v1, &v2);

        assert_eq!(
            sum,
            vec![
                vec![vec![4.0, 3.0], vec![3.0, 7.0]],
                vec![vec![3.0, 7.0], vec![5.0, 3.0]]
            ]
        );
        assert_eq!(
            diff,
            vec![
                vec![vec![2.0, 1.0], vec![-1.0, 1.0]],
                vec![vec![-1.0, 1.0], vec![1.0, 1.0]]
            ]
        )
    }

    #[test]
    fn test_divide() {
        let v = vec![4.0, 6.0];

        let out = divide_vector_float(&v, 2.0);

        assert_eq!(out, vec![2.0, 3.0]);
    }

    #[test]
    fn test_divide_2d() {
        let v = vec![vec![2.0, 4.0], vec![4.0, 6.0]];

        let out = divide_vector_float_2d(&v, 2.0);

        assert_eq!(out, vec![vec![1.0, 2.0], vec![2.0, 3.0]])
    }

    #[test]
    fn test_divide_3d() {
        let v = vec![
            vec![vec![2.0, 4.0], vec![2.0, 4.0]],
            vec![vec![6.0, 4.0], vec![4.0, 2.0]],
        ];

        let out = divide_vector_float_3d(&v, 2.0);

        assert_eq!(
            out,
            vec![
                vec![vec![1.0, 2.0], vec![1.0, 2.0]],
                vec![vec![3.0, 2.0], vec![2.0, 1.0]]
            ]
        );
    }
}

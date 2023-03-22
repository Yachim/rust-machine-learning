/*use rustml::{
    functions::{activation::SIGMOID, cost::MSE, input_normalizations::NO_NORMALIZATION},
    network::{Network, NetworkConstructorType},
};

fn main() {
    // https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
    let training_set = vec![
        (vec![0.0, 0.0, 1.0], vec![0.0]),
        (vec![1.0, 1.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0, 1.0], vec![1.0]),
        (vec![0.0, 1.0, 1.0], vec![0.0]),
    ];
    let mut net = Network::new(
        vec![3, 1],
        NetworkConstructorType::Regression,
        vec![&SIGMOID],
        &MSE,
        &NO_NORMALIZATION,
    );
    net.log_costs = true;

    net.train(training_set, 10000, 1.0, 4);

    let test_set = vec![1.0, 0.0, 0.0];
    net.predict(test_set);

    let out_map = net.get_output();
    let out = out_map.get("res").unwrap();

    println!("output: {out}");
}*/
fn main() {
    unimplemented!();
}

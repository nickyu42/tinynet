#include <iostream>
#include "Network.h"
#include "loader/DataLoader.h"

int main() {
    auto loader = toynet::MNISTLoader();
    auto training_data = loader.load({
                               {"images", "data/mnist/t10k-images.idx3-ubyte"},
                               {"labels", "data/mnist/t10k-labels.idx1-ubyte"}
                       });

    toynet::Network n({784, 30, 10});

    n.SGD(training_data, 30, 10, 3.0);

    std::cout << toynet::valarray_to_json(training_data[0].second);
    std::cout << toynet::valarray_to_json(n.feedforward(training_data[0].first));

    return 0;
}

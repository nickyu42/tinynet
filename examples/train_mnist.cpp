#include <iostream>
#include <fstream>

#include "../Network.h"
#include "../loader/DataLoader.h"

int main() {
    auto loader = toynet::MNISTLoader();
    auto training_data = loader.load({
                                             {"images", "data/mnist/train-images.idx3-ubyte"},
                                             {"labels", "data/mnist/train-labels.idx1-ubyte"}
                                     });

    toynet::Network n({784, 30, 10});

    n.SGD(training_data, 10, 10, 3.0);

    n.dump_parameters("data/mnist/parameters.json");

    return 0;
}

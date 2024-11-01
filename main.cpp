#include <iostream>
#include "Network.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    toynet::Network n({2, 3, 2});

    for (const toynet::Layer &l : n.layers) {
        std::cout << l << std::endl;
    }

    n.backpropogate_and_update(toynet::TrainingSample({1.0, 3.0}, {1.0, 0.0}));

    for (const toynet::Layer &l : n.layers) {
        std::cout << l << std::endl;
    }

    return 0;
}

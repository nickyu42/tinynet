#include <iostream>
#include "Network.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    toynet::Network n({2, 3, 2});

    const std::valarray<double> &out = n.feedforward({1.0, 2.0});

    for (auto x : out) {
        std::cout << x << " ";
    }

    std::cout << std::endl;

    for (const toynet::Layer &l : n.layers) {
        std::cout << l;
    }

    return 0;
}

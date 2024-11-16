#include <emscripten/bind.h>
#include "Network.h"

using namespace emscripten;

void (toynet::Network::*loadFromString)(const std::string&) = &toynet::Network::load_parameters;

std::vector<double> feedforward_wrapper(toynet::Network &net, const std::vector<double> &vec) {
    auto input = std::valarray<double>(vec.data(), vec.size());
    auto res = net.feedforward(input);
    return std::vector<double>(std::begin(res), std::end(res));
}

EMSCRIPTEN_BINDINGS(my_class_bindings) {
        class_<toynet::Network>("Network")
                .constructor<std::vector<unsigned int>>()
                .function("load_parameters", loadFromString)
                .function("feedforward", &toynet::Network::feedforward);

        function("feedforward_wrapper", &feedforward_wrapper);

        register_vector<unsigned int>("IntList");
        register_vector<double>("DoubleList");
}
//
// Created by Nick Yu on 12/09/2024.
//

#include "Network.h"

#include <random>
#include <iostream>

template <typename T>
T sigmoid(const T& z) {
    return 1.0 / (1.0 + exp(-z));
}

std::valarray<double> sigmoid_derivative(const std::valarray<double> &z) {
    auto s = sigmoid(z);
    return s * (1.0 - s);
}

const std::valarray<double> &toynet::Layer::feedforward(const std::valarray<double> &input) {
    assert(input.size() == this->m);

    for (size_t i = 0; i < n; i++) {
        z[i] = this->bias[i];

        for (size_t j = 0; j < this->m; j++) {
            z[i] += this->weights[i * n + j] * input[j];
        }

        // TODO: functionality for alternative activation functions
        activation[i] = sigmoid(z[i]);
    }

    return this->activation;
}

toynet::Layer::Layer(toynet::ActFunc a, size_t n, size_t m) : func(a), n(n), m(m), bias(n), weights(n * m),
                                                              activation(n), z(n), error(n), dC_db(n), dC_dw(n * m) {
    // XXX: Gaussian distribution for now
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0.0, 1.0};

    auto randf = [&d, &gen] { return d(gen); };

    for (size_t i = 0; i < n; i++) {
        bias[i] = randf();

        for (size_t j = 0; j < m; j++) {
            weights[i * n + j] = randf();
        }
    }
}

std::ostream &operator<<(std::ostream &strm, const toynet::Layer &l) {
    strm << "Layer(m=" << l.m << ",n=" << l.n << "," << l.func << ")";

    strm << std::endl << std::endl;

    for (auto w: l.bias) {
        strm << w << " ";
    }

    strm << std::endl << std::endl;

    for (size_t i = 0; i < l.n; i++) {
        for (size_t j = 0; j < l.m; j++) {
            strm << l.weights[i * l.n + j] << " ";
        }
        strm << std::endl;
    }

    return strm;
}

const std::valarray<double> &toynet::Network::feedforward(const std::valarray<double> &input) {
    std::valarray<double> current = input;

    for (auto &layer: this->layers) {
        current = layer.feedforward(current);
    }

    this->activation = current;

    return this->activation;
}

toynet::Network::Network(std::vector<unsigned int> sizes) : activation(sizes[sizes.size() - 1]) {
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        unsigned int m = sizes[i];
        unsigned int n = sizes[i + 1];

        Layer l(Sigmoid, n, m);
        layers.push_back(std::move(l));
    }
}

void
toynet::Network::SGD(std::vector<TrainingSample> training_data, unsigned int epochs, unsigned int mini_batch_size,
                     double eta) {
    std::default_random_engine rng{};
    std::shuffle(training_data.begin(), training_data.end(), rng);

    auto it = training_data.begin();

}

void toynet::Network::update_mini_batch(std::vector<TrainingSample>::iterator mini_batch_begin,
                                        std::vector<TrainingSample>::iterator mini_batch_end, double eta) {
}

void toynet::Network::backpropogate_and_update(const toynet::TrainingSample &sample) {
    // Calculate the error for the last layer
    std::valarray<double> cost_derivative = this->layers.back().activation - sample.second;

    // compute delta = C_gradient hadamard sigmoid_prime
    std::valarray<double> delta = cost_derivative * sigmoid_derivative(this->layers.back().z);

    this->layers.back().dC_db = delta;
    this->layers.back().dC_dw = this->layers[this->layers.size() - 2].activation * delta;

    // propagate the error backwards
}
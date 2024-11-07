#include "Network.h"

#include <random>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <fstream>
#include <cmath>

#include <nlohmann/json.hpp>

template<typename T>
T sigmoid(const T &z) {
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
            z[i] += this->weights[i * m + j] * input[j];
        }

        // TODO: functionality for alternative activation functions
        activation[i] = sigmoid(z[i]);
    }

    return this->activation;
}

toynet::Layer::Layer(toynet::ActFunc a, size_t n, size_t m) : func(a), n(n), m(m), bias(n), weights(n * m),
                                                              activation(n), z(n), dC_db(n), dC_dw(n * m) {
    // XXX: Gaussian distribution for now
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d{0.0, 1.0};

    auto randf = [&d, &gen] { return d(gen); };

    for (size_t i = 0; i < n; i++) {
        bias[i] = randf();

        for (size_t j = 0; j < m; j++) {
            weights[i * m + j] = randf();
        }
    }
}

std::string toynet::valarray_to_json(const std::valarray<double> &a) {
    std::ostringstream strm;
    strm << "[";
    for (size_t i = 0; i < a.size(); ++i) {
        if (i > 0) {
            strm << ", ";
        }
        strm << a[i];
    }
    strm << "]";
    return strm.str();
}

std::ostream &operator<<(std::ostream &strm, const toynet::Layer &l) {
    strm << "{\n"
         << "  \"n\": " << l.n << ",\n"
         << "  \"m\": " << l.m << ",\n"
         << "  \"weights\": " << toynet::valarray_to_json(l.weights) << ",\n"
         << "  \"bias\": " << toynet::valarray_to_json(l.bias) << ",\n"
         << "  \"activation\": " << toynet::valarray_to_json(l.activation) << ",\n"
         << "  \"z\": " << toynet::valarray_to_json(l.z) << ",\n"
         << "  \"dC_dw\": " << toynet::valarray_to_json(l.dC_dw) << ",\n"
         << "  \"dC_db\": " << toynet::valarray_to_json(l.dC_db) << "\n"
         << "}";

    return strm;
}

std::ostream &operator<<(std::ostream &strm, const toynet::Network &n) {
    strm << "[";
    for (size_t i = 0; i < n.layers.size(); ++i) {
        if (i > 0) {
            strm << ", ";
        }
        strm << n.layers[i];
    }
    strm << "]";
    return strm;
}

double quadratic_cost(const std::valarray<double> &x, const std::valarray<double> &y) {
    assert(x.size() == y.size());

    double loss = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        auto d = (x[i] - y[i]);
        loss += d * d;
    }

    return loss / 2;
}

double toynet::Network::compute_loss(const std::valarray<double> &y) {
    return quadratic_cost(y, this->layers.back().activation);
}

const std::valarray<double> &toynet::Network::feedforward(const std::valarray<double> &input) {
    // Set input layer activation
    layers[0].activation = input;

    for (size_t l = 1; l < layers.size(); l++) {
        layers[l].feedforward(layers[l - 1].activation);
    }

    return layers.back().activation;
}

toynet::Network::Network(std::vector<unsigned int> sizes) {
    // Input layer
    layers.push_back(std::move(Layer(Sigmoid, sizes[0], 1)));

    for (size_t i = 0; i < sizes.size() - 1; i++) {
        unsigned int m = sizes[i];
        unsigned int n = sizes[i + 1];

        Layer l(Sigmoid, n, m);
        layers.push_back(std::move(l));
    }
}

void
toynet::Network::SGD(std::vector<TrainingSample> training_data, unsigned int epochs, unsigned int mini_batch_size,
                     double eta, bool write_state) {
    std::default_random_engine rng{};

    std::ofstream out;
    if (write_state) {
        out = std::ofstream("out.json");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(training_data.begin(), training_data.end(), rng);

        for (unsigned int batch_start_i = 0; batch_start_i < mini_batch_size; batch_start_i += mini_batch_size) {
            auto batch_start = training_data.begin() + batch_start_i;
            auto batch_end = batch_start_i + mini_batch_size >= training_data.size()
                             ? training_data.end()
                             : batch_start + batch_start_i;

            update_mini_batch(batch_start, batch_end, eta);
        }

        std::cout << "Epoch [" << epoch << "]" << std::endl;

        // TODO: serialize network state
        if (write_state) {
            out << "{"
                << "\"epoch\":" << epoch << ","
                << *this
                << "}";
        }

    }

    if (write_state) {
        out.close();
    }
}

void toynet::Network::update_mini_batch(std::vector<TrainingSample>::iterator mini_batch_begin,
                                        std::vector<TrainingSample>::iterator mini_batch_end, double eta) {

    std::vector<std::valarray<double>> dC_db;
    std::vector<std::valarray<double>> dC_dw;

    dC_dw.reserve(this->layers.size());
    dC_db.reserve(this->layers.size());

    for (const auto &l: this->layers) {
        dC_dw.push_back(std::move(std::valarray<double>(l.n * l.m)));
        dC_db.push_back(std::move(std::valarray<double>(l.n)));
    }

    for (auto sample = mini_batch_begin; sample != mini_batch_end; sample++) {
        // Calculate error and for this training sample
        backpropogate_and_update(*sample);

        // Sum cost function gradients
        for (size_t i = 0; i < layers.size(); ++i) {
            dC_dw[i] += layers[i].dC_dw;
            dC_db[i] += layers[i].dC_db;
        }
    }

    auto m = static_cast<double>(std::distance(mini_batch_begin, mini_batch_end));

    // update weights and biases
    for (int i = 0; i < layers.size(); ++i) {
        layers[i].weights -= (eta / m) * dC_dw[i];
        layers[i].bias -= (eta / m) * dC_db[i];
    }
}

void toynet::Network::backpropogate_and_update(const toynet::TrainingSample &sample) {
    this->feedforward(sample.first);

    // Calculate the error for the last layer
    std::valarray<double> cost_derivative = this->layers.back().activation - sample.second;

    // compute delta = C_gradient hadamard sigmoid_prime
    std::valarray<double> delta = cost_derivative * sigmoid_derivative(this->layers.back().z);

    this->layers.back().dC_db = delta;
    this->layers.back().dC_dw = this->layers[this->layers.size() - 2].activation * delta;

    // propagate the error backwards
    for (size_t l_i = this->layers.size() - 2; l_i > 0; l_i--) {
        Layer &l = this->layers[l_i];
        Layer &l_next = this->layers[l_i + 1];

        std::valarray<double> t(l.z.size());

        for (size_t row = 0; row < l_next.m; row++) {
            for (size_t col = 0; col < l_next.n; col++) {
                // transposed matrix mul
                t[col] += l_next.weights[row + col * l_next.n] * delta[col];
            }
        }

        // hadamard
        t *= sigmoid_derivative(l.activation);

        delta = t;

        l.dC_db = delta;
        l.dC_dw = this->layers[l_i - 1].activation * delta;
    }
}

void toynet::Network::load_parameters(const std::string &json_state) {
    auto layer_parameters = nlohmann::json::parse(json_state).get<std::vector<nlohmann::json>>();

    // Example for a network of {2, 2, 2}
    //  [
    //      {"weights": [1, 2, 1, 0], "bias": [1, 0]},
    //      {"weights": [2, 2, 0, 1], "bias": [0, 3]}
    //  ]
    for (size_t l = 0; l < layer_parameters.size(); ++l) {
        auto &p = layer_parameters[l];
        auto w = p["weights"].get<std::vector<double>>();
        if (w.size() != this->layers[l + 1].n * this->layers[l + 1].m) {
            throw std::runtime_error("Given weights matrix does not match layer size");
        }
        this->layers[l + 1].weights = std::move(std::valarray<double>(w.data(), w.size()));

        auto b = p["bias"].get<std::vector<double>>();
        if (b.size() != this->layers[l + 1].n) {
            throw std::runtime_error("Given bias vector does not match layer size");
        }
        this->layers[l + 1].bias = std::move(std::valarray<double>(b.data(), b.size()));
    }
}

double vector_norm(const std::valarray<double> &v) {
    return sqrt((v * v).sum());
}

std::vector<double> toynet::Network::gradient_check(const toynet::TrainingSample &sample, double epsilon) {
    std::vector<double> diffs(layers.size() - 1);

    // Calculate and store gradients
    this->backpropogate_and_update(sample);

    for (size_t l = 1; l < layers.size(); ++l) {
        auto &layer = layers[l];
        std::vector<double> gradient_diff_vec = {};

        for (size_t i = 0; i < layer.weights.size(); ++i) {
            auto original_w = layer.weights[i];

            layer.weights[i] = original_w + epsilon;
            feedforward(sample.first);
            auto loss_p = compute_loss(sample.second);

            layer.weights[i] = original_w - epsilon;
            feedforward(sample.first);
            auto loss_n = compute_loss(sample.second);

            // Restore weight
            layer.weights[i] = original_w;

            // Finite differences approximation
            gradient_diff_vec.push_back((loss_p - loss_n) / (2 * epsilon));
        }

        for (size_t i = 0; i < layer.bias.size(); ++i) {
            auto original_b = layer.bias[i];

            layer.bias[i] = original_b + epsilon;
            feedforward(sample.first);
            auto loss_p = compute_loss(sample.second);

            layer.bias[i] = original_b - epsilon;
            feedforward(sample.first);
            auto loss_n = compute_loss(sample.second);

            // Restore bias
            layer.bias[i] = original_b;

            // Finite differences approximation
            gradient_diff_vec.push_back((loss_p - loss_n) / (2 * epsilon));
        }

        std::valarray<double> gradient_diff(gradient_diff_vec.data(), gradient_diff_vec.size());
        std::valarray<double> gradients(layer.weights.size() + layer.bias.size());
        gradients[std::slice(0, layer.weights.size(), 1)] = layer.dC_dw;
        gradients[std::slice(layer.weights.size(), layer.bias.size(), 1)] = layer.dC_db;

        auto diff = vector_norm(gradients - gradient_diff) / (vector_norm(gradient_diff) + vector_norm(gradients));

        std::cout << "Difference for layer " << l << ": " << diff << std::endl;

        diffs.push_back(diff);
    }

    return diffs;
}

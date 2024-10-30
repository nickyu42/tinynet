//
// Created by Nick Yu on 12/09/2024.
//

#ifndef NEURAL_NET_NETWORK_H
#define NEURAL_NET_NETWORK_H

#include <vector>
#include <cstdint>
#include <valarray>

namespace toynet {
    class Layer;
}

std::ostream &operator<<(std::ostream &strm, const toynet::Layer &l);

namespace toynet {
    enum ActFunc {
        Sigmoid
    };

    class Layer {
        size_t n;
        size_t m;
        ActFunc func;

    public:
        std::valarray<double> weights;
        std::valarray<double> bias;
        std::valarray<double> activation;
        std::valarray<double> z;

        std::valarray<double> error;
        std::valarray<double> dC_dw;
        std::valarray<double> dC_db;

        Layer(ActFunc a, size_t n, size_t m);

        friend std::ostream &(::operator<<)(std::ostream &strm, const toynet::Layer &l);

        const std::valarray<double> &feedforward(const std::valarray<double> &input);
    };

    using TrainingSample = std::pair<std::valarray<double>, std::valarray<double>>;

    class Network {
    public:
        std::vector<Layer> layers;
        std::valarray<double> activation;

        const std::valarray<double> &feedforward(const std::valarray<double> &input);
        void backpropogate_and_update(const TrainingSample &sample);

        void SGD(std::vector<TrainingSample> training_data, unsigned int epochs, unsigned int mini_batch_size,
                 double eta);

        void update_mini_batch(std::vector<TrainingSample>::iterator mini_batch_begin,
                               std::vector<TrainingSample>::iterator mini_batch_end, double eta);

        explicit Network(std::vector<unsigned int> sizes);
    };
}

#endif //NEURAL_NET_NETWORK_H

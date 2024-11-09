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
    public:
        size_t n;
        size_t m;
        ActFunc func;

        std::valarray<double> weights;
        std::valarray<double> bias;
        std::valarray<double> activation;
        std::valarray<double> z;

        std::valarray<double> dC_dw;
        std::valarray<double> dC_db;

        Layer(ActFunc a, size_t n, size_t m);

        const std::valarray<double> &feedforward(const std::valarray<double> &input);
    };

    using TrainingSample = std::pair<std::valarray<double>, std::valarray<double>>;

    class Network {
    public:
        std::vector<Layer> layers;

        const std::valarray<double> &feedforward(const std::valarray<double> &input);

        void backpropogate_and_update(const TrainingSample &sample);

        void SGD(std::vector<TrainingSample> training_data, unsigned int epochs, unsigned int mini_batch_size,
                 double eta, bool write_state = false);

        void update_mini_batch(std::vector<TrainingSample>::iterator mini_batch_begin,
                               std::vector<TrainingSample>::iterator mini_batch_end, double eta);

        double compute_loss(const std::valarray<double> &y);

        std::vector<double> gradient_check(const TrainingSample &sample, double epsilon = 1e-5);

        void load_parameters(const std::string &json_state);

        void load_parameters(std::istream &input);

        void dump_parameters(const std::string &filename);

        explicit Network(std::vector<unsigned int> sizes);
    };

    std::string valarray_to_json(const std::valarray<double> &a);
}

#endif //NEURAL_NET_NETWORK_H

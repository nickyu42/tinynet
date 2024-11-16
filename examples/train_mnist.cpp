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

    auto validation_data = loader.load({
                                               {"images", "data/mnist/t10k-images.idx3-ubyte"},
                                               {"labels", "data/mnist/t10k-labels.idx1-ubyte"}
                                       });

    toynet::Network n({784, 128, 128, 10});

    for (int i = 0; i < 50; ++i) {
        n.SGD(training_data, 1, 32, 3.0);

        size_t total_correct = 0;
        for (auto &sample: validation_data) {
            auto act = n.feedforward(sample.first);

            unsigned int predicted_digit = 0;
            double max_activation = std::numeric_limits<double>::lowest();
            for (unsigned int j = 0; j < act.size(); ++j) {
                if (act[j] > max_activation) {
                    predicted_digit = j;
                    max_activation = act[j];
                }
            }

            if (sample.second[predicted_digit] == 1) {
                total_correct++;
            }
        }

        std::cout << i << "," << total_correct << std::endl;
    }

    n.dump_parameters("data/mnist/parameters.json");

    return 0;
}

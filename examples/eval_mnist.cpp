#include <fstream>
#include <limits>
#include <iostream>

#include "../Network.h"
#include "../loader/DataLoader.h"

int main() {
    auto loader = toynet::MNISTLoader();
    auto validation_data = loader.load({
                                               {"images", "data/mnist/t10k-images.idx3-ubyte"},
                                               {"labels", "data/mnist/t10k-labels.idx1-ubyte"}
                                       });

    toynet::Network n({784, 30, 10});

    std::ifstream parameters_file("data/mnist/parameters.json");
    n.load_parameters(parameters_file);
    parameters_file.close();

    auto total_samples = validation_data.size();
    size_t total_correct = 0;
    for (auto &sample: validation_data) {
        auto act = n.feedforward(sample.first);

        unsigned int predicted_digit = 0;
        double max_activation = std::numeric_limits<double>::lowest();
        for (unsigned int i = 0; i < 9; ++i) {
            if (act[i] > max_activation) {
                predicted_digit = i;
                max_activation = act[i];
            }
        }

        if (sample.second[predicted_digit] == 1) {
            total_correct++;
        }
    }

    std::cout << "Total correct: " << total_correct << " / " << total_samples << " = "
              << static_cast<double>(total_correct) / static_cast<double>(total_samples) * 100.0 << "%" << std::endl;

    return 0;
}

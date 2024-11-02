#ifndef NEURAL_NET_DATALOADER_H
#define NEURAL_NET_DATALOADER_H

#include <map>

#include "../Network.h"

namespace toynet {
    class DataLoader {
    public:
        virtual std::vector<toynet::TrainingSample> load(std::map<std::string, std::string> config) = 0;
    };

    class MNISTLoader : DataLoader {
    public:
        std::vector<toynet::TrainingSample> load(std::map<std::string, std::string> config) override;

    private:
        static std::vector<toynet::TrainingSample> load_images(const std::string &images, const std::string &labels);
    };
}

#endif //NEURAL_NET_DATALOADER_H

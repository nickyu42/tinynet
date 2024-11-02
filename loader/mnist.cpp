#include <fstream>
#include <iostream>

#include "DataLoader.h"

std::vector<toynet::TrainingSample> toynet::MNISTLoader::load(std::map<std::string, std::string> config) {
    return load_images(config["images"], config["labels"]);
}

std::vector<toynet::TrainingSample>
toynet::MNISTLoader::load_images(const std::string &images_filepath, const std::string &labels_filepath) {

    std::cout << "Loading data from images=" << images_filepath << ", labels=" << labels_filepath << std::endl;

    // Reading labels
    std::ifstream label_file(labels_filepath, std::ios::binary);
    if (!label_file.is_open()) {
        throw std::runtime_error("Cannot open label file: " + labels_filepath);
    }

    uint32_t magic = 0, size = 0;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = __builtin_bswap32(magic); // Convert to big endian
    if (magic != 2049) {
        throw std::runtime_error("Magic number mismatch, expected 2049, got " + std::to_string(magic));
    }

    label_file.read(reinterpret_cast<char*>(&size), 4);
    size = __builtin_bswap32(size);

    // Reading images
    std::ifstream image_file(images_filepath, std::ios::binary);
    if (!image_file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + images_filepath);
    }

    uint32_t images_size = 0, rows = 0, cols = 0;
    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = __builtin_bswap32(magic);
    if (magic != 2051) {
        throw std::runtime_error("Magic number mismatch, expected 2051, got " + std::to_string(magic));
    }

    image_file.read(reinterpret_cast<char*>(&images_size), 4);
    images_size = __builtin_bswap32(images_size);

    if (images_size != size) {
        throw std::runtime_error("Expected " + std::to_string(size) + " images, got " + std::to_string(images_size));
    }

    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = __builtin_bswap32(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = __builtin_bswap32(cols);

    std::vector<toynet::TrainingSample> samples;
    samples.reserve(size);

    std::vector<uint8_t> image_buffer(rows * cols);
    for (size_t i = 0; i < size; ++i) {
        // Read and vectorize the label
        uint8_t label = 0;
        label_file.read(reinterpret_cast<char*>(&label), 1);
        std::valarray<double> label_vector(0.0, 10);
        label_vector[label] = 1.0;

        // Read and normalize the image
        image_file.read(reinterpret_cast<char*>(image_buffer.data()), rows * cols);
        std::valarray<double> image_vector(image_buffer.size());

        for (size_t j = 0; j < image_buffer.size(); ++j) {
            image_vector[j] = static_cast<double>(image_buffer[j]) / 255.0;
        }

        samples.emplace_back(std::move(image_vector), std::move(label_vector));
    }

    label_file.close();
    image_file.close();

    std::cout << "Successfully loaded " << size << " images!" << std::endl;

    return samples;
}

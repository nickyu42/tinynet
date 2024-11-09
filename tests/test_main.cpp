#include <gtest/gtest.h>

#include "../Network.h"

TEST(NetworkTest, Load) {
    toynet::Network n({2, 3, 2});

    auto p = R"(
        [
            {"weights": [1, 2, 3, 3, 2, 1], "bias": [1, 2, 0]},
            {"weights": [1, 2, 3, 3, 2, 1], "bias": [0, 0]}
        ]
    )";
    n.load_parameters(p);

    std::vector<double> expected_weights = {1, 2, 3, 3, 2, 1};

    for (size_t i = 0; i < expected_weights.size(); ++i) {
        ASSERT_EQ(n.layers[1].weights[i], expected_weights[i]);
    }

    std::vector<double> expected_bias = {1, 2, 0};

    for (size_t i = 0; i < expected_bias.size(); ++i) {
        ASSERT_EQ(n.layers[1].bias[i], expected_bias[i]);
    }
}

TEST(NetworkTest, BasicForward) {
    toynet::Network n({2, 3});

    auto p = R"(
        [
            {"weights": [1, 2, 3, 3, -2, 1], "bias": [-5.5, -10.2, 0.7]}
        ]
    )";
    n.load_parameters(p);

    std::valarray<double> expected_activation = {0.377541, 0.231475, 0.668188};
    std::valarray<double> expected_z = {-0.5, -1.2, 0.7};

    auto activation = n.feedforward({1, 2});

    for (size_t i = 0; i < expected_activation.size(); ++i) {
        ASSERT_NEAR(activation[i], expected_activation[i], 1e-5);
    }

    for (size_t i = 0; i < expected_z.size(); ++i) {
        ASSERT_NEAR(n.layers[1].z[i], expected_z[i], 1e-5);
    }
}

TEST(NetworkTest, BackpropSimple) {
    toynet::Network n({2, 2, 1});

    auto p = R"(
        [
            {"weights": [0.1, 0.2, 0.3, 0.4], "bias": [0.1, 0.2]},
            {"weights": [0.2, 0.3], "bias": [0.1]}
        ]
    )";
    n.load_parameters(p);

    toynet::TrainingSample sample = {{1, 0.5}, {1.0}};

    auto diffs = n.gradient_check(sample);
    for (auto d: diffs) {
        ASSERT_NEAR(d, 0.0,
                    1e-7) << "Difference between approximate gradient and backpropogation gradient should be approximately equal";
    }
}

TEST(NetworkTest, Backprop) {
    toynet::Network n({2, 3, 2});

    auto p = R"(
        [
            {"weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "bias": [0.1, 0.2, 0.3]},
            {"weights": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "bias": [0.1, 0.2]}
        ]
    )";
    n.load_parameters(p);

    toynet::TrainingSample sample = {{1, 0.5}, {1.0, 0.0}};

    auto diffs = n.gradient_check(sample);

    for (auto d: diffs) {
        ASSERT_NEAR(d, 0.0,
                    1e-7) << "Difference between approximate gradient and backpropogation gradient should be approximately equal";
    }
}

TEST(NetworkTest, BackpropRandom) {
    toynet::Network n({2, 3, 2});

    toynet::TrainingSample sample = {{1, 0.5}, {1.0, 0.0}};

    auto diffs = n.gradient_check(sample);

    for (auto d: diffs) {
        ASSERT_NEAR(d, 0.0,
                    1e-7) << "Difference between approximate gradient and backpropogation gradient should be approximately equal";
    }
}

TEST(NetworkTest, BackpropLargeRandom) {
    toynet::Network n({3, 6, 12, 6, 3});

    toynet::TrainingSample sample = {{1, 0.5, 0.0}, {0.0, 1.0, 0.0}};

    auto diffs = n.gradient_check(sample);

    for (auto d: diffs) {
        ASSERT_NEAR(d, 0.0,
                    1e-7) << "Difference between approximate gradient and backpropogation gradient should be approximately equal";
    }
}

TEST(NetworkTest, MiniBatch) {
    toynet::Network n({2, 2, 1});

    auto p = R"(
        [
            {"weights": [0.1, 0.2, 0.3, 0.4], "bias": [0.1, 0.2]},
            {"weights": [0.2, 0.3], "bias": [0.1]}
        ]
    )";
    n.load_parameters(p);

    std::vector<toynet::TrainingSample> samples = {{{1.0, 0.5}, {1.0}}, {{0.5, 1}, {0.1}}};

    n.update_mini_batch(samples.begin(), samples.end(), 3.0);

    for (auto &l : n.layers) {
        std::cout << l << std::endl;
    }
}
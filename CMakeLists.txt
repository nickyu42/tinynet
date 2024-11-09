cmake_minimum_required(VERSION 3.25)
project(neural_net)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add gtest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add json parser
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

add_library(toynet Network.cpp Network.h
        loader/DataLoader.h
        loader/mnist.cpp)
target_link_libraries(toynet nlohmann_json::nlohmann_json)

enable_testing()

add_executable(
        test_main
        tests/test_main.cpp
#        Network.cpp Network.h
#        loader/DataLoader.h
#        loader/mnist.cpp
)
target_link_libraries(
        test_main
        GTest::gtest_main
        nlohmann_json::nlohmann_json
        toynet
)

include(GoogleTest)
gtest_discover_tests(test_main)

add_subdirectory(examples)
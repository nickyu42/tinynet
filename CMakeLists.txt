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

# Optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -flto")

add_library(toynet Network.cpp Network.h
        loader/DataLoader.h
        loader/mnist.cpp)
target_link_libraries(toynet nlohmann_json::nlohmann_json)

enable_testing()

add_executable(
        test_main
        tests/test_main.cpp
)
target_link_libraries(
        test_main
        GTest::gtest_main
        nlohmann_json::nlohmann_json
        toynet
)

if (DEFINED EMSCRIPTEN)
    set(EMSCRIPTEN_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME=\"createModule\" -lembind")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSCRIPTEN_FLAGS}")

    add_executable(toynet_wasm Network.cpp Network.h bindings.cpp)
    target_link_libraries(toynet_wasm nlohmann_json::nlohmann_json)

    set(CMAKE_EXECUTABLE_SUFFIX ".js")

    set_target_properties(toynet_wasm PROPERTIES COMPILE_FLAGS "-Os")
    set_target_properties(toynet_wasm PROPERTIES LINK_FLAGS    "-Os -s WASM=1 --no-entry")
endif()

include(GoogleTest)
gtest_discover_tests(test_main)

add_subdirectory(examples)
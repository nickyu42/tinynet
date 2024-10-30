cmake_minimum_required(VERSION 3.25)
project(neural_net)

set(CMAKE_CXX_STANDARD 11)

add_executable(neural_net main.cpp Network.cpp Network.h)

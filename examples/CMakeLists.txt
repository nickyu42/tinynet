# Specify the directory containing example programs
file(GLOB EXAMPLE_SOURCES *.cpp)

# Iterate over each example source file and create a separate executable
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    # Get the filename without the extension
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

    # Create an executable for each source file
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

    # Link the executable to the library
    target_link_libraries(${EXAMPLE_NAME} PRIVATE toynet)
endforeach()
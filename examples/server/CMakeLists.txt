# ATLAS Persistence Server Example
cmake_minimum_required(VERSION 3.14)

project(atlas-server)

# Find required packages
find_package(Threads REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(../../include)
include_directories(../../src)

# Source files
set(ATLAS_PERSISTENCE_SOURCES
    ../../src/atlas-persistence.cpp
    atlas-persistence-server.cpp
    server.cpp
)

# Create the atlas-server executable
add_executable(atlas-server ${ATLAS_PERSISTENCE_SOURCES})

# Link libraries
target_link_libraries(atlas-server 
    Threads::Threads
    m  # Math library for some systems
)

# Compiler definitions
target_compile_definitions(atlas-server PRIVATE
    ATLAS_ENABLED=1
)

# Add pthread for signal handling
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(atlas-server pthread rt)
endif()

# Set output directory
set_target_properties(atlas-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples/server
)
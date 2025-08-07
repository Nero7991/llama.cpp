# ATLAS sources
set(ATLAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/atlas-types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/atlas-backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/atlas-memory.cpp
)

set(ATLAS_HEADERS
    ../../include/atlas/atlas-types.h
)

# Create ATLAS library
add_library(ggml-atlas ${ATLAS_SOURCES} ${ATLAS_HEADERS})

target_include_directories(ggml-atlas PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_compile_definitions(ggml-atlas PUBLIC GGML_USE_ATLAS)

# Link with base ggml
target_link_libraries(ggml-atlas PUBLIC ggml-base)

# Add CPU backend if enabled
if (GGML_ATLAS_CPU)
    target_compile_definitions(ggml-atlas PUBLIC GGML_ATLAS_CPU)
endif()

# Add CUDA backend if enabled
if (GGML_ATLAS_CUDA AND GGML_CUDA)
    target_compile_definitions(ggml-atlas PUBLIC GGML_ATLAS_CUDA)
    target_link_libraries(ggml-atlas PUBLIC ggml-cuda)
endif()

# Install headers
install(FILES ${ATLAS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/atlas
)

# Install library
install(TARGETS ggml-atlas
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
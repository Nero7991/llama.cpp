# CMake generated Testfile for 
# Source directory: /home/orencollaco/GitHub/llama.cpp/tests/atlas
# Build directory: /home/orencollaco/GitHub/llama.cpp/tests/atlas/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(AtlasUnitTests "/home/orencollaco/GitHub/llama.cpp/tests/atlas/build/test-atlas-comprehensive" "--unit-only")
set_tests_properties(AtlasUnitTests PROPERTIES  TIMEOUT "300" _BACKTRACE_TRIPLES "/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;180;add_test;/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;0;")
add_test(AtlasMemoryModule "/home/orencollaco/GitHub/llama.cpp/tests/atlas/build/test-memory-module")
set_tests_properties(AtlasMemoryModule PROPERTIES  TIMEOUT "120" _BACKTRACE_TRIPLES "/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;183;add_test;/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;0;")
add_test(AtlasOmegaRule "/home/orencollaco/GitHub/llama.cpp/tests/atlas/build/test-omega-rule")
set_tests_properties(AtlasOmegaRule PROPERTIES  TIMEOUT "120" _BACKTRACE_TRIPLES "/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;186;add_test;/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;0;")
add_test(AtlasIntegration "/home/orencollaco/GitHub/llama.cpp/tests/atlas/build/test-atlas-integration")
set_tests_properties(AtlasIntegration PROPERTIES  TIMEOUT "600" _BACKTRACE_TRIPLES "/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;189;add_test;/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;0;")
add_test(AtlasCUDAPerformance "/home/orencollaco/GitHub/llama.cpp/tests/atlas/build/test-atlas-cuda-performance")
set_tests_properties(AtlasCUDAPerformance PROPERTIES  TIMEOUT "900" _BACKTRACE_TRIPLES "/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;200;add_test;/home/orencollaco/GitHub/llama.cpp/tests/atlas/CMakeLists.txt;0;")

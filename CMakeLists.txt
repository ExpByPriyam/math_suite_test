cmake_minimum_required(VERSION 3.16)
project(MathSuite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define our low-level header-only library
add_library(MathSuite INTERFACE)

# Point to the new include directory
target_include_directories(MathSuite INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(MathSuite INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -O3
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        target_compile_options(MathSuite INTERFACE -march=native)
        message(STATUS "Architecture detected: ARM64 (NEON Optimized)")
    else()
        target_compile_options(MathSuite INTERFACE -mavx2 -mfma)
        message(STATUS "Architecture detected: x86_64 (AVX2 Optimized)")
    endif()

elseif(MSVC)
    target_compile_options(MathSuite INTERFACE /W4 /O2)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64")
        target_compile_options(MathSuite INTERFACE /arch:AVX2)
    endif()
endif()

# Point to the new location of main.cpp inside the src directory
add_executable(math_suite_test src/main.cpp)
target_link_libraries(math_suite_test PRIVATE MathSuite)
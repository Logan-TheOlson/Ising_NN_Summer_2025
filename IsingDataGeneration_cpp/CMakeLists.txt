cmake_minimum_required(VERSION 3.15)
project(IsingCSV CXX)

# ---- Options ----
option(ENABLE_OPENMP "Build with OpenMP parallelism" ON)

# Optional tuning knobs (can be set per-profile in CLion)
set(PROGRESS_MS "2000" CACHE STRING "Progress print cadence in ms")
set(BATCH_SIZE "32768" CACHE STRING "RNG batch size (try 16384..65536)")
set(BUFFERED_UNIFORMS "0" CACHE STRING "0: on-the-fly uniforms, 1: buffered")
set(RNG_BACKEND "1" CACHE STRING "0: PCG32, 1: xoshiro256**")

# ---- Build type & language ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Target ----
add_executable(ising_csv ising_generation_withMag.cpp)

# ---- Warnings & optimization ----
if(MSVC)
    target_compile_options(ising_csv PRIVATE /W4 /permissive-)
    target_compile_options(ising_csv PRIVATE
            $<$<CONFIG:Release>:/O2 /DNDEBUG /arch:AVX2 /openmp:llvm>
    )
    # Uncomment if you truly never use RTTI/exceptions:
    # target_compile_options(ising_csv PRIVATE $<$<CONFIG:Release>:/GR->)    # disable RTTI
    # target_compile_options(ising_csv PRIVATE $<$<CONFIG:Release>:/EHsc->)  # disable exceptions
else()
    target_compile_options(ising_csv PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ising_csv PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -fopenmp -fno-exceptions -fno-rtti -funroll-loops>
    )
endif()

# ---- OpenMP (optional) ----
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ising_csv PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(ising_csv PRIVATE USE_OPENMP=1)
    else()
        message(STATUS "OpenMP not found; building without it.")
    endif()
endif()

# ---- LTO / IPO ----
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if(ipo_ok)
    set_property(TARGET ising_csv PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
else()
    message(STATUS "IPO/LTO not enabled: ${ipo_msg}")
endif()

# ---- Tunable macros to the code ----
target_compile_definitions(ising_csv PRIVATE
        PROGRESS_MS=${PROGRESS_MS}
        BATCH_SIZE=${BATCH_SIZE}
        BUFFERED_UNIFORMS=${BUFFERED_UNIFORMS}
        RNG_BACKEND=${RNG_BACKEND}
)

# ---- Output dir ----
set_target_properties(ising_csv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

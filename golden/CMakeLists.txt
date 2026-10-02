# Golden-output generators: link against the local C++ dlib and emit reference
# outputs that the Rust test-suite compares against.
#
# Usage:
#   cmake -S golden -B golden/build -Ddlib_DIR=/path/to/dlib/dlib/build
#   cmake --build golden/build -j
cmake_minimum_required(VERSION 3.16)
project(golden_generators CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dlib CONFIG REQUIRED)

file(GLOB GOLDEN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
foreach(src ${GOLDEN_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE dlib::dlib)
endforeach()

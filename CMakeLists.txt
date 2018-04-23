cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(knn)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive") #supress error in older gcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # build debug symbols

# Include header files from directory 'include'
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost 1.63 COMPONENTS system chrono python numpy REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${DLIB_LIBRARIES})
add_definitions(-D_GLIBCXX_PARALLEL)
PYTHON_ADD_MODULE(KNNTest src/knn.cpp)


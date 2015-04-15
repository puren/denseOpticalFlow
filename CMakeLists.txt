project(denseOpticalFlow)

cmake_minimum_required(VERSION 2.8.7)
set(cmake_test_VERSION_MAJOR 0)
set(cmake_test_VERSION_MINOR 1)
set(cmake_test_VERSION_PATCH 0)

set(exe_name "denseOpticalFlow")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

################################################################
#                    PACKAGES                                  #
################################################################

#find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(Qt4 REQUIRED)
#find_package(Eigen REQUIRED)
#find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(HDF5 REQUIRED)
                                                                                                                                                                                                 
################################################################
#                   INCLUDES                                   #
################################################################
#set(FOREST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rf_lib" CACHE FILEPATH "random forest library path")
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/include)

################################################################
#                   PATH                                       #
################################################################

set(dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set(LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

################################################################
#                   ADD PROJECT FILES                          #
################################################################
file( GLOB  source_files *.cpp )
file( GLOB  header_files *.hpp *.h )

ADD_EXECUTABLE("${exe_name}" ${source_files} ${header_files})


################################################################
#                   LINKING LIBRARIES                          #
################################################################
target_link_libraries("${exe_name}" ${OpenCV_LIBS})


################################################################
#                   LINKING HDF5                          #
################################################################
find_package(HDF5)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIR})
    set(_hdf5_libs hdf5 hdf5_cpp)
else()
    # we did not find it so we fetch it from the web, configure, compile, and link
    include(ExternalProject)
    set(ExternalProjectCMakeArgs
        -DHDF5_BUILD_CPP_LIB=ON
        )
    set(_hdf5_version "1.8.14")
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/hdf5_local/src)
    ExternalProject_Add(hdf5_local
        DOWNLOAD_COMMAND curl http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-${_hdf5_version}.tar.gz | tar xvz -C ${PROJECT_BINARY_DIR}/hdf5_local/src
        CMAKE_ARGS ${ExternalProjectCMakeArgs}
        SOURCE_DIR ${PROJECT_BINARY_DIR}/hdf5_local/src/hdf5-${_hdf5_version}
        BINARY_DIR ${PROJECT_BINARY_DIR}/hdf5_local/build
        STAMP_DIR  ${PROJECT_BINARY_DIR}/hdf5_local/stamp
        TMP_DIR    ${PROJECT_BINARY_DIR}/hdf5_local/tmp
        INSTALL_COMMAND ""
        )
    include_directories(
        ${PROJECT_BINARY_DIR}/hdf5_local/src/hdf5-${_hdf5_version}/c++/src
        ${PROJECT_BINARY_DIR}/hdf5_local/build
        )
    set(_hdf5_libs
        ${PROJECT_BINARY_DIR}/hdf5_local/build/bin/libhdf5_cpp.a
        ${PROJECT_BINARY_DIR}/hdf5_local/build/bin/libhdf5.a
        -ldl
        )
    add_dependencies(writedata.x hdf5_local)
endif()

target_link_libraries(
    "${exe_name}"
    ${_hdf5_libs}
    )
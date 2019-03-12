cmake_minimum_required(VERSION 2.8 )

project(road_sign_recognition)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_definitions("-std=c++11")

find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  )

add_library(${PROJECT_NAME}_roadsign
  include/DetectedMat.h
  include/RoadSignDetector.h
  include/SVMclassifier.h
  src/DetectedMat.cpp
  src/RoadSignDetector.cpp
  src/SVMclassifier.cpp
  )

target_link_libraries(${PROJECT_NAME}_roadsign
  ${OpenCV_LIBS}
  )

add_executable(${PROJECT_NAME}
  src/FinalProject.cpp
  )
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${PROJECT_NAME}_roadsign
  )


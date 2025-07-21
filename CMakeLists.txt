cmake_minimum_required(VERSION 3.8)
project(etrobo_object_detection_tflite)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Find TensorFlow Lite
# find_path(TENSORFLOW_LITE_INCLUDE_DIR
#   NAMES tensorflow/lite/interpreter.h
#   PATHS /usr/include /usr/local/include
# )
# 
# find_library(TENSORFLOW_LITE_LIB
#   NAMES tensorflow-lite
#   PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
# )
# 
# if(NOT TENSORFLOW_LITE_INCLUDE_DIR OR NOT TENSORFLOW_LITE_LIB)
#   message(FATAL_ERROR "TensorFlow Lite not found. Please install tensorflow-lite-dev package.")
# endif()

add_subdirectory(
  "/home/ouwa/work/tensorflow/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

include_directories(
  include
  ${TENSORFLOW_LITE_INCLUDE_DIR}
)

# C++ executable
add_executable(etrobo_object_detection_tflite
  src/etrobo_object_detection_tflite.cpp
)

ament_target_dependencies(etrobo_object_detection_tflite
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

target_link_libraries(etrobo_object_detection_tflite
  ${TENSORFLOW_LITE_LIB}
  tensorflow-lite
)

install(TARGETS
  etrobo_object_detection_tflite
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

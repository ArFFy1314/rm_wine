cmake_minimum_required(VERSION 3.8)
project(rm_wine)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

set(ngraph_DIR "/opt/openvino/runtime/cmake")
set(InferenceEngine_DIR "/opt/openvino/runtime/cmake")

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

add_executable(wine_serial 
  src/wine_serial.cpp 
  src/yolov7_kpt.cpp 
  include/rm_wine/wine_serial.hpp 
  include/rm_wine/yolov7_kpt.hpp)

add_executable(wine_virtual 
  src/img_pub.cpp)

target_include_directories(wine_serial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(wine_serial PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  wine_serial
  "rclcpp"
  "sensor_msgs"
  "auto_aim_interfaces"
  "OpenCV"
  "cv_bridge"
  "InferenceEngine"
  "ngraph"
)

ament_target_dependencies(
  wine_virtual
  "rclcpp"
  "sensor_msgs"
  "auto_aim_interfaces"
  "OpenCV"
  "cv_bridge"
  "image_transport"
)

install(TARGETS
        wine_serial
        wine_virtual
        DESTINATION lib/${PROJECT_NAME})

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
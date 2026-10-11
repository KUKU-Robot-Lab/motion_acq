cmake_minimum_required(VERSION 3.16)
project(motion_acq_nova2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(senseglove_api REQUIRED)

add_executable(nova2_sdk src/nova2_sdk_node.cpp)
target_include_directories(nova2_sdk PRIVATE ${senseglove_api_INCLUDE_DIRS})
target_link_libraries(nova2_sdk ${senseglove_api_LIBRARIES} pthread)
ament_target_dependencies(nova2_sdk rclcpp sensor_msgs std_msgs)
# libsgcore is installed next to senseglove_api; find it at run time without LD_LIBRARY_PATH
set_target_properties(nova2_sdk PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS nova2_sdk DESTINATION lib/${PROJECT_NAME})

ament_package()

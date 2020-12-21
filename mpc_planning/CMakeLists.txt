cmake_minimum_required(VERSION 2.8.3)
project(mpc_planning)
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs tf2 tf2_ros)

find_package(cmake_modules REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS} include)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs tf2 tf2_ros
  DEPENDS Eigen
)
include_directories(${catkin_INCLUDE_DIRS})

add_executable(mpc_planning_node src/mpc_planning_node.cpp)
target_link_libraries(mpc_planning_node ${catkin_LIBRARIES})



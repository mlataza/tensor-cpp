add_executable(basic basic.cpp)
target_link_libraries(basic tensor-cpp)

add_executable(trajectory trajectory.cpp)
target_link_libraries(trajectory tensor-cpp)
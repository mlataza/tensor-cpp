add_executable(ranks ranks.cpp)
target_link_libraries(ranks tensor-cpp)
add_test(NAME ranks COMMAND ranks)

add_executable(sizes sizes.cpp)
target_link_libraries(sizes tensor-cpp)
add_test(NAME sizes COMMAND sizes)

add_executable(indices indices.cpp)
target_link_libraries(indices tensor-cpp)
add_test(NAME indices COMMAND indices)

add_executable(constructor constructor.cpp)
target_link_libraries(constructor tensor-cpp)
add_test(NAME constructor COMMAND constructor)

add_executable(sum sum.cpp)
target_link_libraries(sum tensor-cpp)
add_test(NAME sum COMMAND sum)

add_executable(difference difference.cpp)
target_link_libraries(difference tensor-cpp)
add_test(NAME difference COMMAND difference)

add_executable(scaled scaled.cpp)
target_link_libraries(scaled tensor-cpp)
add_test(NAME scaled COMMAND scaled)

add_executable(iterator iterator.cpp)
target_link_libraries(iterator tensor-cpp)
add_test(NAME iterator COMMAND iterator)

add_executable(shapes shapes.cpp)
target_link_libraries(shapes tensor-cpp)
add_test(NAME shapes COMMAND shapes)

add_executable(transpose transpose.cpp)
target_link_libraries(transpose tensor-cpp)
add_test(NAME transpose COMMAND transpose)

add_executable(multiplication multiplication.cpp)
target_link_libraries(multiplication tensor-cpp)
add_test(NAME multiplication COMMAND multiplication)

add_executable(assignment assignment.cpp)
target_link_libraries(assignment tensor-cpp)
add_test(NAME assignment COMMAND assignment)
cmake_minimum_required(VERSION 2.8)
project( test_of )
find_package( OpenCV REQUIRED )

add_executable( test_of test_of.cpp )
include_directories( ${OpenCV_INCLUDE_DIRS} )  
include_directories( include )
target_link_libraries( test_of ${OpenCV_LIBS} )


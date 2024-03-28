cmake_minimum_required(VERSION 3.1.0)

set(CXX_STANDARD 20)

project(AlgebraWithSTL VERSION 0.1.0)

add_executable(${PROJECT_NAME}
    
    algebra.cpp
    main.cpp
)

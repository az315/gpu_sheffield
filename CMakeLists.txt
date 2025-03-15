cmake_minimum_required(VERSION 3.30)
project(gpu_sheffield C)

set(CMAKE_C_STANDARD 11)

add_executable(gpu_sheffield main.c
        exercise01.c
        exercise04.c
        linked_list.h
        exercise02.c
        exercise03.c)

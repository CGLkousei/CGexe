cmake_minimum_required(VERSION 3.25)
project(CGexe)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(CGexe
        Camera.cpp
        Camera.h
        GLPreview.cpp
        GLPreview.h
        Jpeg.cpp
        Jpeg.h
        Light.cpp
        Light.h
        main.cpp
        random.c
        random.h
        TriMesh.cpp
        TriMesh.h)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(CGexe PUBLIC Eigen3::Eigen)

find_package(OpenGL REQUIRED)
target_link_libraries(CGexe PUBLIC ${OPENGL_LIBRARIES})
message("OPENGL: ${OPENGL_LIBRARIES}")

find_package(FreeGLUT CONFIG REQUIRED)
target_link_libraries(CGexe PUBLIC $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)

find_package(JPEG REQUIRED)
target_link_libraries(CGexe PUBLIC JPEG::JPEG)
cmake_minimum_required(VERSION 3.25)
project(CGexe)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(CGexe
        Camera.cpp
        Camera.h
        GLPreview.cpp
        GLPreview.h
        Light.cpp
        Light.h
        Jpeg.cpp
        Jpeg.h
        main.cpp
        random.c
        random.h
        TriMesh.cpp
        TriMesh.h
        Image.cpp
        Image.h
        Color.cpp
        Color.h
        Renderer.cpp
        Renderer.h
        ParticipatingMedia.cpp
        ParticipatingMedia.h)

#find_package(OpenCV CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE opencv_ml opencv_dnn opencv_core opencv_flann)

find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC opencv_ml opencv_dnn opencv_core opencv_flann opencv_imgcodecs opencv_highgui )

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

find_package(OpenGL REQUIRED)
target_link_libraries(CGexe PUBLIC ${OPENGL_LIBRARIES})
message("OPENGL: ${OPENGL_LIBRARIES}")

find_package(FreeGLUT CONFIG REQUIRED)
target_link_libraries(CGexe PUBLIC $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)

find_package(JPEG REQUIRED)
target_link_libraries(CGexe PUBLIC JPEG::JPEG)
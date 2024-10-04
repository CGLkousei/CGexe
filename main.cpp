
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <unistd.h>
#else

#include <GL/freeglut.h>

#endif

#define _USE_MATH_DEFINES

#include <math.h>
#include <vector>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <ctime>

#include "Camera.h"
#include "TriMesh.h"
#include "GLPreview.h"
#include "Renderer.h"


const int g_FilmWidth = 640;
const int g_FilmHeight = 480;
GLuint g_FilmTexture = 0;

bool g_DrawFilm = true;

std::vector<int> modes = {4};
int mode_index = 0;
unsigned int samples = 5000;
unsigned int nSamplesPerPixel = 1;
bool save_flag = false;

const std::string filename = "mode";
const std::string directoryname = "hair_1";

clock_t start_time;
clock_t end_time;

int width = 640;
int height = 480;

double g_FrameSize_WindowSize_Scale_x = 1.0;
double g_FrameSize_WindowSize_Scale_y = 1.0;

Camera g_Camera;
Renderer g_renderer;
std::vector<AreaLight> g_AreaLights;

Object g_Obj;
Hair g_Hair;

void printCurrentTime(){
    try{
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::cout << "Current time is " << std::ctime(&now_time);
    } catch (const std::exception& e) {
        std::cerr << "error occurred in printCurrentTime(): " << e.what() << std::endl;
    }
}

void initAreaLights() {
    AreaLight light1;
    light1.pos << 0.0, 2.5, 0.0;
    light1.arm_u << 1.0, 0.0, 0.0;
    light1.arm_v = -light1.pos.cross(light1.arm_u);
    light1.arm_v.normalize();
    light1.arm_u = light1.arm_u * 0.7;
    light1.arm_v = light1.arm_v * 0.7;
    light1.color << 1.0, 1.0, 1.0;
    light1.intensity = 40.0;

    g_AreaLights.push_back(light1);
}
void setHair(Hair &hairs, Eigen::Vector3d w){
    const Eigen::Vector3d color(0.3, 0.3, 0.3);
//    const Eigen::Vector3d color(0.9, 0.9, 0.9);
//    const double absorb = 0.5;
    const double absorb = 0.01;
    const double alpha = -5;
    const double beta = 5;
    const double eta = 1.55;
    const double radius = 0.05;

    for(int i = 0; i < hairs.hairs.size(); i++){
        hairs.hairs[i].setRadius(radius);
        hairs.hairs[i].setMaterial(color, absorb, alpha, beta, eta);
        hairs.hairs[i].setUVector();
        hairs.hairs[i].setWVector(w);
    }
}
void changeMode(const unsigned int samples){
    if(g_renderer.g_CountBuffer[0] >= samples){
        save_flag = true;
        end_time = clock();
    }
}
void saveImg(std::string filename, std::string directory){
    if(save_flag){
        const double rendering_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
        std::string time_str = std::to_string(static_cast<int>(rendering_time));
        std::string file_str = filename + std::to_string(modes[mode_index]) + "_" + time_str + "s_" + std::to_string(samples) + "sample";

        //make the directory
        if(!std::filesystem::exists(directory)){
            if(!std::filesystem::create_directories(directory)){
                std::cerr << "Failed to create directory: " << directory << std::endl;
                return;
            }
        }

        g_renderer.saveImg( directory + "/" + file_str);
        std::cout << "Rendering mode " << modes[mode_index] << " takes " << time_str << " second." << std::endl << std::endl;

        mode_index++;
        if(mode_index >= modes.size()) {
            glutLeaveMainLoop();
        }

        g_renderer.resetFilm();
        g_renderer.clearRayTracedResult();
        save_flag = false;
        start_time = clock();
    }
}

void initFilm() {
    glGenTextures(1, &g_FilmTexture);
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_FilmWidth, g_FilmHeight, 0, GL_RGB, GL_FLOAT, g_renderer.g_FilmBuffer);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void updateFilm() {
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_FilmWidth, g_FilmHeight, GL_RGB, GL_FLOAT, g_renderer.g_FilmBuffer);
}

void idle() {
#ifdef __APPLE__
    //usleep( 1000*1000 / 60 );
    usleep( 1000 / 60 );
#else
    Sleep(1000.0 / 60.0);
#endif
    if(!save_flag) {
        g_renderer.rendering(modes[mode_index]);
        g_renderer.updateFilm();
        updateFilm();
    }

    saveImg(filename, directoryname);
    changeMode(samples);

    glutPostRedisplay();
}

void display() {
    glViewport(0, 0, width * g_FrameSize_WindowSize_Scale_x, height * g_FrameSize_WindowSize_Scale_y);

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    projection_and_modelview(g_Camera, width, height);

    glEnable(GL_DEPTH_TEST);

    drawFloor();

    computeGLShading(g_Obj, g_AreaLights);
    drawObject(g_Obj);

    drawLights(g_AreaLights);

    if (g_DrawFilm)
        drawFilm(g_Camera, g_FilmTexture);

    glDisable(GL_DEPTH_TEST);

    glutSwapBuffers();
}

void resize(int w, int h) {
    width = w;
    height = h;
}

int main(int argc, char *argv[]) {
    g_Camera.setEyePoint(Eigen::Vector3d{0.0, 1.0, 5.5});
    g_Camera.lookAt(Eigen::Vector3d{0.0, 0.5, 0.0}, Eigen::Vector3d{0.0, 1.0, 0.0});
    initAreaLights();

    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutCreateWindow("Hello world!!");

    // With retina display, frame buffer size is twice the window size.
    // Viewport size should be set on the basis of the frame buffer size, rather than the window size.
    // g_FrameSize_WindowSize_Scale_x and g_FrameSize_WindowSize_Scale_y account for this factor.
    GLint dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, dims);
    std::cout << "dims( " << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << ")" << std::endl;

    g_FrameSize_WindowSize_Scale_x = double(dims[2]) / double(width);
    g_FrameSize_WindowSize_Scale_y = double(dims[3]) / double(height);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(resize);

    initFilm();
    loadObj("../obj/hair_one_line.obj", g_Obj, g_Hair);

    setHair(g_Hair, Eigen::Vector3d{0.0, 1.0, 0.0});
//    for(int i = 0; i < g_Hair.hairs.size(); i++){
//        std::cout << g_Hair.hairs[i].vertices.size() << std::endl;
//        std::cout << g_Hair.hairs[i].lines.size() << std::endl;
//        for(int j = 0; j < g_Hair.hairs[i].lines.size(); j++){
//            std::cout << g_Hair.hairs[i].lines[j].transpose() << std::endl;
//        }
//        for(int j = 0; j < g_Hair.hairs[i].vertices.size(); j++){
//            std::cout << g_Hair.hairs[i].vertices[j].transpose() << std::endl;
//        }
//    }

    g_renderer.setNsamples(nSamplesPerPixel, samples);
    g_renderer.set3Dscene(g_Camera, g_Obj, g_AreaLights, g_Hair);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    start_time = clock();

    glutMainLoop();
    return 0;
}

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

//RayTracingInternalData g_RayTracingInternalData;

bool g_DrawFilm = true;

std::vector<int> modes = {6};
int index = 0;
unsigned int samples = 1000;
unsigned int nSamplesPerPixel = 1000;
bool save_flag = false;

const std::string filename = "mode__";
const std::string directoryname = "Bidirectional3";

clock_t start_time;
clock_t end_time;

int width = 640;
int height = 480;

int g_MM_LIGHT_idx = 0;
int mx, my;

double g_FrameSize_WindowSize_Scale_x = 1.0;
double g_FrameSize_WindowSize_Scale_y = 1.0;

Camera g_Camera;
Renderer g_renderer;
std::vector<AreaLight> g_AreaLights;
std::vector<ParticipatingMedia> g_ParticipatingMedia;

Object g_Obj;

void printCurrentTime(){
    try {
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
    light1.intensity = 30.0;

    g_AreaLights.push_back(light1);
}
void initParticipatingMedias(){
    ParticipatingMedia pm;
    pm.pos << 0.0f, 0.0f, 0.0f;
    pm.color << 0.78f, 0.78f, 0.78f;
    pm.radius = 10.0f;
    pm.extinction = 0.2;
    pm.albedo = 0.9;
    pm.hg_g = 0.9;

//    ParticipatingMedia pm2;
//    pm2.pos << 1.0f, 1.0f, -0.05f;
//    pm2.color << 1.00f, 0.0f, 0.00f;
//    pm2.radius = 1.0f;
//    pm2.extinction = 0.55;
//    pm2.albedo = 0.9;
//    pm2.hg_g = 0.9;

    g_ParticipatingMedia.push_back(pm);
//    g_ParticipatingMedia.push_back(pm2);
}
void changeMode(const unsigned int samples, const int limit, std::string filename, std::string directory){
    if(g_renderer.g_CountBuffer[0] >= samples){
        save_flag = true;
        end_time = clock();
    }
}
void saveImg(const int limit, std::string filename, std::string directory){
    if(save_flag){
        const double rendering_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
        std::string time_str = std::to_string(static_cast<int>(rendering_time));
        std::string file_str = filename + std::to_string(modes[index]) + "_" + time_str + "s_" + std::to_string(samples) + "sample";

        //make the directory
        if(!std::filesystem::exists(directory)){
            if(!std::filesystem::create_directories(directory)){
                std::cerr << "Failed to create directory: " << directory << std::endl;
                return;
            }
        }

        g_renderer.saveImg( directory + "/" + file_str);
        std::cout << "Rendering mode " << modes[index] << " takes " << time_str << " second." << std::endl << std::endl;

        index++;
        if(index >= limit)
            glutLeaveMainLoop();

        g_renderer.resetFilm();
        g_renderer.clearRayTracedResult();
        save_flag = false;
        printCurrentTime();
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
        g_renderer.rendering(modes[index]);
        g_renderer.updateFilm();
        updateFilm();
    }

    saveImg(modes.size(), filename, directoryname);
    changeMode(samples, modes.size(), filename, directoryname);

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
    g_Camera.setEyePoint(Eigen::Vector3d{0.0, 1.0, 4.5});
    g_Camera.lookAt(Eigen::Vector3d{0.0, 0.5, 0.0}, Eigen::Vector3d{0.0, 1.0, 0.0});
    initAreaLights();
    initParticipatingMedias();

    glutInit(&argc, argv);
    glutInitWindowSize(width, height);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutCreateWindow("Hello world!!");

    // With retina display, frame buffer size is twice the window size.
    // Viewport size should be set on the basis of the frame buffer size, rather than the window size.
    // g_FrameSize_WindowSize_Scale_x and g_FrameSize_WindowSize_Scale_y account for this factor.
    GLint dims[4] = {0};
    glGetIntegerv(GL_VIEWPORT, dims);
    std::cout << "dims( " << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << ")" << std::endl << std::endl;

    g_FrameSize_WindowSize_Scale_x = double(dims[2]) / double(width);
    g_FrameSize_WindowSize_Scale_y = double(dims[3]) / double(height);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutReshapeFunc(resize);

    initFilm();
    loadObj("../obj/room.obj", g_Obj);

    g_renderer.setNsamples(nSamplesPerPixel, samples);
    g_renderer.set3Dscene(g_Camera, g_Obj, g_AreaLights, g_ParticipatingMedia);

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    printCurrentTime();
    start_time = clock();

    glutMainLoop();
    return 0;
}

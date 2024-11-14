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
std::vector<int> path_lengths = {5, 6};
int mode_index = 0;
int path_index = 0;
unsigned int samples = 5 * 1e3;
unsigned int nSamplesPerPixel = 1e3;
bool save_flag = false;

const std::string filename = "length_";
const std::string directoryname = "Bidirectional_Debug3";

clock_t start_time;
clock_t end_time;

int width = 640;
int height = 480;

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
void changeMode(const unsigned int samples){
    if(g_renderer.g_CountBuffer[0] >= samples){
        save_flag = true;
        end_time = clock();
    }
}

void updateFilm() {
    std::cout << "Start" << std::endl;
    glBindTexture(GL_TEXTURE_2D, g_FilmTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g_FilmWidth, g_FilmHeight, GL_RGB, GL_FLOAT, g_renderer.g_FilmBuffer);
    std::cout << "End" << std::endl;
}

void saveImg(std::string filename, std::string directory){
        if (save_flag) {
            for(int s = 0; s < path_lengths[path_index]; s++) {
                std::cout << "Hi" << std::endl;

                const double rendering_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
                std::string time_str = std::to_string(static_cast<int>(rendering_time));
                std::string file_str = filename + std::to_string(s + 1) + "_" + time_str + "s_" +
                                       std::to_string(samples) + "sample";

                //make the directory
                if (!std::filesystem::exists(directory)) {
                    if (!std::filesystem::create_directories(directory)) {
                        std::cerr << "Failed to create directory: " << directory << std::endl;
                        return;
                    }
                }
                const int index = s * g_FilmWidth * g_FilmHeight;
                for(int j = 0; j < g_FilmWidth * g_FilmHeight; j++){
                    if (g_renderer.g_CountBuffer[j] > 0) {
                        g_renderer.g_FilmBuffer[j * 3] = g_renderer.g_AccumulationBuffer[j * 3 + index] / g_renderer.g_CountBuffer[j];
                        g_renderer.g_FilmBuffer[j * 3 + 1] = g_renderer.g_AccumulationBuffer[j * 3 + 1 + index] / g_renderer.g_CountBuffer[j];
                        g_renderer.g_FilmBuffer[j * 3 + 2] = g_renderer.g_AccumulationBuffer[j * 3 + 2 + index] / g_renderer.g_CountBuffer[j];
                    }
                    else {
                        g_renderer.g_FilmBuffer[j * 3] = 0.0;
                        g_renderer.g_FilmBuffer[j * 3 + 1] = 0.0;
                        g_renderer.g_FilmBuffer[j * 3 + 2] = 0.0;
                    }
                }

                std::cout << "No. " << s << std::endl;
                std::cout << "Accum: " << g_renderer.g_AccumulationBuffer[index] << std::endl;
                std::cout << "Film: " << g_renderer.g_FilmBuffer[0] << std::endl;
                std::cout << "Count: " << g_renderer.g_CountBuffer[0] << std::endl;

                Sleep(1e6 / 60.0);
                updateFilm();

                g_renderer.saveImg(directory + "/" + file_str);
                std::cout << "Rendering mode " << path_lengths[path_index] << " takes " << time_str << " second." << std::endl
                          << std::endl;


            }

            mode_index++;
            if (mode_index >= modes.size()) {
                mode_index = 0;
                path_index++;

                g_renderer.path_length = path_lengths[path_index];
                free(g_renderer.g_AccumulationBuffer);
                g_renderer.g_AccumulationBuffer =(float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3 * path_lengths[path_index]);

                if (path_index >= path_lengths.size())
                    glutLeaveMainLoop();
            }

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

void idle() {
#ifdef __APPLE__
    //usleep( 1000*1000 / 60 );
    usleep( 1000 / 60 );
#else
    Sleep(1000.0 / 60.0);
#endif
    if(!save_flag) {
        g_renderer.rendering(modes[mode_index]);
//        g_renderer.updateFilm();
//        updateFilm();
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
    g_renderer.set3Dscene(g_Camera, g_Obj, g_AreaLights, g_ParticipatingMedia, path_lengths);

//    Ray ray; ray.depth = 0;
//    g_Camera.screenView(1, 1, ray);
//    ray.prev_mesh_idx = -99;
//    ray.prev_primitive_idx = -1;
//    RayHit in_RayHit;
//    g_renderer.rayTracing(g_Obj, g_AreaLights, ray, in_RayHit);
//    Ray new_ray; new_ray.depth = 0;
//    new_ray.prev_mesh_idx = -99;
//    new_ray.prev_primitive_idx = -1;
//    new_ray.o = ray.o + in_RayHit.t * ray.d;
//    new_ray.d = - ray.d;
//    RayHit new_rayhit;
//    Eigen::Vector2i pixels = g_renderer.rayCameraIntersect(g_renderer.g_Camera, new_ray, new_rayhit);
//    if(new_rayhit.t < in_RayHit.t * 2){
//        std::cout << "in_RayHit.t: " << in_RayHit.t << std::endl;
//        std::cout << "new_RayHit.t: " << new_rayhit.t << std::endl;
//        std::cout << "Pixel: " << pixels.transpose() << std::endl;
//    }
//    else{
//        std::cout << "BIG!!" << std::endl;
//    }

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    printCurrentTime();
    start_time = clock();

    glutMainLoop();
    return 0;
}

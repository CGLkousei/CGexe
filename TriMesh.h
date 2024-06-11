
#ifndef TriMesh_h
#define TriMesh_h

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <unistd.h>
#else

#include <GL/freeglut.h>

#endif

#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#define EIGEN_DONT_VECTORIZE

#include <Eigen/Core>
#include <vector>
struct Material {
    GLuint texture;
    Eigen::Vector3d color;
    float kd;
    float ks;
    float kt;
    double eta;
    double m;

    Eigen::Vector3d getKd() const;
    Eigen::Vector3d getKs() const;
    Eigen::Vector3d getKt() const;
};

struct TriMesh {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertices;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertex_normals;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> tex_coords;
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> triangles;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertex_colors;

    Material material;
};

struct Object {
    std::vector<TriMesh> meshes;
};

void resetMesh(TriMesh &io_Mesh);

void printMesh(const TriMesh &in_Mesh);

void extendMesh(TriMesh &io_BaseMesh, const TriMesh &in_SecondMesh);

GLuint prepareTextureFromJpegFile(const char *in_FileName);

void resizeObj(Object &io_object, const Eigen::Vector3d &in_min, const Eigen::Vector3d &in_max);

bool loadObj(const std::string &in_filename, Object &out_object);

#endif /* TriMesh_h */

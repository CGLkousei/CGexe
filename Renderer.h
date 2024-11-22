//
// Created by kouse on 2024/05/16.
//

#ifndef CGEXE_RENDERER_H
#define CGEXE_RENDERER_H

#include "Camera.h"
#include "Light.h"
#include "TriMesh.h"

struct RayTracingInternalData {
    int nextPixel_i;
    int nextPixel_j;
};
struct RayHit {
    double t;
    double alpha;
    double beta;
    double h;
    int mesh_idx; // < -1: no intersection, -1: area light, >= 0: object_mesh
    int primitive_idx; // < 0: no intersection
    bool isFront;
    bool isHair;
    Eigen::Vector3d n;
};

class Renderer {
public:
    const int g_FilmWidth = 640;
    const int g_FilmHeight = 480;
    float *g_FilmBuffer = nullptr;
    float *g_AccumulationBuffer = nullptr;
    int *g_CountBuffer = nullptr;
    RayTracingInternalData g_RayTracingInternalData;
    int nSamplesPerPixel = 1;

    Camera g_Camera;
    std::vector<AreaLight> g_AreaLights;
    Object g_Obj;
    Hair g_Hair;

    Renderer();
    Renderer(Camera camera, Object obj, std::vector<AreaLight> lights);

    void set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights, Hair hair);
    void setNsamples(const unsigned int nSample, const unsigned int samples);

    void resetFilm();
    void updateFilm();
    void saveImg(const std::string filename);
    void clearRayTracedResult();
    void stepToNextPixel(RayTracingInternalData &io_data);
    void rayTriangleIntersect(const TriMesh &in_Mesh, const int in_Triangle_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayAreaLightIntersect(const std::vector<AreaLight> &in_AreaLights, const int in_Light_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayHairIntersect(const TriCurb &in_Curb, const int in_Line_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const Ray &in_Ray, RayHit &io_Hit);

    void rendering(const int mode);

    Eigen::Vector3d sampleRandomPoint(const AreaLight &in_Light);

    Eigen::Vector3d computeRayHitNormal(const Object &in_Object, const RayHit &in_Hit);

    Eigen::Vector3d computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair);
    Eigen::Vector3d computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, bool first);
    Eigen::Vector3d computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, bool first);

    Eigen::Vector3d computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const int mode);
    Eigen::Vector3d computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const int mode);

    double getLightProbability(const std::vector<AreaLight> &in_AreaLights);
    double getDiffuseProbablitity(const Eigen::Vector3d normal, const Eigen::Vector3d out_dir);
    double getBlinnPhongProbablitity(const Eigen::Vector3d in_dir, const Eigen::Vector3d normal, const Eigen::Vector3d out_dir, const double m);

    double diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth);
    double blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m, const int depth);
    double refractionSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double eta, const int depth);
    double marschnerSample(const Eigen::Vector3d &in_x, Ray &out_ray, const RayHit &rayHit, const Hair &in_Hair, const double theta, const double phi, const double c, const int depth, int p);
    double marschnerSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth);

    double FrDielectric(double gamma, double etaI, double etaT) const;
    double getTransmittance(double absorption, double h, double theta, double phi) const;
};

#endif //CGEXE_RENDERER_H

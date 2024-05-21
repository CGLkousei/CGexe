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
    int mesh_idx; // < -1: no intersection, -1: area light, >= 0: object_mesh
    int primitive_idx; // < 0: no intersection
    bool isFront;
};

class Renderer {
public:
    const int g_FilmWidth = 640;
    const int g_FilmHeight = 480;
    float *g_FilmBuffer = nullptr;
    float *g_AccumulationBuffer = nullptr;
    int *g_CountBuffer = nullptr;
    RayTracingInternalData g_RayTracingInternalData;
    const int nSamplesPerPixel = 1;

    Camera g_Camera;
    std::vector<AreaLight> g_AreaLights;
    Object g_Obj;

    Renderer();
    Renderer(Camera camera, Object obj, std::vector<AreaLight> lights);

    void set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights);

    void resetFilm();
    void saveImg(const std::string filename);
    void clearRayTracedResult();
    void stepToNextPixel(RayTracingInternalData &io_data);
    void rayTriangleIntersect(const TriMesh &in_Mesh, const int in_Triangle_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayAreaLightIntersect(const std::vector<AreaLight> &in_AreaLights, const int in_Light_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Ray &in_Ray, RayHit &io_Hit);

    void rendering();
    void rendering(const int mode);

    Eigen::Vector3d sampleRandomPoint(const AreaLight &in_Light);

    Eigen::Vector3d computeDirectLighting(const std::vector<AreaLight> &in_AreaLights, const Eigen::Vector3d &in_x,
                                          const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_w_eye,
                                          const RayHit &in_ray_hit, const Object &in_Object, const Material &in_Material);
    Eigen::Vector3d computeRayHitNormal(const Object &in_Object, const RayHit &in_Hit);
    Eigen::Vector3d computeDiffuseReflection(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_w_eye,
                                             const RayHit &in_ray_hit, const Object &in_Object, const Material &in_Material,
                                             const std::vector<AreaLight> &in_AreaLights);
    Eigen::Vector3d computeReflection(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_w_eye,
                                      const RayHit &in_ray_hit, const Object &in_Object, const Material &in_Material,
                                      const std::vector<AreaLight> &in_AreaLights);
    Eigen::Vector3d computeRefraction(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_w_eye,
                                      const RayHit &in_ray_hit, const Object &in_Object, const Material &in_Material,
                                      const std::vector<AreaLight> &in_AreaLights);
    Eigen::Vector3d computeShading(const Ray &in_Ray, const RayHit &in_RayHit, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights);

    Eigen::Vector3d computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights);
    Eigen::Vector3d computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first);
    Eigen::Vector3d computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first);

    Eigen::Vector3d computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights,const Object &in_Object, const int mode);
    Eigen::Vector3d computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights,const Object &in_Object, const int mode);
    double getLightProbability(const std::vector<AreaLight> &in_AreaLights);
    double getDiffuseProbablitity(const Eigen::Vector3d normal, const Eigen::Vector3d out_dir);
    double getBlinnPhongProbablitity(const Eigen::Vector3d in_dir, const Eigen::Vector3d normal, const Eigen::Vector3d out_dir, const double m);

    double diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object);
    double blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m);
};


#endif //CGEXE_RENDERER_H

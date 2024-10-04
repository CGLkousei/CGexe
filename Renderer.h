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
struct SubPath {
    Eigen::Vector3d x;
    Eigen::Vector3d in_dir;
    Eigen::Vector3d contribute;
    Eigen::Vector3d radiance;
    RayHit rh;
    int materialMode;
    double probability;
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
    int path_length = 0;

    Camera g_Camera;
    std::vector<AreaLight> g_AreaLights;
    std::vector<ParticipatingMedia> g_ParticipatingMedia;
    Object g_Obj;
    std::vector<SubPath> g_SubPath;

    Renderer();
    Renderer(Camera camera, Object obj, std::vector<AreaLight> lights);

    void set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights, std::vector<ParticipatingMedia> media);
    void setNsamples(const unsigned int nSample, const unsigned int samples);

    void resetFilm();
    void updateFilm();
    void saveImg(const std::string filename);
    void clearRayTracedResult();
    void stepToNextPixel(RayTracingInternalData &io_data);

    void rayTriangleIntersect(const TriMesh &in_Mesh, const int in_Triangle_idx, const Ray &in_Ray, RayHit &out_Result);
    void rayAreaLightIntersect(const std::vector<AreaLight> &in_AreaLights, const int in_Light_idx, const Ray &in_Ray, RayHit &out_Result);
    Eigen::Vector2i rayCameraIntersect(const Camera &in_Camera, const Ray &in_Ray, RayHit &out_Result);

    void rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Ray &in_Ray, RayHit &io_Hit);
    void rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const std::vector<ParticipatingMedia> &all_medias, const Ray &in_Ray, RayHit &io_Hit);

    void rendering(const int mode);

    Eigen::Vector3d computeRayHitNormal(const Object &in_Object, const RayHit &in_Hit);

    Eigen::Vector3d computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights);
    Eigen::Vector3d computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first);
    Eigen::Vector3d computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first);
    Eigen::Vector3d computeBPT(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const std::vector<SubPath> &in_SubPath, bool first);

    Eigen::Vector3d computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<ParticipatingMedia> &all_media);
    Eigen::Vector3d computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<ParticipatingMedia> &all_media, bool first);

    Eigen::Vector3d computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights,const Object &in_Object, const int mode);
    Eigen::Vector3d computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const int mode);
    Eigen::Vector3d computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const std::vector<ParticipatingMedia> &all_medias, const Object &in_Object, const int mode);

    double getLightProbability(const std::vector<AreaLight> &in_AreaLights);
    double getDiffuseProbability(const Eigen::Vector3d normal, const Eigen::Vector3d out_dir);
    double getBlinnPhongProbability(const Eigen::Vector3d in_dir, const Eigen::Vector3d normal, const Eigen::Vector3d out_dir, const double m);
    double getPhaseProbability(const Eigen::Vector3d in_dir, const Eigen::Vector3d out_dir, const double hg_g);

    Eigen::Vector3d sampleRandomPoint(const AreaLight &in_Light);
    void diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth);
    void blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m, const int depth);
    void scatteringSaple(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_direction, Ray &out_ray, const int p_index, const double hg_g, const int depth);
    void HemisphericSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const int light_index);

    bool isInParticipatingMedia(const ParticipatingMedia &media, const Eigen::Vector3d &in_point);
    double getFreePath(const std::vector<ParticipatingMedia> &all_medias, const Eigen::Vector3d &in_point, int &index);

    Eigen::Vector3d BidirectinalPathTrace(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const std::vector<SubPath> &in_SubPath, const int mode);
    void LightTracing(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<SubPath> &in_subpath);
    Eigen::Vector3d setRadiance(const std::vector<AreaLight> &in_AreaLights, const std::vector<SubPath> &in_SubPath, const int index, const int light_index);
    void setProbability(std::vector<SubPath> &in_Subpath);
    Eigen::Vector3d calcGeometry(const Eigen::Vector3d &dir, const Object &in_Object, const std::vector<SubPath> &in_SubPath, const int index);
};


#endif //CGEXE_RENDERER_H

//
// Created by kouse on 2024/05/16.
//

#include <cstdlib>
#include <iostream>
#include "Renderer.h"
#include "random.h"
#include "Image.h"

#define __FAR__ 1.0e33
#define __PI__ 	3.14159265358979323846
#define MAX_RAY_DEPTH INT_MAX

Renderer::Renderer() {
    g_FilmBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_AccumulationBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_CountBuffer = (int *) malloc(sizeof(int) * g_FilmWidth * g_FilmHeight);

    resetFilm();
    clearRayTracedResult();
}
Renderer::Renderer(Camera camera, Object obj, std::vector<AreaLight> lights) {
    g_Camera = std::move(camera);
    g_Obj = std::move(obj);
    g_AreaLights = std::move(lights);
}

void Renderer::set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights, std::vector<ParticipatingMedia> media) {
    g_Camera = std::move(camera);
    g_Obj = std::move(obj);
    g_AreaLights = std::move(lights);
    g_ParticipatingMedia = std::move(media);

    g_FilmBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_AccumulationBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_CountBuffer = (int *) malloc(sizeof(int) * g_FilmWidth * g_FilmHeight);
}

void Renderer::setNsamples(const unsigned int nSample, const unsigned int samples) {
    if(nSample <= 0){
        nSamplesPerPixel = 1;
    }
    else{
        if(nSample > samples){
            nSamplesPerPixel = samples;
        }
        else{
            nSamplesPerPixel = nSample;
        }
    }
}

void Renderer::resetFilm() {
    memset(g_AccumulationBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    memset(g_CountBuffer, 0, sizeof(int) * g_FilmWidth * g_FilmHeight);
}

void Renderer::updateFilm() {
    for (int i = 0; i < g_FilmWidth * g_FilmHeight; i++) {
        if (g_CountBuffer[i] > 0) {
            g_FilmBuffer[i * 3] = g_AccumulationBuffer[i * 3] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 1] = g_AccumulationBuffer[i * 3 + 1] / g_CountBuffer[i];
            g_FilmBuffer[i * 3 + 2] = g_AccumulationBuffer[i * 3 + 2] / g_CountBuffer[i];
        }
        else {
            g_FilmBuffer[i * 3] = 0.0;
            g_FilmBuffer[i * 3 + 1] = 0.0;
            g_FilmBuffer[i * 3 + 2] = 0.0;
        }
    }
}

void Renderer::saveImg(const std::string filename) {
    GLubyte *g_ImgBuffer = new GLubyte[g_FilmWidth * g_FilmHeight * 3];

    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, g_FilmWidth, g_FilmHeight, GL_RGB, GL_UNSIGNED_BYTE, g_ImgBuffer);

    Image image(g_FilmWidth, g_FilmHeight, g_ImgBuffer);
    image.save(filename);
    image.generateCSV(filename);
}

void Renderer::stepToNextPixel(RayTracingInternalData &io_data) {
    io_data.nextPixel_i++;
    if (io_data.nextPixel_i >= g_FilmWidth) {
        io_data.nextPixel_i = 0;
        io_data.nextPixel_j++;

        if (io_data.nextPixel_j >= g_FilmHeight) {
            io_data.nextPixel_j = 0;
        }
    }
}

void Renderer::clearRayTracedResult() {
    g_RayTracingInternalData.nextPixel_i = -1;
    g_RayTracingInternalData.nextPixel_j = 0;

    memset(g_FilmBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
}

void Renderer::rayTriangleIntersect(const TriMesh &in_Mesh, const int in_Triangle_idx, const Ray &in_Ray, RayHit &out_Result) {
    out_Result.t = __FAR__;

    const Eigen::Vector3d v1 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](0)];
    const Eigen::Vector3d v2 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](1)];
    const Eigen::Vector3d v3 = in_Mesh.vertices[in_Mesh.triangles[in_Triangle_idx](2)];

    Eigen::Vector3d triangle_normal = (v1 - v3).cross(v2 - v3);
    triangle_normal.normalize();

    bool isFront = true;

    const double denominator = triangle_normal.dot(in_Ray.d);
    if (denominator >= 0.0)
        isFront = false;

    const double t = triangle_normal.dot(v3 - in_Ray.o) / denominator;

    if (t <= 0.0)
        return;

    const Eigen::Vector3d x = in_Ray.o + t * in_Ray.d;

    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = v1 - v3;
    A.col(1) = v2 - v3;

    Eigen::Matrix2d ATA = A.transpose() * A;
    const Eigen::Vector2d b = A.transpose() * (x - v3);

    const Eigen::Vector2d alpha_beta = ATA.inverse() * b;

    if (alpha_beta.x() < 0.0 || 1.0 < alpha_beta.x() || alpha_beta.y() < 0.0 || 1.0 < alpha_beta.y() ||
        1.0 - alpha_beta.x() - alpha_beta.y() < 0.0 || 1.0 < 1.0 - alpha_beta.x() - alpha_beta.y())
        return;

    out_Result.t = t;
    out_Result.alpha = alpha_beta.x();
    out_Result.beta = alpha_beta.y();
    out_Result.isFront = isFront;
}

void Renderer::rayAreaLightIntersect(const std::vector<AreaLight> &in_AreaLights, const int in_Light_idx, const Ray &in_Ray,
                                     RayHit &out_Result) {
    out_Result.t = __FAR__;

    const Eigen::Vector3d pos = in_AreaLights[in_Light_idx].pos;
    const Eigen::Vector3d arm_u = in_AreaLights[in_Light_idx].arm_u;
    const Eigen::Vector3d arm_v = in_AreaLights[in_Light_idx].arm_v;

    Eigen::Vector3d light_normal = arm_u.cross(arm_v);
    light_normal.normalize();

    bool isFront = true;

    const double denominator = light_normal.dot(in_Ray.d);
    if (denominator >= 0.0)
        isFront = false;

    const double t = light_normal.dot(pos - in_Ray.o) / denominator;

    if (t <= 0.0)
        return;

    const Eigen::Vector3d x = in_Ray.o + t * in_Ray.d;

    // rescale uv coordinates such that they reside in [-1, 1], when the hit point is inside of the light.
    const double u = (x - pos).dot(arm_u) / arm_u.squaredNorm();
    const double v = (x - pos).dot(arm_v) / arm_v.squaredNorm();

    if (u < -1.0 || 1.0 < u || v < -1.0 || 1.0 < v) return;

    out_Result.t = t;
    out_Result.alpha = u;
    out_Result.beta = v;
    out_Result.isFront = isFront;
}

Eigen::Vector2i Renderer::rayCameraIntersect(const Camera &in_Camera, const Ray &in_Ray, RayHit &out_Result) {
    out_Result.t = __FAR__;

    const Eigen::Vector3d ray_to_camera = in_Camera.getEyePoint() - in_Ray.o;
    const Eigen::Vector3d parallel = ray_to_camera.dot(in_Ray.d) * in_Ray.d;
    const double distance = (ray_to_camera - parallel).norm();

    if(distance > 1)
        return Eigen::Vector2i::Zero();

    const double camera_t = parallel.norm();
    if(camera_t <= 0.0)
        return Eigen::Vector2i::Zero();

    const Eigen::Vector3d n = -in_Camera.getZVector();

    const double denominator = n.dot(in_Ray.d);
    if(denominator >= 0.0)
        return Eigen::Vector2i::Zero();

    const Eigen::Vector3d left_up_point = in_Camera.getEyePoint() + n * in_Camera.getFocalLength();
    const double film_t = n.dot(left_up_point - in_Ray.o) / denominator;

    if(film_t <= 0.0)
        return Eigen::Vector2i::Zero();

    const Eigen::Vector3d x = in_Ray.o + film_t * in_Ray.d;
    const Eigen::Vector3d center_to_hit = x - in_Camera.getCenterPoint();

    const Eigen::Vector3d m_x_Vector = in_Camera.getXVector();
    const Eigen::Vector3d m_y_Vector = in_Camera.getYVector();

    const double halfScreenWidth = in_Camera.getScreenWidth() * 0.5;
    const double halfScreenHeight = in_Camera.getScreenHeight() * 0.5;

    const double x_length = (center_to_hit.dot(m_x_Vector) * m_x_Vector).norm();
    const double y_length = (center_to_hit.dot(m_y_Vector) * m_y_Vector).norm();

    if((abs(x_length) - halfScreenWidth < 1e-6) && (abs(y_length - halfScreenHeight < 1e-6))){
        out_Result.t = film_t;

        int x_pixel = static_cast<int>((x_length + halfScreenWidth) / halfScreenWidth / 2.0 * g_FilmWidth);
        x_pixel = (x_pixel == g_FilmWidth) ? x_pixel-1 : x_pixel;
        int y_pixel = static_cast<int>((y_length + halfScreenHeight) / halfScreenHeight / 2.0 * g_FilmHeight);
        y_pixel = (y_pixel == g_FilmHeight) ? y_pixel-1 : y_pixel;

        return Eigen::Vector2i{x_pixel, y_pixel};
    }

    return Eigen::Vector2i::Zero();
}

void Renderer::rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Ray &in_Ray, RayHit &io_Hit) {
    double t_min = __FAR__;
    double alpha_I = 0.0, beta_I = 0.0;
    int mesh_idx = -99;
    int primitive_idx = -1;
    bool isFront = true;

    for (int m = 0; m < in_Object.meshes.size(); m++) {
        for (int k = 0; k < in_Object.meshes[m].triangles.size(); k++) {
            if (m == in_Ray.prev_mesh_idx && k == in_Ray.prev_primitive_idx) continue;

            RayHit temp_hit;
            rayTriangleIntersect(in_Object.meshes[m], k, in_Ray, temp_hit);
            if (temp_hit.t < t_min) {
                t_min = temp_hit.t;
                alpha_I = temp_hit.alpha;
                beta_I = temp_hit.beta;
                mesh_idx = m;
                primitive_idx = k;
                isFront = temp_hit.isFront;
            }
        }
    }

    for (int l = 0; l < in_AreaLights.size(); l++) {
        if (-1 == in_Ray.prev_mesh_idx && l == in_Ray.prev_primitive_idx) continue;

        RayHit temp_hit;
        rayAreaLightIntersect(in_AreaLights, l, in_Ray, temp_hit);
        if (temp_hit.t < t_min) {
            t_min = temp_hit.t;
            alpha_I = temp_hit.alpha;
            beta_I = temp_hit.beta;
            mesh_idx = -1;
            primitive_idx = l;
            isFront = temp_hit.isFront;
        }
    }

    io_Hit.t = t_min;
    io_Hit.alpha = alpha_I;
    io_Hit.beta = beta_I;
    io_Hit.mesh_idx = mesh_idx;
    io_Hit.primitive_idx = primitive_idx;
    io_Hit.isFront = isFront;
}
void Renderer::rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const std::vector<ParticipatingMedia> &all_medias, const Ray &in_Ray, RayHit &io_Hit) {
    double t_min = __FAR__;
    double alpha_I = 0.0, beta_I = 0.0;
    int mesh_idx = -99;
    int primitive_idx = -1;
    bool isFront = true;

    for (int m = 0; m < in_Object.meshes.size(); m++) {
        for (int k = 0; k < in_Object.meshes[m].triangles.size(); k++) {
            if (m == in_Ray.prev_mesh_idx && k == in_Ray.prev_primitive_idx) continue;

            RayHit temp_hit;
            rayTriangleIntersect(in_Object.meshes[m], k, in_Ray, temp_hit);
            if (temp_hit.t < t_min) {
                t_min = temp_hit.t;
                alpha_I = temp_hit.alpha;
                beta_I = temp_hit.beta;
                mesh_idx = m;
                primitive_idx = k;
                isFront = temp_hit.isFront;
            }
        }
    }

    for (int l = 0; l < in_AreaLights.size(); l++) {
        if (-1 == in_Ray.prev_mesh_idx && l == in_Ray.prev_primitive_idx) continue;

        RayHit temp_hit;
        rayAreaLightIntersect(in_AreaLights, l, in_Ray, temp_hit);
        if (temp_hit.t < t_min) {
            t_min = temp_hit.t;
            alpha_I = temp_hit.alpha;
            beta_I = temp_hit.beta;
            mesh_idx = -1;
            primitive_idx = l;
            isFront = temp_hit.isFront;
        }
    }

    int p_index;
    const double s = getFreePath(all_medias, in_Ray.o, p_index);
    if(t_min > s){
        t_min = s;
        mesh_idx = -2;
        primitive_idx = p_index;
    }

    io_Hit.t = t_min;
    io_Hit.alpha = alpha_I;
    io_Hit.beta = beta_I;
    io_Hit.mesh_idx = mesh_idx;
    io_Hit.primitive_idx = primitive_idx;
    io_Hit.isFront = isFront;
}

Eigen::Vector3d Renderer::sampleRandomPoint(const AreaLight &in_Light) {
    const double r1 = 2.0 * randomMT() - 1.0;
    const double r2 = 2.0 * randomMT() - 1.0;
    return in_Light.pos + r1 * in_Light.arm_u + r2 * in_Light.arm_v;
}

Eigen::Vector3d Renderer::computeRayHitNormal(const Object &in_Object, const RayHit &in_Hit) {
    const int v1_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](0);
    const int v2_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](1);
    const int v3_idx = in_Object.meshes[in_Hit.mesh_idx].triangles[in_Hit.primitive_idx](2);

    const Eigen::Vector3d n1 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v1_idx];
    const Eigen::Vector3d n2 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v2_idx];
    const Eigen::Vector3d n3 = in_Object.meshes[in_Hit.mesh_idx].vertex_normals[v3_idx];

    const double gamma = 1.0 - in_Hit.alpha - in_Hit.beta;
    Eigen::Vector3d n = in_Hit.alpha * n1 + in_Hit.beta * n2 + gamma * n3;
    n.normalize();

    if (!in_Hit.isFront) n = -n;

    return n;
}

//original function for main.
void Renderer::rendering(const int mode) {
    for(int i = 0; i < g_FilmWidth * g_FilmHeight; i++){
        stepToNextPixel(g_RayTracingInternalData);

        int pixel_flat_idx =
                g_RayTracingInternalData.nextPixel_j * g_FilmWidth + g_RayTracingInternalData.nextPixel_i;

        Eigen::Vector3d I = Eigen::Vector3d::Zero();

        for (int k = 0; k < nSamplesPerPixel; k++) {
            double p_x = (g_RayTracingInternalData.nextPixel_i + randomMT()) / g_FilmWidth;
            double p_y = (g_RayTracingInternalData.nextPixel_j + randomMT()) / g_FilmHeight;

            Ray ray; ray.depth = 0;
            g_Camera.screenView(p_x, p_y, ray);
            ray.prev_mesh_idx = -99;
            ray.prev_primitive_idx = -1;

            g_SubPath.clear();

            switch(mode){
                case 1: {
                    I += computePathTrace(ray, g_Obj, g_AreaLights);
                    break;
                }
                case 2: {
                    I += computeNEE(ray, g_Obj, g_AreaLights, true);
                    break;
                }
                case 3: {
                    I += computeMIS(ray, g_Obj, g_AreaLights, true);
                    break;
                }
                case 4: {
                    I += computePathTrace(ray, g_Obj, g_AreaLights, g_ParticipatingMedia);
                    break;
                }
                case 5: {
                    I += computeMIS(ray, g_Obj, g_AreaLights, g_ParticipatingMedia, true);
                    break;
                }
                case 6: {
                    int light_index = static_cast<int>(randomMT() * g_AreaLights.size());
                    if(light_index == g_AreaLights.size()){
//                        std::cerr << "light_index gets over g_AreaLights.size()" << std::endl;
                        light_index--;
                    }

                    const Eigen::Vector3d light_cross = g_AreaLights[light_index].arm_u.cross(g_AreaLights[light_index].arm_v);
                    const Eigen::Vector3d light_normal = light_cross.normalized();

                    const Eigen::Vector3d light_point = sampleRandomPoint(g_AreaLights[light_index]);
                    Ray light_ray; RayHit light_rayHit;

                    HemisphericSample(light_point, light_normal, light_ray, light_index);
                    LightTracing(light_ray, g_Obj, g_AreaLights, g_SubPath);

                    const double cosine = light_ray.d.dot(light_normal);
                    light_rayHit.mesh_idx = -1;
                    light_rayHit.primitive_idx = light_index;
                    light_rayHit.isFront = true;
                    g_SubPath[0].rh = light_rayHit;
                    g_SubPath[0].contribute = Eigen::Vector3d::Ones() * 2.0f * __PI__;
                    g_SubPath[0].x = light_point;
                    g_SubPath[0].materialMode = 0;
                    g_SubPath[0].probability = light_cross.norm() * 4.0f;
                    for(int i = 0; i < g_SubPath.size(); i++){
                        g_SubPath[i].radiance = setRadiance(g_AreaLights, g_SubPath, i, light_index);
                    }

                    setProbability(g_SubPath);

                    I += computeBPT(ray, g_Obj, g_AreaLights, g_SubPath, true);
                    break;
                }
                case 7: {
                    int light_index = static_cast<int>(randomMT() * g_AreaLights.size());
                    if(light_index == g_AreaLights.size()){
//                        std::cerr << "light_index gets over g_AreaLights.size()" << std::endl;
                        light_index--;
                    }

                    const Eigen::Vector3d light_cross = g_AreaLights[light_index].arm_u.cross(g_AreaLights[light_index].arm_v);
                    const double light_area = light_cross.norm() * 4.0f;
                    Eigen::Vector3d Light_intensity = g_AreaLights[light_index].intensity * g_AreaLights[light_index].color * light_area;
                    const Eigen::Vector3d light_normal = light_cross.normalized();

                    const Eigen::Vector3d light_point = sampleRandomPoint(g_AreaLights[light_index]);
                    Ray light_ray; RayHit light_rayHit;

                    double pdf = HemisphericSample(light_point, light_normal, light_ray, light_index);
                    Light_intensity /= pdf;

                    Eigen::Vector2i pixels;
                    I += Light_intensity.cwiseProduct(computeLightTrace(light_ray, g_Obj, g_AreaLights, pixels));

                    pixel_flat_idx = pixels.y() * g_FilmWidth + pixels.x();
                    break;
                }
            }
        }

        g_AccumulationBuffer[pixel_flat_idx * 3] += I.x();
        g_AccumulationBuffer[pixel_flat_idx * 3 + 1] += I.y();
        g_AccumulationBuffer[pixel_flat_idx * 3 + 2] += I.z();
        g_CountBuffer[pixel_flat_idx] += nSamplesPerPixel;
    }
}

Eigen::Vector3d Renderer::computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (!in_RayHit.isFront)
            return Eigen::Vector3d::Zero();

        return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
    }

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    const double r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computePathTrace(new_ray, in_Object, in_AreaLights).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computePathTrace(new_ray, in_Object, in_AreaLights).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first) {
    if(in_Ray.depth >= MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (in_RayHit.isFront && first)
            return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;

        return Eigen::Vector3d::Zero();
    }

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    double r = randomMT() * (kd + ks);

    if(r < kd)
        I += computeDirectLighting(in_Ray, in_RayHit, in_AreaLights, in_Object, 1);
    else if(r < kd + ks)
        I += computeDirectLighting(in_Ray, in_RayHit, in_AreaLights, in_Object, 2);

    r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computeNEE(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeNEE(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (in_RayHit.isFront) {
            if (first) {
                return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
            else{
                //pathTrace
                const Eigen::Vector3d n_light = in_AreaLights[in_RayHit.primitive_idx].arm_u.cross(in_AreaLights[in_RayHit.primitive_idx].arm_v);
                const double cosine = std::max<double>(0.0f, n_light.dot(- in_Ray.d));
                const double distance = (x - in_Ray.o).norm();
                const double path_pdf = in_Ray.pdf;
                const double nee_pdf = getLightProbability(in_AreaLights) * distance * distance / cosine;
                const double MIS_weight = (path_pdf * path_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                const double MIS_weight = 0.5f;

                return MIS_weight * in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
        }
        return Eigen::Vector3d::Zero();
    }

    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);
    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    double r = randomMT() * (kd + ks);

    if(r < kd)
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, in_Object, 1);
    else if(r < kd + ks)
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, in_Object, 2);

    r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computeMIS(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f)
            return I;

        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeMIS(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeBPT(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const std::vector<SubPath> &in_SubPath, bool first) {
    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;

    //いったん重みは考えない
    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (in_RayHit.isFront && first)
            return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
        else if(in_RayHit.isFront)
            return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color / (in_Ray.depth + 1);
        return Eigen::Vector3d::Zero();
    }

    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);
    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    double r = randomMT() * (kd + ks);

    if(r < kd)
        I += BidirectinalPathTrace(in_Ray, in_RayHit, in_AreaLights, in_Object, in_SubPath, 1);
    else if(r < kd + ks)
        I += BidirectinalPathTrace(in_Ray, in_RayHit, in_AreaLights, in_Object, in_SubPath, 2);

    r = randomMT();
    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computeBPT(new_ray, in_Object, in_AreaLights, in_SubPath, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f)
            return I;

        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeBPT(new_ray, in_Object, in_AreaLights, in_SubPath, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeLightTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, Eigen::Vector2i &pixels) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();


    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
//        const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
//        const Eigen::Vector3d n = in_AreaLights[in_RayHit.primitive_idx].arm_u.cross(in_AreaLights[in_RayHit.primitive_idx].arm_v).normalized();
//        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
//        return computeLightTrace(new_ray, in_Object, in_AreaLights, pixels);

        return Eigen::Vector3d::Zero();
    }

    RayHit view_hit;
    pixels = rayCameraIntersect(g_Camera, in_Ray, view_hit);
    if (in_RayHit.t > view_hit.t){
        return Eigen::Vector3d::Ones();
    }

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);


    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    const double r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computeLightTrace(new_ray, in_Object, in_AreaLights, pixels).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeLightTrace(new_ray, in_Object, in_AreaLights, pixels).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<ParticipatingMedia> &all_media) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, all_media, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (!in_RayHit.isFront)
            return Eigen::Vector3d::Zero();

        return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
    }

    Eigen::Vector3d I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;

    if(in_RayHit.mesh_idx == -2){
        const Eigen::Vector3d n =  in_Ray.d;

        const int p_index = in_RayHit.primitive_idx;
        ParticipatingMedia *p = &all_media[p_index];
        const double albedo = p->albedo;
        const double r = randomMT();
        if(r >= albedo)
            return Eigen::Vector3d::Zero();

        scatteringSaple(x, in_Ray.d, new_ray, p_index, p->hg_g, in_Ray.depth);
        I += computePathTrace(new_ray, in_Object, in_AreaLights, all_media).cwiseProduct(p->color) / albedo;
    }
    else{
        const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);
        const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
        const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
        const double r = randomMT();

        if(r < kd){
            diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
            I += computePathTrace(new_ray, in_Object, in_AreaLights, all_media).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
        }
        else if(r < kd + ks){
            const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
            blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
            if(new_ray.pdf < 0.0f)
                return I;

            const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
            I += computePathTrace(new_ray, in_Object, in_AreaLights, all_media).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
        }
    }

    return I;
}

Eigen::Vector3d Renderer::computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<ParticipatingMedia> &all_media, bool first) {
    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, all_media, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (in_RayHit.isFront) {
            if (first) {
                return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
            else{
                //pathTrace
                const Eigen::Vector3d n_light = in_AreaLights[in_RayHit.primitive_idx].arm_u.cross(in_AreaLights[in_RayHit.primitive_idx].arm_v);
                const double cosine = std::max<double>(0.0f, n_light.dot(- in_Ray.d));
                const double distance = (x - in_Ray.o).norm();
                const double path_pdf = in_Ray.pdf;
                const double nee_pdf = getLightProbability(in_AreaLights) * distance * distance / cosine;
                const double MIS_weight = (path_pdf * path_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                const double MIS_weight = 0.0f;

                return MIS_weight * in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
        }
        return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    if(in_RayHit.mesh_idx == -2){
        //participating media
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, all_media, in_Object, 0);

        const int p_index = in_RayHit.primitive_idx;
        ParticipatingMedia *p = &all_media[p_index];
        const double albedo = p->albedo;
        const double r = randomMT();
        if(r >= albedo)
            return Eigen::Vector3d::Zero();

        scatteringSaple(x, in_Ray.d, new_ray, p_index, p->hg_g, in_Ray.depth);
        I += computeMIS(new_ray, in_Object, in_AreaLights, all_media, false).cwiseProduct(p->color) / albedo;
    }
    else{
        const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

        const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
        const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
        double r = randomMT() * (kd + ks);

        if(r < kd)
            I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, all_media, in_Object, 1);
        else if(r < kd + ks)
            I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, all_media, in_Object, 2);

        r = randomMT();

        if(r < kd){
            diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
            I += computeMIS(new_ray, in_Object, in_AreaLights, all_media, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
        }
        else if(r < kd + ks){
            const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
            blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
            if(new_ray.pdf < 0.0f)
                return I;

            const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
            I += computeMIS(new_ray, in_Object, in_AreaLights, all_media, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
        }
    }

    return I;
}

Eigen::Vector3d Renderer::computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const int mode) {
    Eigen::Vector3d I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    for(int i = 0; i < in_AreaLights.size(); i++) {
        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d x_L = p_light - x;
        const double dist = x_L.norm();
        x_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cos_light = n_light.dot(-x_L);
        if(cos_light <= 0.0) continue;

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx == -1 && rh.primitive_idx == i) {
            const double cos_x =  x_L.dot(n);
            if(cos_x <= 0.0) continue;
            const double G = (cos_x * cos_light) / (dist * dist);

            switch(mode){
                case 1: {
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / __PI__;
                    I += area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
                case 2: {
                    const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
                    const Eigen::Vector3d halfVector = ((-1 * in_Ray.d) + x_L).normalized();
                    const double cosine = std::max<double>(0.0f, n.dot(halfVector));
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__);
                    I += area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
            }
        }
    }

    return I;
}
Eigen::Vector3d Renderer::computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const int mode) {
    Eigen::Vector3d I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    for(int i = 0; i < in_AreaLights.size(); i++) {
        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d x_L = p_light - x;
        const double dist = x_L.norm();
        x_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cos_light = n_light.dot(-x_L);
        if (cos_light <= 0.0) continue;

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx == -1 && rh.primitive_idx == i) {
            const double cos_x = n.dot(x_L);
            if (cos_x <= 0.0) continue;
            const double G = (cos_x * cos_light) / (dist * dist);

            switch(mode){
                case 1: {
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / __PI__;

                    const double path_pdf = getDiffuseProbability(n, x_L);
                    const double nee_pdf = (dist * dist) / (area * cos_light);
                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                    const double MIS_weight = 0.5f;

                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
                case 2: {
                    const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;

                    const Eigen::Vector3d halfVector = ((-1 * in_Ray.d) + x_L).normalized();
                    const double cosine = std::max<double>(0.0f, n.dot(halfVector));
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__);

                    const double path_pdf = getBlinnPhongProbability(in_Ray.d, n, x_L, m);
                    const double nee_pdf = (dist * dist) / (area * cos_light);
                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                    const double MIS_weight = 0.5f;

                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
            }
        }
    }

    return I;
}
Eigen::Vector3d Renderer::computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const std::vector<ParticipatingMedia> &all_medias, const Object &in_Object, const int mode) {
    Eigen::Vector3d I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    Eigen::Vector3d n;
    if(in_RayHit.mesh_idx == -2)
        n = Eigen::Vector3d::Ones();
    else
        n = computeRayHitNormal(in_Object, in_RayHit);

    int p_index;
    const double s = getFreePath(all_medias, x, p_index);

    for(int i = 0; i < in_AreaLights.size(); i++) {
        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d x_L = p_light - x;
        const double distance = x_L.norm();
        x_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cos_light = n_light.dot(-x_L);
        if (cos_light <= 0.0) continue;

        if(s < distance)
            continue;

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx == -1 && rh.primitive_idx == i) {
            const double cos_x = n.dot(x_L);
            if (cos_x <= 0.0) continue;
            const double G = (cos_x * cos_light) / (distance * distance);
            const double nee_pdf = (distance * distance) / (area * cos_light);

            switch(mode){
                case 0:{
                    const double path_pdf = getPhaseProbability(in_Ray.d, x_L, all_medias[p_index].hg_g);
                    const Eigen::Vector3d BSDF = all_medias[p_index].color * path_pdf;  //pdfがBSDFに相当

                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                    const double MIS_weight = 1.0f;
                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
                case 1: {
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / __PI__;
                    const double path_pdf = getDiffuseProbability(n, x_L);

                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                    const double MIS_weight = 1.0f;
                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
                case 2: {
                    const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;

                    const Eigen::Vector3d halfVector = ((-1 * in_Ray.d) + x_L).normalized();
                    const double cosine = std::max<double>(0.0f, n.dot(halfVector));
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__);
                    const double path_pdf = getBlinnPhongProbability(in_Ray.d, n, x_L, m);
                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);
//                    const double MIS_weight = 1.0f;
                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
            }
        }
    }
    return I;
}

double Renderer::getLightProbability(const std::vector<AreaLight> &in_AreaLights) {
    double pdf = 0.0f;

    for(int i = 0; i < in_AreaLights.size(); i++){
        const Eigen::Vector3d light_cross = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = light_cross.norm() * 4.0;

        pdf += area;
    }

    return 1.0f / pdf;
}
double Renderer::getDiffuseProbability(const Eigen::Vector3d normal, const Eigen::Vector3d out_dir) {
    const double cosine = std::max<double>(0.0f, normal.dot(out_dir));

    return cosine / __PI__;
}
double Renderer::getBlinnPhongProbability(const Eigen::Vector3d in_dir, const Eigen::Vector3d normal,
                                          const Eigen::Vector3d out_dir, const double m) {
    const Eigen::Vector3d halfVector = ((-1.0f * in_dir) + out_dir).normalized();
    const double cosine = std::max<double>(0.0f, normal.dot(halfVector));

    return (m + 1) * pow(cosine, m) / (2.0 * __PI__);
}
double Renderer::getPhaseProbability(const Eigen::Vector3d in_dir, const Eigen::Vector3d out_dir, const double hg_g) {
    const double cosine = in_dir.dot(out_dir);
    return (1 - hg_g * hg_g) / (4.0f * __PI__ * pow((1 + hg_g * hg_g - 2.0f * hg_g * cosine), 1.5f));
}

void Renderer::diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(sqrt(randomMT()));
    const double phi = randomMT() * 2.0f * __PI__;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d x_L = _dx * bn + _dy * in_n + _dz * cn;
    x_L.normalize();
    out_ray.o = in_x;
    out_ray.d = x_L;
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;
    out_ray.depth = depth + 1;
    out_ray.pdf = cos(theta) / __PI__;
}
void Renderer::blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m, const int depth) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(pow(randomMT(), 1.0f / (m + 1.0f)));
    const double phi = randomMT() * 2.0f * __PI__;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d halfVector = _dx * bn + _dy * in_n + _dz * cn;
    halfVector.normalize();

    const Eigen::Vector3d o_parallel = (-1 * in_direction).dot(halfVector) * halfVector;
    const Eigen::Vector3d o_vertical = (-1 * in_direction) - o_parallel;

    out_ray.o = in_x;
    out_ray.d = (o_parallel - o_vertical).normalized();
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;
    out_ray.depth = depth + 1;


    if(out_ray.d.dot(bn) < 0.0f)
        out_ray.pdf = -1.0f;
    else
        out_ray.pdf = (m + 1) * pow(cos(theta), m) / (2.0f * __PI__);
}

void Renderer::scatteringSaple(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_direction, Ray &out_ray, const int p_index, const double hg_g, const int depth) {
    Eigen::Vector3d bn;
    if(fabs(in_direction.x()) > 1e-4)
        bn = Eigen::Vector3d::UnitY().cross(in_direction).normalized();
    else
        bn = Eigen::Vector3d::UnitX().cross(in_direction).normalized();

    const Eigen::Vector3d cn = bn.cross(in_direction);

    double cosine;
    if(fabs(hg_g) > 1e-4){
        cosine = 2.0f * randomMT() - 1.0f;
    }
    else{
        double f = (1.0f - hg_g * hg_g) / (1.0f - hg_g + 2.0f * hg_g * randomMT());
        cosine = (1.0f + hg_g * hg_g - f * f) / (2.0f * hg_g);
    }

    const double cosine_t = (cosine < -1.0f) ? -1.0f : ((cosine > 1.0f) ? 1.0f : cosine);
    const double theta = acos(cosine_t);
    const double phi = 2.0f * __PI__ * randomMT();

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d d = _dx * bn + _dy * in_direction + _dz * cn;
    d.normalize();

    out_ray.o = in_x;
    out_ray.d = d;
    out_ray.prev_mesh_idx = -2;
    out_ray.prev_primitive_idx = p_index;
    out_ray.depth = depth + 1;
    out_ray.pdf = (1 - hg_g * hg_g) / pow((1 + hg_g * hg_g - 2.0f * hg_g * cos(theta)), 1.5f);
}

double Renderer::HemisphericSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const int light_index) {
    Eigen::Vector3d bn;
    if(fabs(in_n.x()) > 1e-3)
        bn = Eigen::Vector3d::UnitY().cross(in_n).normalized();
    else
        bn = Eigen::Vector3d::UnitX().cross(in_n).normalized();

    const Eigen::Vector3d cn = in_n.cross(bn);

    const double theta = acos(randomMT());
    const double phi = randomMT() * 2.0f * __PI__;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d x_L = _dx * bn + _dy * in_n + _dz * cn;
    x_L.normalize();
    out_ray.o = in_x;
    out_ray.d = x_L;
    out_ray.prev_mesh_idx = -1;
    out_ray.prev_primitive_idx = light_index;
    out_ray.depth = 1;
    out_ray.pdf = 1.0f / (2.0f * __PI__);

    return out_ray.pdf;
}

bool Renderer::isInParticipatingMedia(const ParticipatingMedia &media, const Eigen::Vector3d &in_point) {
    double distance = (media.pos - in_point).norm();

    if(media.radius > distance)
        return true;

    return false;
}

double Renderer::getFreePath(const std::vector<ParticipatingMedia> &all_medias, const Eigen::Vector3d &in_point, int &index) {
    double s_min = DBL_MAX;
    index = -1;
    for(int i = 0; i < all_medias.size(); i++){
        if(!isInParticipatingMedia(all_medias[i], in_point))
            continue;

        double extinction = all_medias[i].extinction;
        if(extinction > 1e-6) {
            const double s = - log(randomMT()) / extinction;
            if(s_min > s){
                s_min = s;
                index = i;
            }
        }
    }

    return s_min;
}

Eigen::Vector3d Renderer::BidirectinalPathTrace(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const std::vector<SubPath> &in_SubPath, const int mode) {
    Eigen::Vector3d I = Eigen::Vector3d::Zero();
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);
    const int depth = in_Ray.depth;

    for(int i = 0; i < in_SubPath.size(); i++){
        if(i > in_SubPath.size() - 1 || i < 0) return Eigen::Vector3d::Zero();

        Eigen::Vector3d connect_dir = in_SubPath[i].x - x;
        const double dist = connect_dir.norm();
        connect_dir.normalize();

        Eigen::Vector3d connect_normal;
        if(in_SubPath[i].rh.mesh_idx == -1)
            connect_normal = (in_AreaLights[in_SubPath[i].rh.primitive_idx].arm_u.cross(in_AreaLights[in_SubPath[i].rh.primitive_idx].arm_v)).normalized();
        else if(in_SubPath[i].rh.mesh_idx >= 0)
            connect_normal = computeRayHitNormal(in_Object, in_SubPath[i].rh);

        const double connect_cos = connect_normal.dot(-connect_dir);
        if(connect_cos <= 0.0) continue;

        // shadow test
        Ray _ray;
        _ray.o = x;
        _ray.d = connect_dir;
        _ray.prev_mesh_idx = in_RayHit.mesh_idx;
        _ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit _rh;
        rayTracing(in_Object, in_AreaLights, _ray, _rh);
        if(_rh.mesh_idx == in_SubPath[i].rh.mesh_idx && _rh.primitive_idx == in_SubPath[i].rh.primitive_idx){
            //接続が成功
            const double cos_x = n.dot(connect_dir);
            if(cos_x <= 0.0) continue;

            const double G = (cos_x * connect_cos) / (dist * dist);

            switch(mode){
                case 1: {
                    const Eigen::Vector3d connect_BSDF = (in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / __PI__).cwiseProduct(G * calcGeometry(-connect_dir, in_Object, in_SubPath, i));
                    I += in_SubPath[i].radiance.cwiseProduct(connect_BSDF) / (depth + i + 2);
                    break;
                }
                case 2: {
                    const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
                    const Eigen::Vector3d halfVector = ((-1 * in_Ray.d) + connect_dir).normalized();
                    const double cosine = std::max<double>(0.0f, n.dot(halfVector));
                    const Eigen::Vector3d connect_BSDF = (in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__)).cwiseProduct(G * calcGeometry(-connect_dir, in_Object, in_SubPath, i));
                    I += in_SubPath[i].radiance.cwiseProduct(connect_BSDF) / (depth + i + 2);
                    break;
                }
            }
        }
    }

    return I;
}

void Renderer::LightTracing(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, std::vector<SubPath> &in_subpath) {
    int depth = in_Ray.depth;
    if(in_Ray.depth > MAX_RAY_DEPTH){
        in_subpath.resize(depth);
        return;
    }

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0){
        in_subpath.resize(depth);
        return;
    }

    if(in_RayHit.mesh_idx == -1){
        in_subpath.resize(depth);
        return;
    }

    //viewPointに衝突した場合の処理はここに書く
    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    const double r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        LightTracing(new_ray, in_Object, in_AreaLights, in_subpath);

        in_subpath[depth].contribute = in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / kd;
        in_subpath[depth].materialMode = 1;
        in_subpath[depth].probability = in_Ray.pdf;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(new_ray.pdf < 0.0f){
            in_subpath.resize(depth + 1);
            return;
        }
        LightTracing(new_ray, in_Object, in_AreaLights, in_subpath);

        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        in_subpath[depth].contribute = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f) / (ks * (m + 1.0f));
        in_subpath[depth].materialMode = 2;
        in_subpath[depth].probability = in_Ray.pdf;
    }
    else{
        in_subpath.resize(depth);
        return;
    }

    in_subpath[depth].in_dir = in_Ray.d;
    in_subpath[depth].x = new_ray.o;
    in_subpath[depth].rh = in_RayHit;

    return;
}

Eigen::Vector3d Renderer::setRadiance(const std::vector<AreaLight> &in_AreaLights, const std::vector<SubPath> &in_SubPath, const int index, const int light_index) {
    const double pdf = in_AreaLights[light_index].arm_u.cross(in_AreaLights[light_index].arm_v).norm() * 4.0f;
    Eigen::Vector3d I = in_AreaLights[light_index].intensity * in_AreaLights[light_index].color * pdf;

    for(int i = 0; i < index; i++){
        I = I.cwiseProduct(in_SubPath[i].contribute);
    }

    return I;
}

void Renderer::setProbability(std::vector<SubPath> &in_Subpath) {
    for(int i = 1; i < in_Subpath.size(); i++){
        in_Subpath[i].probability *= in_Subpath[i - 1].probability;
    }
}

Eigen::Vector3d Renderer::calcGeometry(const Eigen::Vector3d &dir, const Object &in_Object, const std::vector<SubPath> &in_SubPath, const int index) {
    SubPath _s = in_SubPath[index];
    RayHit _rh = _s.rh;

    switch(_s.materialMode){
        case 0: {
            return Eigen::Vector3d::Ones();
        }
        case 1: {
            return in_Object.meshes[_rh.mesh_idx].material.getKd() / __PI__;
        }
        case 2: {
            const Eigen::Vector3d n = computeRayHitNormal(in_Object, _rh);
            const double m = in_Object.meshes[_rh.mesh_idx].material.m;
            const Eigen::Vector3d halfVector = (dir + _s.in_dir).normalized();
            const double cosine = std::max<double>(0.0f, n.dot(halfVector));
            return in_Object.meshes[_rh.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__);
        }
    }

    std::cerr << "error happens in calcGeometry()" << std::endl;
    exit(1);
}
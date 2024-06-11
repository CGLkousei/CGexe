//
// Created by kouse on 2024/05/16.
//

#include <cstdlib>
#include "Renderer.h"
#include "random.h"
#include "Image.h"

#define __FAR__ 1.0e33
#define __PI__ 	3.14159265358979323846

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

void Renderer::set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights) {
    g_Camera = std::move(camera);
    g_Obj = std::move(obj);
    g_AreaLights = std::move(lights);

    g_FilmBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_AccumulationBuffer = (float *) malloc(sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    g_CountBuffer = (int *) malloc(sizeof(int) * g_FilmWidth * g_FilmHeight);
}

void Renderer::resetFilm() {
    memset(g_AccumulationBuffer, 0, sizeof(float) * g_FilmWidth * g_FilmHeight * 3);
    memset(g_CountBuffer, 0, sizeof(int) * g_FilmWidth * g_FilmHeight);
}

void Renderer::saveImg(const std::string filename) {
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

        const int pixel_flat_idx =
                g_RayTracingInternalData.nextPixel_j * g_FilmWidth + g_RayTracingInternalData.nextPixel_i;

        Eigen::Vector3d I = Eigen::Vector3d::Zero();

        for (int k = 0; k < nSamplesPerPixel; k++) {
            double p_x = (g_RayTracingInternalData.nextPixel_i + randomMT()) / g_FilmWidth;
            double p_y = (g_RayTracingInternalData.nextPixel_j + randomMT()) / g_FilmHeight;

            Ray ray;
            g_Camera.screenView(p_x, p_y, ray);
            ray.prev_mesh_idx = -99;
            ray.prev_primitive_idx = -1;


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
            }
        }

        g_AccumulationBuffer[pixel_flat_idx * 3] += I.x();
        g_AccumulationBuffer[pixel_flat_idx * 3 + 1] += I.y();
        g_AccumulationBuffer[pixel_flat_idx * 3 + 2] += I.z();
        g_CountBuffer[pixel_flat_idx] += nSamplesPerPixel;

        g_FilmBuffer[i * 3] = g_AccumulationBuffer[i * 3] / g_CountBuffer[i];
        g_FilmBuffer[i * 3 + 1] = g_AccumulationBuffer[i * 3 + 1] / g_CountBuffer[i];
        g_FilmBuffer[i * 3 + 2] = g_AccumulationBuffer[i * 3 + 2] / g_CountBuffer[i];
    }
}

Eigen::Vector3d Renderer::computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights) {
    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx < 0) // the ray has hit an area light
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
        diffuseSample(x, n, new_ray, in_RayHit, in_Object);
        I += computePathTrace(new_ray, in_Object, in_AreaLights).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, in_Object.meshes[in_RayHit.mesh_idx].material.m);
        if(pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computePathTrace(new_ray, in_Object, in_AreaLights).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine) / ks;
    }

    return I;
}

Eigen::Vector3d Renderer::computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first) {
    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx < 0) // the ray has hit an area light
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
    const double r = randomMT();

    if(r < kd){
        I += computeDirectLighting(in_Ray, in_RayHit, in_AreaLights, in_Object, 1);
        diffuseSample(x, n, new_ray, in_RayHit, in_Object);
        I += computeNEE(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        I += computeDirectLighting(in_Ray, in_RayHit, in_AreaLights, in_Object, 2);
        const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, in_Object.meshes[in_RayHit.mesh_idx].material.m);
        if(pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeNEE(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine) / ks;
    }

    return I;
}

Eigen::Vector3d Renderer::computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, bool first) {
    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;

    if (in_RayHit.mesh_idx < 0) // the ray has hit an area light
    {
        if (in_RayHit.isFront) {
            if (first) {
                return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
            else{
                //pathTrace
                const Eigen::Vector3d n_light = in_AreaLights[in_RayHit.primitive_idx].arm_u.cross(in_AreaLights[in_RayHit.primitive_idx].arm_v);
                const double cosine = std::max<double>(0.0f, n_light.dot(-1.0f * in_Ray.d));
                const double distance = (x - in_Ray.o).norm();
                const double path_pdf = in_Ray.pdf;
                const double nee_pdf = getLightProbability(in_AreaLights) * distance * distance / cosine;
                const double MIS_weight = (path_pdf * path_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);

                return MIS_weight * in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
            }
        }
        return Eigen::Vector3d::Zero();
    }

    const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);
    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
    const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
    const double r = randomMT();

    if(r < kd){
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, in_Object, 1);
        const double pdf = diffuseSample(x, n, new_ray, in_RayHit, in_Object);
        new_ray.pdf = pdf;
        I += computeMIS(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_AreaLights, in_Object, 2);
        const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, in_Object.meshes[in_RayHit.mesh_idx].material.m);
        if(pdf < 0.0f)
            return I;
        new_ray.pdf = pdf;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeMIS(new_ray, in_Object, in_AreaLights, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine) / ks;
    }

    return I;
}

Eigen::Vector3d Renderer::computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const std::vector<AreaLight> &in_AreaLights, const Object &in_Object, const int mode) {
    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    for(int i = 0; i < in_AreaLights.size(); i++) {
        const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
        const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d x_L = p_light - x;
        const double dist = x_L.norm();
        x_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cos_light = n_light.dot(-x_L);
        if (cos_light <= 0.0) continue;
        const double cos_x = std::max<double>(0.0, x_L.dot(n));
        const double G = (cos_x * cos_light) / (dist * dist);

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx < 0 && rh.primitive_idx == i) {
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
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 1) * pow(cosine, m) / (2.0f * __PI__);
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

    for(int i = 0; i < in_AreaLights.size(); i++) {
        const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
        const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

        const Eigen::Vector3d p_light = sampleRandomPoint(in_AreaLights[i]);
        Eigen::Vector3d x_L = p_light - x;
        const double dist = x_L.norm();
        x_L.normalize();

        Eigen::Vector3d n_light = in_AreaLights[i].arm_u.cross(in_AreaLights[i].arm_v);
        const double area = n_light.norm() * 4.0;
        n_light.normalize();
        const double cos_light = n_light.dot(-x_L);
        if (cos_light <= 0.0) continue;
        const double cos_x = std::max<double>(0.0, x_L.dot(n));
        const double G = (cos_x * cos_light) / (dist * dist);

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, ray, rh);
        if (rh.mesh_idx < 0 && rh.primitive_idx == i) {
            switch(mode){
                case 1: {
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKd() / __PI__;

                    const double path_pdf = getDiffuseProbablitity(n, x_L);
                    const double nee_pdf = (dist * dist) / (area * cos_light);
                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);

                    I += MIS_weight * area * in_AreaLights[i].intensity * in_AreaLights[i].color.cwiseProduct(BSDF * G);
                    break;
                }
                case 2: {
                    const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;

                    const Eigen::Vector3d halfVector = ((-1 * in_Ray.d) + x_L).normalized();
                    const double cosine = std::max<double>(0.0f, n.dot(halfVector));
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 1) * pow(cosine, m) / (2.0f * __PI__);

                    const double path_pdf = getBlinnPhongProbablitity(in_Ray.d, n, x_L, m);
                    const double nee_pdf = (dist * dist) / (area * cos_light);
                    const double MIS_weight = (nee_pdf * nee_pdf) / (nee_pdf * nee_pdf + path_pdf * path_pdf);

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
double Renderer::getDiffuseProbablitity(const Eigen::Vector3d normal, const Eigen::Vector3d out_dir) {
    const double cosine = std::max<double>(0.0f, normal.dot(out_dir));

    return cosine / __PI__;
}
double Renderer::getBlinnPhongProbablitity(const Eigen::Vector3d in_dir, const Eigen::Vector3d normal, const Eigen::Vector3d out_dir, const double m) {
    const Eigen::Vector3d halfVector = ((-1.0f * in_dir) + out_dir).normalized();
    const double cosine = std::max<double>(0.0f, normal.dot(halfVector));

    return (m + 1) * pow(cosine, m) / (2.0 * __PI__);
}

double Renderer::diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(sqrt(randomMT()));
    const double phi = randomMT() * 2.0 * __PI__;

    const double _dx = sin(theta) * cos(phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(phi);

    Eigen::Vector3d x_L = _dx * bn + _dy * in_n + _dz * cn;
    x_L.normalize();
    out_ray.o = in_x;
    out_ray.d = x_L;
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;

    return cos(theta) / __PI__;
}
double Renderer::blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(pow((1.0f - randomMT()), 1.0f / (m + 1.0f)));
    const double phi = randomMT() * 2.0 * __PI__;

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

    if(out_ray.d.dot(bn) < 0.0f)
        return -1.0f;

    return (m + 1) * pow(cos(theta), m) / (2.0f * __PI__);
}
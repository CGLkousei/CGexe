//
// Created by kouse on 2024/05/16.
//

#include <cstdlib>
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

void Renderer::set3Dscene(Camera camera, Object obj, std::vector<AreaLight> lights, Hair hair) {
    g_Camera = std::move(camera);
    g_Obj = std::move(obj);
    g_AreaLights = std::move(lights);
    g_Hair = std::move(hair);

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

void Renderer::rayHairIntersect(const TriCurb &in_Curb, const int in_Line_idx, const Ray &in_Ray, RayHit &out_Result) {
    out_Result.t = __FAR__;

    const Eigen::Vector3d v1 = in_Curb.vertices[in_Curb.lines[in_Line_idx].x()];
    const Eigen::Vector3d v2 = in_Curb.vertices[in_Curb.lines[in_Line_idx].y()];

    const double radius = in_Curb.radius;

    bool isFront = true;

    const Eigen::Vector3d constant_A = in_Ray.o - v1;
    const Eigen::Vector3d v1_to_v2 = v2 - v1;
    const double norm = v1_to_v2.norm();
    const Eigen::Vector3d d = in_Ray.d;

    const Eigen::Vector3d d_cross = d.cross(v1_to_v2);
    const Eigen::Vector3d A_cross = constant_A.cross(v1_to_v2);

    const double a = d_cross.dot(d_cross);
    const double b = A_cross.dot(d_cross);
    const double c = A_cross.dot(A_cross) - radius * radius * norm * norm;

//    const double a = 1.0f;
//    const double b = d.dot(in_Ray.o - v1);
//    const double c = (in_Ray.o - v1).dot(in_Ray.o - v1) - radius * radius;

//    const Eigen::Vector3d d = (v2 - v1).normalized();
//    const Eigen::Vector3d o = in_Ray.o;
//    const double minus = o.dot(d) - v1.dot(d);
//    const double a = 4.0f;
//    const double b = 2.0f * d.dot(in_Ray.o - v1) + 2.0f * minus;
//    const double c = (o - v1).dot(o - v1) + 2.0f * minus * (o - v1).dot(d) + minus * minus - radius * radius;

    const double discriminant = b * b - a * c;

    if(discriminant < 1e-6)
        return;

    const Eigen::Array2d distances{(-b - sqrt(discriminant)) / a, (-b + sqrt(discriminant)) / a};
    if((distances < 1e-6).all()) return;

    const double t = distances[0] > 1e-6 ? distances[0] : distances[1];
    const Eigen::Vector3d point = in_Ray.o + d * t;

    const double u = (point - v1).dot(v1_to_v2) / (norm * norm);
    if(u < 0 || u > 1)
        return;

    const Eigen::Vector3d parallel = u * v1_to_v2;
    Eigen::Vector3d n = (point - parallel).normalized();

    if(in_Ray.d.dot(n) > 0.0f) {
        n = -n;
        isFront = false;
    }

    const double cos_gamma = n.dot(-d);

    out_Result.t = t;
    out_Result.isFront = isFront;
    out_Result.n = n;
    out_Result.h = std::sqrt(1 - cos_gamma * cos_gamma);
}

void Renderer::rayTracing(const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const Ray &in_Ray, RayHit &io_Hit) {
    double t_min = __FAR__;
    double alpha_I = 0.0, beta_I = 0.0;
    int mesh_idx = -99;
    int primitive_idx = -1;
    bool isFront = true;
    bool isHair = false;
    double h = 0.0;
    Eigen::Vector3d n = Eigen::Vector3d::Zero();

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
                isHair = false;
                h = 0.0;
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
            isHair = false;
            h = 0.0;
        }
    }

    for (int m = 0; m < in_Hair.hairs.size(); m++) {
        for (int k = 0; k < in_Hair.hairs[m].lines.size(); k++) {
            if(m == in_Ray.prev_mesh_idx && k == in_Ray.prev_primitive_idx) continue;

            RayHit temp_hit;
            rayHairIntersect(in_Hair.hairs[m], k, in_Ray, temp_hit);
            if(temp_hit.t < t_min){
                t_min = temp_hit.t;
                mesh_idx = m;
                primitive_idx = k;
                isFront = temp_hit.isFront;
                isHair = true;
                n = temp_hit.n;
                h = temp_hit.h;
            }
        }
    }

    io_Hit.t = t_min;
    io_Hit.alpha = alpha_I;
    io_Hit.beta = beta_I;
    io_Hit.mesh_idx = mesh_idx;
    io_Hit.primitive_idx = primitive_idx;
    io_Hit.isFront = isFront;
    io_Hit.isHair = isHair;
    io_Hit.n = n;
    io_Hit.h = h;
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
                    I += computePathTrace(ray, g_Obj, g_AreaLights, g_Hair);
                    break;
                }
                case 2: {
                    I += computeNEE(ray, g_Obj, g_AreaLights, g_Hair, true);
                    break;
                }
                case 3: {
                    I += computeMIS(ray, g_Obj, g_AreaLights, g_Hair, true);
                    break;
                }
                case 4: {
                    RayHit in_RayHit;
                    rayTracing(g_Obj, g_AreaLights, g_Hair, ray, in_RayHit);
                    if(in_RayHit.isHair) {
                        I = g_Hair.hairs[in_RayHit.mesh_idx].hair_material.color;
                    }
                    else if(in_RayHit.mesh_idx == -1)
                        I = g_AreaLights[in_RayHit.primitive_idx].intensity * g_AreaLights[in_RayHit.primitive_idx].color;
                    else
                        I = g_Obj.meshes[in_RayHit.mesh_idx].material.color;
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

Eigen::Vector3d Renderer::computePathTrace(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Hair, in_Ray, in_RayHit);

    if(in_RayHit.primitive_idx < 0)
        return Eigen::Vector3d::Zero();

    if (in_RayHit.mesh_idx == -1) // the ray has hit an area light
    {
        if (!in_RayHit.isFront)
            return Eigen::Vector3d::Zero();

        return in_AreaLights[in_RayHit.primitive_idx].intensity * in_AreaLights[in_RayHit.primitive_idx].color;
    }

    const Eigen::Vector3d x = in_Ray.o + in_RayHit.t * in_Ray.d;
    Eigen::Vector3d I = Eigen::Vector3d::Zero();

    if (in_RayHit.isHair){
        const Eigen::Vector3d n = in_RayHit.n;
        const double h = in_RayHit.h;
        const double pdf = marschnerSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);

        const Eigen::Vector3d u = in_Hair.hairs[in_RayHit.mesh_idx].u[in_RayHit.primitive_idx];
        Eigen::Vector3d d_parallel_u = in_Ray.d.dot(u) * u;
        Eigen::Vector3d d_vertical_u = x - d_parallel_u;

        const double cosine_phi_i = n.dot(- d_vertical_u.normalized());
        const double phi_i = acos(cosine_phi_i);
        const double sine_phi_i = sin(phi_i);

        const Eigen::Vector3d w = in_Hair.hairs[in_RayHit.mesh_idx].w[in_RayHit.primitive_idx];
        Eigen::Vector3d d_parallel_w = in_Ray.d.dot(w) * w;
        Eigen::Vector3d d_vertical_w = x - d_parallel_w;

        const double cosine_theta_i = in_Hair.hairs[in_RayHit.mesh_idx].v[in_RayHit.primitive_idx].dot(- d_vertical_w.normalized());
        const double theta_i = acos(cosine_theta_i);
        const double sine_theta_i = sin(theta_i);

        d_parallel_w = new_ray.d.dot(w) * w;
        d_vertical_w = x - d_parallel_w;

        const double theta_r = acos(in_Hair.hairs[in_RayHit.mesh_idx].v[in_RayHit.primitive_idx].dot(- d_vertical_w.normalized()));

        const double theta_d = (theta_r - theta_i) / 2.0;
        const double theta_h = (theta_r + theta_i) / 2.0;

        const double cos_d = std::cos(theta_d);
        const double sin_d = std::sin(theta_d);
        const double eta = in_Hair.hairs[in_RayHit.mesh_idx].hair_material.eta;
        const double eta1 = std::sqrt(eta * eta - sin_d * sin_d) / cos_d;
        const double eta2 = eta * eta * cos_d / std::sqrt(eta * eta - sin_d * sin_d);
        const double theta_t = eta1 / eta2 * sin(theta_i);
        const double phi_t = eta1 / eta2 * sin(phi_i);

        const double A0 = FrDielectric(phi_i, eta1, eta2);
        const double fr = FrDielectric(phi_t, 1/eta1, 1/eta2);
        const double Transmittance = getTransmittance(in_Hair.hairs[in_RayHit.mesh_idx].hair_material.absorb, h, theta_t, phi_t);
        const double A1 = (1.0 - A0) * (1.0 - A0) * Transmittance;
        const double A2 = (1.0 - A0) * (1.0 - A0) * fr * Transmittance * Transmittance;

        const double c = asin(1 / eta1);
        const double cosine_r = std::cos(theta_r);
        const double sine_r = std::sin(theta_r);
        const double cosine_d = std::cos(theta_d);

        const double r = randomMT() * (A0 + A1 + A2);
        if(r < A0){
            //R mode
            const double micro_variation = -2.0f / sqrt(1.0f - h * h);
            const double Np = A0 / (2.0f * micro_variation);
            const double Mp = in_Hair.hairs[in_RayHit.mesh_idx].hair_material.getMp(0, theta_h);

            I += computePathTrace(new_ray, in_Object, in_AreaLights, in_Hair).cwiseProduct(in_Hair.hairs[in_RayHit.mesh_idx].hair_material.color) * Np * Mp * sine_r * cosine_r / (pdf * A0 * cosine_d * cosine_d);
        }
        else if(r < A0 + A1){
            const double micro_variation = ((6.0 / __PI__) - 2.0) - (24.0 * c * phi_i * phi_i / __PI__ / __PI__ / __PI__) / sqrt(1.0f - h * h);
            const double Np = A0 / (2.0f * micro_variation);
            const double Mp = in_Hair.hairs[in_RayHit.mesh_idx].hair_material.getMp(0, theta_h);

            I += computePathTrace(new_ray, in_Object, in_AreaLights, in_Hair).cwiseProduct(in_Hair.hairs[in_RayHit.mesh_idx].hair_material.color) * Np * Mp * sine_r * cosine_r / (pdf * A0 * cosine_d * cosine_d);
        }
        else{
            const double micro_variation = ((12.0 / __PI__) - 2.0) - (48.0 * c * phi_i * phi_i / __PI__ / __PI__ / __PI__) / sqrt(1.0f - h * h);
            const double Np = A0 / (2.0f * micro_variation);
            const double Mp = in_Hair.hairs[in_RayHit.mesh_idx].hair_material.getMp(0, theta_h);

            I += computePathTrace(new_ray, in_Object, in_AreaLights, in_Hair).cwiseProduct(in_Hair.hairs[in_RayHit.mesh_idx].hair_material.color) * Np * Mp * sine_r * cosine_r / (pdf * A0 * cosine_d * cosine_d);
        }
    }
    else {
        const Eigen::Vector3d n = computeRayHitNormal(in_Object, in_RayHit);

        const double kd = in_Object.meshes[in_RayHit.mesh_idx].material.kd;
        const double ks = in_Object.meshes[in_RayHit.mesh_idx].material.ks;
        const double r = randomMT();

        if(r < kd){
            diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
            I += computePathTrace(new_ray, in_Object, in_AreaLights, in_Hair).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
        }
        else if(r < kd + ks){
            const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
            const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
            if(pdf < 0.0f)
                return I;
            const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
            I += computePathTrace(new_ray, in_Object, in_AreaLights, in_Hair).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
        }
    }

    return I;
}

Eigen::Vector3d Renderer::computeNEE(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, bool first) {
    if(in_Ray.depth >= MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Hair, in_Ray, in_RayHit);

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
        I += computeDirectLighting(in_Ray, in_RayHit, in_Object, in_AreaLights, in_Hair, 1);
    else if(r < kd + ks)
        I += computeDirectLighting(in_Ray, in_RayHit, in_Object, in_AreaLights, in_Hair, 2);

    r = randomMT();

    if(r < kd){
        diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        I += computeNEE(new_ray, in_Object, in_AreaLights, in_Hair, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(pdf < 0.0f)
            return I;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeNEE(new_ray, in_Object, in_AreaLights, in_Hair, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeMIS(const Ray &in_Ray, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, bool first) {
    if(in_Ray.depth > MAX_RAY_DEPTH)
        return Eigen::Vector3d::Zero();

    Ray new_ray; RayHit in_RayHit;
    rayTracing(in_Object, in_AreaLights, in_Hair, in_Ray, in_RayHit);

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
    double r = randomMT() * (kd + ks);

    if(r < kd)
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_Object, in_AreaLights, in_Hair, 1);
    else if(r < kd + ks)
        I += computeDirectLighting_MIS(in_Ray, in_RayHit, in_Object, in_AreaLights, in_Hair, 2);

    r = randomMT();

    if(r < kd){
        const double pdf = diffuseSample(x, n, new_ray, in_RayHit, in_Object, in_Ray.depth);
        new_ray.pdf = pdf;
        I += computeMIS(new_ray, in_Object, in_AreaLights, in_Hair, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKd()) / kd;
    }
    else if(r < kd + ks){
        const double m = in_Object.meshes[in_RayHit.mesh_idx].material.m;
        const double pdf = blinnPhongSample(x, n, in_Ray.d, new_ray, in_RayHit, in_Object, m, in_Ray.depth);
        if(pdf < 0.0f)
            return I;
        new_ray.pdf = pdf;
        const double cosine = std::max<double>(0.0f, n.dot(new_ray.d));
        I += computeMIS(new_ray, in_Object, in_AreaLights, in_Hair, false).cwiseProduct(in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * cosine * (m + 2.0f)) / (ks * (m + 1.0f));
    }

    return I;
}

Eigen::Vector3d Renderer::computeDirectLighting(const Ray &in_Ray, const RayHit &in_RayHit, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const int mode) {
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

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, in_Hair, ray, rh);
        if (rh.mesh_idx == -1 && rh.primitive_idx == i) {
            const double cos_x = std::max<double>(0.0, x_L.dot(n));
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
Eigen::Vector3d Renderer::computeDirectLighting_MIS(const Ray &in_Ray, const RayHit &in_RayHit, const Object &in_Object, const std::vector<AreaLight> &in_AreaLights, const Hair &in_Hair, const int mode) {
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

        // shadow test
        Ray ray;
        ray.o = x;
        ray.d = x_L;
        ray.prev_mesh_idx = in_RayHit.mesh_idx;
        ray.prev_primitive_idx = in_RayHit.primitive_idx;
        RayHit rh;
        rayTracing(in_Object, in_AreaLights, in_Hair, ray, rh);
        if (rh.mesh_idx == -1 && rh.primitive_idx == i) {
            const double cos_x = std::max<double>(0.0, x_L.dot(n));
            const double G = (cos_x * cos_light) / (dist * dist);

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
                    const Eigen::Vector3d BSDF = in_Object.meshes[in_RayHit.mesh_idx].material.getKs() * (m + 2.0f) * pow(cosine, m) / (2.0f * __PI__);

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

    return (m + 1) * pow(cosine, m) / 2.0 * __PI__;
}

double Renderer::diffuseSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth) {
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
    out_ray.depth = depth + 1;

    return cos(theta) / __PI__;
}
double Renderer::blinnPhongSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double m, const int depth) {
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

    const Eigen::Vector3d o_parallel = (-in_direction).dot(halfVector) * halfVector;
    const Eigen::Vector3d o_vertical = (-in_direction) - o_parallel;

    out_ray.o = in_x;
    out_ray.d = (o_parallel - o_vertical).normalized();
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;
    out_ray.depth = depth + 1;

    if(out_ray.d.dot(in_n) < 0.0f)
        return -1.0f;

    return (m + 1) * pow(cos(theta), m) / (2.0f * __PI__);
}

double Renderer::refractionSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, const Eigen::Vector3d &in_direction, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const double eta, const int depth) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double e_dot_n = in_n.dot(-in_direction);
    const double inside_sqrt = 1.0f - eta * eta * (1.0f - e_dot_n * e_dot_n);

    Eigen::Vector3d x_L;
    if(inside_sqrt < 0.0){
        //全反射
        x_L = 2.0 * in_n * e_dot_n + in_direction;
    }
    else{
        //透過
        x_L = - in_n * inside_sqrt - eta * (-in_direction - e_dot_n * in_n);
    }

    x_L.normalize();
    out_ray.o = in_x;
    out_ray.d = x_L;
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;
    out_ray.depth = depth + 1;

    return eta * eta / e_dot_n;
}

double Renderer::marschnerSample(const Eigen::Vector3d &in_x, Ray &out_ray, const RayHit &rayHit, const Hair &in_Hair,
                                const double theta, const double phi, const double c, const int depth, int p) {
    p = p > 2 ? p : 2;
    const double _phi = phi + (6.0f * p * c / __PI__ - 2.0f) * phi - (8.0f * p * c * pow(phi, 3.0) / pow(__PI__, 3.0)) + (p * __PI__);

    const double _dx = sin(theta) * cos(_phi);
    const double _dy = cos(theta);
    const double _dz = sin(theta) * sin(_phi);

    Eigen::Vector3d x_L = Eigen::Vector3d{_dx, _dy, _dz};
    x_L.normalize();
    out_ray.o = in_x;
    out_ray.d = x_L;
    out_ray.prev_mesh_idx = rayHit.mesh_idx;
    out_ray.prev_primitive_idx = rayHit.primitive_idx;
    out_ray.depth = depth + 1;

    return 1.0f;
}

double Renderer::marschnerSample(const Eigen::Vector3d &in_x, const Eigen::Vector3d &in_n, Ray &out_ray, const RayHit &rayHit, const Object &in_Object, const int depth) {
    Eigen::Vector3d bn =
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].x()] -
            in_Object.meshes[rayHit.mesh_idx].vertices[in_Object.meshes[rayHit.mesh_idx].triangles[rayHit.primitive_idx].z()];

    bn.normalize();

    const Eigen::Vector3d cn = bn.cross(in_n);

    const double theta = acos(1.0 - 2.0 *randomMT());
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
    out_ray.depth = depth + 1;

    return 1.0 / 4.0 / __PI__;
}

double Renderer::FrDielectric(double gamma, double etaI, double etaT) const {
    const double sineI = std::sin(gamma);
    const double cosineI = sqrt(1.0 - sineI * sineI);
    const double sineT = etaI / etaT * sineI;
    const double cosineT = sqrt(1.0 - sineT * sineT);

    if(sineT >= 1.0f)
        return 1.0f;

    const double Fp = (etaI * cosineI - etaT * cosineT) / (etaI * cosineI + etaT * cosineT);
    const double Fs = (etaT * cosineI - etaI * cosineT) / (etaT * cosineI + etaI * cosineT);

    return (Fp * Fp + Fs * Fs) * 0.5f;
}

double Renderer::getTransmittance(double absorption, double h, double theta, double phi) const {
    const double _absorption = absorption / std::cos(theta);
    return exp(-2.0f * _absorption * (1 + std::cos(2.0 * phi)));
}
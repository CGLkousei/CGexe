
#define _CRT_SECURE_NO_WARNINGS

#include "TriMesh.h"
#include "Jpeg.h"
#include <iostream>
#include <string.h>
#include <map>

struct InternalVertices {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertices;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vertex_normals;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> tex_coords;
};

struct InternalTriangles {
    std::vector<Eigen::Matrix3i, Eigen::aligned_allocator<Eigen::Matrix3i>> triangles;
    // v0, v1, v2
    // vt0, vt1, vt2
    // vn0, vn1, vn2

    std::string material_name;
    int material_id;
};

struct InternalMaterial {
    std::string material_name;
    std::string texture_name;
    Eigen::Vector3d color;
    float kd;
    float ks;
    float kt;

    double eta;
    double m;

    bool valid;
};

struct VertexIDs {
    int vid;
    int vtid;
    int vnid;
};

bool operator<(const VertexIDs &lhs, const VertexIDs &rhs) {
    return (lhs.vid < rhs.vid) || (lhs.vid == rhs.vid && lhs.vtid < rhs.vtid) ||
           (lhs.vid == rhs.vid && lhs.vtid == rhs.vtid && lhs.vnid < rhs.vnid);
}

void resetInternalVertices(InternalVertices &io_vertices) {
    io_vertices.vertices.clear();
    io_vertices.vertex_normals.clear();
    io_vertices.tex_coords.clear();
}

void resetInternalTriangles(InternalTriangles &io_triangles) {
    io_triangles.triangles.clear();

    io_triangles.material_name = "";
}

void resetMaterial(Material &io_Material) {
    io_Material.texture = 0;
    io_Material.color = Eigen::Vector3d::Zero();
    io_Material.kd = 0.0;
    io_Material.ks = 0.0;
    io_Material.kt = 0.0;
}

void resetMesh(TriMesh &io_Mesh) {
    io_Mesh.vertices.clear();
    io_Mesh.vertex_normals.clear();
    io_Mesh.vertex_colors.clear();
    io_Mesh.tex_coords.clear();
    io_Mesh.triangles.clear();

    resetMaterial(io_Mesh.material);
}

void printMesh(const TriMesh &in_Mesh) {
    std::cout << "--- Mesh ---" << std::endl;
    std::cout << "  vertices" << std::endl;
    for (int i = 0; i < in_Mesh.vertices.size(); i++) {
        std::cout << "v(" << i << "): " << in_Mesh.vertices[i].transpose() << std::endl;
    }

    for (int i = 0; i < in_Mesh.triangles.size(); i++) {
        std::cout << "t(" << i << "): " << in_Mesh.triangles[i].transpose() << std::endl;
    }
}

// extend the base mesh by the second mesh
void extendMesh(TriMesh &io_BaseMesh, const TriMesh &in_SecondMesh) {
    const size_t vid_offset = io_BaseMesh.vertices.size();

    const size_t num_v_to_extend = in_SecondMesh.vertices.size();
    for (size_t i = 0; i < num_v_to_extend; i++) {
        io_BaseMesh.vertices.push_back(in_SecondMesh.vertices[i]);
        io_BaseMesh.vertex_normals.push_back(in_SecondMesh.vertex_normals[i]);
        io_BaseMesh.vertex_colors.push_back(in_SecondMesh.vertex_colors[i]);
        io_BaseMesh.tex_coords.push_back(in_SecondMesh.tex_coords[i]);
    }

    const size_t num_t_to_extend = in_SecondMesh.triangles.size();
    for (size_t i = 0; i < num_t_to_extend; i++) {
        const Eigen::Vector3i triangle_vids = in_SecondMesh.triangles[i];
        io_BaseMesh.triangles.push_back(triangle_vids.array() + vid_offset);
    }

    io_BaseMesh.material.kd = in_SecondMesh.material.kd;
    io_BaseMesh.material.texture = in_SecondMesh.material.texture;
}

GLuint prepareTextureFromJpegFile(const char *in_FileName) {
    JpegData data;
    readJpegData(data, in_FileName);

    if (data.ch != 3) {
        std::cout << "Error: Currently only images with three color channels are supported." << std::endl;
        std::cout << "The loaded image has " << data.ch << " channels." << std::endl;

        finalizeJpegData(data);
        exit(-1);
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, data.width, data.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    finalizeJpegData(data);

    return texture;
}

bool loadMtl(const std::string &in_filename, std::vector<InternalMaterial> &io_internal_materials) {
    FILE *f;

    const int BUF_MAX = 4096;

    // text buffer for storing a single line of the input file
    char line[BUF_MAX];

    // text buffer for storing material name
    char material_name[BUF_MAX];

    // text buffer for storing texture name
    char texture_name[BUF_MAX];

    f = fopen(in_filename.c_str(), "r");
    if (!f) {
        std::cout << "Could not open mtl file: " << in_filename << std::endl;
        return false;
    }

    InternalMaterial material;
    material.valid = false;

    while (fgets(line, BUF_MAX, f) != NULL) {
        if (strncmp(line, "newmtl", 6) == 0) {
            if (material.valid) {
                io_internal_materials.push_back(material);
                material.valid = false;
            }

            sscanf(line, "newmtl %s", material_name);
            if (strlen(material_name) == 0)
                continue;

            material.material_name = material_name;
            material.texture_name = "";
            material.color << 1.0, 1.0, 1.0;
            material.kd = 0.8;
            material.ks = 0.0;
            material.kt = 0.0;
            material.eta = 1.0;
            material.m = 100.0;
            material.valid = true;
        }

        // read in albedo
        if (line[0] == 'K') {
            // diffuse reflectance (Kd)
            if (line[1] == 'd') {
                float r, g, b;
                sscanf(line, "Kd %f %f %f", &r, &g, &b);

                material.color << r, g, b;
            }
            else if (line[1] == 's') {
                float r, g, b;
                sscanf(line, "Ks %f %f %f", &r, &g, &b);

                material.ks = r;
            }
        }

        if (strncmp(line, "Ni", 2) == 0) {
            float eta;
            sscanf(line, "Ni %f", &eta);

            material.eta = eta;
        }

        if (line[0] == 'd') {
            float Kt;
            sscanf(line, "d %f", &Kt);

            material.kt = Kt;
        }

        if (line[0] == 'm'){
            float m;
            sscanf(line, "m %f", &m);

            material.m = m;
        }

        if (strncmp(line, "Pr", 2) == 0) {
            float Pr;
            sscanf(line, "Pr %f", &Pr);

            material.kd = Pr;
        }

        if (strncmp(line, "map_Kd", 6) == 0) {
            sscanf(line, "map_Kd %s", texture_name);

            if (strlen(texture_name) == 0)
                continue;

            material.texture_name = texture_name;
        }

    }

    if (material.valid) {
        io_internal_materials.push_back(material);
        material.valid = false;
    }

    fclose(f);
    return true;
}

bool _loadObj(const std::string &in_filename, InternalVertices &out_internal_vertices,
              std::vector<InternalTriangles> &out_internal_triangles,
              std::vector<InternalMaterial> &out_internal_materials) {
    FILE *f = NULL;
    const int BUF_MAX = 4096;

    // text buffer for storing a single line of the input file
    char line[BUF_MAX];

    // text buffer for storing material name
    char material_name[BUF_MAX];

    f = fopen(in_filename.c_str(), "r");
    if (!f) {
        std::cout << "Could not open obj file: " << in_filename << std::endl;
        return false;
    }

    resetInternalVertices(out_internal_vertices);

    out_internal_triangles.clear();
    out_internal_materials.clear();

    InternalTriangles triangles;

    while (fgets(line, BUF_MAX, f) != NULL) {
        // load the material file
        if (strncmp(line, "mtllib", 6) == 0) {
            sscanf(line, "mtllib %s", material_name);

            if (strlen(material_name) > 0) {
                if (!loadMtl(material_name, out_internal_materials)) {
                    std::cout << "Could not read mtl file: " << material_name << std::endl;
                    fclose(f);
                    return false;
                }
            }
        }

        // read in material name
        if (strncmp(line, "usemtl", 6) == 0) {
            sscanf(line, "usemtl %s", material_name);

            // "usemtl" command starts a new polygon group
            // if there is an existing non-empty polygon group, save it to the vector, and
            // generate a new empty polygon group.
            if (triangles.triangles.size() > 0) {
                out_internal_triangles.push_back(triangles);
                resetInternalTriangles(triangles);
            }

            // set the specified material name
            triangles.material_name = material_name;
        }

        // read in vertex info
        if (line[0] == 'v') {
            // if the line starts with 'vn', read in the vertex material
            if (line[1] == 'n') {
                float x, y, z;
                sscanf(line, "vn %f %f %f", &x, &y, &z);
                out_internal_vertices.vertex_normals.emplace_back(x, y, z);
            }
                // if the line starts with 'vt', read in the texture coordinates
            else if (line[1] == 't') {
                float u, v;
                sscanf(line, "vt %f %f", &u, &v);
                out_internal_vertices.tex_coords.emplace_back(u, 1.0 - v);
            }
                // if the line starts with 'v', read in the vertex position
            else {
                float x, y, z;
                sscanf(line, "v %f %f %f", &x, &y, &z);
                out_internal_vertices.vertices.emplace_back(x, y, z);
            }
        }

        // read in polygon info
        if (line[0] == 'f') {
            char *tp = strtok(&(line[2]), " ");
            std::vector<Eigen::Vector3i> vertices;
            while (tp != NULL) {
                int vid = -1, vtid = -1, vnid = -1;
                sscanf(tp, "%d/%d/%d", &vid, &vtid, &vnid);

                // indices in obj files start from 1
                vertices.emplace_back(vid - 1, vtid - 1, vnid - 1);

                tp = strtok(NULL, " ");
            }

            // A face in an obj file is a general polygon with the number of vertices larger than 2.
            // If the number of vertices of a face is larger than 3, subdivide the polygon into a set of triangles.
            for (int i = 0; i < vertices.size() - 2; i++) {
                Eigen::Matrix3i vertex;
                vertex.col(0) = vertices[0];
                vertex.col(1) = vertices[i + 1];
                vertex.col(2) = vertices[i + 2];

                triangles.triangles.push_back(vertex);
            }
        }
    }

    if (triangles.triangles.size() > 0) {
        out_internal_triangles.push_back(triangles);
    }

    fclose(f);
    return true;
}

bool loadObj(const std::string &in_filename, Object &out_object) {
    InternalVertices internal_vertices;
    std::vector<InternalTriangles> internal_triangles;
    std::vector<InternalMaterial> internal_materials;

    _loadObj(in_filename, internal_vertices, internal_triangles, internal_materials);

    // relate the polygon group name with the material name
    for (int i = 0; i < internal_triangles.size(); i++) {
        internal_triangles[i].material_id = -1;
        for (int k = 0; k < internal_materials.size(); k++) {
            if (strcmp(internal_triangles[i].material_name.c_str(), internal_materials[k].material_name.c_str()) == 0) {
                internal_triangles[i].material_id = k;
                continue;
            }
        }
    }

    // Prepare the mesh
    // An obj mesh may have different numbers of vertex positions, texture coordinates, and vertex normals
    // In our mesh data structure, two vertices are the same if and only if the positions, texture coordinates and vertex normals are all the same; if any of these are different, we treat them as two different vertices.

    out_object.meshes.clear();

    for (int i = 0; i < internal_triangles.size(); i++) {
        // use a map to distinguish vertices
        std::map<VertexIDs, int> vertex_idx_map;
        int nVertices = 0;
        for (int j = 0; j < internal_triangles[i].triangles.size(); j++) {
            for (int k = 0; k < 3; k++) {
                Eigen::Vector3i __v = internal_triangles[i].triangles[j].col(k);
                VertexIDs _v;
                _v.vid = __v(0);
                _v.vtid = __v(1);
                _v.vnid = __v(2);

                // Find if an vertex with (the combination of) the vid, vtid, and vnid already exists.
                // If no, that is a new vertex so insert it with the its index (= nVertices) to the map.
                if (vertex_idx_map.find(_v) == vertex_idx_map.end()) {
                    vertex_idx_map.insert(std::pair<VertexIDs, int>(_v, nVertices));
                    nVertices++;
                }
            }
        }

        TriMesh mesh;
        mesh.vertices.resize(nVertices);
        mesh.vertex_normals.resize(nVertices);
        mesh.tex_coords.resize(nVertices);
        mesh.vertex_colors.resize(nVertices);

        // Each vertex stored in the map is a valid vertex, and we add it to the mesh
        for (auto q = vertex_idx_map.begin(); q != vertex_idx_map.end(); q++) {
            VertexIDs _v = q->first;
            int vid = q->second;

            mesh.vertices[vid] = internal_vertices.vertices[_v.vid];
            mesh.tex_coords[vid] = internal_vertices.tex_coords[_v.vtid];
            mesh.vertex_normals[vid] = internal_vertices.vertex_normals[_v.vnid];
        }

        for (int j = 0; j < internal_triangles[i].triangles.size(); j++) {
            Eigen::Vector3i vids{0, 0, 0};
            for (int k = 0; k < 3; k++) {
                Eigen::Vector3i __v = internal_triangles[i].triangles[j].col(k);
                VertexIDs _v;
                _v.vid = __v(0);
                _v.vtid = __v(1);
                _v.vnid = __v(2);

                // Find the vertex index
                auto q = vertex_idx_map.find(_v);
                if (q == vertex_idx_map.end()) {
                    std::cout << "This should not happen. This is a bug." << std::endl;
                }

                vids(k) = q->second;
            }

            mesh.triangles.push_back(vids);
        }

        // If a positive material index is attached to the mesh, there is a valid material,
        // so we read the information.
        // If valid, also read the texture.

        if (internal_triangles[i].material_id >= 0) {
            mesh.material.color = internal_materials[internal_triangles[i].material_id].color;
            mesh.material.kd = internal_materials[internal_triangles[i].material_id].kd;
            mesh.material.ks = internal_materials[internal_triangles[i].material_id].ks;
            mesh.material.kt = internal_materials[internal_triangles[i].material_id].kt;
            mesh.material.eta = internal_materials[internal_triangles[i].material_id].eta;
            mesh.material.m = internal_materials[internal_triangles[i].material_id].m;

            if (internal_materials[internal_triangles[i].material_id].texture_name.length() > 0) {
                mesh.material.texture = prepareTextureFromJpegFile(
                        internal_materials[internal_triangles[i].material_id].texture_name.c_str());
            }
            else {
                mesh.material.texture = 0;
            }
        }
        else {
            mesh.material.color << 0.8, 0.8, 0.8;
            mesh.material.kd = 1.0;
            mesh.material.ks = 0.0;
            mesh.material.kt = 0.0;
            mesh.material.eta = 1.0;
            mesh.material.m = 100.0;
            mesh.material.texture = 0;
        }

        out_object.meshes.push_back(mesh);
    }

    return true;
}

// resize the object to fit the specified region.
void resizeObj(Object &io_object, const Eigen::Vector3d &in_min, const Eigen::Vector3d &in_max) {
    Eigen::Vector3d obj_min{1.0e33, 1.0e33, 1.0e33};
    Eigen::Vector3d obj_max{-1.0e33, -1.0e33, -1.0e33};

    for (int i = 0; i < io_object.meshes.size(); i++) {
        for (int j = 0; j < io_object.meshes[i].vertices.size(); j++) {
            obj_min = obj_min.cwiseMin(io_object.meshes[i].vertices[j]);
            obj_max = obj_max.cwiseMax(io_object.meshes[i].vertices[j]);
        }
    }

    Eigen::Vector3d obj_center = (obj_max + obj_min) * 0.5;
    Eigen::Vector3d obj_half_width = (obj_max - obj_min) * 0.5;

    Eigen::Vector3d region_center = (in_max + in_min) * 0.5;
    Eigen::Vector3d region_half_width = (in_max - in_min) * 0.5;

    const double scale = (region_half_width.cwiseQuotient(obj_half_width)).minCoeff();

    for (int i = 0; i < io_object.meshes.size(); i++) {
        for (int j = 0; j < io_object.meshes[i].vertices.size(); j++) {
            io_object.meshes[i].vertices[j] = (io_object.meshes[i].vertices[j] - obj_center) * scale + region_center;
        }
    }
}

Eigen::Vector3d Material::getKd() const {
    return kd * color;
}
Eigen::Vector3d Material::getKs() const{
    return ks * color;
}
Eigen::Vector3d Material::getKt() const{
    return kt * color;
}

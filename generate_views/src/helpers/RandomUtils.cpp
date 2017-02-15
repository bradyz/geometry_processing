#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>                              // scale
#include <glm/gtx/string_cast.hpp>                                // to_string


using namespace std;
using namespace glm;

namespace glm {
  std::ostream& operator<<(std::ostream& os, const glm::vec2& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::vec3& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::vec4& v) {
    os << glm::to_string(v);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::mat4& v) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j)
        os << std::setprecision(3) << v[j][i] << "\t";
      os << std::endl;
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const glm::mat3& v) {
    os << glm::to_string(v);
    return os;
  }
}  // namespace glm

vector<vec4> getVertexNormals (const vector<vec4>& vertices,
                               const vector<uvec3>& faces) {
  vector<vec4> normals(vertices.size());
  for (const uvec3& face: faces) {
    int v1 = face[0];
    int v2 = face[1];
    int v3 = face[2];
    vec3 a = vec3(vertices[v1]);
    vec3 b = vec3(vertices[v2]);
    vec3 c = vec3(vertices[v3]);
    vec3 u = normalize(b - a);
    vec3 v = normalize(c - a);
    vec4 n = vec4(normalize(cross(u, v)), 0.0f);
    normals[v1] += n;
    normals[v2] += n;
    normals[v3] += n;
  }
  for (int i = 0; i < normals.size(); ++i) {
    normals[i] = normalize(normals[i]);
    normals[i][3] = 0.0f;
  }
  return normals;
}

void LoadOBJ(const string& file,
             vector<vec4>& vertices,
             vector<uvec3>& faces,
             vector<vec4>& normals) {
  vertices.clear();
  faces.clear();
  normals.clear();
  fstream ifs(file);
  string buffer;
  while (getline(ifs, buffer)) {
    stringstream ss(buffer);
    char type;
    ss >> type;
    if (type == 'f') {
      int i, j, k;
      ss >> i >> j >> k;
      faces.push_back(uvec3(i-1, j-1, k-1));
    }
    else if (type == 'v') {
      float x, y, z;
      ss >> x >> y >> z;
      vertices.push_back(vec4(x, y, z, 1.0));
    }
  }
  normals = getVertexNormals(vertices, faces);
  cout << "Loaded " << file << " successfully." << endl;
  cout << "\t" << vertices.size() << " vertices." << endl;
  cout << "\t" << faces.size() << " faces." << endl;
}

void LoadOBJWithNormals(const string& file,
                        vector<vec4>& vertices,
                        vector<uvec3>& faces,
                        vector<vec4>& normals) {
  vertices.clear();
  faces.clear();
  normals.clear();
  ifstream ifs(file);
  string buffer;
  while (getline(ifs, buffer)) {
    string type;
    stringstream ss(buffer);
    ss >> type;
    if (type == "v") {
      double x, y, z;
      ss >> x >> y >> z;
      vertices.push_back(vec4(x, y, z, 1.0));
    }
    else if (type == "vn"){
      double x, y, z;
      ss >> x >> y >> z;
      normals.push_back(vec4(x, y, z, 1.0));
    }
    else if (type == "f") {
      // of the form 22283//22283
      uvec3 idx;
      for (int i = 0; i < 3; ++i) {
        string tmp;
        ss >> tmp;
        for (int j = 0; j < tmp.size() && tmp[j] != '/'; ++j) {
          idx[i] = idx[i] * 10 + int(tmp[j]) - '0';
        }
        idx[i] -= 1;
      }
      faces.push_back(idx);
    }
  }
  cout << "Loaded " << file << " successfully." << endl;
  cout << "\t" << vertices.size() << " vertices." << endl;
  cout << "\t" << faces.size() << " faces." << endl;
}

string loadShader (const string& filename) {
  cout << "Loading shader: " << filename;
  ifstream file(filename);
  cout << (file.fail() ? " failed." : " succeeded.") << endl;
  stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void fixSphereVertices (vector<vec4>& sphere_vertices) {
  mat4 T = translate(vec3(0.0, -1.0, 0.0));
  mat4 S = scale(vec3(10.0, 10.0, 10.0));
  for (vec4& vertex: sphere_vertices)
    vertex = T * S * vertex;
}

struct vec4_sort {
  bool operator() (const vec4& a, const vec4& b) const {
    for (int i = 0; i < 4; ++i) {
      if (a[i] == b[i])
        continue;
      return a[i] < b[i];
    }
    return false;
  }
};

void fixDuplicateVertices (vector<vec4>& vertices, vector<uvec3>& faces) {
  vector<vec4> v;
  vector<uvec3> f;

  map<vec4, int, vec4_sort> dupeVertCheck;
  for (const vec4& vert : vertices) {
    if (dupeVertCheck.find(vert) == dupeVertCheck.end()) {
      dupeVertCheck[vert] = dupeVertCheck.size();
      v.push_back(vert);
    }
  }

  for (const uvec3& face : faces) {
    uvec3 new_face;
    for (int i = 0; i < 3; ++i)
      new_face[i] = dupeVertCheck[vertices[face[i]]];
    f.push_back(new_face);
  }

  cout << "Removed: " << (vertices.size() - v.size()) << " vertices." << endl;

  vertices = v;
  faces = f;
}

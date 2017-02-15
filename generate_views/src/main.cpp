#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <libgen.h>

#include <glm/glm.hpp>

#include <png++/png.hpp>

#include <Eigen/Core>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/per_vertex_normals.h>

#include "helpers/RandomUtils.h"

#include "render/OpenGLStuff.h"
#include "render/Phong.h"

using namespace std;
using namespace glm;
using namespace Eigen;

string FILES_TO_PROCESS = "first argument";
string OUT_DIR = "second argument";

PhongProgram phongP(&view_matrix, &projection_matrix);

vector<vec4> object_vertices;
vector<uvec3> object_faces;
vector<vec4> object_normals;

void translate(vector<vec4> &points, const vec4 &translation) {
  for (vec4 &point : points)
    point += translation;
}

void scale(vector<vec4> &points, float factor) {
  for (vec4 &point : points) {
    for (int i = 0; i < 3; i++)
      point[i] *= factor;
  }
}

vector<float> min_bounds(const vector<vec4> &points) {
  vector<float> result(3);
  for (int i = 0; i < 3; i++)
    result[i] = points[0][i];
  for (const vec4 &point : points) {
    for (int i = 0; i < 3; i++)
      result[i] = std::min(result[i], point[i]);
  }
  return result;
}

vector<float> max_bounds(const vector<vec4> &points) {
  vector<float> result(3);
  for (int i = 0; i < 3; i++)
    result[i] = points[0][i];
  for (const vec4 &point : points) {
    for (int i = 0; i < 3; i++)
      result[i] = std::max(result[i], point[i]);
  }
  return result;
}

// Resizes the bounding box of points into bounding box with max length 1.0.
void expand(vector<vec4> &points) {
  vector<float> min_vals = min_bounds(points);
  vector<float> max_vals = max_bounds(points);
  int axis = 0;
  for (int i = 0; i < 3; i++) {
    if (max_vals[i] - min_vals[i] > max_vals[axis] - min_vals[axis])
      axis = i;
  }
  float factor = 1.0 / (max_vals[axis] - min_vals[axis]);
  scale(points, factor);
}

void preprocess(vector<vec4> &points) {
  // Modelnet10 faces from the top. Rotate by 90 degrees.
  for (vec4 &point : points) {
    float y = point[1];
    float z = point[2];
    point[1] = z;
    point[2] = y;
  }

  // Translate bounding box corner to origin.
  vector<float> min_vals = min_bounds(points);
  vec4 translation(-min_vals[0], -min_vals[1], -min_vals[2], 0.0f);
  translate(points, translation);

  // Make the bounding box touch the unit cube.
  expand(points);

  // Translate so bounding box center is at the origin.
  translate(points, vec4(-0.5f, -0.5f, -0.5f, 0.0f));
}

void writePNG(vector<unsigned char> &in, const char *fn, int w, int h) {
 png::image< png::rgb_pixel > image(w, h);
 int offset = 0;
 for (size_t y = 0; y < image.get_height(); ++y) {
     for (size_t x = 0; x < image.get_width(); ++x) {
         image[h-y-1][x] = png::rgb_pixel(in[offset+0],
                                          in[offset+1],
                                          in[offset+2]);
         offset += 3;
     }
 }
 image.write(fn);
}

void generate(string &filename, float r, int theta_samples, int phi_samples) {
  cout << filename << endl;

  float d_phi   = (M_PI) / (float) (phi_samples + 1);
  float d_theta = (2 * M_PI) / (float) (theta_samples);

  for (int i = 1; i <= phi_samples; i++) {
    for (int j = 1; j <= theta_samples; j++) {
      float phi   = i * d_phi;
      float theta = j * d_theta;

      vec3 eye_pos = vec3(r * sin(phi) * cos(theta),
                          r * cos(phi),
                          r * sin(phi) * sin(theta));

      // Pass in the new eye position.
      if (!keepLoopingOpenGL(eye_pos))
        return;

      phongP.draw(object_vertices, object_faces, object_normals, mat4(),
                  RED, vec4(eye, 1.0f));

      endLoopOpenGL();

      // Must be done after buffers are swapped.
      vector<unsigned char> pixels(3 * window_width * window_height);
      glReadPixels(0, 0, window_width, window_height, GL_RGB,
                   GL_UNSIGNED_BYTE, &pixels[0]);

      stringstream ss;
      ss << OUT_DIR << filename << "_" << i << "_" << j << ".png";
      writePNG(pixels, ss.str().c_str(), window_width, window_height);
    }
  }
}

// Uses libigl to read into an Eigen matrix then converts back to glm.
void readMesh(string filename, bool is_obj) {
  MatrixXd vertices;
  MatrixXi faces;
  MatrixXd normals;

  if (is_obj)
    igl::readOBJ(filename, vertices, faces);
  else
    igl::readOFF(filename, vertices, faces);

  igl::per_vertex_normals(vertices, faces, normals);

  object_vertices.clear();
  object_faces.clear();
  object_normals.clear();

  for (int i = 0; i < vertices.rows(); i++) {
    vec4 tmp;
    for (int j = 0; j < 3; j++)
      tmp[j] = vertices(i, j);
    tmp[3] = 1.0f;
    object_vertices.push_back(tmp);
  }

  for (int i = 0; i < faces.rows(); i++) {
    vec3 tmp;
    for (int j = 0; j < 3; j++)
      tmp[j] = faces(i, j);
    object_faces.push_back(tmp);
  }

  for (int i = 0; i < normals.rows(); i++) {
    vec4 tmp;
    for (int j = 0; j < 3; j++)
      tmp[j] = normals(i, j);
    object_normals.push_back(tmp);
  }
}

int main (int argc, char* argv[]) {
  FILES_TO_PROCESS = argv[1];
  OUT_DIR = argv[2];

  ifstream process_file(FILES_TO_PROCESS);

  // OpenGL Setup.
  initOpenGL();
  phongP.setup();

  string buffer;
  while (getline(process_file, buffer)) {
    stringstream ss(buffer);
    string content = ss.str();

    // Set boolean to false if the mesh is an off file.
    readMesh(content, false);
    preprocess(object_vertices);

    // Generate all viewpoints.
    generate(content, 3.0f, 5, 5);
  }

  cleanupOpenGL();
}

#ifndef PHONG_H
#define PHONG_H

#include <vector>

#include <glm/glm.hpp>

#include "helpers/RandomUtils.h"
#include "render/Program.h"

struct PhongProgram: public Program {
  GLint obj_color_location;
  GLint eye_location;

  PhongProgram (glm::mat4* view_p, glm::mat4* proj_p) :
    Program(view_p, proj_p) {
  }

  virtual void setup();
  void draw (const std::vector<glm::vec4>& vertices,
             const std::vector<glm::uvec3>& faces,
             const std::vector<glm::vec4>& normals,
             const glm::mat4& model, const glm::vec4& color,
             const glm::vec4& eye);
};

#endif

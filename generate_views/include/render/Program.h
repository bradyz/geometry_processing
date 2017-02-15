#ifndef PROGRAM_H
#define PROGRAM_H

#include <glm/glm.hpp>

#include <GL/glew.h>

struct Program {
  int vaoIndex;

  GLuint programId;

  GLint projection_matrix_location;
  GLint model_matrix_location;
  GLint view_matrix_location;

  const glm::mat4& view;
  const glm::mat4& proj;

  GLint light_position_location;

  Program (glm::mat4* view_p, glm::mat4* proj_p) :
    view(*view_p), proj(*proj_p) { }

  virtual void setup() = 0;
};

#endif

#include <iostream>
#include <string>

#include <GL/glew.h>

#include <glm/glm.hpp>

#include "helpers/RandomUtils.h"
#include "render/OpenGLStuff.h"
#include "render/Phong.h"

using namespace std;

void PhongProgram::setup () {
  this->vaoIndex = kPhongVao;

  // shaders
  string vertex_shader = loadShader(PHONG_VERT);
  GLuint vertex_shader_id = setupShader(vertex_shader.c_str(), GL_VERTEX_SHADER);

  string geometry_shader = loadShader(PHONG_GEOM);
  GLuint geometry_shader_id = setupShader(geometry_shader.c_str(), GL_GEOMETRY_SHADER);

  string fragment_shader = loadShader(PHONG_FRAG);
  GLuint fragment_shader_id = setupShader(fragment_shader.c_str(), GL_FRAGMENT_SHADER);

  // program
  GLuint& program_id = this->programId;
  CHECK_GL_ERROR(program_id = glCreateProgram());
  CHECK_GL_ERROR(glAttachShader(program_id, vertex_shader_id));
  CHECK_GL_ERROR(glAttachShader(program_id, geometry_shader_id));
  CHECK_GL_ERROR(glAttachShader(program_id, fragment_shader_id));

  // attributes
  CHECK_GL_ERROR(glBindAttribLocation(program_id, 0, "vertex_position"));
  CHECK_GL_ERROR(glBindAttribLocation(program_id, 1, "vertex_normal"));
  CHECK_GL_ERROR(glBindFragDataLocation(program_id, 0, "fragment_color"));

  glLinkProgram(program_id);
  CHECK_GL_PROGRAM_ERROR(program_id);

  // uniforms
  GLint& projection_matrix_location = this->projection_matrix_location;
  CHECK_GL_ERROR(projection_matrix_location = glGetUniformLocation(program_id, "projection"));
  GLint& model_matrix_location = this->model_matrix_location;
  CHECK_GL_ERROR(model_matrix_location = glGetUniformLocation(program_id, "model"));
  GLint& view_matrix_location = this->view_matrix_location;
  CHECK_GL_ERROR(view_matrix_location = glGetUniformLocation(program_id, "view"));

  GLint& light_position_location = this->light_position_location;
  CHECK_GL_ERROR(light_position_location = glGetUniformLocation(program_id, "light_position"));

  GLint& obj_color_location = this->obj_color_location;
  CHECK_GL_ERROR(obj_color_location = glGetUniformLocation(program_id, "obj_color"));

  GLint& eye_location = this->eye_location;
  CHECK_GL_ERROR(eye_location = glGetUniformLocation(program_id, "eye"));
}

void PhongProgram::draw (const vector<glm::vec4>& vertices,
                         const vector<glm::uvec3>& faces,
                         const vector<glm::vec4>& normals,
                         const glm::mat4& model, const glm::vec4& color,
                         const glm::vec4& eye) {
  CHECK_GL_ERROR(glUseProgram(this->programId));

  CHECK_GL_ERROR(glUniformMatrix4fv(this->model_matrix_location, 1, GL_FALSE,
                                    &model[0][0]));
  CHECK_GL_ERROR(glUniformMatrix4fv(this->view_matrix_location, 1, GL_FALSE,
                                    &this->view[0][0]));
  CHECK_GL_ERROR(glUniformMatrix4fv(this->projection_matrix_location, 1,
                                    GL_FALSE, &this->proj[0][0]));

  CHECK_GL_ERROR(glUniform4fv(this->light_position_location, 1,
                 &LIGHT_POSITION[0]));

  CHECK_GL_ERROR(glUniform4fv(this->obj_color_location, 1, &color[0]));
  CHECK_GL_ERROR(glUniform4fv(this->eye_location, 1, &eye[0]));

  CHECK_GL_ERROR(glBindVertexArray(array_objects[this->vaoIndex]));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER,
                              buffer_objects[this->vaoIndex][kVertexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * vertices.size() * 4,
                              &vertices[0], GL_STATIC_DRAW));

  CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(0));

  CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER,
                              buffer_objects[this->vaoIndex][kVertexNormalBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
                              sizeof(float) * normals.size() * 4,
                              &normals[0], GL_STATIC_DRAW));

  CHECK_GL_ERROR(glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0));
  CHECK_GL_ERROR(glEnableVertexAttribArray(1));

  CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                              buffer_objects[this->vaoIndex][kIndexBuffer]));
  CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                              sizeof(uint32_t) * faces.size() * 3,
                              &faces[0], GL_STATIC_DRAW));

  CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, faces.size() * 3,
                                GL_UNSIGNED_INT, 0));
}

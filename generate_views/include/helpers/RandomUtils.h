#ifndef RANDOMUTILS_H
#define RANDOMUTILS_H

#include <string>
#include <vector>
#include <iostream>

#include <glm/glm.hpp>

const glm::vec4 GREEN = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
const glm::vec4 BLUE = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
const glm::vec4 WHITE = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
const glm::vec4 CYAN = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
const glm::vec4 RED = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
const glm::vec4 BROWN = glm::vec4(0.72f, 0.60f, 0.41f, 1.0f);
const glm::vec4 BLACK = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

void LoadOBJ(const std::string& file,
             std::vector<glm::vec4>& vertices,
             std::vector<glm::uvec3>& faces,
             std::vector<glm::vec4>& normals);

void LoadOBJWithNormals(const std::string& file,
                        std::vector<glm::vec4>& vertices,
                        std::vector<glm::uvec3>& faces,
                        std::vector<glm::vec4>& normals);

std::string loadShader(const std::string& file);

std::vector<glm::vec4> getVertexNormals (const std::vector<glm::vec4>& vertices,
                                         const std::vector<glm::uvec3>& faces);

void fixDuplicateVertices (std::vector<glm::vec4>& v,
                           std::vector<glm::uvec3>& f);

namespace glm {
  std::ostream& operator<<(std::ostream& os, const glm::vec2& v);
  std::ostream& operator<<(std::ostream& os, const glm::vec3& v);
  std::ostream& operator<<(std::ostream& os, const glm::vec4& v);
  std::ostream& operator<<(std::ostream& os, const glm::mat3& v);
  std::ostream& operator<<(std::ostream& os, const glm::mat4& v);
}

#endif

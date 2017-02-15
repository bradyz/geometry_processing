#include <string>
#include <iostream>
#include <sstream>

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "render/Program.h"
#include "render/OpenGLStuff.h"
#include "helpers/RandomUtils.h"

using namespace std;

enum {
  kMouseModeCamera,
  kNumMouseModes
};

const std::string window_title = "Collision Detection";

GLuint array_objects[kNumVaos];
GLuint buffer_objects[kNumVaos][kNumVbos];

glm::vec4 LIGHT_POSITION = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);

const float kNear = 1.0f;
const float kFar = 100.0f;
const float kFov = 45.0f;
float camera_distance = 2.0f;

glm::vec3 eye = glm::vec3(5.0f, 5.0f, 5.0f);
glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);

glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
glm::vec3 look = eye - center;
glm::vec3 tangent = glm::cross(up, look);
glm::mat3 orientation = glm::mat3(tangent, up, look);

glm::mat4 view_matrix;
glm::mat4 projection_matrix;

const int WIDTH = 112;
const int HEIGHT = 112;
int window_width = WIDTH;
int window_height = HEIGHT;

bool do_action = false;

GLFWwindow* window;

const char* OpenGlErrorToString(GLenum error) {
  switch (error) {
    case GL_NO_ERROR:
      return "GL_NO_ERROR";
      break;
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
      break;
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
      break;
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
      break;
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
      break;
    default:
      return "Unknown Error";
      break;
  }
  return "Unknown Error";
}

GLuint setupShader (const char* shaderName, GLenum shaderType) {
  GLuint shaderId = 0;
  CHECK_GL_ERROR(shaderId = glCreateShader(shaderType));
  CHECK_GL_ERROR(glShaderSource(shaderId, 1, &shaderName, nullptr));
  glCompileShader(shaderId);
  CHECK_GL_SHADER_ERROR(shaderId);
  return shaderId;
}

void ErrorCallback (int error, const char* description) {
  cerr << "GLFW Error: " << description << "\n";
}

void KeyCallback (GLFWwindow* window, int key, int scancode, int action, int mods) {
  do_action = false;
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
  else if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
    do_action = true;
  }
}

void initOpenGL () {
  if (!glfwInit())
    exit(EXIT_FAILURE);

  glfwSetErrorCallback(ErrorCallback);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);

  window = glfwCreateWindow(window_width, window_height,
                            &window_title[0], nullptr, nullptr);

  CHECK_SUCCESS(window != nullptr);

  glfwMakeContextCurrent(window);
  CHECK_SUCCESS(glewInit() == GLEW_OK);
  glGetError();

  glfwSetKeyCallback(window, KeyCallback);
  glfwSwapInterval(1);

  cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
  cout << "OpenGL version supported:" << glGetString(GL_VERSION) << "\n";

  // Setup our VAOs.
  CHECK_GL_ERROR(glGenVertexArrays(kNumVaos, array_objects));

  // Generate buffer objects
  for (int i = 0; i < kNumVaos; ++i)
    CHECK_GL_ERROR(glGenBuffers(kNumVbos, &buffer_objects[i][0]));

  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_MULTISAMPLE);
  glEnable(GL_BLEND);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_POLYGON_SMOOTH);

  glDepthFunc(GL_LESS);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

bool keepLoopingOpenGL (glm::vec3 eye_pos) {
  if (glfwWindowShouldClose(window))
    return false;

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glfwGetFramebufferSize(window, &window_width, &window_height);
  glViewport(0, 0, window_width, window_height);

  float fovy = static_cast<float>(kFov * (M_PI / 180.0f));
  float aspect = static_cast<float>(window_width) / window_height;

  LIGHT_POSITION = glm::vec4(eye_pos * 2.0f, 1.0f);

  view_matrix = glm::lookAt(eye_pos, center, up);
  projection_matrix = glm::perspective(fovy, aspect, kNear, kFar);

  return true;
}

void endLoopOpenGL () {
  glfwPollEvents();
  glfwSwapBuffers(window);
}

void cleanupOpenGL () {
  glfwDestroyWindow(window);
  glfwTerminate();
  exit(EXIT_SUCCESS);
}

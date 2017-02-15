#version 330 core

uniform vec4 light_position;

in vec4 vertex_position;
in vec4 vertex_normal;

out vec4 vs_light_direction;
out vec4 vs_vertex_normal;
out vec4 vs_world_pos;

void main() {
  gl_Position = vertex_position;
  vs_light_direction = light_position - gl_Position;
  vs_vertex_normal = vertex_normal;
  vs_world_pos = vertex_position;
}

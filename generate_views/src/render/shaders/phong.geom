#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 projection;
uniform mat4 model;
uniform mat4 view;

in vec4 vs_light_direction[];
in vec4 vs_vertex_normal[];
in vec4 vs_world_pos[];

out vec4 normal;
out vec4 light;
out vec4 world_pos;

void main() {
  for (int i = 0; i < gl_in.length(); i++) {
    normal = model * vs_vertex_normal[i];
    light = vs_light_direction[i];
    world_pos = vs_world_pos[i];
    gl_Position = projection * view * model * gl_in[i].gl_Position;
    EmitVertex();
  }
  EndPrimitive();
}

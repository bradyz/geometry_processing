#version 330 core

uniform vec4 obj_color;
uniform vec4 eye;

in vec4 world_pos;
in vec4 normal;
in vec4 light;

out vec4 fragment_color;

void main() {
  vec4 to_eye = normalize(eye - world_pos);
  float dot_nl = dot(normalize(light), normalize(normal));
  dot_nl = max(dot_nl, -dot_nl);

  vec4 H = normalize(normalize(light) + to_eye);

  float KS = pow(clamp(dot(normalize(normal), H), 0.0f, 1.0f), 100);
  float KD = clamp(dot_nl, 0.0f, 1.0f);

  fragment_color = clamp(vec4((KD + KS) * obj_color.xyz, obj_color.w),
                         0.0f, 1.0f);
}

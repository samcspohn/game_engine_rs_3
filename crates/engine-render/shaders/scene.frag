#version 450

// Fragment shader — warm-orange base colour with a single directional light.

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 f_color;

void main() {
    vec3 light_dir = normalize(vec3(1.0, 2.0, 3.0));
    float ambient = 0.20;
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    vec3 base = vec3(0.85, 0.55, 0.20); // warm orange
    f_color = vec4(base * (ambient + diffuse), 1.0);
}

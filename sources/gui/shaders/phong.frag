#version 330 core

uniform vec3 lightpos;
uniform bool phong_enabled;
uniform mat4 mvp;

in fData
{
  vec3 color;
  vec3 pos;
  vec3 normal;
} frag;

out vec4 out_color;

// TODO: paramatrize some stuff inside here
vec3 phong_shading(vec3 incolor, vec3 normal)
{
  vec3 lightpos_cam = (mvp * vec4(lightpos, 1.0f)).xyz;

  vec3 diffuse = vec3(0.0);
  vec3 specular = vec3(0.0);

  // ambient term
  vec3 ambient = 0.3 * incolor;

  // diffuse color
  vec3 kd = incolor;

  // specular color
  vec3 ks = vec3(1.0, 1.0, 1.0);

  // diffuse term
  vec3 lightDir = normalize(lightpos_cam - frag.pos);
  float NdotL = dot(normal, lightDir);

  if (NdotL > 0.0)
      diffuse = kd * NdotL;

  // specular term
  vec3 rVector = normalize(2.0 * normal * dot(normal, lightDir) - lightDir);
  vec3 viewVector = normalize(-frag.pos);
  float RdotV = dot(rVector, viewVector);

  if (RdotV > 0.0)
      specular = ks * pow(RdotV, 32) * .05f;

  return ambient + diffuse + specular;
}

void main()
{
  if (phong_enabled)
  {
    out_color = vec4(phong_shading(frag.color, frag.normal), 1.);
  }
  else
  {
    out_color = vec4(frag.color, 1.);
  }
}

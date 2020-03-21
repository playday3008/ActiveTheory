{@}BasicGLTextBatch.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uUIColor;

#!VARYINGS
varying vec2 vUv;
// varying vec3 v_uColor;
varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = mix(uUIColor, vec3(1.0), smoothstep(0.5, 1.0, v_uAlpha)*0.5);
    gl_FragColor.a = alpha * v_uAlpha;
}
{@}BasicGLUI.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec2 uv = vUv;

    gl_FragColor = vec4(uColor, 1.0);
    gl_FragColor.a = texture2D(tMap, uv).r * uAlpha;
}{@}BasicGLUIBatch.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;
varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.rgb *= uColor;
    gl_FragColor.a *= v_uAlpha;
}
{@}Icon.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec2 uv = vUv;
    vec4 texel = texture2D(tMap, uv);
    gl_FragColor = vec4(uColor, 1.0);
    gl_FragColor.a = texel.r * uAlpha;
}{@}IconAlpha.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    vec2 uv = vUv;
    vec4 texel = texture2D(tMap, uv);
    gl_FragColor = texel;
    gl_FragColor.rgb *= uColor;
    gl_FragColor.a *= uAlpha;
}{@}AntimatterCopy.fs{@}uniform sampler2D tDiffuse;

varying vec2 vUv;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}{@}AntimatterCopy.vs{@}varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}{@}AntimatterPass.vs{@}void main() {
    gl_Position = vec4(position, 1.0);
}{@}AntimatterPosition.vs{@}uniform sampler2D tPos;

void main() {
    vec4 decodedPos = texture2D(tPos, position.xy);
    vec3 pos = decodedPos.xyz;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = 0.02 * (1000.0 / length(mvPosition.xyz));
    gl_Position = projectionMatrix * mvPosition;
}{@}AntimatterBasicFrag.fs{@}void main() {
    gl_FragColor = vec4(1.0);
}{@}antimatter.glsl{@}vec3 getData(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv).xyz;
}

vec4 getData4(sampler2D tex, vec2 uv) {
    return texture2D(tex, uv);
}{@}blendmodes.glsl{@}float blendColorDodge(float base, float blend) {
    return (blend == 1.0)?blend:min(base/(1.0-blend), 1.0);
}
vec3 blendColorDodge(vec3 base, vec3 blend) {
    return vec3(blendColorDodge(base.r, blend.r), blendColorDodge(base.g, blend.g), blendColorDodge(base.b, blend.b));
}
vec3 blendColorDodge(vec3 base, vec3 blend, float opacity) {
    return (blendColorDodge(base, blend) * opacity + base * (1.0 - opacity));
}
float blendColorBurn(float base, float blend) {
    return (blend == 0.0)?blend:max((1.0-((1.0-base)/blend)), 0.0);
}
vec3 blendColorBurn(vec3 base, vec3 blend) {
    return vec3(blendColorBurn(base.r, blend.r), blendColorBurn(base.g, blend.g), blendColorBurn(base.b, blend.b));
}
vec3 blendColorBurn(vec3 base, vec3 blend, float opacity) {
    return (blendColorBurn(base, blend) * opacity + base * (1.0 - opacity));
}
float blendVividLight(float base, float blend) {
    return (blend<0.5)?blendColorBurn(base, (2.0*blend)):blendColorDodge(base, (2.0*(blend-0.5)));
}
vec3 blendVividLight(vec3 base, vec3 blend) {
    return vec3(blendVividLight(base.r, blend.r), blendVividLight(base.g, blend.g), blendVividLight(base.b, blend.b));
}
vec3 blendVividLight(vec3 base, vec3 blend, float opacity) {
    return (blendVividLight(base, blend) * opacity + base * (1.0 - opacity));
}
float blendHardMix(float base, float blend) {
    return (blendVividLight(base, blend)<0.5)?0.0:1.0;
}
vec3 blendHardMix(vec3 base, vec3 blend) {
    return vec3(blendHardMix(base.r, blend.r), blendHardMix(base.g, blend.g), blendHardMix(base.b, blend.b));
}
vec3 blendHardMix(vec3 base, vec3 blend, float opacity) {
    return (blendHardMix(base, blend) * opacity + base * (1.0 - opacity));
}
float blendLinearDodge(float base, float blend) {
    return min(base+blend, 1.0);
}
vec3 blendLinearDodge(vec3 base, vec3 blend) {
    return min(base+blend, vec3(1.0));
}
vec3 blendLinearDodge(vec3 base, vec3 blend, float opacity) {
    return (blendLinearDodge(base, blend) * opacity + base * (1.0 - opacity));
}
float blendLinearBurn(float base, float blend) {
    return max(base+blend-1.0, 0.0);
}
vec3 blendLinearBurn(vec3 base, vec3 blend) {
    return max(base+blend-vec3(1.0), vec3(0.0));
}
vec3 blendLinearBurn(vec3 base, vec3 blend, float opacity) {
    return (blendLinearBurn(base, blend) * opacity + base * (1.0 - opacity));
}
float blendLinearLight(float base, float blend) {
    return blend<0.5?blendLinearBurn(base, (2.0*blend)):blendLinearDodge(base, (2.0*(blend-0.5)));
}
vec3 blendLinearLight(vec3 base, vec3 blend) {
    return vec3(blendLinearLight(base.r, blend.r), blendLinearLight(base.g, blend.g), blendLinearLight(base.b, blend.b));
}
vec3 blendLinearLight(vec3 base, vec3 blend, float opacity) {
    return (blendLinearLight(base, blend) * opacity + base * (1.0 - opacity));
}
float blendLighten(float base, float blend) {
    return max(blend, base);
}
vec3 blendLighten(vec3 base, vec3 blend) {
    return vec3(blendLighten(base.r, blend.r), blendLighten(base.g, blend.g), blendLighten(base.b, blend.b));
}
vec3 blendLighten(vec3 base, vec3 blend, float opacity) {
    return (blendLighten(base, blend) * opacity + base * (1.0 - opacity));
}
float blendDarken(float base, float blend) {
    return min(blend, base);
}
vec3 blendDarken(vec3 base, vec3 blend) {
    return vec3(blendDarken(base.r, blend.r), blendDarken(base.g, blend.g), blendDarken(base.b, blend.b));
}
vec3 blendDarken(vec3 base, vec3 blend, float opacity) {
    return (blendDarken(base, blend) * opacity + base * (1.0 - opacity));
}
float blendPinLight(float base, float blend) {
    return (blend<0.5)?blendDarken(base, (2.0*blend)):blendLighten(base, (2.0*(blend-0.5)));
}
vec3 blendPinLight(vec3 base, vec3 blend) {
    return vec3(blendPinLight(base.r, blend.r), blendPinLight(base.g, blend.g), blendPinLight(base.b, blend.b));
}
vec3 blendPinLight(vec3 base, vec3 blend, float opacity) {
    return (blendPinLight(base, blend) * opacity + base * (1.0 - opacity));
}
float blendReflect(float base, float blend) {
    return (blend == 1.0)?blend:min(base*base/(1.0-blend), 1.0);
}
vec3 blendReflect(vec3 base, vec3 blend) {
    return vec3(blendReflect(base.r, blend.r), blendReflect(base.g, blend.g), blendReflect(base.b, blend.b));
}
vec3 blendReflect(vec3 base, vec3 blend, float opacity) {
    return (blendReflect(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendGlow(vec3 base, vec3 blend) {
    return blendReflect(blend, base);
}
vec3 blendGlow(vec3 base, vec3 blend, float opacity) {
    return (blendGlow(base, blend) * opacity + base * (1.0 - opacity));
}
float blendOverlay(float base, float blend) {
    return base<0.5?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend));
}
vec3 blendOverlay(vec3 base, vec3 blend) {
    return vec3(blendOverlay(base.r, blend.r), blendOverlay(base.g, blend.g), blendOverlay(base.b, blend.b));
}
vec3 blendOverlay(vec3 base, vec3 blend, float opacity) {
    return (blendOverlay(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendHardLight(vec3 base, vec3 blend) {
    return blendOverlay(blend, base);
}
vec3 blendHardLight(vec3 base, vec3 blend, float opacity) {
    return (blendHardLight(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendPhoenix(vec3 base, vec3 blend) {
    return min(base, blend)-max(base, blend)+vec3(1.0);
}
vec3 blendPhoenix(vec3 base, vec3 blend, float opacity) {
    return (blendPhoenix(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendNormal(vec3 base, vec3 blend) {
    return blend;
}
vec3 blendNormal(vec3 base, vec3 blend, float opacity) {
    return (blendNormal(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendNegation(vec3 base, vec3 blend) {
    return vec3(1.0)-abs(vec3(1.0)-base-blend);
}
vec3 blendNegation(vec3 base, vec3 blend, float opacity) {
    return (blendNegation(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendMultiply(vec3 base, vec3 blend) {
    return base*blend;
}
vec3 blendMultiply(vec3 base, vec3 blend, float opacity) {
    return (blendMultiply(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendAverage(vec3 base, vec3 blend) {
    return (base+blend)/2.0;
}
vec3 blendAverage(vec3 base, vec3 blend, float opacity) {
    return (blendAverage(base, blend) * opacity + base * (1.0 - opacity));
}
float blendScreen(float base, float blend) {
    return 1.0-((1.0-base)*(1.0-blend));
}
vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(blendScreen(base.r, blend.r), blendScreen(base.g, blend.g), blendScreen(base.b, blend.b));
}
vec3 blendScreen(vec3 base, vec3 blend, float opacity) {
    return (blendScreen(base, blend) * opacity + base * (1.0 - opacity));
}
float blendSoftLight(float base, float blend) {
    return (blend<0.5)?(2.0*base*blend+base*base*(1.0-2.0*blend)):(sqrt(base)*(2.0*blend-1.0)+2.0*base*(1.0-blend));
}
vec3 blendSoftLight(vec3 base, vec3 blend) {
    return vec3(blendSoftLight(base.r, blend.r), blendSoftLight(base.g, blend.g), blendSoftLight(base.b, blend.b));
}
vec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {
    return (blendSoftLight(base, blend) * opacity + base * (1.0 - opacity));
}
float blendSubtract(float base, float blend) {
    return max(base+blend-1.0, 0.0);
}
vec3 blendSubtract(vec3 base, vec3 blend) {
    return max(base+blend-vec3(1.0), vec3(0.0));
}
vec3 blendSubtract(vec3 base, vec3 blend, float opacity) {
    return (blendSubtract(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendExclusion(vec3 base, vec3 blend) {
    return base+blend-2.0*base*blend;
}
vec3 blendExclusion(vec3 base, vec3 blend, float opacity) {
    return (blendExclusion(base, blend) * opacity + base * (1.0 - opacity));
}
vec3 blendDifference(vec3 base, vec3 blend) {
    return abs(base-blend);
}
vec3 blendDifference(vec3 base, vec3 blend, float opacity) {
    return (blendDifference(base, blend) * opacity + base * (1.0 - opacity));
}
float blendAdd(float base, float blend) {
    return min(base+blend, 1.0);
}
vec3 blendAdd(vec3 base, vec3 blend) {
    return min(base+blend, vec3(1.0));
}
vec3 blendAdd(vec3 base, vec3 blend, float opacity) {
    return (blendAdd(base, blend) * opacity + base * (1.0 - opacity));
}{@}cheapblur.fs{@}vec4 blur(sampler2D tDiffuse, vec2 uv, float sampleDist, float strength) {
    float samples[6];
    samples[0] = -0.08;
    samples[1] = -0.03;
    samples[2] = -0.01;
    samples[3] =  0.01;
    samples[4] =  0.03;
    samples[5] =  0.08;

    vec2 dir = normalize(0.5 - uv);
    vec4 texel = texture2D(tDiffuse, uv);
    vec4 sum = texel;

    for (int i = 0; i < 6; i++) {
        sum += texture2D(tDiffuse, uv + (dir * samples[i] * sampleDist * strength));
    }

    sum /= 6.0;
    return sum;
}{@}conditionals.glsl{@}vec4 when_eq(vec4 x, vec4 y) {
  return 1.0 - abs(sign(x - y));
}

vec4 when_neq(vec4 x, vec4 y) {
  return abs(sign(x - y));
}

vec4 when_gt(vec4 x, vec4 y) {
  return max(sign(x - y), 0.0);
}

vec4 when_lt(vec4 x, vec4 y) {
  return max(sign(y - x), 0.0);
}

vec4 when_ge(vec4 x, vec4 y) {
  return 1.0 - when_lt(x, y);
}

vec4 when_le(vec4 x, vec4 y) {
  return 1.0 - when_gt(x, y);
}

vec3 when_eq(vec3 x, vec3 y) {
  return 1.0 - abs(sign(x - y));
}

vec3 when_neq(vec3 x, vec3 y) {
  return abs(sign(x - y));
}

vec3 when_gt(vec3 x, vec3 y) {
  return max(sign(x - y), 0.0);
}

vec3 when_lt(vec3 x, vec3 y) {
  return max(sign(y - x), 0.0);
}

vec3 when_ge(vec3 x, vec3 y) {
  return 1.0 - when_lt(x, y);
}

vec3 when_le(vec3 x, vec3 y) {
  return 1.0 - when_gt(x, y);
}

vec2 when_eq(vec2 x, vec2 y) {
  return 1.0 - abs(sign(x - y));
}

vec2 when_neq(vec2 x, vec2 y) {
  return abs(sign(x - y));
}

vec2 when_gt(vec2 x, vec2 y) {
  return max(sign(x - y), 0.0);
}

vec2 when_lt(vec2 x, vec2 y) {
  return max(sign(y - x), 0.0);
}

vec2 when_ge(vec2 x, vec2 y) {
  return 1.0 - when_lt(x, y);
}

vec2 when_le(vec2 x, vec2 y) {
  return 1.0 - when_gt(x, y);
}

float when_eq(float x, float y) {
  return 1.0 - abs(sign(x - y));
}

float when_neq(float x, float y) {
  return abs(sign(x - y));
}

float when_gt(float x, float y) {
  return max(sign(x - y), 0.0);
}

float when_lt(float x, float y) {
  return max(sign(y - x), 0.0);
}

float when_ge(float x, float y) {
  return 1.0 - when_lt(x, y);
}

float when_le(float x, float y) {
  return 1.0 - when_gt(x, y);
}

vec4 and(vec4 a, vec4 b) {
  return a * b;
}

vec4 or(vec4 a, vec4 b) {
  return min(a + b, 1.0);
}

vec4 Not(vec4 a) {
  return 1.0 - a;
}

vec3 and(vec3 a, vec3 b) {
  return a * b;
}

vec3 or(vec3 a, vec3 b) {
  return min(a + b, 1.0);
}

vec3 Not(vec3 a) {
  return 1.0 - a;
}

vec2 and(vec2 a, vec2 b) {
  return a * b;
}

vec2 or(vec2 a, vec2 b) {
  return min(a + b, 1.0);
}


vec2 Not(vec2 a) {
  return 1.0 - a;
}

float and(float a, float b) {
  return a * b;
}

float or(float a, float b) {
  return min(a + b, 1.0);
}

float Not(float a) {
  return 1.0 - a;
}{@}contrast.glsl{@}void applyContrast(inout vec3 color, float contrast, float brightness) {
	vec3 colorContrasted = color * contrast;
	color.rgb = colorContrasted + vec3(brightness, brightness, brightness);
}{@}curl.glsl{@}float CNrange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

float CNnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += CNrange(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += CNrange(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}

vec3 snoiseVec3( vec3 x ){

    float s  = CNnoise(vec3( x ));
    float s1 = CNnoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
    float s2 = CNnoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
    vec3 c = vec3( s , s1 , s2 );
    return c;

}

vec3 curlNoise( vec3 p ){

    const float e = 1e-1;
    vec3 dx = vec3( e   , 0.0 , 0.0 );
    vec3 dy = vec3( 0.0 , e   , 0.0 );
    vec3 dz = vec3( 0.0 , 0.0 , e   );

    vec3 p_x0 = snoiseVec3( p - dx );
    vec3 p_x1 = snoiseVec3( p + dx );
    vec3 p_y0 = snoiseVec3( p - dy );
    vec3 p_y1 = snoiseVec3( p + dy );
    vec3 p_z0 = snoiseVec3( p - dz );
    vec3 p_z1 = snoiseVec3( p + dz );

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    return normalize( vec3( x , y , z ) * divisor );
}{@}depthvalue.fs{@}float getDepthValue(sampler2D tDepth, vec2 uv, float n, float f) {
    return (2.0 * n) / (f + n - texture2D(tDepth, uv).x * (f - n));
}{@}eases.glsl{@}#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef HALF_PI
#define HALF_PI 1.5707963267948966
#endif

float backInOut(float t) {
  float f = t < 0.5
    ? 2.0 * t
    : 1.0 - (2.0 * t - 1.0);

  float g = pow(f, 3.0) - f * sin(f * PI);

  return t < 0.5
    ? 0.5 * g
    : 0.5 * (1.0 - g) + 0.5;
}

float backIn(float t) {
  return pow(t, 3.0) - t * sin(t * PI);
}

float backOut(float t) {
  float f = 1.0 - t;
  return 1.0 - (pow(f, 3.0) - f * sin(f * PI));
}

float bounceOut(float t) {
  const float a = 4.0 / 11.0;
  const float b = 8.0 / 11.0;
  const float c = 9.0 / 10.0;

  const float ca = 4356.0 / 361.0;
  const float cb = 35442.0 / 1805.0;
  const float cc = 16061.0 / 1805.0;

  float t2 = t * t;

  return t < a
    ? 7.5625 * t2
    : t < b
      ? 9.075 * t2 - 9.9 * t + 3.4
      : t < c
        ? ca * t2 - cb * t + cc
        : 10.8 * t * t - 20.52 * t + 10.72;
}

float bounceIn(float t) {
  return 1.0 - bounceOut(1.0 - t);
}

float bounceInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
    : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
}

float circularInOut(float t) {
  return t < 0.5
    ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
    : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}

float circularIn(float t) {
  return 1.0 - sqrt(1.0 - t * t);
}

float circularOut(float t) {
  return sqrt((2.0 - t) * t);
}

float cubicInOut(float t) {
  return t < 0.5
    ? 4.0 * t * t * t
    : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
}

float cubicIn(float t) {
  return t * t * t;
}

float cubicOut(float t) {
  float f = t - 1.0;
  return f * f * f + 1.0;
}

float elasticInOut(float t) {
  return t < 0.5
    ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
    : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
}

float elasticIn(float t) {
  return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
}

float elasticOut(float t) {
  return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
}

float expoInOut(float t) {
  return t == 0.0 || t == 1.0
    ? t
    : t < 0.5
      ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
      : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}

float expoIn(float t) {
  return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
}

float expoOut(float t) {
  return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
}

float linear(float t) {
  return t;
}

float quadraticInOut(float t) {
  float p = 2.0 * t * t;
  return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}

float quadraticIn(float t) {
  return t * t;
}

float quadraticOut(float t) {
  return -t * (t - 2.0);
}

float quarticInOut(float t) {
  return t < 0.5
    ? +8.0 * pow(t, 4.0)
    : -8.0 * pow(t - 1.0, 4.0) + 1.0;
}

float quarticIn(float t) {
  return pow(t, 4.0);
}

float quarticOut(float t) {
  return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
}

float qinticInOut(float t) {
  return t < 0.5
    ? +16.0 * pow(t, 5.0)
    : -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0;
}

float qinticIn(float t) {
  return pow(t, 5.0);
}

float qinticOut(float t) {
  return 1.0 - (pow(t - 1.0, 5.0));
}

float sineInOut(float t) {
  return -0.5 * (cos(PI * t) - 1.0);
}

float sineIn(float t) {
  return sin((t - 1.0) * HALF_PI) + 1.0;
}

float sineOut(float t) {
  return sin(t * HALF_PI);
}
{@}ColorMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 color;

#!VARYINGS

#!SHADER: ColorMaterial.vs
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: ColorMaterial.fs
void main() {
    gl_FragColor = vec4(color, 1.0);
}{@}DebugCamera.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vColor;

#!SHADER: DebugCamera.vs
void main() {
    vColor = mix(vec3(1.0), vec3(1.0, 0.0, 0.0), step(position.z, 0.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: DebugCamera.fs
void main() {
    gl_FragColor = vec4(vColor, 1.0);
}{@}ScreenQuad.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: ScreenQuad.vs
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: ScreenQuad.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a = 1.0;
}{@}TestMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vNormal;

#!SHADER: TestMaterial.vs
void main() {
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: TestMaterial.fs
void main() {
    gl_FragColor = vec4(vNormal, 1.0);
}{@}TextureMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: TextureMaterial.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: TextureMaterial.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.rgb /= gl_FragColor.a;
}{@}BlitPass.fs{@}void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
    gl_FragColor.a = 1.0;
}{@}NukePass.vs{@}varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}{@}ShadowDepth.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: ShadowDepth.vs
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: ShadowDepth.fs
void main() {
    gl_FragColor = vec4(vec3(gl_FragCoord.z), 1.0);
}{@}instance.vs{@}vec3 transformNormal(vec3 n, vec4 orientation) {
    vec3 ncN = cross(orientation.xyz, n);
    n = ncN * (2.0 * orientation.w) + (cross(orientation.xyz, ncN) * 2.0 + n);
    return n;
}

vec3 transformPosition(vec3 position, vec3 offset, vec3 scale, vec4 orientation) {
    vec3 pos = position;
    pos *= scale;

    pos = pos + 2.0 * cross(orientation.xyz, cross(orientation.xyz, pos) + orientation.w * pos);
    pos += offset;
    return pos;
}

vec3 transformPosition(vec3 position, vec3 offset, vec4 orientation) {
    vec3 pos = position;

    pos = pos + 2.0 * cross(orientation.xyz, cross(orientation.xyz, pos) + orientation.w * pos);
    pos += offset;
    return pos;
}

vec3 transformPosition(vec3 position, vec3 offset, float scale, vec4 orientation) {
    return transformPosition(position, offset, vec3(scale), orientation);
}

vec3 transformPosition(vec3 position, vec3 offset) {
    return position + offset;
}

vec3 transformPosition(vec3 position, vec3 offset, float scale) {
    vec3 pos = position * scale;
    return pos + offset;
}

vec3 transformPosition(vec3 position, vec3 offset, vec3 scale) {
    vec3 pos = position * scale;
    return pos + offset;
}{@}lights.fs{@}vec3 worldLight(vec3 pos, vec3 vpos) {
    vec4 mvPos = modelViewMatrix * vec4(vpos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}{@}lights.vs{@}vec3 worldLight(vec3 pos) {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}

vec3 worldLight(vec3 lightPos, vec3 localPos) {
    vec4 mvPos = modelViewMatrix * vec4(localPos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(lightPos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}{@}shadows.fs{@}float shadowCompare(sampler2D map, vec2 coords, float compare) {
    return step(compare, texture2D(map, coords).r);
}

float shadowLerp(sampler2D map, vec2 coords, float compare, float size) {
    const vec2 offset = vec2(0.0, 1.0);

    vec2 texelSize = vec2(1.0) / size;
    vec2 centroidUV = floor(coords * size + 0.5) / size;

    float lb = shadowCompare(map, centroidUV + texelSize * offset.xx, compare);
    float lt = shadowCompare(map, centroidUV + texelSize * offset.xy, compare);
    float rb = shadowCompare(map, centroidUV + texelSize * offset.yx, compare);
    float rt = shadowCompare(map, centroidUV + texelSize * offset.yy, compare);

    vec2 f = fract( coords * size + 0.5 );

    float a = mix( lb, lt, f.y );
    float b = mix( rb, rt, f.y );
    float c = mix( a, b, f.x );

    return c;
}

float srange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

float shadowrandom(vec3 vin) {
    vec3 v = vin * 0.1;
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += srange(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += srange(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}

float shadowLookup(sampler2D map, vec3 coords, float size, float compare, vec3 wpos) {
    float shadow = 1.0;

    #if defined(SHADOW_MAPS)
    bvec4 inFrustumVec = bvec4 (coords.x >= 0.0, coords.x <= 1.0, coords.y >= 0.0, coords.y <= 1.0);
    bool inFrustum = all(inFrustumVec);
    bvec2 frustumTestVec = bvec2(inFrustum, coords.z <= 1.0);
    bool frustumTest = all(frustumTestVec);

    if (frustumTest) {
        vec2 texelSize = vec2(1.0) / size;

        float dx0 = -texelSize.x;
        float dy0 = -texelSize.y;
        float dx1 = +texelSize.x;
        float dy1 = +texelSize.y;

        float rnoise = shadowrandom(wpos) * 0.0015;
        dx0 += rnoise;
        dy0 -= rnoise;
        dx1 += rnoise;
        dy1 -= rnoise;

        #if defined(SHADOWS_MED)
        shadow += shadowCompare(map, coords.xy + vec2(0.0, dy0), compare);
//        shadow += shadowCompare(map, coords.xy + vec2(dx1, dy0), compare);
        shadow += shadowCompare(map, coords.xy + vec2(dx0, 0.0), compare);
        shadow += shadowCompare(map, coords.xy, compare);
        shadow += shadowCompare(map, coords.xy + vec2(dx1, 0.0), compare);
//        shadow += shadowCompare(map, coords.xy + vec2(dx0, dy1), compare);
        shadow += shadowCompare(map, coords.xy + vec2(0.0, dy1), compare);
        shadow /= 5.0;

        #elif defined(SHADOWS_HIGH)
        shadow = shadowLerp(map, coords.xy + vec2(dx0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy, compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, 0.0), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(0.0, dy1), compare, size);
        shadow += shadowLerp(map, coords.xy + vec2(dx1, dy1), compare, size);
        shadow /= 9.0;

        #else
        shadow = shadowCompare(map, coords.xy, compare);
        #endif
    }

    #endif

    return clamp(shadow, 0.0, 1.0);
}

vec3 transformShadowLight(vec3 pos, vec3 vpos) {
    vec4 mvPos = modelViewMatrix * vec4(vpos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(pos, 1.0);
    return normalize(worldPosition.xyz - mvPos.xyz);
}

float getShadow(vec3 pos, vec3 normal, float bias) {
    float shadow = 1.0;
    #if defined(SHADOW_MAPS)

    #pragma unroll_loop
    for (int i = 0; i < SHADOW_COUNT; i++) {
        vec4 shadowMapCoords = shadowMatrix[i] * vec4(pos, 1.0);
        vec3 coords = (shadowMapCoords.xyz / shadowMapCoords.w) * vec3(0.5) + vec3(0.5);

        float lookup = shadowLookup(shadowMap[i], coords, shadowSize[i], coords.z - bias, pos);
        lookup += mix(1.0 - step(0.002, dot(transformShadowLight(shadowLightPos[i], pos), normal)), 0.0, step(999.0, normal.x));
        shadow *= clamp(lookup, 0.0, 1.0);
    }

    #endif
    return shadow;
}

float getShadow(vec3 pos, vec3 normal) {
    return getShadow(pos, normal, 0.0);
}

float getShadow(vec3 pos, float bias) {
    return getShadow(pos, vec3(99999.0), bias);
}

float getShadow(vec3 pos) {
    return getShadow(pos, vec3(99999.0), 0.0);
}{@}FXAA.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec2 v_rgbNW;
varying vec2 v_rgbNE;
varying vec2 v_rgbSW;
varying vec2 v_rgbSE;
varying vec2 v_rgbM;

#!SHADER: FXAA.vs

varying vec2 vUv;

void main() {
    vUv = uv;

    vec2 fragCoord = uv * resolution;
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);

    gl_Position = vec4(position, 1.0);
}

#!SHADER: FXAA.fs

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif
#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif
#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
            vec2 v_rgbNW, vec2 v_rgbNE,
            vec2 v_rgbSW, vec2 v_rgbSE,
            vec2 v_rgbM) {
    vec4 color;
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec3 rgbNW = texture2D(tex, v_rgbNW).xyz;
    vec3 rgbNE = texture2D(tex, v_rgbNE).xyz;
    vec3 rgbSW = texture2D(tex, v_rgbSW).xyz;
    vec3 rgbSE = texture2D(tex, v_rgbSE).xyz;
    vec4 texColor = texture2D(tex, v_rgbM);
    vec3 rgbM  = texColor.xyz;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;

    vec3 rgbA = 0.5 * (
        texture2D(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)).xyz);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture2D(tex, fragCoord * inverseVP + dir * -0.5).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * 0.5).xyz);

    float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = vec4(rgbA, texColor.a);
    else
        color = vec4(rgbB, texColor.a);
    return color;
}

void main() {
    vec2 fragCoord = vUv * resolution;
    gl_FragColor = fxaa(tDiffuse, fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);
    gl_FragColor.a = 1.0;
}{@}gaussianblur.fs{@}vec4 blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.411764705882353) * direction;
  vec2 off2 = vec2(3.2941176470588234) * direction;
  vec2 off3 = vec2(5.176470588235294) * direction;
  color += texture2D(image, uv) * 0.1964825501511404;
  color += texture2D(image, uv + (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv - (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv + (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv - (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv + (off3 / resolution)) * 0.010381362401148057;
  color += texture2D(image, uv - (off3 / resolution)) * 0.010381362401148057;
  return color;
}

vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture2D(image, uv) * 0.29411764705882354;
  color += texture2D(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture2D(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color;
}

vec4 blur9(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * direction;
  vec2 off2 = vec2(3.2307692308) * direction;
  color += texture2D(image, uv) * 0.2270270270;
  color += texture2D(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture2D(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture2D(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
}{@}glscreenprojection.glsl{@}vec2 frag_coord(vec4 glPos) {
    return ((glPos.xyz / glPos.w) * 0.5 + 0.5).xy;
}

vec2 getProjection(vec3 pos, mat4 projMatrix) {
    vec4 mvpPos = projMatrix * vec4(pos, 1.0);
    return frag_coord(mvpPos);
}

void applyNormal(inout vec3 pos, mat4 projNormalMatrix) {
    vec3 transformed = vec3(projNormalMatrix * vec4(pos, 0.0));
    pos = transformed;
}{@}DefaultText.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS

varying vec2 vUv;

#!SHADER: DefaultText.vs

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: DefaultText.fs

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = uColor;
    gl_FragColor.a = alpha * uAlpha;
}
{@}msdf.glsl{@}float msdf(sampler2D tMap, vec2 uv) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;

    // TODO: fallback for fwidth for webgl1 (need to enable ext)
    float d = fwidth(signedDist);
    float alpha = smoothstep(-d, d, signedDist);
    if (alpha < 0.01) discard;
    return alpha;
}

float strokemsdf(sampler2D tMap, vec2 uv, float stroke, float padding) {
    vec3 tex = texture2D(tMap, uv).rgb;
    float signedDist = max(min(tex.r, tex.g), min(max(tex.r, tex.g), tex.b)) - 0.5;
    float t = stroke;
    float alpha = smoothstep(-t, -t + padding, signedDist) * smoothstep(t, t - padding, signedDist);
    return alpha;
}{@}GLUIBatch.glsl{@}#!ATTRIBUTES
attribute vec2 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));
    pos.xy *= scale;
    pos.xy += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(1.0);
}{@}GLUIBatchText.glsl{@}#!ATTRIBUTES
attribute vec2 offset;
attribute vec2 scale;
attribute float rotation;
//attributes

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;
//varyings

#!SHADER: Vertex

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
    oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
    oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
    0.0,                                0.0,                                0.0,                                1.0);
}

void main() {
    vUv = uv;
    //vdefines

    vec3 pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(position, 1.0));
    pos.xy *= scale;
    pos.xy += offset;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = v_uColor;
    gl_FragColor.a = alpha * v_uAlpha;
}
{@}GLUIObject.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: GLUIObject.vs
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIObject.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= uAlpha;
}{@}GLUIObjectMask.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uAlpha;
uniform vec4 mask;

#!VARYINGS
varying vec2 vUv;
varying vec2 vWorldPos;

#!SHADER: GLUIObjectMask.vs
void main() {
    vUv = uv;
    vWorldPos = (modelMatrix * vec4(position.xy, 0.0, 1.0)).xy;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: GLUIObjectMask.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= uAlpha;

    if (vWorldPos.x > mask.x + mask.z) discard;
    if (vWorldPos.x < mask.x) discard;
    if (vWorldPos.y > mask.y) discard;
    if (vWorldPos.y < mask.y - mask.w) discard;
}{@}levelmask.glsl{@}float levelChannel(float inPixel, float inBlack, float inGamma, float inWhite, float outBlack, float outWhite) {
    return (pow(((inPixel * 255.0) - inBlack) / (inWhite - inBlack), inGamma) * (outWhite - outBlack) + outBlack) / 255.0;
}

vec3 levels(vec3 inPixel, float inBlack, float inGamma, float inWhite, float outBlack, float outWhite) {
    vec3 o = vec3(1.0);
    o.r = levelChannel(inPixel.r, inBlack, inGamma, inWhite, outBlack, outWhite);
    o.g = levelChannel(inPixel.g, inBlack, inGamma, inWhite, outBlack, outWhite);
    o.b = levelChannel(inPixel.b, inBlack, inGamma, inWhite, outBlack, outWhite);
    return o;
}

float animateLevels(float inp, float t) {
    float inBlack = 0.0;
    float inGamma = range(t, 0.0, 1.0, 0.0, 3.0);
    float inWhite = range(t, 0.0, 1.0, 20.0, 255.0);
    float outBlack = 0.0;
    float outWhite = 255.0;

    float mask = 1.0 - levels(vec3(inp), inBlack, inGamma, inWhite, outBlack, outWhite).r;
    mask = max(0.0, min(1.0, mask));
    return mask;
}{@}LightVolume.glsl{@}#!ATTRIBUTES
attribute vec3 offset;
attribute vec4 attribs;

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform vec3 uColor;
uniform vec3 uUIColor;

uniform ubo {
    float uScale;
    float uSeparation;
    float uAlpha;
    float uMaskScale;
    float uRotateSpeed;
    float uRotateTexture;
    float uNoiseScale;
    float uNoiseSpeed;
    float uNoiseRange;
    float uOffset;
    float uScrollX;
    float uScrollY;
    float uHueShift;
};

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;
varying vec4 vAttribs;
varying float vOffset;

#!SHADER: LightVolume.vs

#require(instance.vs)
#require(rotation.glsl)

void main() {
    vec3 pos = transformPosition(position, offset * uSeparation, uScale);
    pos = vec3(vec4(pos, 1.0) * rotationMatrix(vec3(0.0, 0.0, 1.0), radians(360.0 * 0.1 * offset.z * uOffset)));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    vUv = uv;
    vPos = pos;
    vAttribs = attribs;
    vOffset = offset.z * 10.0;
}

#!SHADER: LightVolume.fs

#require(rgb2hsv.fs)
#require(range.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = rgb2hsv(uUIColor);
    //color += (vUv.y * uHueShift * 0.01);
    color.y *= 0.6;
    color = hsv2rgb(color);

    float noise = cnoise(vPos * uNoiseScale + (time * uNoiseSpeed));
    vec2 uv = scaleUV(vUv, vec2(range(noise, -1.0, 1.0, 0.9, 1.1)));
    uv += noise * uNoiseRange * 0.1;

    //vec2 fluid = texture2D(tFluid, gl_FragCoord.xy / resolution).xy * texture2D(tFluidMask, gl_FragCoord.xy / resolution).r;
    //uv -= fluid * 0.00025;
//
    float mask = texture2D(tMask, uv).r;
    float alpha = texture2D(tMap, uv).r;
//
    if (mask < 0.001) discard;
//
//    vec2 uv = scaleUV(vUv, vec2(uMaskScale));
//

    ////    uv.x += sin(time * 0.04) * 0.3;
//
//    uv = rotateUV(uv, uRotateSpeed * time * range(vAttribs.x, 0.0, 1.0, 0.5, 1.5));
//    uv.x += time * uScrollX * 0.1 * range(vAttribs.y, 0.0, 1.0, 0.5, 1.5);
//    uv.y += time * uScrollY * 0.1 * range(vAttribs.z, 0.0, 1.0, 0.5, 1.5);
//

    alpha *= mask;
    alpha *= crange(getNoise(vUv, time), 0.0, 1.0, 0.85, 1.0);

    gl_FragColor = vec4(color, alpha * uAlpha);
}{@}luma.fs{@}float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}{@}lut.fs{@}vec4 lookup(in vec4 textureColor, in sampler2D lookupTable) {
    mediump float blueColor = textureColor.b * 63.0;

    mediump vec2 quad1;
    quad1.y = floor(floor(blueColor) / 8.0);
    quad1.x = floor(blueColor) - (quad1.y * 8.0);

    mediump vec2 quad2;
    quad2.y = floor(ceil(blueColor) / 8.0);
    quad2.x = ceil(blueColor) - (quad2.y * 8.0);

    highp vec2 texPos1;
    texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.r);
    texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.g);

    texPos1.y = 1.0-texPos1.y;

    highp vec2 texPos2;
    texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.r);
    texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * textureColor.g);

    texPos2.y = 1.0-texPos2.y;

    lowp vec4 newColor1 = texture2D(lookupTable, texPos1);
    lowp vec4 newColor2 = texture2D(lookupTable, texPos2);

    lowp vec4 newColor = mix(newColor1, newColor2, fract(blueColor));
    return newColor;
}

vec3 lookup(in vec3 textureColor, in sampler2D lookupTable) {
    return lookup(vec4(textureColor, 1.0), lookupTable).rgb;
}{@}matcap.vs{@}vec2 reflectMatcap(vec3 position, mat4 modelViewMatrix, mat3 normalMatrix, vec3 normal) {
    vec4 p = vec4(position, 1.0);
    
    vec3 e = normalize(vec3(modelViewMatrix * p));
    vec3 n = normalize(normalMatrix * normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
        pow(r.x, 2.0) +
        pow(r.y, 2.0) +
        pow(r.z + 1.0, 2.0)
    );
    
    vec2 uv = r.xy / m + .5;
    
    return uv;
}

vec2 reflectMatcap(vec3 position, mat4 modelViewMatrix, vec3 normal) {
    vec4 p = vec4(position, 1.0);
    
    vec3 e = normalize(vec3(modelViewMatrix * p));
    vec3 n = normalize(normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
                         pow(r.x, 2.0) +
                         pow(r.y, 2.0) +
                         pow(r.z + 1.0, 2.0)
                         );
    
    vec2 uv = r.xy / m + .5;
    
    return uv;
}

vec2 reflectMatcap(vec4 mvPos, vec3 normal) {
    vec3 e = normalize(vec3(mvPos));
    vec3 n = normalize(normal);
    vec3 r = reflect(e, n);
    float m = 2.0 * sqrt(
                         pow(r.x, 2.0) +
                         pow(r.y, 2.0) +
                         pow(r.z + 1.0, 2.0)
                         );

    vec2 uv = r.xy / m + .5;

    return uv;
}{@}BasicMirror.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMirrorReflection;
uniform mat4 uMirrorMatrix;

#!VARYINGS
varying vec4 vMirrorCoord;

#!SHADER: BasicMirror.vs
void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: BasicMirror.fs
void main() {
    gl_FragColor = texture2DProj(tMirrorReflection, vMirrorCoord);
}{@}MouseFlowMapBlend.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform sampler2D uTexture;
uniform sampler2D uStamp;
uniform float uSpeed;
uniform float uFirstDraw;

#!VARYINGS

varying vec2 vUv;

#!SHADER: MouseFlowMapBlend.vs

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: MouseFlowMapBlend.fs

vec3 blend(vec3 base, vec3 blend, float opacity) {
    return blend + (base * (1.0 - opacity));
}

#require(range.glsl)

void main() {
    vec3 prev = texture2D(uTexture, vUv).rgb;
    prev = prev * 2.0 - 1.0;
    float amount = crange(length(prev.rg), 0.0, 0.4, 0.0, 1.0);
    amount = 0.5 + 0.48 * (1.0 - pow(1.0 - amount, 3.0));
    prev *= amount;
    prev = prev * 0.5 + 0.5;

    // blue not used
    prev.b = 0.5;

    vec4 tex = texture2D(uStamp, vUv);
    gl_FragColor.rgb = blend(prev, tex.rgb, tex.a);

    // Force a grey on first draw to have init values
    gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(0.5), uFirstDraw);
    gl_FragColor.a = 1.0;
}
{@}MouseFlowMapStamp.glsl{@}#!ATTRIBUTES

#!UNIFORMS

uniform vec2 uVelocity;
uniform float uFalloff;
uniform float uAlpha;
uniform float uAspect;

#!VARYINGS

varying vec2 vUv;

#!SHADER: MouseFlowMapStamp.vs

void main() {
    vUv = uv;
    vec3 pos = position;
    pos.x *= 1.0 / uAspect;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

    #!SHADER: MouseFlowMapStamp.fs

void main() {
    gl_FragColor.rgb = vec3(uVelocity * 0.5 + 0.5, 1.0);
    gl_FragColor.a = smoothstep(0.5, 0.499 - (uFalloff * 0.499), length(vUv - 0.5)) * uAlpha;
}
{@}flowmap.fs{@}float getFlowMask(sampler2D map, vec2 uv) {
    vec2 flow = texture2D(map, uv).rg;
    return clamp(length(flow.rg * 2.0 - 1.0), 0.0, 1.0);
}

vec2 getFlow(sampler2D map, vec2 uv) {
    vec2 flow = texture2D(map, uv).rg * 2.0 - 1.0;
    flow.y *= -1.0;
    return flow;
}{@}NeutrinoBase.fs{@}#require(neutrino.glsl)

uniform float uLerp;

//params

void main() {
    vec2 uv = getUV();
    vec4 pdata = getData4(tInput, uv);
    vec3 index = getData(tIndices, uv);
    vec4 activePos = getData4(tActive, uv);
    vec4 attribs = getData4(tAttribs, uv);
    vec4 random = attribs;
    vec3 pos = pdata.xyz;

    float CHAIN = index.x;
    float LINE = index.y;

    if (pdata.w > 0.9) { //head of the chain

        if (activePos.a < 0.01) {
            gl_FragColor = vec4(pos, pdata.w); //still
            return;
        }

        if (activePos.a > 0.7 && activePos.a < 0.8 || activePos.a > 0.05 && activePos.a < 0.15) { //if its in the initial state
            pos = activePos.xyz;
            gl_FragColor = vec4(pos, pdata.w);
            return;
        }

        if (activePos.a > 0.25) { //OK to move!
        //main
        }

    } else {

        float followIndex = getIndex(LINE, CHAIN-1.0);
        vec3 followPos = getData(tInput, getUVFromIndex(followIndex));

        float headIndex = getIndex(LINE, 0.0);
        vec4 headActive = getData4(tActive, getUVFromIndex(headIndex));

        if (headActive.a < 0.01) { //still
            gl_FragColor = vec4(pos, pdata.w);
            return;
        }

        if (headActive.a > 0.7 && headActive.a < 0.8 || headActive.a > 0.05 && headActive.a < 0.15) { //still in the init state
            pos = headActive.xyz;
            gl_FragColor = vec4(pos, pdata.w);
            return;
        }

        pos += (followPos - pos) * uLerp;

    }

    gl_FragColor = vec4(pos, pdata.w);
 }{@}NeutrinoTube.glsl{@}#!ATTRIBUTES
attribute float angle;
attribute vec2 tuv;
attribute float cIndex;
attribute float cNumber;

#!UNIFORMS
uniform sampler2D tPositions;
uniform sampler2D tLife;
uniform float radialSegments;
uniform float thickness;
uniform float taper;

#!VARYINGS
varying float vLength;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying vec3 vPos;
varying vec2 vUv;
varying float vIndex;
varying float vLife;

#!SHADER: NeutrinoTube.vs

#define PI 3.1415926535897932384626433832795

//neutrinoparams

#require(neutrino.glsl)
#require(range.glsl)
#require(conditionals.glsl)

void createTube(vec2 volume, out vec3 offset, out vec3 normal) {
    float posIndex = getIndex(cNumber, cIndex);
    float nextIndex = getIndex(cNumber, cIndex + 1.0);

    vLength = cIndex/(lineSegments-1.0);
    vIndex = cIndex;

    vec3 current = texture2D(tPositions, getUVFromIndex(posIndex)).xyz;
    vec3 next = texture2D(tPositions, getUVFromIndex(nextIndex)).xyz;

    vec3 T = normalize(next - current);
    vec3 B = normalize(cross(T, next + current));
    vec3 N = -normalize(cross(B, T));

    float tubeAngle = angle;
    float circX = cos(tubeAngle);
    float circY = sin(tubeAngle);

    volume *= mix(crange(vLength, 1.0 - taper, 1.0, 1.0, 0.0) * crange(vLength, 0.0, taper, 0.0, 1.0), 1.0, when_eq(taper, 0.0));

    normal.xyz = normalize(B * circX + N * circY);
    offset.xyz = current + B * volume.x * circX + N * volume.y * circY;
}

void main() {
    float headIndex = getIndex(cNumber, 0.0);
    float life = texture2D(tLife, getUVFromIndex(headIndex)).z;
    vLife = life;

    vec2 iuv = getUVFromIndex(headIndex);

    float scale = 1.0;
    //neutrinovs
    vec2 volume = vec2(thickness * 0.065 * scale);

    vec3 transformed;
    vec3 objectNormal;
    createTube(volume, transformed, objectNormal);

    vec3 transformedNormal = normalMatrix * objectNormal;
    vNormal = normalize(transformedNormal);
    vUv = tuv.yx;

    vec3 pos = transformed;
    vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
    vViewPosition = -mvPosition.xyz;
    vPos = pos;
    gl_Position = projectionMatrix * mvPosition;

    //neutrinovspost
}

#!SHADER: NeutrinoTube.fs
void main() {
    gl_FragColor = vec4(1.0);
}{@}neutrino.glsl{@}uniform sampler2D tIndices;
uniform sampler2D tActive;
uniform sampler2D tAttribs;
uniform float textureSize;
uniform float lineSegments;

vec2 getUVFromIndex(float index) {
    float size = textureSize;
    vec2 uv = vec2(0.0);
    float p0 = index / size;
    float y = floor(p0);
    float x = p0 - y;
    uv.x = x;
    uv.y = y / size;
    return uv;
}

float getIndex(float line, float chain) {
    return (line * lineSegments) + chain;
}{@}normalmap.glsl{@}vec3 unpackNormal( vec3 eye_pos, vec3 surf_norm, sampler2D normal_map, float intensity, float scale, vec2 uv ) {
    surf_norm = normalize(surf_norm);
    
    vec3 q0 = dFdx( eye_pos.xyz );
    vec3 q1 = dFdy( eye_pos.xyz );
    vec2 st0 = dFdx( uv.st );
    vec2 st1 = dFdy( uv.st );
    
    vec3 S = normalize( q0 * st1.t - q1 * st0.t );
    vec3 T = normalize( -q0 * st1.s + q1 * st0.s );
    vec3 N = normalize( surf_norm );
    
    vec3 mapN = texture2D( normal_map, uv * scale ).xyz * 2.0 - 1.0;
    mapN.xy *= intensity;
    mat3 tsn = mat3( S, T, N );
    return normalize( tsn * mapN );
}

//mvPosition.xyz, normalMatrix * normal, normalMap, intensity, scale, uv{@}PBR.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: PBR.vs

#require(pbr.vs)

void main() {
    setupPBR(position);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: PBR.fs

#require(pbr.fs)

void main() {
    gl_FragColor = getPBR();
}{@}pbr.fs{@}uniform sampler2D tBaseColor;
uniform vec2 uEnv;

uniform sampler2D tMRO;
uniform vec3 uMRO;

uniform sampler2D tNormal;
uniform vec2 uNormalScale;

uniform sampler2D tLUT;
uniform sampler2D tEnvDiffuse;
uniform sampler2D tEnvSpecular;
uniform float uHDR;

uniform vec3 uLightDirection;

const float PI = 3.14159265359;
const float PI2 = 6.28318530718;
const float RECIPROCAL_PI = 0.31830988618;
const float RECIPROCAL_PI2 = 0.15915494;
const float LOG2 = 1.442695;
const float EPSILON = 1e-6;
const float LN2 = 0.6931472;

const float ENV_LODS = 7.0;

float pDarken = 1.0;
vec3 pColor = vec3(1.0);

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vMPos;

vec4 SRGBtoLinear(vec4 srgb) {
    vec3 linOut = pow(srgb.xyz, vec3(2.2));
    return vec4(linOut, srgb.w);
}

vec4 RGBEToLinear(in vec4 value) {
    return vec4(value.rgb * exp2(value.a * 255.0 - 128.0), 1.0);
}

vec4 RGBMToLinear(in vec4 value) {
    float maxRange = 6.0;
    return vec4(value.xyz * value.w * maxRange, 1.0);
}

vec4 RGBDToLinear(in vec4 value, in float maxRange) {
    return vec4(value.rgb * ((maxRange / 255.0) / value.a), 1.0);
}

vec3 linearToSRGB(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

vec3 getNormal() {
    vec3 pos_dx = vec3(dFdx(vMPos.x), dFdx(vMPos.y), dFdx(vMPos.z));
    vec3 pos_dy = vec3(dFdy(vMPos.x), dFdy(vMPos.y), dFdy(vMPos.z));
    vec3 tex_dx = vec3(dFdx(vUv.x), dFdx(vUv.y), dFdx(0.0));
    vec3 tex_dy = vec3(dFdy(vUv.x), dFdy(vUv.y), dFdy(0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

    vec3 ng = normalize(vNormal);

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn = mat3(t, b, ng);

    vec3 n = texture2D(tNormal, vUv * uNormalScale.y).rgb;
    n = normalize(tbn * ((2.0 * n - 1.0) * vec3(uNormalScale.x, uNormalScale.x, 1.0)));

    return n;
}

vec3 specularReflection(vec3 specularEnvR0, vec3 specularEnvR90, float VdH) {
    return specularEnvR0 + (specularEnvR90 - specularEnvR0) * pow(clamp(1.0 - VdH, 0.0, 1.0), 5.0);
}

float geometricOcclusion(float NdL, float NdV, float roughness) {
    float r = roughness;

    float attenuationL = 2.0 * NdL / (NdL + sqrt(r * r + (1.0 - r * r) * (NdL * NdL)));
    float attenuationV = 2.0 * NdV / (NdV + sqrt(r * r + (1.0 - r * r) * (NdV * NdV)));
    return attenuationL * attenuationV;
}

float microfacetDistribution(float roughness, float NdH) {
    float roughnessSq = roughness * roughness;
    float f = (NdH * roughnessSq - NdH) * NdH + 1.0;
    return roughnessSq / (PI * f * f);
}

vec2 cartesianToPolar(vec3 n) {
    vec2 uv;
    uv.x = atan(n.z, n.x) * RECIPROCAL_PI2 + 0.5;
    uv.y = asin(n.y) * RECIPROCAL_PI + 0.5;
    return uv;
}

vec4 autoToLinear(vec4 texel) {
    vec4 rgbm = RGBMToLinear(texel);
    vec4 srgb = SRGBtoLinear(texel);
    return mix(srgb, rgbm, uHDR);
}

vec3 getIBLContribution(float NdV, float roughness, vec3 n, vec3 reflection, vec3 diffuseColor, vec3 specularColor) {
    vec3 brdf = SRGBtoLinear(texture2D(tLUT, vec2(NdV, roughness))).rgb;

    vec3 diffuseLight = autoToLinear(texture2D(tEnvDiffuse, cartesianToPolar(n))).rgb;

    // Sample 2 levels and mix between to get smoother degradation
    float blend = roughness * ENV_LODS;
    float level0 = floor(blend);
    float level1 = min(ENV_LODS, level0 + 1.0);
    blend -= level0;

    // Sample the specular env map atlas depending on the roughness value
    vec2 uvSpec = cartesianToPolar(n);
    uvSpec.y /= 2.0;

    vec2 uv0 = uvSpec;
    vec2 uv1 = uvSpec;

    uv0 /= pow(2.0, level0);
    uv0.y += 1.0 - exp(-LN2 * level0);

    uv1 /= pow(2.0, level1);
    uv1.y += 1.0 - exp(-LN2 * level1);

    vec3 specular0 = autoToLinear(texture2D(tEnvSpecular, uv0)).rgb;
    vec3 specular1 = autoToLinear(texture2D(tEnvSpecular, uv1)).rgb;
    vec3 specularLight = mix(specular0, specular1, blend);

    vec3 diffuse = diffuseLight * diffuseColor;
    vec3 specular = specularLight * (specularColor * brdf.x + brdf.y);

    // A value to be able to push the strength and mimic HDR
    specular *= (1.0 + uEnv.y * specularLight);
    return diffuse + specular;
}

vec3 getMRO() {
    return texture2D(tMRO, vUv).rgb;
}

vec3 calculatePBR(vec3 baseColor) {
    // rgb = [metallic, roughness, occlusion] - still have a available
    vec4 mroSample = texture2D(tMRO, vUv);
    float metallic = clamp(mroSample.r * uMRO.x, 0.04, 1.0);
    float roughness = clamp(mroSample.g * uMRO.y, 0.04, 1.0);

    vec3 diffuseColor = baseColor * 0.96 * (1.0 - metallic);
    vec3 specularColor = mix(vec3(0.04), baseColor, metallic);

    float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvR0 = specularColor.rgb;
    vec3 specularEnvR90 = vec3(reflectance90);

    vec3 N = getNormal();
    vec3 V = normalize(cameraPosition - vMPos);
    vec3 L = normalize(uLightDirection);
    vec3 H = normalize(L + V);
    vec3 reflection = -normalize(reflect(V, N));

    float NdL = clamp(dot(N, L), 0.001, 1.0);
    float NdV = clamp(abs(dot(N, V)), 0.001, 1.0);
    float NdH = clamp(dot(N, H), 0.0, 1.0);
    float LdH = clamp(dot(L, H), 0.0, 1.0);
    float VdH = clamp(dot(V, H), 0.0, 1.0);

    vec3 F = specularReflection(specularEnvR0, specularEnvR90, VdH);
    float G = geometricOcclusion(NdL, NdV, roughness);
    float D = microfacetDistribution(roughness, NdH);

    vec3 diffuseContrib = (1.0 - F) * (diffuseColor / PI);
    vec3 specContrib = F * G * D / (4.0 * NdL * NdV);
    vec3 color = NdL * (diffuseContrib + specContrib) * pDarken;

    color += getIBLContribution(NdV, roughness, N, reflection, diffuseColor, specularColor) * pColor * uEnv.x;

    return mix(color, color * mroSample.b, uMRO.z);
}

vec4 getPBR() {
    vec4 baseColor = SRGBtoLinear(texture2D(tBaseColor, vUv));
    vec3 color = calculatePBR(baseColor.rgb);
    return vec4(linearToSRGB(color), baseColor.a);
}


vec4 getPBR(vec3 inputColor) {
    vec4 baseColor = SRGBtoLinear(vec4(inputColor, 1.0));
    vec3 color = calculatePBR(baseColor.rgb);
    return vec4(linearToSRGB(color), 1.0);
}{@}pbr.vs{@}uniform sampler2D tBaseColor;
uniform vec3 uBaseColor;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vMPos;

void setupPBR(vec3 pos) {
    vUv = uv;
    vec4 mPos = modelMatrix * vec4(pos, 1.0);
    vMPos = mPos.xyz / mPos.w;
    vNormal = normalMatrix * normal;
}{@}range.glsl{@}float range(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

float crange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(range(oldValue, oldMin, oldMax, newMin, newMax), min(newMax, newMin), max(newMin, newMax));
}{@}rgb2hsv.fs{@}vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}{@}rgbshift.fs{@}vec4 getRGB(sampler2D tDiffuse, vec2 uv, float angle, float amount) {
    vec4 texel;

    #test Tests.rgbShift()
    vec2 offset = vec2(cos(angle), sin(angle)) * amount;
    vec4 r = texture2D(tDiffuse, uv + offset);
    vec4 g = texture2D(tDiffuse, uv);
    vec4 b = texture2D(tDiffuse, uv - offset);
    texel = vec4(r.r, g.g, b.b, g.a);
    #endtest

    #test !Tests.rgbShift()
    texel = texture2D(tDiffuse, uv);
    #endtest

    return texel;
}{@}rotation.glsl{@}mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}{@}SceneLayout.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;
uniform float uAlpha;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= texture2D(tMask, vUv).r * uAlpha;
}{@}simplenoise.glsl{@}float getNoise(vec2 uv, float time) {
    float x = uv.x * uv.y * time * 1000.0;
    x = mod(x, 13.0) * mod(x, 123.0);
    float dx = mod(x, 0.01);
    float amount = clamp(0.1 + dx * 100.0, 0.0, 1.0);
    return amount;
}

highp float random(vec2 co) {
    highp float a = 12.9898;
    highp float b = 78.233;
    highp float c = 43758.5453;
    highp float dt = dot(co.xy, vec2(a, b));
    highp float sn = mod(dt, 3.14);
    return fract(sin(sn) * c);
}

float cnoise(vec3 v) {
    float t = v.z * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += range(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += range(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}

float cnoise(vec2 v) {
    float t = v.x * 0.3;
    v.y *= 0.8;
    float noise = 0.0;
    float s = 0.5;
    noise += range(sin(v.x * 0.9 / s + t * 10.0) + sin(v.x * 2.4 / s + t * 15.0) + sin(v.x * -3.5 / s + t * 4.0) + sin(v.x * -2.5 / s + t * 7.1), -1.0, 1.0, -0.3, 0.3);
    noise += range(sin(v.y * -0.3 / s + t * 18.0) + sin(v.y * 1.6 / s + t * 18.0) + sin(v.y * 2.6 / s + t * 8.0) + sin(v.y * -2.6 / s + t * 4.5), -1.0, 1.0, -0.3, 0.3);
    return noise;
}{@}simplex2d.glsl{@}//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
    return mod289(((x*34.0)+1.0)*x);
}

float snoise(vec2 v)
{
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    
    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    
    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
                     + i.x + vec3(0.0, i1.x, 1.0 ));
    
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    
    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
    
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    
    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    
    // Compute final noise value at P
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}{@}simplex3d.glsl{@}// Description : Array and textureless GLSL 2D/3D/4D simplex
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v) {
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    i = mod289(i);
    vec4 p = permute( permute( permute(
          i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
        + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

//float surface(vec3 coord) {
//    float n = 0.0;
//    n += 1.0 * abs(snoise(coord));
//    n += 0.5 * abs(snoise(coord * 2.0));
//    n += 0.25 * abs(snoise(coord * 4.0));
//    n += 0.125 * abs(snoise(coord * 8.0));
//    float rn = 1.0 - n;
//    return rn * rn;
//}{@}transformUV.glsl{@}vec2 transformUV(vec2 uv, float a[9]) {

    // Convert UV to vec3 to apply matrices
	vec3 u = vec3(uv, 1.0);

    // Array consists of the following
    // 0 translate.x
    // 1 translate.y
    // 2 skew.x
    // 3 skew.y
    // 4 rotate
    // 5 scale.x
    // 6 scale.y
    // 7 origin.x
    // 8 origin.y

    // Origin before matrix
    mat3 mo1 = mat3(
        1, 0, -a[7],
        0, 1, -a[8],
        0, 0, 1);

    // Origin after matrix
    mat3 mo2 = mat3(
        1, 0, a[7],
        0, 1, a[8],
        0, 0, 1);

    // Translation matrix
    mat3 mt = mat3(
        1, 0, -a[0],
        0, 1, -a[1],
    	0, 0, 1);

    // Skew matrix
    mat3 mh = mat3(
        1, a[2], 0,
        a[3], 1, 0,
    	0, 0, 1);

    // Rotation matrix
    mat3 mr = mat3(
        cos(a[4]), sin(a[4]), 0,
        -sin(a[4]), cos(a[4]), 0,
    	0, 0, 1);

    // Scale matrix
    mat3 ms = mat3(
        1.0 / a[5], 0, 0,
        0, 1.0 / a[6], 0,
    	0, 0, 1);

	// apply translation
   	u = u * mt;

	// apply skew
   	u = u * mh;

    // apply rotation relative to origin
    u = u * mo1;
    u = u * mr;
    u = u * mo2;

    // apply scale relative to origin
    u = u * mo1;
    u = u * ms;
    u = u * mo2;

    // Return vec2 of new UVs
    return u.xy;
}

vec2 rotateUV(vec2 uv, float r, vec2 origin) {
    vec3 u = vec3(uv, 1.0);

    mat3 mo1 = mat3(
        1, 0, -origin.x,
        0, 1, -origin.y,
        0, 0, 1);

    mat3 mo2 = mat3(
        1, 0, origin.x,
        0, 1, origin.y,
        0, 0, 1);

    mat3 mr = mat3(
        cos(r), sin(r), 0,
        -sin(r), cos(r), 0,
        0, 0, 1);

    u = u * mo1;
    u = u * mr;
    u = u * mo2;

    return u.xy;
}

vec2 rotateUV(vec2 uv, float r) {
    return rotateUV(uv, r, vec2(0.5));
}

vec2 translateUV(vec2 uv, vec2 translate) {
    vec3 u = vec3(uv, 1.0);
    mat3 mt = mat3(
        1, 0, -translate.x,
        0, 1, -translate.y,
        0, 0, 1);

    u = u * mt;
    return u.xy;
}

vec2 scaleUV(vec2 uv, vec2 scale, vec2 origin) {
    vec3 u = vec3(uv, 1.0);

    mat3 mo1 = mat3(
        1, 0, -origin.x,
        0, 1, -origin.y,
        0, 0, 1);

    mat3 mo2 = mat3(
        1, 0, origin.x,
        0, 1, origin.y,
        0, 0, 1);

    mat3 ms = mat3(
        1.0 / scale.x, 0, 0,
        0, 1.0 / scale.y, 0,
        0, 0, 1);

    u = u * mo1;
    u = u * ms;
    u = u * mo2;
    return u.xy;
}

vec2 scaleUV(vec2 uv, vec2 scale) {
    return scaleUV(uv, scale, vec2(0.5));
}
{@}LightBlur.fs{@}uniform vec2 uResolution;
uniform vec2 uDir;

#require(gaussianblur.fs)

void main() {
    gl_FragColor = blur9(tDiffuse, vUv, uResolution, uDir);
}{@}VolumetricLight.fs{@}uniform vec2 lightPos;
uniform float fExposure;
uniform float fDecay;
uniform float fDensity;
uniform float fWeight;
uniform float fClamp;

const int iSamples = 20;

void main() {
    vec2 deltaTextCoord = vUv - lightPos;
    deltaTextCoord *= 1.0  / float(iSamples) * fDensity;
    vec2 coord = vUv;

    float illuminationDecay = 1.0;
    vec4 color = vec4(0.0);

    for (int i = 0; i < iSamples; i++) {
        coord -= deltaTextCoord;
        vec4 texel = texture2D(tDiffuse, coord);
        texel *= illuminationDecay * fWeight;

        color += texel;
        illuminationDecay *= fDecay;
    }

    color *= fExposure;
    color = clamp(color, 0.0, fClamp);
    gl_FragColor = color;
}{@}HomeListComposite.fs{@}uniform sampler2D tDepth;
uniform float uDist;
uniform float uStrength;
uniform float uGlobalStrength;
uniform float uGlobalBoost;
uniform vec4 uStrengthMap;
uniform vec4 uDarkMap;

#require(depthvalue.fs)
#require(range.glsl)
#require(cheapblur.fs)

void main() {
    float depth = getDepthValue(tDepth, vUv, 0.01, 1.0);
    float strength = crange(depth, uStrengthMap.x, uStrengthMap.y, uStrengthMap.z, uStrengthMap.w);
    float darken = crange(depth, uDarkMap.x, uDarkMap.y, uDarkMap.z, uDarkMap.w);
    gl_FragColor = blur(tDiffuse, vUv, uDist * .1, strength * uStrength + uGlobalStrength) * darken * uGlobalBoost;
}{@}HomeParticleShader.glsl{@}#!ATTRIBUTES
attribute vec4 random;

#!UNIFORMS
uniform float DPR;
uniform float uSize;
uniform float uAlpha;
uniform sampler2D tMap;
uniform vec3 uColor;

#!VARYINGS
varying float vAlpha;

#!SHADER: Vertex

#require(range.glsl)

void main() {
    vec3 pos = getPos();
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    vec3 worldPos = vec3(modelMatrix * vec4(pos, 1.0));

    vAlpha = random.w;
    vAlpha *= crange(pos.z, 0.8, 1.0, 1.0, 0.0) * crange(pos.z, 0.0, 0.2, 0.0, 1.0);
    vAlpha *= crange(length(worldPos - cameraPosition), 0.5, 2.0, 0.0, 1.0);

    gl_PointSize = 0.02 * DPR * crange(random.z, 0.0, 1.0, 0.5, 1.5) * uSize * (1000.0 / length(mvPosition.xyz));
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: Fragment
void main() {
    vec2 uv = gl_PointCoord.xy;
    float mask = texture2D(tMap, uv).r;
    gl_FragColor = vec4(uColor, vAlpha * mask * uAlpha);
}{@}PBRElement.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform float uBaseDarken;
uniform float uLightAffect;
uniform float uPostDarken;
uniform float uFlipNormal;
uniform float uLightDist;
uniform float uBackLit;
uniform float uShadowBias;
uniform float uLogoAffect;
uniform float uWind;
uniform vec2 uFade;
uniform vec2 uRough;

#!VARYINGS
varying vec3 vPos;
varying vec3 vWorldPos;

#!SHADER: Vertex

#require(pbr.vs)
#require(lighting.vs)
#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 pos = position;
    if (uWind > 0.0) pos += cnoise(pos*0.3 + time*0.15) * uWind;

    setupPBR(pos);
    setupLight(pos);

    vPos = pos;

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vWorldPos = worldPos.xyz;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: Fragment

#require(pbr.fs)
#require(shadows.fs)
#require(lighting.fs)
#require(rgb2hsv.fs)
#require(range.glsl)
#require(simplenoise.glsl)
#require(conditionals.glsl)

vec3 getColor(vec3 mro) {
    vec3 color;
    vec3 baseColor = texture2D(tBaseColor, vUv).rgb * uBaseDarken;

    #test Tests.usePBR()
    color = getPBR(baseColor).rgb;
    #endtest

    #test !Tests.usePBR()
    color = baseColor * 0.1;
    #endtest

    return color;
}

void main() {
    pDarken = 0.1;
    lNormal = getNormal() * uFlipNormal;
    lPhong = true;
    lAreaToPoint = true;
    pColor = lightColor[0].rgb;

    //pColor *= 1.0 + uLogoAffect*0.5*sin(time*50.0)*sin(time*12.0) * smoothstep(-0.7, 0.0, vWorldPos.y) * smoothstep(1.0, 0.0, vWorldPos.y);


    lPhongColor = pColor;

    #test Tests.useLambert()
    lPhong = false;
    #endtest

    vec3 mro = getMRO();
    mro.y = crange(mro.y, uRough.x, uRough.y, 0.0, 1.0);
    lPhongShininess = mro.y;
    lPhongColor += mro.y;




    vec3 color = getColor(mro);

    vec3 areaColor = rgb2hsv(getAreaLightColor(mro.y));
    areaColor.x += 0.02 * cnoise(vPos * 0.01);
    areaColor = hsv2rgb(areaColor);
    areaColor *= pow(crange(length(vWorldPos), 0.0, uLightDist, 1.0, 0.0), 1.3) * smoothstep(0.0, 0.5, uBackLit * vWorldPos.z);

    #test Tests.usePBR()
    areaColor *= mro.y;
    #endtest

    color += areaColor * uLightAffect;
    color += 0.0125 * when_lt(vWorldPos.z, 0.0);

    //color = mix(color, areaColor, uLogoAffect);



//    color *= getShadow(vPos, 1.0*uShadowBias / 2048.0);
    color *= uPostDarken * crange(vWorldPos.z, uFade.x, uFade.y, 1.0, 0.0);

    gl_FragColor = vec4(color, 1.0);
}
{@}PBRGround.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform mat4 uMirrorMatrix;
uniform sampler2D tMirrorReflection;
//uniform sampler2D tMask;
uniform float uMirrorStrength;
uniform float uAreaStrength;
uniform float uMirrorRough;
uniform float uShadowStrength;
uniform float uBaseDarken;
uniform float uPostDarken;
uniform float uNoise;
uniform vec2 uFade;
uniform vec2 uRough;
uniform float uLogoAffect;

#!VARYINGS
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vViewDir;
varying vec4 vMirrorCoord;
varying float vStrength;

#!SHADER: Vertex

#require(pbr.vs)
#require(lighting.vs)
#require(range.glsl)

void main() {
    setupLight(position);
    setupPBR(position);

    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;

    vPos = position;
    vStrength = crange(length(worldPos), 0.0, 2.0, 5.0, 1.0);
    vWorldPos = worldPos.xyz;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    vViewDir = -mvPosition.xyz;
}

#!SHADER: Fragment

#require(pbr.fs)
#require(lighting.fs)
#require(range.glsl)
#require(shadows.fs)
#require(rgb2hsv.fs)
#require(simplenoise.glsl)
#require(conditionals.glsl)
#require(lights.fs)

vec3 getColor(vec3 mro) {
    vec3 color;
    vec3 baseColor = texture2D(tBaseColor, vUv).rgb * uBaseDarken;

    #test Tests.usePBR()
    color = getPBR(baseColor).rgb;
    #endtest

    #test !Tests.usePBR()
    color = baseColor * 0.1;
    #endtest

    return color;
}

vec3 getAreaColor(vec3 mro) {
    #test !Tests.useAreaLight()
    lAreaToPoint = true;
    #endtest

    vec3 areaColor = rgb2hsv(getAreaLightColor(mro.y));
    areaColor.x += 0.02 * cnoise(vPos * 0.01);
    areaColor = hsv2rgb(areaColor);

    #test !Tests.useAreaLight()
    float worldLen = length(vWorldPos);
    areaColor *= crange(worldLen, 0.0, 3.0, 1.0, 0.0) * smoothstep(0.0, 0.5, vWorldPos.z);
    areaColor *= crange(worldLen, 0.0, 1.4, 1.6, 1.0);
    #endtest

    return areaColor;
}

void main() {
    setupLight();

    lPhong = true;
    lNormal = getNormal();
    pDarken = 0.1;
    pColor = lightColor[0].rgb;

    #test Tests.useLambert()
    lPhong = false;
    #endtest

    vec3 mro = getMRO();
    mro.y = crange(mro.y, uRough.x, uRough.y, 0.0, 1.0);

    lPhongShininess = mro.y;
    lPhongColor += mro.y * 2.0;

    vec3 baseColor = texture2D(tBaseColor, vUv).rgb * uBaseDarken;

    vec3 areaColor = getAreaColor(mro);

    vec3 color = getColor(mro);

    color += areaColor * uAreaStrength * vStrength;

    #test Tests.renderMirror()
    vec4 mirrorCoord = vMirrorCoord;
    mirrorCoord.xz += lNormal.xy * uMirrorRough * 0.1;

    vec3 mirror = texture2DProj(tMirrorReflection, mirrorCoord).rgb;
    color += mirror * uMirrorStrength * crange(1.0 - mro.y, 0.0, 1.0, 0.2, 1.0);
    #endtest


    //color += uLogoAffect;


//    color *= mix(1.0, getShadow(vPos, 1.0 / 2048.0), uShadowStrength);
    color *= uPostDarken * crange(vWorldPos.z, uFade.x, uFade.y, 1.0, 0.0);
//    color *= texture2D(tMask, vUv).r * mro.z;
    color *= mro.z;
    gl_FragColor = vec4(color, 1.0);
}{@}SceneBG.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMask;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    float mask = texture2D(tMask, vUv).r;
    gl_FragColor = vec4(uColor * mask, 1.0);
}{@}SceneComposite.fs{@}uniform sampler2D tGlow;
uniform sampler2D tGlowMask;
uniform sampler2D tDepth;
uniform sampler2D tLogo;
uniform float uGlowStrength;
uniform float uFogStrength;
uniform float uFogBrightness;
uniform float uRGBStrength;
uniform float uDistortion;
uniform float uDesat;
uniform float uVignette;
uniform float uOcclusion;
uniform float uFar;
uniform float uDPR;
uniform vec2 uHue;
uniform vec2 uBackDarken;
uniform vec2 uFog;
uniform vec3 uFogColor;
uniform float uLogoAffect;
uniform float uSpaceHold;
uniform float uHome;
uniform float uIntro;
uniform float uPhone;

#require(range.glsl)
#require(simplenoise.glsl)
#require(depthvalue.fs)
#require(rgb2hsv.fs)
#require(rgbshift.fs)
#require(transformUV.glsl)
#require(blendmodes.glsl)
#require(eases.glsl)
#require(mousefluid.fs)

void main() {
    float dist = length(vUv - vec2(0.5));
    float scale = crange(dist, 0.2, 0.5, 1.0, 1.0 + (0.01 * uDistortion));
    float warpNoise = crange(cnoise(vec3(vUv*3.0, time*0.2)), -1.0, 1.0, 0.0, 1.0);
    float bulge = smoothstep(0.7, 0.35, dist)*(0.15 + sin(time) * 0.04 + warpNoise*0.02) * 0.8;
    scale = mix(scale, 1.05+bulge, uLogoAffect*0.5);

    scale += uIntro*0.12;

    float extraRGBStrength = crange(dist, 0.2, 0.5, 1.0, 2.0);
    vec2 uv = scaleUV(vUv, vec2(scale));

    vec3 fluidPlusMask = getFluidVelocityMask();
    vec2 fluid = fluidPlusMask.xy;
    float fluidMask = fluidPlusMask.z;
    uv += fluid * 0.0001 * uLogoAffect * fluidMask;
    extraRGBStrength += fluidMask*0.1;

    vec4 logoDistortion = texture2D(tLogo, vUv);
    logoDistortion.x = crange(logoDistortion.x, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    logoDistortion.y = crange(logoDistortion.y, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    uv += 0.1 * logoDistortion.xy * uLogoAffect;

    vec3 glow = vec3(0.0);
    float glowMask = 0.0;
    float liquidRGB = smoothstep(0.2,1.0,fluidMask)*0.002*uLogoAffect;
    vec3 color = getRGB(tDiffuse, uv, 0.3, 0.001 * uRGBStrength * extraRGBStrength + liquidRGB + uLogoAffect*0.001).rgb;

    #test Tests.screenGlow()
    glow = texture2D(tGlow, uv).rgb;
    glowMask = texture2D(tGlowMask, uv).a;
    #endtest

    float depth = getDepthValue(tDepth, uv, 0.1, uFar);
    float backDarken = crange(depth, uBackDarken.x, uBackDarken.y, 1.0, 0.0);

    vec3 fogColor = rgb2hsv(uFogColor);
    fogColor.x += 0.1 * cnoise(vec3(vUv * 0.05, time*0.2));
    fogColor.y = mix(fogColor.y, fogColor.y*1.5, uSpaceHold);

    float noise = cnoise(vec3(vUv*1.2, time*0.2));
    fogColor.z = uFogBrightness + noise*0.3;
    fogColor.z *= 1.0-uIntro*0.8;
    fogColor = hsv2rgb(fogColor);

    float fog = crange(depth, uFog.x, uFog.y, 0.0, 1.0);
    float fogBlend = fog * crange(glowMask, 0.0, 1.0, 0.18, 1.0) * uFogStrength * crange(backDarken, 0.0, 1.0, 0.2, 1.0);
    color = mix(color, fogColor, fogBlend*1.5);
    color += glow * crange(glowMask, 0.0, 1.0, 0.5, 1.0) * (uGlowStrength + uLogoAffect*0.2) * (1.0 - smoothstep(depth, depth+0.1, uOcclusion));
    color *= mix(1.0, getNoise(vUv * uDPR, time), fogBlend*0.2);
    //color *= sineOut(crange(dist, 0.0, 0.5, 1.0, uVignette));

    color += 0.1 * logoDistortion.a * crange(logoDistortion.z, 0.0, 1.0, 0.5, 1.0) * uLogoAffect;

    float staticNoise = range(getNoise(vUv * uDPR * 3., time*0.1), 0.0, 1.0, -1.0, 1.0);
    color += staticNoise*0.01;

    color = mix(color, blendMultiply(color, uFogColor), mix(0.1 + (1.0-uHome)*0.9, 1.0, uSpaceHold));
    color = mix(color, blendOverlay(color*mix(0.6, 0.8, uHome)*mix(1.0, 0.8, uPhone), uFogColor), mix(mix(0.5 + (1.0-uHome)*0.6, 1.0, uLogoAffect), 0.7, uSpaceHold));
    color = mix(color, blendSoftLight(color, uFogColor), smoothstep(0.7, 1.0, uSpaceHold)*0.5);
    color = mix(color, blendOverlay(color, uFogColor), sineIn(crange(dist, mix(0.3, 0.22, uPhone), 0.8, 0.0, mix(1.0, 1.0, uPhone))));

    color = rgb2hsv(color);
    color.y *= 1.0-uIntro*0.9;
    color.x += crange(vUv.x+vUv.y, 0.0, 2.0, -0.05, 0.05);
    color.x += sineIn(crange(dist, 0.1, 0.8, 0.0, 0.25));
    color.z = mix(color.z, pow(color.z, 2.0), uIntro*smoothstep(0.0, 0.5, dist));
    color = hsv2rgb(color);

    vec3 gray = vec3(0.5,0.5,0.5);
    color = ((color - gray) * 1.04) + gray; // increase contrast

    gl_FragColor = vec4(color, 1.0);
}{@}PBRForestSnow.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform float uBaseDarken;
uniform float uNoiseScale;
uniform float uSparkle;
uniform float uSparkleStep;
uniform float uFresnel;
uniform float uPow;
uniform float uFPow;
uniform float uPostDarken;

#!VARYINGS

#!SHADER: Vertex
#require(pbr.vs)
#require(lighting.vs)

void main() {
    setupLight(position);
    setupPBR(position);

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vPos = position;
    vViewDir = -mvPosition.xyz;
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: Fragment

#require(pbr.fs)
#require(lighting.fs)
#require(rgb2hsv.fs)
#require(range.glsl)
#require(simplenoise.glsl)

float getFresnel(vec3 normal, vec3 viewDir, float power) {
    float d = dot(normalize(normal), normalize(viewDir));
    return 1.0 - pow(abs(d), power);
}

vec3 getColor(vec3 mro) {
    vec3 color;
    vec3 baseColor = texture2D(tBaseColor, vUv).rgb * uBaseDarken;

    #test Tests.usePBR()
    color = getPBR(baseColor).rgb;
    #endtest

    #test !Tests.usePBR()
    color = baseColor * 0.1;
    #endtest

    return color;
}

vec3 getAreaColor(vec3 mro) {
    #test !Tests.useAreaLight()
    lAreaToPoint = true;
    #endtest

    vec3 areaColor = rgb2hsv(getAreaLightColor(10.0));
    areaColor.x += 0.02 * cnoise(vPos * 0.01);
    areaColor = hsv2rgb(areaColor);

    #test !Tests.useAreaLight()
    float worldLen = length(vWorldPos);
    areaColor *= crange(worldLen, 0.0, 3.0, 1.0, 0.0) * smoothstep(0.0, 0.5, vWorldPos.z);
    areaColor *= crange(worldLen, 0.0, 1.4, 1.6, 1.0);
    #endtest

    return areaColor;
}

void main() {
    setupLight();

    lNormal = getNormal();
    vec3 mro = getMRO();

    vec3 normal2;
    float scale = uNoiseScale;
    normal2.x = cnoise(vPos.xx * scale);
    normal2.y = cnoise(vPos.yy * scale);
    normal2.z = cnoise(vPos.zz * scale);

//    lPhongShininess = mro.y;
//    lPhong = true;

    vec3 light = getAreaColor(mro);
    light *= mix(1.0, uSparkle, step(uSparkleStep * 0.1, getFresnel(normal2, vViewDir, uPow)));
    light *= mix(1.0, uFresnel, getFresnel(lNormal, vViewDir, uFPow));
    light *= mro.z * uPostDarken;

    gl_FragColor = vec4(light, 1.0);
}{@}PBRSnow.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform float uNoiseScale;
uniform float uPow;
uniform float uFPow;
uniform float uBaseDarken;
uniform float uPostDarken;
uniform float uSparkle;
uniform float uSparkleStep;
uniform float uFresnel;
uniform vec2 uFade;
uniform vec2 uRough;

#!VARYINGS
varying vec3 vPos;
varying vec3 vViewDir;

#!SHADER: Vertex

#require(pbr.vs)
#require(lighting.vs)

void main() {
    setupLight(position);
    setupPBR(position);

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vPos = position;
    vViewDir = -mvPosition.xyz;
    vNormal = normalMatrix * normal;
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(pbr.fs)
#require(lighting.fs)
#require(rgb2hsv.fs)

float getFresnel(vec3 normal, vec3 viewDir, float power) {
    float d = dot(normalize(normal), normalize(viewDir));
    return 1.0 - pow(abs(d), power);
}

vec3 getColor(vec3 mro) {
    vec3 color;
    vec3 baseColor = texture2D(tBaseColor, vUv).rgb * uBaseDarken;

    #test Tests.usePBR()
    color = getPBR(baseColor).rgb;
    #endtest

    #test !Tests.usePBR()
    color = baseColor * 0.1;
    #endtest

    return color;
}

vec3 getAreaColor(vec3 mro) {
    #test !Tests.useAreaLight()
    lAreaToPoint = true;
    #endtest

    vec3 areaColor = rgb2hsv(getAreaLightColor(mro.y));
    areaColor.x += 0.02 * cnoise(vPos * 0.01);
    areaColor = hsv2rgb(areaColor);

    #test !Tests.useAreaLight()
    float worldLen = length(vWorldPos);
    areaColor *= crange(worldLen, 0.0, 3.0, 1.0, 0.0) * smoothstep(0.0, 0.5, vWorldPos.z);
    areaColor *= crange(worldLen, 0.0, 1.4, 1.6, 1.0);
    #endtest

    return areaColor;
}

vec3 getNormal2() {
    vec3 pos_dx = vec3(dFdx(vMPos.x), dFdx(vMPos.y), dFdx(vMPos.z));
    vec3 pos_dy = vec3(dFdy(vMPos.x), dFdy(vMPos.y), dFdy(vMPos.z));
    vec3 tex_dx = vec3(dFdx(vUv.x), dFdx(vUv.y), dFdx(0.0));
    vec3 tex_dy = vec3(dFdy(vUv.x), dFdy(vUv.y), dFdy(0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) / (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t);

    vec3 ng = normalize(vNormal);

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn = mat3(t, b, ng);

    vec3 n = vec3(0.0);
    n = normalize(tbn * ((2.0 * n - 1.0) * vec3(uNormalScale.x, uNormalScale.x, 1.0)));

    return n;
}

void main() {
    setupLight();

    lNormal = getNormal2();
    vec3 normal = getNormal();

    vec3 normal2;
    float scale = uNoiseScale;
    normal2.x = cnoise(vPos.xx * scale);
    normal2.y = cnoise(vPos.yy * scale);
    normal2.z = cnoise(vPos.zz * scale);

    vec3 mro = getMRO();

    lPhongShininess = mro.y;
    lPhong = true;

    vec3 light = getAreaColor(mro);
    light *= mix(1.0, uSparkle, step(uSparkleStep * 0.1, getFresnel(normal2, vViewDir, uPow)));
    light *= mix(1.0, uFresnel, getFresnel(normal, vViewDir, uFPow));
    light *= uPostDarken * crange(vWorldPos.z, uFade.x, uFade.y, 1.0, 0.0);

    gl_FragColor = vec4(light, 1.0);
}{@}ForestComposite.fs{@}uniform sampler2D tGlow;
uniform sampler2D tGlowMask;
uniform sampler2D tDepth;
uniform sampler2D tLogo;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform float uGlowStrength;
uniform float uFogStrength;
uniform float uFogBrightness;
uniform float uRGBStrength;
uniform float uDistortion;
uniform float uDesat;
uniform float uVignette;
uniform vec2 uHue;
uniform vec2 uBackDarken;
uniform vec2 uFog;
uniform vec3 uFogColor;
uniform float uLogoAffect;

#require(range.glsl)
#require(simplenoise.glsl)
#require(depthvalue.fs)
#require(rgb2hsv.fs)
#require(rgbshift.fs)
#require(transformUV.glsl)
#require(eases.glsl)

void main() {
    float dist = length(vUv - vec2(0.5));
    float scale = crange(dist, 0.2, 0.5, 1.0, 1.0 + (0.01 * uDistortion));
    scale = mix(scale, 1.2, uLogoAffect);

    float extraRGBStrength = crange(dist, 0.2, 0.5, 1.0, 2.0);
    vec2 uv = scaleUV(vUv, vec2(scale));

    float fluidMask = 0.0;
    #test Tests.mouseFluid()
    fluidMask = texture2D(tFluidMask, vUv).r;
    vec2 fluid = texture2D(tFluid, vUv).xy * fluidMask;
    uv += fluid * 0.0002 * uLogoAffect;
    extraRGBStrength += fluidMask*0.1;
    #endtest

    vec4 logoDistortion = texture2D(tLogo, vUv);
    logoDistortion.x = crange(logoDistortion.x, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    logoDistortion.y = crange(logoDistortion.y, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    uv += 0.1 * logoDistortion.xy * uLogoAffect;

    vec3 glow = vec3(0.0);
    float glowMask = 1.0;
    float liquidRGB = smoothstep(0.2,1.0,fluidMask)*0.005*uLogoAffect;
    vec3 color = getRGB(tDiffuse, uv, 0.3, 0.001 * uRGBStrength * extraRGBStrength + liquidRGB).rgb;

    #test Tests.screenGlow()
        glow = texture2D(tGlow, uv).rgb;
        glowMask = texture2D(tGlowMask, uv).a;
    #endtest

    float depth = getDepthValue(tDepth, uv, 0.1, 5.0);

    float backDarken = crange(depth, uBackDarken.x, uBackDarken.y, 1.0, 0.0);

    vec3 fogColor = rgb2hsv(uFogColor);
    fogColor.x += uHue.x * cnoise(vUv * uHue.y);
    fogColor.y -= uDesat;
    fogColor.z = uFogBrightness;
    fogColor = hsv2rgb(fogColor);

    float fog = crange(depth, uFog.x, uFog.y, 0.0, 1.0);
    float fogBlend = fog * crange(glowMask, 0.0, 1.0, 0.25, 1.0) * uFogStrength * crange(backDarken, 0.0, 1.0, 0.2, 1.0);
    color = mix(color, fogColor, fogBlend);

    color += glow * crange(glowMask, 0.0, 1.0, 0.5, 1.0) * (uGlowStrength + uLogoAffect*0.2) * step(0.6, depth);

    color *= mix(1.0, getNoise(vUv, time), fogBlend);
    color *= sineOut(crange(dist, 0.0, 0.5, 1.0, uVignette));

    color += 0.075 * logoDistortion.a * crange(logoDistortion.z, 0.0, 1.0, 0.5, 1.0) * uLogoAffect;

    gl_FragColor = vec4(color, 1.0);
}{@}HomeLogoNormal.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec3 vNormal;
varying vec2 vUv;

#!SHADER: Vertex

#require(range.glsl)

void main() {
    vUv = uv;

    vNormal = normalMatrix * normal;
    vNormal.x = crange(vNormal.x, -1.0, 1.0, 0.0, 1.0);
    vNormal.y = crange(vNormal.y, -1.0, 1.0, 0.0, 1.0);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec2 uv = vUv;
    uv *= 1.0+cnoise(vec3(vUv, time*0.2))*0.3;
    float depth = texture2D(tMap, uv).r;
    gl_FragColor = vec4(vNormal.xy, smoothstep(0.1, 1.0, depth), 1.0);
    gl_FragColor.rgb += 0.1;
}{@}HomeWater.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform mat4 uMirrorMatrix;
uniform sampler2D tMirrorReflection;
uniform sampler2D tWaterNormal;
uniform sampler2D tMask;
uniform float uMirrorStrength;
uniform float uMirrorDistort;
uniform float uAreaStrength;
uniform float uNormalDistort;
uniform float uWaterSpeed;
uniform float uUVScale;
uniform float uDPR;
uniform float uNormalBlend;

#!VARYINGS
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vMVPos;
varying vec3 vWorldPos;
varying vec3 vPos;
varying vec4 vMirrorCoord;

#!SHADER: Vertex

#require(water.vs)
#require(lighting.vs)

void main() {
    setupLight(position);
    vec3 pos = calculateWaterPos();

    vec4 worldPos = modelMatrix * vec4(pos, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;
    vUv = uv;
    vPos = pos;
    vWorldPos = worldPos.xyz;

    vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mvPos;

    vMVPos = mvPos.xyz;
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(conditionals.glsl)
#require(lighting.fs)
#require(rgb2hsv.fs)

vec4 getNoise(vec2 uv){
    float time = time * 0.1 * uWaterSpeed;
    vec2 uv0 = (uv/103.0)+vec2(time/17.0, time/29.0);
    vec2 uv1 = uv/107.0-vec2(time/-19.0, time/31.0);
    vec2 uv2 = uv/vec2(897.0, 983.0)+vec2(time/101.0, time/97.0);
    vec2 uv3 = uv/vec2(991.0, 877.0)-vec2(time/109.0, time/-113.0);
    float scale = uUVScale;
    vec4 noise = (texture2D(tWaterNormal, uv0 * scale)) +
                 (texture2D(tWaterNormal, uv1 * scale)) +
                 (texture2D(tWaterNormal, uv2 * scale)) +
                 (texture2D(tWaterNormal, uv3 * scale));
    return noise*0.5-1.0;
}

void main() {
    setupLight();
    lAreaToPoint = true;

    float mask = texture2D(tMask, vUv).r;

    vec4 noise = getNoise(vUv * uDPR);
    vec3 waternormal = normalize(noise.xzy * vec3(2.0, 1.0, 2.0));

//    vec2 waternormals = texture2D(tWaterNormal, vUv * uUVScale + time * 0.1).xy;
//    waternormals.x = range(waternormals.x, 0.0, 1.0, -1.0, 1.0);
//    waternormals.y = range(waternormals.y, 0.0, 1.0, -1.0, 1.0);

    lNormal = normalize(vNormal + waternormal);

    vec4 mirrorCoord = vMirrorCoord;
    mirrorCoord.xz += waternormal.xy * 0.1 * uMirrorDistort;
    mirrorCoord.xz += vNormal.xy * 0.1 * uNormalDistort;

    vec3 areaColor = rgb2hsv(getAreaLightColor());
    areaColor.x += 0.02 * cnoise(vPos * 0.002);
    areaColor = hsv2rgb(areaColor);
    areaColor *= crange(noise.z, 0.5, 1.0, 0.0, 1.0);

    areaColor *= pow(crange(length(vWorldPos), 0.0, 3.0, 1.0, 0.0) * smoothstep(-0.5, 0.0, vWorldPos.z), 1.5);
    areaColor *= crange(abs(vWorldPos.x), 0.5, 2.5, 1.0, 0.0);

    vec3 color = getPointLightColor();

    vec3 reflectionColor = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        vec4 coord = mirrorCoord;
        coord.x += cos(degrees((float(i) / 10.0) * 360.0)) * 0.01;
        coord.z += sin(degrees((float(i) / 10.0) * 360.0)) * 0.01;
        reflectionColor += texture2DProj(tMirrorReflection, coord).rgb;
    }
    reflectionColor /= 5.0;

    color += reflectionColor * uMirrorStrength * 0.1;
    color += areaColor * uAreaStrength * 0.6;

//    gl_FragColor = vec4(vNormal, 1.0);
    gl_FragColor = vec4(color, mask);
}{@}WaterComposite.fs{@}uniform sampler2D tGlow;
uniform sampler2D tGlowMask;
uniform sampler2D tDepth;
uniform sampler2D tLogo;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform float uGlowStrength;
uniform float uFogStrength;
uniform float uFogBrightness;
uniform float uRGBStrength;
uniform float uDistortion;
uniform float uDesat;
uniform float uVignette;
uniform float uLogoAffect;
uniform float uHome;
uniform vec2 uHue;
uniform vec2 uFog;
uniform vec3 uFogColor;

#require(range.glsl)
#require(simplenoise.glsl)
#require(depthvalue.fs)
#require(rgb2hsv.fs)
#require(rgbshift.fs)
#require(transformUV.glsl)
#require(eases.glsl)
#require(blendmodes.glsl)

void main() {
    float dist = length(vUv - vec2(0.5));
    float scale = crange(dist, 0.2, 0.5, 1.0, 1.0 + (0.01 * uDistortion));
    scale = mix(scale, 1.2, uLogoAffect);

    float extraRGBStrength = crange(dist, 0.2, 0.5, 1.0, 2.0);
    vec2 uv = scaleUV(vUv, vec2(scale));

    float fluidMask = texture2D(tFluidMask, vUv).r;
    vec2 fluid = texture2D(tFluid, vUv).xy * fluidMask;
    uv += fluid * 0.0002 * uLogoAffect;
    extraRGBStrength += fluidMask*0.1;

    vec4 logoDistortion = texture2D(tLogo, vUv);
    logoDistortion.x = crange(logoDistortion.x, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    logoDistortion.y = crange(logoDistortion.y, 0.0, 1.0, -1.0, 1.0) * logoDistortion.z;
    uv += 0.1 * logoDistortion.xy * uLogoAffect;

    vec3 glow = vec3(0.0);
    float glowMask = 1.0;
    float liquidRGB = smoothstep(0.2,1.0,fluidMask)*0.005*uLogoAffect;
    vec3 color = getRGB(tDiffuse, uv, fluidMask+time*0.2, 0.001 * uRGBStrength * extraRGBStrength + liquidRGB).rgb;

    #test Tests.screenGlow()
    glow = texture2D(tGlow, uv).rgb;
    #endtest

    glowMask = texture2D(tGlowMask, uv).a;

    float depth = getDepthValue(tDepth, uv, 0.1, 20.0);

    vec3 fogColor = rgb2hsv(uFogColor);
    fogColor.x += uHue.x * cnoise(vUv * uHue.y);
    fogColor.y *= uDesat;
    fogColor.z = uFogBrightness;
    fogColor = hsv2rgb(fogColor);

    float fog = sineIn(crange(depth, uFog.x, uFog.y, 0.0, 1.0));
    float fog2 = sineIn(crange(depth, uFog.x, uFog.y, 0.55, 1.5));
    float fogBlend = fog * crange(glowMask, 0.0, 1.0, 0.25, 1.0) * uFogStrength;
    float fogBlend2 = fog * crange(glowMask, 0.0, 1.0, 0.25, 1.0) * uFogStrength * crange(depth, 0.6, 0.97, 1.0, 0.0);
    color = mix(color, fogColor, fogBlend);
    color = mix(color, fogColor, fogBlend2);

    color = mix(color, blendOverlay(color*mix(0.45, 0.75, uHome), uFogColor), 0.5 + (1.0-uHome)*0.5);

    color += glow * crange(glowMask, 0.0, 1.0, 0.5, 1.0) * (uGlowStrength + uLogoAffect*0.2);

    color *= sineOut(crange(dist, 0.0, 0.5, 1.0, uVignette));

    color += 0.075 * logoDistortion.a * crange(logoDistortion.z, 0.0, 1.0, 0.5, 1.0) * uLogoAffect;

    gl_FragColor = vec4(color, 1.0);
}{@}WaterDistance.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMirrorReflection;
uniform sampler2D tWaterNormal;
uniform mat4 uMirrorMatrix;
uniform float uWaterSpeed;
uniform float uMirrorDistort;
uniform float uMirrorStrength;
uniform float uUVScale;

#!VARYINGS
varying vec4 vMirrorCoord;
varying vec2 vUv;

#!SHADER: Vertex

#require(lighting.vs)

void main() {
    setupLight(position);

    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;
    vUv = uv;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(lighting.fs)

vec4 getNoise(vec2 uv){
    float time = time * 0.001 * uWaterSpeed;
    vec2 uv0 = (uv/103.0)+vec2(time/17.0, time/29.0);
    vec2 uv1 = uv/107.0-vec2(time/-19.0, time/31.0);
    vec2 uv2 = uv/vec2(897.0, 983.0)+vec2(time/101.0, time/97.0);
    vec2 uv3 = uv/vec2(991.0, 877.0)-vec2(time/109.0, time/-113.0);
    float scale = uUVScale * 10.0;
    vec4 noise = (texture2D(tWaterNormal, uv0 * scale)) +
                 (texture2D(tWaterNormal, uv1 * scale)) +
                 (texture2D(tWaterNormal, uv2 * scale)) +
                 (texture2D(tWaterNormal, uv3 * scale));
    return noise*0.5-1.0;
}

void main() {
    setupLight();

    vec4 noise = getNoise(vUv);
    vec3 waternormal = normalize(noise.xzy * vec3(2.0, 1.0, 2.0));

    lNormal = waternormal;

    vec4 mirrorCoord = vMirrorCoord;
    mirrorCoord.xz += waternormal.xy * uMirrorDistort;

    vec3 color = getPointLightColor();

    vec3 reflectionColor = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        vec4 coord = mirrorCoord;
        coord.x += cos(degrees((float(i) / 10.0) * 360.0)) * 0.01;
        coord.z += sin(degrees((float(i) / 10.0) * 360.0)) * 0.01;
        reflectionColor += texture2DProj(tMirrorReflection, coord).rgb;
    }
    reflectionColor /= 5.0;
    color += reflectionColor * uMirrorStrength;

    gl_FragColor = vec4(color, 1.0);
}{@}WaterSky.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform float uNoiseScale;
uniform float uNoiseStrength;
uniform float uMinNoise;
uniform vec3 uColor;

#!VARYINGS
varying vec3 vPos;
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)
#require(blendmodes.glsl)

void main() {
    float uv = crange(vPos.y, 0.3, 1.0, 0.0, 1.0);
    uv += cnoise(vPos * uNoiseScale) * 0.1 * uNoiseStrength;

    vec3 color = texture2D(tMap, vec2(uv)).rgb;
    color *= crange(getNoise(vUv, time), 0.0, 1.0, uMinNoise, 1.0);

    color = rgb2hsv(color);
    color.y = 0.0;
    color = hsv2rgb(color);

    color = blendOverlay(color, uColor);

    color *= 0.6;


    gl_FragColor = vec4(color, 1.0);
}{@}WaterStars.glsl{@}#!ATTRIBUTES
attribute vec4 random;

#!UNIFORMS
uniform float uSize;
uniform float uScale;
uniform sampler2D tMap;

#!VARYINGS
varying vec4 vRandom;

#!SHADER: Vertex

#require(range.glsl)

void main() {
    vec3 pos = getPos();
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);

    float size = uSize * crange(random.x, 0.0, 1.0, 1.0 - uScale, 1.0 + uScale);

    vRandom = random;

    gl_PointSize = size * (1000.0 / length(mvPosition.xyz));
    gl_Position = projectionMatrix * mvPosition;
}

#!SHADER: Fragment

#require(range.glsl)
#require(conditionals.glsl)

void main() {
    vec3 map = texture2D(tMap, gl_PointCoord).rgb;
    float alpha = map.r;
    alpha = mix(alpha, map.g, when_gt(vRandom.y, 0.5));
    alpha = mix(alpha, map.b, vRandom.z);

    vec3 color = vec3(crange(length(gl_PointCoord), 0.0, 0.5, 0.0, 1.0));

    gl_FragColor = vec4(color, alpha * 0.5);
}{@}ScreenCubes.glsl{@}#!ATTRIBUTES
attribute vec2 uvIndex;
attribute vec3 lookup;

#!UNIFORMS
uniform vec3 uScale;
uniform vec3 uColor;
uniform float uUVScale;
uniform vec2 uAspectScale;
uniform sampler2D tMap;
uniform sampler2D tLogo;
uniform sampler2D tProjectLogo;
uniform sampler2D tPos;
uniform sampler2D tFlowmap;
uniform sampler2D tRGB;
uniform float uRGBScale;
uniform vec2 uRGB;
uniform vec2 uProjectLogoScale;
uniform float uRGBStrength;
uniform float uHoldValue;
uniform float uSpaceValue;
uniform float uDPR;
uniform float uIntro;
uniform float uIntro2;
uniform float uIntro3;
uniform float uChange;
uniform float uHome;
uniform float uHomeHover;
uniform float uLogoScale;
uniform float uLogoAlpha;
uniform float uProjectLogoAlpha;
uniform float uResizing;
uniform float uNoCubes;
uniform float uScreenBrighten;
uniform vec2 uBorder;

#!VARYINGS
varying vec2 vUv2;
varying vec2 vUv3;
varying vec2 rUV;
varying float vFront;
varying float vFlow;
varying float vIndex;
varying vec2 vUv;

#!SHADER: ScreenCubesQuad.vs

varying vec2 vUv;
varying vec2 vUv2;

#require(transformUV.glsl)

void main() {
    vUv = scaleUV(uv, uAspectScale);
    vUv2 = uv;
    vFront = 1.0;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: ScreenCubesQuad.fs

// FALLBACK VERSION
varying vec2 vUv;
varying vec2 vUv2;

#require(rgb2hsv.fs)
#require(range.glsl)
#require(simplenoise.glsl)
#require(blendmodes.glsl)
#require(transformUV.glsl)
void main() {
   float staticNoise = range(getNoise(vUv * 5., time*0.1), 0.0, 1.0, -1.0, 1.0);
    float homeHoverTransition = smoothstep(0.0, 0.5, uHomeHover) * smoothstep(1.0, 0.5, uHomeHover);
    vec2 videoUV = scaleUV(vUv, vec2(1.0+uHomeHover*0.05));
    videoUV *= 1.0 + cnoise(vec3(vUv*200.0, time*5.0)) * homeHoverTransition * 0.01;

    vec3 video = texture2D(tMap, videoUV).rgb;

    vec3 rgb = texture2D(tRGB, vUv * uRGBScale).rgb;
    vec3 baseLogo = texture2D(tLogo, scaleUV(vUv, vec2(uLogoScale, uLogoScale))).rgb;
    baseLogo = mix(vec3(0.), baseLogo, uLogoAlpha);
    baseLogo = blendMultiply(baseLogo, uColor);
    baseLogo *= (1.0-uHoldValue);

    vec3 blended = video * rgb;
    video = mix(video, blended, mix(uRGB.x, uRGB.y, uRGBStrength)*0.3);

    vec3 pbr = texture2D(tMap, rUV).rgb * uHoldValue;
    vec3 color = mix(pbr, video, vFront);
    color = mix(color, blendMultiply(color, uColor), 0.3);

    float borderAlpha = 0.0;
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv2.x, uBorder.x, uBorder.x+0.0001, 1.0, 0.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv2.x, 1.0-uBorder.x, 1.0-uBorder.x+0.001, 0.0, 1.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv2.y, uBorder.y, uBorder.y+0.0001, 1.0, 0.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv2.y, 1.0-uBorder.y, 1.0-uBorder.y+0.001, 0.0, 1.0));
    vec3 border = uColor*borderAlpha;

    // Intro
    vec3 logoColor = mix(uColor, vec3(1.0), 0.5);
    vec3 intro = vec3(0.5) + staticNoise*(0.4+smoothstep(0.5, 0.0, abs(uIntro-0.5))*0.5);
    intro = blendMultiply(intro, logoColor);
    intro = mix(intro, blendScreen(intro, video), 1.0-uIntro);
    intro = mix(intro, blendAdd(intro, video), 1.0-uIntro);
    intro = mix(intro, vec3(1.0), smoothstep(0.8, 1.0, uIntro)*0.2);
    intro = rgb2hsv(intro);
    intro.y *= 1.0-uIntro;
    intro = hsv2rgb(intro);
    intro += border*(0.5+sin(time*0.2)*0.5);
    color = mix(color, intro, uIntro);

    // Home Button Hover
    vec3 hover = color;
    hover = blendMultiply(hover, uColor);
    hover += border;
    hover = blendAdd(hover, baseLogo);
    hover += homeHoverTransition*0.1;
    hover = mix(hover, uColor, homeHoverTransition*0.3);
    //hover += staticNoise * crange(uHomeHover, 0.0, 0.5, 0.0, 1.0) * crange(uHomeHover, 0.5, 1.0, 1.0, 0.0);
    color = mix(color, hover, uHomeHover);

    // Hold Overlay
    vec3 hold = color*0.6;
    //hold = mix(hold, blendGlow(hold, video), smoothstep(0.0, 0.9, uHoldValue)*0.5);
    float noise = smoothstep(0.1, 1.0, crange(cnoise(vec3(vUv*1.5, time*0.5+vIndex*2.0)), -1.0, 1.0, 0.0, 1.0));
    hold = mix(hold, blendAdd(hold, uColor), noise*uHoldValue*0.6);
    color = mix(color, hold, uHoldValue);
    color = mix(color, hold, homeHoverTransition);


    // Border

    // Darken for volumetric brightening
    color = mix(color, blendSoftLight(color, uColor), (1.0-uHome)*0.6);
    color *= mix(mix(0.6, 0.72, uHome)*0.7, 1.1, uScreenBrighten);

    //color = uColor*0.8;

    vec2 logoUv = scaleUV(vUv, uProjectLogoScale) + (.5 - staticNoise) * .2 * (1.-uProjectLogoAlpha);
    vec4 projectLogoTexel = texture2D(tProjectLogo, logoUv);
    float logoDiscard = 0.0;
    float limit = .001;
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.x, limit, limit+0.0001, 1.0, 0.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.x, 1.0-limit, 1.0-limit+0.001, 0.0, 1.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.y, limit, limit+0.0001, 1.0, 0.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.y, 1.0-limit, 1.0-limit+0.001, 0.0, 1.0));

    // Final Stuff
    color = mix(color, blendAdd(color, baseLogo), crange(sin(time*0.3), -1.0, 1.0, 0.0, 1.0-uHoldValue)*uHome);
    color += border*(0.1+sin(time*0.2)*0.1)*(1.0-uHoldValue)*uHome;
    color += sin(time*21.0)*0.001+sin(time*50.0)*0.001+sin(time*80.0)*0.001;
    color += staticNoise*0.03 + homeHoverTransition*staticNoise*0.05;
    color += projectLogoTexel.r * projectLogoTexel.a * (1.-logoDiscard) * uColor * uProjectLogoAlpha * .5;
    color = mix(color, uColor, homeHoverTransition*0.1);

    color = mix(color, uColor, uResizing*0.9);

    vec3 space = uColor*0.8;
    space -= baseLogo*uHome;
    color = mix(color, space, uSpaceValue);
    color = mix(color, space, uIntro2);

    vec3 start = vec3(staticNoise*0.3);
    start += baseLogo*uHome;
    start += border;
    color = mix(color, start, uIntro3);


    gl_FragColor = vec4(color, 1.0 - vFront);
}

#!SHADER: ScreenCubesQuadInstance.vs

varying vec2 vUv;
varying float vAlpha;
attribute vec3 offset;

#require(instance.vs)
#require(conditionals.glsl)

void main() {
    vUv = uv;
    vFront = 1.0;
    vAlpha = pow(0.3 * (1.0 - (offset.z / 5.0)), 2.0);

    vec3 pos = transformPosition(position, offset);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: ScreenCubesQuadInstance.fs

varying vec2 vUv;
varying float vAlpha;

#require(rgb2hsv.fs)

void main() {
    vec3 video = texture2D(tMap, vUv).rgb;
    vec3 rgb = texture2D(tRGB, vUv * uRGBScale).rgb;

    vec3 blended = video * rgb;
    video = mix(video, blended, mix(uRGB.x, uRGB.y, uRGBStrength));

    gl_FragColor = vec4(video, vAlpha);
}


#!SHADER: Vertex

#require(instance.vs)
#require(range.glsl)
#require(conditionals.glsl)
#require(transformUV.glsl)
#require(glscreenprojection.glsl)
#require(rotation.glsl)
#require(matcap.vs)
#require(flowmap.fs)
#require(simplenoise.glsl)

vec2 getUV() {
    vec2 vuv;
    float xMin = 0.0 + uvIndex.x;
    float xMax = uUVScale + uvIndex.x;
    float yMin = 0.0 + uvIndex.y;
    float yMax = uUVScale + uvIndex.y;

    vuv.x = crange(position.x, -0.5, 0.5, xMin, xMax);
    vuv.y = crange(position.y, -0.5, 0.5, yMin, yMax);
    return vuv;
}

void main() {
    vUv = uv;
    vUv2 = scaleUV(getUV(), uAspectScale);
    vUv3 = getUV();
    vFront = when_gt(normal.z, max(normal.y, normal.x));
    vIndex = uvIndex.x*uvIndex.y;

    vec3 offset = texture2D(tPos, lookup.xy).xyz;

    vec2 screen = getProjection(vec3(modelViewMatrix * vec4(offset, 1.0)), projectionMatrix);
    vec2 flow = getFlow(tFlowmap, screen);
//    vFlow = length(flow);

    float flowStrength = length(flow);
//    flowStrength *= mix(1.0, 0.0, when_lt(flowStrength, 0.05));
    flowStrength = crange(flowStrength, 0.0, 0.1, 0.0, 1.0) * uHoldValue;
    vFlow = flowStrength;

    mat4 rot = rotationMatrix(vec3(-flow.xy, flow.x), radians(flowStrength * 180.0));
    vec3 rpos = vec3(rot * vec4(position, 1.0));
    vec3 rnormal = vec3(rot * vec4(normal, 1.0));
    vec3 pos = transformPosition(rpos, offset, uScale);

    //pos += cnoise(pos*5.0+time*0.1)*0.2*uHoldUniform;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    rUV = reflectMatcap(pos, modelViewMatrix, normalMatrix, rnormal);
}

#!SHADER: Fragment

#require(rgb2hsv.fs)
#require(range.glsl)
#require(simplenoise.glsl)
#require(blendmodes.glsl)
#require(transformUV.glsl)

void main() {
    float staticNoise = range(getNoise(vUv2 * 3., time*0.1), 0.0, 1.0, -1.0, 1.0);
    float homeHoverTransition = smoothstep(0.0, 0.5, uHomeHover) * smoothstep(1.0, 0.5, uHomeHover);
    vec2 videoUV = scaleUV(vUv2, vec2(1.0+uHomeHover*0.05));
    videoUV *= 1.0 + cnoise(vec3(vUv2*200.0, time*5.0)) * homeHoverTransition * 0.01;

    vec3 video = texture2D(tMap, videoUV).rgb;

    vec3 rgb = texture2D(tRGB, vUv2 * uRGBScale).rgb;
    vec3 baseLogo = texture2D(tLogo, scaleUV(vUv2, vec2(uLogoScale, uLogoScale))).rgb;
    baseLogo = mix(vec3(0.), baseLogo, uLogoAlpha);
    baseLogo = blendMultiply(baseLogo, uColor);
    baseLogo *= smoothstep(0.8, 1.0, uHome);

    vec3 blended = video * rgb;
    video = mix(video, blended, mix(uRGB.x, uRGB.y, uRGBStrength)*0.5);

    vec3 pbr = texture2D(tMap, rUV).rgb * uHoldValue;
    vec3 color = mix(pbr, video, vFront);
    color = mix(color, blendMultiply(color, uColor), 0.2);

    float borderAlpha = 0.0;
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv3.x, uBorder.x, uBorder.x+0.0001, 1.0, 0.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv3.x, 1.0-uBorder.x, 1.0-uBorder.x+0.001, 0.0, 1.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv3.y, uBorder.y, uBorder.y+0.0001, 1.0, 0.0));
    borderAlpha = mix(borderAlpha, 1.0, crange(vUv3.y, 1.0-uBorder.y, 1.0-uBorder.y+0.001, 0.0, 1.0));
    vec3 border = uColor*borderAlpha;

    // Intro
    vec3 logoColor = mix(uColor, vec3(1.0), 0.5);
    vec3 intro = uColor*0.5;
    //intro += staticNoise*(0.4+smoothstep(0.5, 0.0, abs(uIntro-0.5))*0.5)*0.9;
    intro = blendMultiply(intro, logoColor);
    intro = mix(intro, blendScreen(intro, video), 1.0-uIntro);
    intro = mix(intro, blendAdd(intro, video), 1.0-uIntro);
    intro = mix(intro, vec3(1.0), smoothstep(0.8, 1.0, uIntro)*0.2);
    intro = rgb2hsv(intro);
    intro.y *= 1.0-uIntro;
    intro = hsv2rgb(intro);
    intro += border;
    color = mix(color, intro, uIntro);

    // Home Button Hover
    vec3 hover = color;
    hover = blendMultiply(hover, uColor);
    hover += border;
    hover = blendAdd(hover, baseLogo);
    hover += homeHoverTransition*0.1;
    hover = mix(hover, uColor, homeHoverTransition*0.6);
    //hover += staticNoise * crange(uHomeHover, 0.0, 0.5, 0.0, 1.0) * crange(uHomeHover, 0.5, 1.0, 1.0, 0.0);
    color = mix(color, hover, uHomeHover);

    // Hold Overlay
    vec3 hold = video;
   //hold = mix(hold, blendGlow(hold, video), smoothstep(0.0, 0.9, uHoldValue)*0.5);
    //float noise = crange(cnoise(vec3(vUv2, time*0.5+vIndex*2.0)), -1.0, 1.0, 0.0, 1.0);
    //hold = mix(hold, blendAdd(hold, uColor), noise*uHoldValue*0.1);
    hold = mix(hold, blendOverlay(hold, uColor), uHoldValue*0.3);
    color = mix(color, hold, uHoldValue);
    color = mix(color, hold, homeHoverTransition);


    // Darken for volumetric brightening
    color = mix(color, blendSoftLight(color, uColor), (1.0-uHome)*0.6);
    color *= mix(mix(0.6, 0.72, uHome)*0.7, 1.1, uScreenBrighten);
    //color = uColor*0.8;

    vec2 logoUv = scaleUV(vUv2, uProjectLogoScale) + (.5 - staticNoise) * .1 * (1.-uProjectLogoAlpha);
    vec4 projectLogoTexel = texture2D(tProjectLogo, logoUv);
    float logoDiscard = 0.0;
    float limit = .001;
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.x, limit, limit+0.0001, 1.0, 0.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.x, 1.0-limit, 1.0-limit+0.001, 0.0, 1.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.y, limit, limit+0.0001, 1.0, 0.0));
    logoDiscard = mix(logoDiscard, 1.0, crange(logoUv.y, 1.0-limit, 1.0-limit+0.001, 0.0, 1.0));

    // Final Stuff
    color = mix(color, blendAdd(color, baseLogo), crange(sin(time*0.3), -1.0, 1.0, 0.0, 1.0-uHoldValue)*uHome);
    color += border*(0.25+sin(time*0.4)*0.25)*(1.0-uHoldValue);
    color += sin(time*21.0)*0.001+sin(time*50.0)*0.001+sin(time*80.0)*0.001;
    color += projectLogoTexel.r * projectLogoTexel.a * (1.-logoDiscard) * uColor * uProjectLogoAlpha * .5;
    color = mix(color, uColor, homeHoverTransition*0.1);
    color = mix(color, uColor*0.8, uResizing);

    // Space Hold
    vec3 space = uColor*0.8;
    space -= baseLogo*uHome*(1.0-uResizing);
    color = mix(color, space, smoothstep(0.7, 1.0, uSpaceValue));
    color = mix(color, space, uIntro2);

    vec3 start = vec3(staticNoise*0.3);
    start += baseLogo*uHome;
    start += border;
    color = mix(color, start, uIntro3);


    //color += staticNoise*0.03 + homeHoverTransition*staticNoise*0.05;
    gl_FragColor = vec4(color, 1.0 - vFront);
}

{@}ScreenVolumetric.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    vec3 color = texture2D(tMap, vUv).rgb;
//    float mask = texture2D(tMask, vUv).r;
//
//    color *= 1.0 - (step(0.01, mask) * 0.3);

    gl_FragColor = vec4(color, 1.0);
}{@}BarHitShader.glsl{@}#!ATTRIBUTES
// varying vec3 a_uColor;

#!UNIFORMS
// uniform sampler2D tMap;
// uniform vec3 uUIColor;

#!VARYINGS
// varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor = vec4(vec3(1.), 0.);
}
{@}BarShader.glsl{@}#!ATTRIBUTES
varying vec3 a_uColor;

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uUIColor;

#!VARYINGS
varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = uUIColor;
    color += random(vec2(time))*0.2;

    gl_FragColor = vec4(uUIColor, v_uAlpha);
    gl_FragColor.rgb = mix(uUIColor, vec3(1.0), smoothstep(0.5, 1.0, v_uAlpha)*0.5);
}
{@}WorkVideo.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tGlitch;
uniform sampler2D tBoxes;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform vec2 uAspectScale;
uniform float uGlitchScale;
uniform float uGlitchScroll;
uniform float uGlitchBlend;
uniform float uPanEnabled;
uniform float uPan;
uniform vec2 uNoise;
uniform float uStatic;
uniform float uRandom;
uniform vec3 uColor;
uniform float uBoxBlend;
uniform float uHold;
uniform float uBrighten;

#!VARYINGS
varying vec2 vUv;
varying vec2 baseUv;

#!SHADER: Vertex

#require(transformUV.glsl)

void main() {
    baseUv = uv;
    vUv = scaleUV(uv, uAspectScale);

    float panOffset = max(1./(uAspectScale.x) * .5, 0.) * uPanEnabled;
    vUv = translateUV(vUv, vec2(uPan * panOffset, 0.));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(blendmodes.glsl)
#require(rgbshift.fs)
#require(rgb2hsv.fs)
#require(eases.glsl)

vec3 getGlitch(vec2 fluid) {
    vec2 uv = vUv * uGlitchScale;
    uv.y -= time * uGlitchScroll * 0.04 + fluid.y*0.002;
    uv.x -= time * uGlitchScroll * 0.03 + fluid.x*0.002;
    uv.x += cnoise(vUv * uNoise.x + time*0.1) * uNoise.y * 0.1;

    return texture2D(tGlitch, uv).rgb;
}

void main() {
    float fluidMask = smoothstep(0.1, 0.8, texture2D(tFluidMask, baseUv).r);
    float fluidOutline = smoothstep(0.0, 0.6, fluidMask)*smoothstep(0.8, 0.6, fluidMask);

    vec2 fluid = texture2D(tFluid, baseUv).xy * fluidMask;

    vec2 uv = vUv;
    uv += fluid*0.0015*fluidMask*uHold;

    float gradient = vUv.y*0.5 + vUv.x*0.5;
    gradient = 1.0-gradient;

    //vec3 color = texture2D(tMap, uv).rgb;
    vec3 color = getRGB(tMap, uv, 0.3, 0.001 + fluidMask*0.001).rgb;
    color = mix(color, vec3(0.0), smoothstep(0.15,1.0,gradient)*0.8);

    vec3 glitchFluid = getGlitch(fluid);
    vec3 glitch = blendOverlay(color, glitchFluid);
    color = mix(color, glitch, uGlitchBlend * 0.05 + fluidMask*0.);
    color += blendOverlay(uColor, glitchFluid)*fluidOutline*(0.22+0.5*uHold)*(1.0-uBrighten);

    float noise = getNoise(uv, time);
    vec3 noiseColor = blendOverlay(color, color + noise);
    color = mix(color, noiseColor, uStatic * 0.03 + fluidMask*0.05 + fluidOutline*0.07);

    color = rgb2hsv(color);
    color = hsv2rgb(color);

    float dist = length(vUv - vec2(0.5));
    float vignette = sineIn(crange(dist, 0.05, 0.8, 0.0, 1.0));
    vec3 colorOverlay = color;
    //colorOverlay = mix(colorOverlay, blendMultiply(colorOverlay, uColor), 0.4);
    colorOverlay = mix(colorOverlay, blendOverlay(colorOverlay*(0.3+vignette*0.4), uColor), mix(1.0, 0.0, fluidMask*uHold*0.7+fluidOutline*0.5*uHold));
    color = mix(color, colorOverlay, mix(1.0, 0.1, uBrighten));

    color = rgb2hsv(color);
    color.x += crange(vUv.x+vUv.y, 0.0, 2.0, -0.1, 0.1)*(1.0-uBrighten)*0.8;
    color = hsv2rgb(color);

    //color = mix(color, blendOverlay(color, uColor), fluidMask*0.1);

    //color *= 1.0+fluidMask*0.1;


    // float boxes = step(0.9, texture2D(tBoxes, (vUv * 1.3) + vec2(uRandom * 10.0)).r);
    // boxes *= crange(cnoise(vUv + time), -1.0, 1.0, 0.0, 1.0);
//    vec3 boxColor = blendOverlay(color, color + boxes);
//    color = mix(color, boxColor, uBoxBlend * 0.1);


    //vec3 fluid = texture2D(tFluid, vUv).rgb;
    //color += fluid;



    gl_FragColor = vec4(color, 1.0);
}{@}WorkItemText.fs{@}uniform float uFillAlpha;
uniform float uStrokeAlpha;
uniform float uStroke;
uniform float uStrokePadding;
uniform float uOpacity;
uniform float uAnimate;
uniform float uFaded;
uniform vec3 uUIColor;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

#require(transformUV.glsl)

void main() {
    vec2 boundUv = getBoundingUV();
    //boundUv = scaleUV(boundUv, vec2(2.0,2.0));
//    float fluidMask = smoothstep(0.0, 0.1, texture2D(tFluidMask, boundUv).r);
//    vec2 fluid = texture2D(tFluid, boundUv).xy * fluidMask;
    vec2 uv = vUv;
    //uv -= fluid*0.000005;

    float fill = msdf(tMap, uv);
    float stroke = strokemsdf(tMap, uv, uStroke, uStrokePadding);


    float alpha = mix(uFillAlpha, uStrokeAlpha, stroke);
    float noise = cnoise(vec3(uv*5.0, time*0.3));


    //alpha = mix(alpha, fill-noise*0.1, (1.0-uAnimate)*0.2);
    //alpha += noise*(stroke*0.15)*uAnimate;
    alpha = mix(alpha*0.8, alpha, uAnimate);

    alpha *= crange(noise, -1.0, 1.0, smoothstep(0.5, 1.0, uFaded), 1.0);
    //alpha *= 1.0-smoothstep(0.5, 1.0, uFaded)*smoothstep(0.5, 0.0, uFaded)*0.5;

    //alpha += stroke*fluidMask;

    //alpha *= 0.6 + sin(time*0.3)*0.5;
    //alpha += fluidMask;

    //float animate = 1.0-uAnimate;
    //alpha += crange(uv.x, animate, animate+0.5, 1.0*animate, 0.0);

    gl_FragColor.rgb = uUIColor;//vec3(getBoundingUV(), 1.0);
    gl_FragColor.a = clamp(alpha * uAlpha * uOpacity * vTrans * uFaded, 0.0, 1.0);
}
{@}WorkSceneComposite.fs{@}uniform sampler2D tVolumetricLight;
uniform sampler2D tFog;
uniform sampler2D tGlitch;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform float uFogRGB;
uniform float uFogBlend;
uniform float uGlitchScale;
uniform float uDistortion;
uniform vec3 uColor;

#require(range.glsl)
#require(rgbshift.fs)
#require(luma.fs)
#require(blendmodes.glsl)
#require(simplenoise.glsl)
#require(transformUV.glsl)
#require(conditionals.glsl)

vec3 getGlitch() {
    vec2 uv = vUv;
    uv.x -= time * 0.3;
    return texture2D(tGlitch, uv).rgb;
}

vec2 getUV(vec3 texel) {
    vec3 glitch = getGlitch();
    vec2 uv = vUv;

    uv.y += glitch.r * uDistortion * 0.1;

    return uv;
}

void main() {
    vec2 d = vUv - vec2(0.5);
    float angle = atan(d.y, d.x);

    vec2 fuv = vUv;
    vec2 fluid = texture2D(tFluid, vUv).xy * texture2D(tFluidMask, vUv).r;
    fuv -= fluid * 0.0005;

    vec3 fog = texture2D(tFog, fuv).rgb;

    vec2 luv = scaleUV(fuv, 1.0 + (0.2 * vec2(crange(fog.r, 0.0, 1.0, -1.0, 1.0))));

    vec3 light = getRGB(tVolumetricLight, luv, 0.3, 0.002).rgb * crange(fog.g, 0.0, 1.0, 0.5, 1.0) * 0.7;
    vec3 diffuse = texture2D(tDiffuse, vUv).rgb;

    vec2 uv = getUV(diffuse + light + fog);

    fog = blendOverlay(getRGB(tFog, uv, angle, 0.004 * uFogRGB).rgb, light) * uFogBlend;
    fog = mix(fog * uColor, blendScreen(fog, fog * uColor), 0.5);
    fog = mix(fog, light, crange(length(d), 0.1, 0.5, 1.0, 0.5));

    vec3 texel = texture2D(tDiffuse, uv).rgb;
    vec3 color = texel + fog;

    gl_FragColor = vec4(color, 1.0);
}{@}WorkSceneBG.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uNoiseScale;
uniform float uDarken;
uniform float uHue;
uniform float uLightness;
uniform float uTimeScale;
uniform float uVignette;
uniform float uNoiseMin;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)

void main() {
    float noise = range(cnoise((vUv * uNoiseScale) + time*uTimeScale*0.1), -1.0, 1.0, 0.0, 1.0);

    vec3 color = rgb2hsv(uColor);
    color.z -= uLightness * noise;
    color.x -= 0.1 * uHue * noise;
    color = hsv2rgb(color);

    float dist = length(vUv - vec2(0.5));
    color *= range(dist, 0.5, 0.0, 1.0, uVignette);

    color *= uDarken;

    color *= range(getNoise(vUv * 4.0, time), 0.0, 1.0, uNoiseMin, 1.0);

    gl_FragColor = vec4(color, 1.0);
}{@}WorkSceneFloor.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tBG;
uniform mat4 uMirrorMatrix;
uniform sampler2D tMirrorReflection;
uniform float uMirrorStrength;
uniform float uLightStrength;
uniform float uDarken;
uniform vec3 uColor;

#!VARYINGS
varying float vFog;
varying vec4 vMirrorCoord;

#!SHADER: Vertex

#require(range.glsl)
#require(eases.glsl)
#require(lighting.vs)

void main() {
    setupLight(position);

    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vMirrorCoord = uMirrorMatrix * worldPos;

    vFog = sineIn(crange(uv.y, 0.2, 0.5, 0.0, 1.0));
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(lighting.fs)

void main() {
    setupLight();

    vec3 bg = texture2D(tBG, gl_FragCoord.xy / resolution).rgb;
    vec3 color = uColor * uDarken;

    color += getAreaLightColor() * uLightStrength;

    vec3 mirror = texture2DProj(tMirrorReflection, vMirrorCoord).rgb;
    color += mirror * uMirrorStrength;

    color = mix(color, bg, vFog);
    gl_FragColor = vec4(color, 1.0);
}{@}WorkScreenFogQuads.glsl{@}#!ATTRIBUTES
attribute vec3 lookup;

#!UNIFORMS
uniform sampler2D tPos;
uniform sampler2D tMap;
uniform sampler2D tRandom;
uniform vec4 uQuaternion;
uniform float uAlpha;
uniform float uScale;
uniform float uNoiseScale;
uniform float uNoiseStrength;
uniform float uNoiseTime;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;
varying vec4 vRandom;
varying float vAlpha;
varying vec3 vPos;

#!SHADER: Vertex

#require(instance.vs)
#require(range.glsl)
#require(simplenoise.glsl)
#require(rotation.glsl)
#require(rgb2hsv.fs)

void main() {
    vUv = uv;
    vRandom = texture2D(tRandom, lookup.xy);
    vAlpha = uAlpha * 0.1 * crange(vRandom.w, 0.2, 1.0, 0.5, 1.0);

    float scale = uScale * crange(vRandom.x, 0.0, 1.0, 0.5, 1.5);

    vec3 offset = texture2D(tPos, lookup.xy).xyz;
    vec3 pos = transformPosition(position, offset, scale, uQuaternion);

    float rotation = radians((360.0 * vRandom.y) + time*vRandom.z);
    pos = vec3(rotationMatrix(vec3(0.0, 0.0, 1.0), rotation) * vec4(pos, 1.0));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    vPos = pos;
}

#!SHADER: Fragment

#require(range.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

vec2 getUV() {
    float noise = cnoise((vPos * uNoiseScale) + time*uNoiseTime);
    float scale = 1.0 + (noise * uNoiseStrength * 0.1);

    return scaleUV(vUv, vec2(scale));
}

void main() {
    float mask = texture2D(tMap, getUV()).r;
    float padding = 0.3;
    mask *= crange(vUv.x, 0.0, padding, 0.0, 1.0) * crange(vUv.x, 1.0 - padding, 1.0, 1.0, 0.0);
    mask *= crange(vUv.y, 0.0, padding, 0.0, 1.0) * crange(vUv.y, 1.0 - padding, 1.0, 1.0, 0.0);

    gl_FragColor = vec4(uColor, mask * vAlpha);
}{@}WorkItemBorder.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform float uAlpha;
uniform float uVisible;
uniform vec2 uBorder;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec2 uv = vUv;

    float alpha = 0.0;
    alpha = mix(alpha, 1.0, crange(uv.x, uBorder.x, uBorder.x+0.0001, 1.0, 0.0));
    alpha = mix(alpha, 1.0, crange(uv.x, 1.0-uBorder.x, 1.0-uBorder.x+0.0001, 0.0, 1.0));
    alpha = mix(alpha, 1.0, crange(uv.y, uBorder.y, uBorder.y+0.0001, 1.0, 0.0));
    alpha = mix(alpha, 1.0, crange(uv.y, 1.0-uBorder.y, 1.0-uBorder.y+0.0001, 0.0, 1.0));

    alpha = clamp(alpha, 0.0, 1.0);
    alpha *= uVisible;

    vec3 color = uColor;
    //color -= (1.0-uVisible)*random(vec2(time));


    gl_FragColor = vec4(uColor, alpha);
}{@}WorkItemButtonBg.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uUIColor;
uniform float uAlpha;
uniform float uHover;
uniform float uThickness;
uniform vec2 uAspect;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)

float border(vec2 uv, float thickness) {
    float padding = 0.6;
    float left = smoothstep(thickness*(1.0+padding)*uAspect.x, thickness*uAspect.x, uv.x) * smoothstep(0.0, thickness*padding*uAspect.x, uv.x);
    float right = smoothstep(1.0-thickness*(1.0+padding)*uAspect.x, 1.0-thickness*uAspect.x, uv.x) * smoothstep(1.0, 1.0-thickness*padding*uAspect.x, uv.x);
    float bottom = smoothstep(1.0-thickness*(1.0+padding)*uAspect.y, 1.0-thickness*uAspect.y, uv.y) * smoothstep(1.0, 1.0-thickness*padding*uAspect.y, uv.y);
    float top = smoothstep(thickness*(1.0+padding)*uAspect.y, thickness*uAspect.y, uv.y) * smoothstep(0.0, thickness*padding*uAspect.y, uv.y);

    // Fade overlaps
    left *= smoothstep(0.0, thickness*(1.0)*uAspect.y, uv.y) * smoothstep(1.0, 1.0-thickness*(1.0)*uAspect.y, uv.y);
    right *= smoothstep(0.0, thickness*(1.0)*uAspect.y, uv.y) * smoothstep(1.0, 1.0-thickness*(1.0)*uAspect.y, uv.y);
    bottom *= smoothstep(0.0, thickness*(1.0)*uAspect.x, uv.x) * smoothstep(1.0, 1.0-thickness*(1.0)*uAspect.x, uv.x);
    top *= smoothstep(0.0, thickness*(1.0)*uAspect.x, uv.x) * smoothstep(1.0, 1.0-thickness*(1.0)*uAspect.x, uv.x);

    float lines = left+right+bottom+top;
    return clamp(lines, 0.0, 1.0);
}

void main() {
    vec3 color = uUIColor;
    float alpha = border(vUv, uThickness);

    float flicker = sin(time*15.0)*sin(time*30.0)*sin(time*4.0)*uHover;
    color = rgb2hsv(color);
    color.x += flicker*0.08;
    color.z += -0.1+flicker*0.08;
    color = hsv2rgb(color);

    float fill = cnoise(vec3(vUv*0.5, time));
    fill = crange(fill, uHover, uHover+0.1, 0.0, 1.0);

    alpha = mix(alpha, 1.0, uHover);
    alpha = min(alpha, 1.);
    //alpha = mix(alpha, alpha*0.5, 1.0-uHover);

    gl_FragColor.rgb = color;
    gl_FragColor.a = uAlpha * alpha * (0.6+uHover*0.4);
}
{@}WorkItemButtonTxt.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uUIColor;

uniform float uAlpha;
uniform float uHover;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
#require(range.glsl)
#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);


    vec3 color = mix(uUIColor, vec3(.0), uHover);

    alpha *= uAlpha;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha * uAlpha;
}
{@}WorkListUIItem.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform vec3 uColor;
uniform float uHovered;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    vec3 color = uColor;
    color -= random(vec2(time))*uHovered*0.2;

    float alpha = 0.5+0.5*uHovered;

    gl_FragColor = vec4(color, alpha);
}{@}AntimatterSpawn.fs{@}uniform float uMaxCount;
uniform float uSetup;
uniform float decay;
uniform vec2 decayRandom;
uniform sampler2D tLife;
uniform sampler2D tAttribs;

#require(range.glsl)

void main() {
    vec2 uv = getUV();
    vec4 data = getData4(tInput, uv);

    if (getIndex() > uMaxCount) {
        gl_FragColor = vec4(9999.0);
        return;
    }

    vec4 life = texture2D(tLife, uv);
    vec4 random = texture2D(tAttribs, uv);
    if (life.x > 0.5) {
        data.x = 1.0;
        data.yzw = life.yzw;
    } else {
        data.x -= 0.005 * decay * crange(random.w, 0.0, 1.0, decayRandom.x, decayRandom.y);
    }

    if (uSetup > 0.5) {
        data = vec4(0.0);
    }

    gl_FragColor = data;
}{@}advectionManualFilteringShader.fs{@}varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
void main () {
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    gl_FragColor = dissipation * bilerp(uSource, coord, dyeTexelSize);
    gl_FragColor.a = 1.0;
}{@}advectionShader.fs{@}varying vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;
void main () {
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    gl_FragColor = dissipation * texture2D(uSource, coord);
    gl_FragColor.a = 1.0;
}{@}backgroundShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
uniform float aspectRatio;
#define SCALE 25.0
void main () {
    vec2 uv = floor(vUv * SCALE * vec2(aspectRatio, 1.0));
    float v = mod(uv.x + uv.y, 2.0);
    v = v * 0.1 + 0.8;
    gl_FragColor = vec4(vec3(v), 1.0);
}{@}clearShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
}{@}colorShader.fs{@}uniform vec4 color;
void main () {
    gl_FragColor = color;
}{@}curlShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
}{@}displayShader.fs{@}varying vec2 vUv;
uniform sampler2D uTexture;
void main () {
    vec3 C = texture2D(uTexture, vUv).rgb;
    float a = max(C.r, max(C.g, C.b));
    gl_FragColor = vec4(C, a);
}{@}divergenceShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uVelocity;
void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
//    if (vL.x < 0.0) { L = -C.x; }
//    if (vR.x > 1.0) { R = -C.x; }
//    if (vT.y > 1.0) { T = -C.y; }
//    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
}{@}fluidBase.vs{@}varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform vec2 texelSize;

void main () {
    vUv = uv;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(position, 1.0);
}{@}gradientSubtractShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
vec2 boundary (vec2 uv) {
    return uv;
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
}{@}pressureShader.fs{@}varying highp vec2 vUv;
varying highp vec2 vL;
varying highp vec2 vR;
varying highp vec2 vT;
varying highp vec2 vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
vec2 boundary (vec2 uv) {
    return uv;
    // uncomment if you use wrap or repeat texture mode
    // uv = min(max(uv, 0.0), 1.0);
    // return uv;
}
void main () {
    float L = texture2D(uPressure, boundary(vL)).x;
    float R = texture2D(uPressure, boundary(vR)).x;
    float T = texture2D(uPressure, boundary(vT)).x;
    float B = texture2D(uPressure, boundary(vB)).x;
    float C = texture2D(uPressure, vUv).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
}{@}splatShader.fs{@}varying vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec3 bgColor;
uniform vec2 point;
uniform float radius;
uniform float canRender;
uniform float uAdd;

float blendScreen(float base, float blend) {
    return 1.0-((1.0-base)*(1.0-blend));
}

vec3 blendScreen(vec3 base, vec3 blend) {
    return vec3(blendScreen(base.r, blend.r), blendScreen(base.g, blend.g), blendScreen(base.b, blend.b));
}

void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    base *= canRender;

    vec3 outColor = mix(blendScreen(base, splat), base + splat, uAdd);
    gl_FragColor = vec4(outColor, 1.0);
}{@}vorticityShader.fs{@}varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
//    force.y += 400.3;
    vec2 vel = texture2D(uVelocity, vUv).xy;
    gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
}{@}GPUCompute.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS

#!SHADER: Vertex
void main() {
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
uniform sampler2D tMap;
void main() {
    gl_FragColor = texture2D(tMap, gl_FragCoord.xy / resolution);
}{@}AreaLights.glsl{@}mat3 transposeMat3( const in mat3 m ) {
	mat3 tmp;
	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );
	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );
	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );
	return tmp;
}

// Real-Time Polygonal-Light Shading with Linearly Transformed Cosines
// by Eric Heitz, Jonathan Dupuy, Stephen Hill and David Neubelt
// code: https://github.com/selfshadow/ltc_code/
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE  = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS  = 0.5 / LUT_SIZE;
	float dotNV = clamp( dot( N, V ), 0.0, 1.0 );
	// texture parameterized by sqrt( GGX alpha ) and sqrt( 1 - cos( theta ) )
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}

float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	// Real-Time Area Lighting: a Journey from Research to Production (p.102)
	// An approximation of the form factor of a horizon-clipped rectangle.
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}

vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	// rational polynomial approximation to theta / sin( theta ) / 2PI
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}

vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	// bail if point is on back side of plane of light
	// assumes ccw winding order of light vertices
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	// construct orthonormal basis around N
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 ); // negated from paper; possibly due to a different handedness of world coordinate system
	// compute transform
	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );
	// transform rect
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	// project rect onto sphere
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	// calculate vector form factor
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	// adjust for horizon clipping
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
/*
	// alternate method of adjusting for horizon clipping (see referece)
	// refactoring required
	float len = length( vectorFormFactor );
	float z = vectorFormFactor.z / len;
	const float LUT_SIZE  = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS  = 0.5 / LUT_SIZE;
	// tabulated horizon-clipped sphere, apparently...
	vec2 uv = vec2( z * 0.5 + 0.5, len );
	uv = uv * LUT_SCALE + LUT_BIAS;
	float scale = texture2D( ltc_2, uv ).w;
	float result = len * scale;
*/
	return vec3( result );
}{@}Lighting.glsl{@}#!ATTRIBUTES

#!UNIFORMS
vec3 lNormal;
bool lPhong = false;
bool lAreaToPoint = false;
float lPhongShininess = 0.0;
float lPhongAttenuation = 1.0;
vec3 lPhongColor = vec3(1.0);
uniform sampler2D tLTC1;
uniform sampler2D tLTC2;

#!VARYINGS
varying vec3 vPos;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying vec3 vViewDir;

#!SHADER: lighting.vs

void setupLight(vec3 pos, vec3 n) {
    vPos = pos;
    vNormal = normalize(normalMatrix * n);
    vWorldPos = vec3(modelMatrix * vec4(pos, 1.0));
    vViewDir = -vec3(modelViewMatrix * vec4(pos, 1.0));

    lNormal = vNormal;
}

void setupLight(vec3 pos) {
    setupLight(pos, normal);
}

    #!SHADER: lighting.fs

    #require(LightingCommon.glsl)

void setupLight() {
    lNormal = vNormal;
}

vec3 getCombinedColor() {
    vec3 color = vec3(0.0);

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        if (lProps.w < 1.0) continue;

        if (lProps.w < 1.1) {
            lightDirectional(color, lColor, lPos, lData, lData2, lData3, lProps);
        } else if (lProps.w < 2.1) {
            lightPoint(color, lColor, lPos, lData, lData2, lData3, lProps);
        } else if (lProps.w < 3.1) {
            lightCone(color, lColor, lPos, lData, lData2, lData3, lProps);
        } else if (lProps.w < 4.1) {
            lightArea(color, lColor, lPos, lData, lData2, lData3, lProps);
        }
    }

    return lclamp(color);
}

vec3 getPointLightColor() {
    vec3 color = vec3(0.0);

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        if (lProps.w > 1.9 && lProps.w < 2.1) {
            lightPoint(color, lColor, lPos, lData, lData2, lData3, lProps);
        }
    }

    return lclamp(color);
}

vec3 getAreaLightColor(float roughness) {
    vec3 color = vec3(0.0);

    #test Lighting.fallbackAreaToPointTest()
    lAreaToPoint = true;
    #endtest

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        lData.w *= roughness;

        if (lProps.w > 3.9 && lProps.w < 4.1) {
            if (lAreaToPoint) {
                lightPoint(color, lColor, lPos, lData, lData2, lData3, lProps);
            } else {
                lightArea(color, lColor, lPos, lData, lData2, lData3, lProps);
            }
        }
    }

    return lclamp(color);
}

vec3 getAreaLightColor() {
    return getAreaLightColor(1.0);
}

vec3 getSpotLightColor() {
    vec3 color = vec3(0.0);

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        if (lProps.w > 2.9 && lProps.w < 3.1) {
            lightCone(color, lColor, lPos, lData, lData2, lData3, lProps);
        }
    }

    return lclamp(color);
}

vec3 getDirectionalLightColor() {
    vec3 color = vec3(0.0);

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        if (lProps.w > 0.9 && lProps.w < 1.1) {
            lightDirectional(color, lColor, lPos, lData, lData2, lData3, lProps);
        }
    }

    return lclamp(color);
}

vec3 getStandardColor() {
    vec3 color = vec3(0.0);

    #pragma unroll_loop
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 lColor = lightColor[i].rgb;
        vec3 lPos = lightPos[i].rgb;
        vec4 lData = lightData[i];
        vec4 lData2 = lightData2[i];
        vec4 lData3 = lightData3[i];
        vec4 lProps = lightProperties[i];

        if (lProps.w < 1.0) continue;

        if (lProps.w < 1.1) {
            lightDirectional(color, lColor, lPos, lData, lData2, lData3, lProps);
        } else if (lProps.w < 2.1) {
            lightPoint(color, lColor, lPos, lData, lData2, lData3, lProps);
        }
    }

    return lclamp(color);
}{@}LightingCommon.glsl{@}#require(AreaLights.glsl)

vec3 lworldLight(vec3 lightPos, vec3 localPos) {
    vec4 mvPos = modelViewMatrix * vec4(localPos, 1.0);
    vec4 worldPosition = viewMatrix * vec4(lightPos, 1.0);
    return worldPosition.xyz - mvPos.xyz;
}

float lrange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    float newRange = newMax - newMin;
    return (((oldValue - oldMin) * newRange) / oldRange) + newMin;
}

vec3 lclamp(vec3 v) {
    return clamp(v, vec3(0.0), vec3(1.0));
}

float lcrange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    return clamp(lrange(oldValue, oldMin, oldMax, newMin, newMax), min(newMax, newMin), max(newMin, newMax));
}

    #require(Phong.glsl)

void lightDirectional(inout vec3 color, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps) {
    float shininess = lData.x;
    float specularity = lData.y;
    float strength = lProps.x;

    vec3 lDir = lworldLight(lPos, vPos);
    vec3 halfDirection = lDir + vViewDir;
    float diffuse = max(dot(lNormal, lDir), 0.0);
    float halfDot = max(dot(lNormal, halfDirection), 0.0);
    float specular = max(pow(halfDot, shininess), 0.0);

    color += lColor * 0.1 * strength * (diffuse + diffuse * specular * specularity);
}

void lightPoint(inout vec3 color, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return;

    vec3 lDir = lworldLight(lPos, vPos);

    float falloff = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);
    if (lPhong) {
        color += falloff * phong(lProps.x, lColor, lPhongColor, lPhongShininess, lPhongAttenuation, lNormal, normalize(lDir), vViewDir, lProps.z);
    } else {
        float volume = dot(normalize(lDir), lNormal);
        volume = lcrange(volume, 0.0, 1.0, lProps.z, 1.0);
        color += lColor * volume * lProps.x * falloff;
    }
}

void lightCone(inout vec3 color, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return;

    vec3 lDir = lworldLight(lPos, vPos);
    vec3 sDir = degrees(-lData.xyz);
    float radius = lData.w;
    vec3 surfacePos = vWorldPos;
    vec3 surfaceToLight = normalize(lPos - surfacePos);
    float lightToSurfaceAngle = degrees(acos(dot(-surfaceToLight, normalize(sDir))));
    float attenuation = 1.0;

    vec3 nColor = color;
    lightPoint(nColor, lColor, lPos, lData, lData2, lData3, lProps);

    float featherMin = 1.0 - lData2.x*0.1;
    float featherMax = 1.0 + lData2.x*0.1;

    attenuation *= smoothstep(lightToSurfaceAngle*featherMin, lightToSurfaceAngle*featherMax, radius);

    nColor *= attenuation;

    color += nColor;
}

void lightArea(inout vec3 color, vec3 lColor, vec3 lPos, vec4 lData, vec4 lData2, vec4 lData3, vec4 lProps) {
    float dist = length(vWorldPos - lPos);
    if (dist > lProps.y) return;

    vec3 normal = lNormal;
    vec3 viewDir = normalize(vViewDir);
    vec3 position = -vViewDir;
    float roughness = lData.w;
    vec3 mPos = lData.xyz;
    vec3 halfWidth = lData2.xyz;
    vec3 halfHeight = lData3.xyz;

    float falloff = pow(lcrange(dist, 0.0, lProps.y, 1.0, 0.0), 2.0);

    vec3 rectCoords[ 4 ];
    rectCoords[ 0 ] = mPos + halfWidth - halfHeight;
    rectCoords[ 1 ] = mPos - halfWidth - halfHeight;
    rectCoords[ 2 ] = mPos - halfWidth + halfHeight;
    rectCoords[ 3 ] = mPos + halfWidth + halfHeight;

    vec2 uv = LTC_Uv( normal, viewDir, roughness );

    vec4 t1 = texture2D( tLTC1, uv );
    vec4 t2 = texture2D( tLTC2, uv );

    mat3 mInv = mat3(
    vec3( t1.x, 0, t1.y ),
    vec3(    0, 1,    0 ),
    vec3( t1.z, 0, t1.w )
    );

    vec3 fresnel = ( lColor * t2.x + ( vec3( 1.0 ) - lColor ) * t2.y );
    color += lColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords ) * falloff * lProps.x;
    color += lColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords ) * falloff * lProps.x;
}{@}LitMaterial.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;
varying vec3 vPos;

#!SHADER: Vertex

#require(lighting.vs)

void main() {
    vUv = uv;
    vPos = position;
    setupLight(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(lighting.fs)
#require(shadows.fs)

void main() {
    setupLight();

    vec3 color = texture2D(tMap, vUv).rgb;
    color *= getShadow(vPos);

    color += getCombinedColor();

    gl_FragColor = vec4(color, 1.0);
}{@}Phong.glsl{@}float saturate(float v) {
    return clamp(v, 0.0, 1.0);
}

float dPhong(float shininess, float dotNH) {
    return (shininess * 0.5 + 1.0) * pow(dotNH, shininess);
}

vec3 schlick(vec3 specularColor, float dotLH) {
    float fresnel = exp2((-5.55437 * dotLH - 6.98316) * dotLH);
    return (1.0 - specularColor) * fresnel + specularColor;
}

vec3 calcBlinnPhong(vec3 specularColor, float shininess, vec3 normal, vec3 lightDir, vec3 viewDir) {
    vec3 halfDir = normalize(lightDir + viewDir);
    
    float dotNH = saturate(dot(normal, halfDir));
    float dotLH = saturate(dot(lightDir, halfDir));

    vec3 F = schlick(specularColor, dotLH);
    float G = 0.85;
    float D = dPhong(shininess, dotNH);
    
    return F * G * D;
}

vec3 calcBlinnPhong(vec3 specularColor, float shininess, vec3 normal, vec3 lightDir, vec3 viewDir, float minTreshold) {
    vec3 halfDir = normalize(lightDir + viewDir);

    float dotNH = saturate(dot(normal, halfDir));
    float dotLH = saturate(dot(lightDir, halfDir));

    dotNH = lrange(dotNH, 0.0, 1.0, minTreshold, 1.0);
    dotLH = lrange(dotLH, 0.0, 1.0, minTreshold, 1.0);

    vec3 F = schlick(specularColor, dotLH);
    float G = 0.85;
    float D = dPhong(shininess, dotNH);

    return F * G * D;
}

vec3 phong(float amount, vec3 diffuse, vec3 specular, float shininess, float attenuation, vec3 normal, vec3 lightDir, vec3 viewDir, float minThreshold) {
    float cosineTerm = saturate(lrange(dot(normal, lightDir), 0.0, 1.0, minThreshold, 1.0));
    vec3 brdf = calcBlinnPhong(specular, shininess, normal, lightDir, viewDir, minThreshold);
    return brdf * amount * diffuse * attenuation * cosineTerm;
}{@}mousefluid.fs{@}uniform sampler2D tFluid;
uniform sampler2D tFluidMask;

vec2 getFluidVelocity() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return texture2D(tFluid, vUv).xy * fluidMask;
}

vec3 getFluidVelocityMask() {
    float fluidMask = smoothstep(0.1, 0.7, texture2D(tFluidMask, vUv).r);
    return vec3(texture2D(tFluid, vUv).xy * fluidMask, fluidMask);
}{@}ProtonAntimatter.fs{@}uniform sampler2D tOrigin;
uniform sampler2D tAttribs;
uniform float uMaxCount;
//uniforms

#require(range.glsl)
//requires

void main() {
    vec2 uv = getUV();
    vec3 origin = getData(tOrigin, uv);
    vec4 inputData = getData4(tInput, uv);
    vec3 pos = inputData.xyz;
    vec4 random = getData4(tAttribs, uv);
    float data = inputData.w;

    if (getIndex() > uMaxCount) {
        gl_FragColor = vec4(9999.0);
        return;
    }

    //code

    gl_FragColor = vec4(pos, data);
}{@}ProtonAntimatterLifecycle.fs{@}uniform sampler2D tOrigin;
uniform sampler2D tAttribs;
uniform sampler2D tSpawn;
uniform float uMaxCount;
//uniforms

#require(range.glsl)
//requires

void main() {
    vec2 uv = getUV();
    vec3 origin = getData(tOrigin, uv);
    vec4 inputData = getData4(tInput, uv);
    vec3 pos = inputData.xyz;
    vec4 random = getData4(tAttribs, uv);
    float data = inputData.w;

    if (getIndex() > uMaxCount) {
        gl_FragColor = vec4(9999.0);
        return;
    }

    vec4 spawn = texture2D(tSpawn, uv);

    if (spawn.x > 0.9999) {
        pos = spawn.yzw;
        gl_FragColor = vec4(pos, data);
        return;
    }

    //abovespawn
    if (spawn.x <= 0.0) {
        pos.x = 9999.0;
        gl_FragColor = vec4(pos, data);
        return;
    }

    //abovecode
    //code

    gl_FragColor = vec4(pos, data);
}{@}ProtonNeutrino.fs{@}//uniforms

#require(range.glsl)
//requires

void main() {
    //code
}{@}ShadowInspector.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(depthvalue.fs)

void main() {
    gl_FragColor = vec4(vec3(getDepthValue(tMap, vUv, 10.0, 51.0)), 1.0);
}{@}Text3D.glsl{@}#!ATTRIBUTES
attribute vec3 animation;

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uColor;
uniform float uAlpha;
uniform vec3 uTranslate;
uniform vec3 uRotate;
uniform float uTransition;
uniform float uWordCount;
uniform float uLineCount;
uniform float uLetterCount;
uniform float uByWord;
uniform float uByLine;
uniform float uPadding;
uniform vec3 uBoundingMin;
uniform vec3 uBoundingMax;

#!VARYINGS
varying float vTrans;
varying vec2 vUv;
varying vec3 vPos;

#!SHADER: Vertex

#require(range.glsl)
#require(eases.glsl)
#require(rotation.glsl)
#require(conditionals.glsl)

void main() {
    vUv = uv;
    vTrans = 1.0;

    vec3 pos = position;

    if (uTransition < 5.0) {
        float padding = uPadding;
        float letter = (animation.x + 1.0) / uLetterCount;
        float word = (animation.y + 1.0) / uWordCount;
        float line = (animation.z + 1.0) / uLineCount;

        float letterTrans = crange(uTransition, letter - padding, letter + padding, 0.0, 1.0);
        float wordTrans = crange(uTransition, word - padding, word + padding, 0.0, 1.0);
        float lineTrans = crange(uTransition, line - padding, line + padding, 0.0, 1.0);

        vTrans = mix(cubicOut(letterTrans), cubicOut(wordTrans), uByWord);
        vTrans = mix(vTrans, cubicOut(lineTrans), uByLine);

        float invTrans = (1.0 - vTrans);
        vec3 nRotate = normalize(uRotate);
        vec3 axisX = vec3(1.0, 0.0, 0.0);
        vec3 axisY = vec3(0.0, 1.0, 0.0);
        vec3 axisZ = vec3(0.0, 0.0, 1.0);
        vec3 axis = mix(axisX, axisY, when_gt(nRotate.y, nRotate.x));
        axis = mix(axis, axisZ, when_gt(nRotate.z, nRotate.x));
        pos = vec3(vec4(position, 1.0) * rotationMatrix(axis, radians(max(max(uRotate.x, uRotate.y), uRotate.z) * invTrans)));
        pos += uTranslate * invTrans;
    }

    vPos = pos;

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(msdf.glsl)
#require(simplenoise.glsl)

vec2 getBoundingUV() {
    vec2 uv;
    uv.x = crange(vPos.x, uBoundingMin.x, uBoundingMax.x, 0.0, 1.0);
    uv.y = crange(vPos.y, uBoundingMin.y, uBoundingMax.y, 0.0, 1.0);
    return uv;
}

void main() {
    float alpha = msdf(tMap, vUv);

    //float noise = 0.5 + smoothstep(-1.0, 1.0, cnoise(vec3(vUv*50.0, time* 0.3))) * 0.5;

    gl_FragColor.rgb = uColor;
    gl_FragColor.a = alpha * uAlpha * vTrans;
}
{@}VideoGridTexture.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec2 uPos;
uniform vec2 uSize;

#!VARYINGS
varying vec2 vUv;

#!SHADER: VideoGridTexture.vs

#require(transformUV.glsl)

void main() {
    vUv = uv;
    vUv = scaleUV( vUv, uSize );
    vUv = translateUV( vUv, uPos );
    gl_Position = vec4(position, 1.0);
}

#!SHADER: VideoGridTexture.fs
void main() {
    gl_FragColor = texture2D(tMap, vUv);
}{@}InteractiveWaterHeightmap.fs{@}uniform vec2 mousePos;
uniform float mouseSize;
uniform float viscosityConstant;
uniform float WIDTH;
uniform float BOUNDS;

#ifndef PI
#define PI 3.141592653589793
#endif

void main()	{
    vec2 cellSize = 1.0 / resolution;

    vec2 uv = gl_FragCoord.xy * cellSize;

    // heightmapValue.x == height from previous frame
    // heightmapValue.y == height from penultimate frame
    // heightmapValue.z, heightmapValue.w not used
    vec4 heightmapValue = texture2D( heightmap, uv );

    // Get neighbours
    vec4 north = texture2D( heightmap, uv + vec2( 0.0, cellSize.y ) );
    vec4 south = texture2D( heightmap, uv + vec2( 0.0, - cellSize.y ) );
    vec4 east = texture2D( heightmap, uv + vec2( cellSize.x, 0.0 ) );
    vec4 west = texture2D( heightmap, uv + vec2( - cellSize.x, 0.0 ) );

    // https://web.archive.org/web/20080618181901/http://freespace.virgin.net/hugo.elias/graphics/x_water.htm

    float newHeight = ( ( north.x + south.x + east.x + west.x ) * 0.5 - heightmapValue.y ) * viscosityConstant;

    // Mouse influence
    float mousePhase = clamp( length( ( uv - vec2( 0.5 ) ) * BOUNDS - vec2( mousePos.x, - mousePos.y ) ) * PI / mouseSize, 0.0, PI );
    newHeight += ( cos( mousePhase ) + 1.0 ) * 0.28;

    heightmapValue.y = heightmapValue.x;
    heightmapValue.x = newHeight;

    gl_FragColor = heightmapValue;

}{@}WaterShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS

#!VARYINGS
varying vec3 vNormal;

#!SHADER: Vertex

#require(water.vs)

void main() {
    vec3 pos = calculateWaterPos();
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}

#!SHADER: Fragment
void main() {
    gl_FragColor = vec4(vNormal, 1.0);
}{@}water.vs{@}uniform sampler2D heightmap;
uniform float heightScale;
uniform float WIDTH;
uniform float BOUNDS;

vec3 calculateWaterPos() {
    vec2 cellSize = vec2( 1.0 / WIDTH, 1.0 / WIDTH );
    vec3 objectNormal = vec3(
                        ( texture2D( heightmap, uv + vec2( - cellSize.x, 0 ) ).x - texture2D( heightmap, uv + vec2( cellSize.x, 0 ) ).x ) * WIDTH / BOUNDS,
                        ( texture2D( heightmap, uv + vec2( 0, - cellSize.y ) ).x - texture2D( heightmap, uv + vec2( 0, cellSize.y ) ).x ) * WIDTH / BOUNDS,
                        1.0 );


    vNormal = normalize(normalMatrix * objectNormal);

    float heightValue = texture2D(heightmap, uv).x;
    vec3 pos = position;
    pos.z += heightValue * heightScale;

    return pos;
}{@}AboutCover.fs{@}uniform sampler2D tRt;
uniform sampler2D tGlitch;
uniform sampler2D tFluid;
uniform sampler2D tFluidMask;
uniform sampler2D tLogo;
uniform float uGlitchScale;
uniform float uDistort;
uniform float uTransition;
uniform float uLogoAlpha;
uniform float uHome;
uniform float uPhone;
uniform vec3 uColor;
uniform vec3 uUIColor;

#require(blendmodes.glsl)
#require(range.glsl)
#require(simplenoise.glsl)
#require(rgbshift.fs)
#require(rgb2hsv.fs)
#require(transformUV.glsl)

void main() {
    vec2 uv = vUv;
    uv = scaleUV(uv, vec2(1.0-uTransition*0.03, 1.0), vec2(1.0, 0.5));

    float fluidMask = texture2D(tFluidMask, vUv).r * smoothstep(0.5, 1.0, uTransition);
    vec2 fluid = texture2D(tFluid, vUv).xy * fluidMask;
    uv += fluid * 0.005 * uTransition;

    vec2 distUv = mix(uv, vUv, uTransition);
    float glitch = texture2D(tGlitch, distUv * uGlitchScale).r;
    vec2 disp = vec2(0., glitch) * (1.-uTransition) * .3;
    disp += vec2(sin(time * 3.), sin(time * .2 + .6532)) * .001 * (1.-uTransition);
    vec4 rt = texture2D(tRt, distUv + disp); //+ fluid*0.005

    float wobble = 0.1 + cnoise(vec3(vUv*0.2, time*0.05))*0.08;
    uv.x += glitch * uDistort * wobble * uTransition;
    uv.y += glitch * uDistort * wobble * uTransition * 0.2;
    uv += fluid * 0.007 * uTransition;

    vec3 texel = getRGB(tDiffuse, uv, time*0.2, fluidMask*0.06*uTransition).rgb;
    vec3 color = mix(texel, blendSubtract(uUIColor, texel), uTransition);
    color = mix(color, color*(0.3+uHome*0.5), uTransition);
    color = mix(color, vec3(0.24), 0.4 * uTransition);

    //color = mix(color, blendOverlay(color, texel), fluidMask*uTransition);

    vec2 logoUV = vUv;
    logoUV += fluid * 0.005 * uTransition;
    logoUV *= 1.0-uLogoAlpha*0.05;
    logoUV = scaleUV(logoUV, vec2(1.0, resolution.x/resolution.y) * mix(.9, 1.2, uPhone));
    logoUV = translateUV(logoUV, vec2(0.22, 0.0));
    logoUV *= 1.0 + cnoise(vec3(vUv*1.2, time*0.4)) * 0.01;

    vec3 logo = texture2D(tLogo, logoUV).rgb;
    logo = uUIColor*logo.r;

    float maskOutline = smoothstep(0.0, 0.5, fluidMask) * smoothstep(1.0, 0.5, fluidMask);

    float logoAlpha = (0.08+cnoise(vec3(vUv*2.0, time*0.4))*0.04+smoothstep(0.1, 0.102, maskOutline)*0.15)*smoothstep(0.6, 1.0, uTransition);
    logoAlpha *= uLogoAlpha;

    color = mix(color, blendAdd(color, logo), logoAlpha);
    color = mix(color, blendOverlay(color, uColor), uTransition*0.3);

    //color = rgb2hsv(color);
    //color.x += fluidMask*0.1;
    //color.y *= 1.0-0.3 * uTransition;
    //color.z = crange(color.z, 0.0, 1.0, 0.02, 0.8);
    //color = hsv2rgb(color);
    //color = mix(color, blendSubtract(color, vec3(0.0)), smoothstep(0.7, 0.71, fluidMask)*0.5);
    //color = mix(color, uUIColor, 0.2 * uTransition);

    gl_FragColor = vec4(color + rt.rgb * uTransition, 1.0);
}{@}AboutObject.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uUIColor;
// uniform vec3 uColor0;
// uniform vec3 uColor1;
// uniform vec3 uColor2;
uniform float uAlpha;
uniform float uNoiseScale;
uniform float uNoiseSpeed;
uniform float uHover;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
#require(range.glsl)
#require(simplenoise.glsl)

void main() {
    float alpha = texture2D(tMap, vUv).r;

    vec2 screenUV = gl_FragCoord.xy / resolution;
    // float noise = cnoise(vec3(screenUV*uNoiseScale + .4324, time*uNoiseSpeed*0.3 + .345));
    // float noise2 = cnoise(vec3(screenUV*uNoiseScale*0.8, time*uNoiseSpeed*0.3));
    float noiseOver = cnoise(vec3(screenUV*uNoiseScale*10., time*uNoiseSpeed*.7)) * uHover;

    vec3 color = uUIColor;
    // color = mix(color, uColor1, clamp(noise, 0., 1.));
    // color = mix(color, uColor2, clamp(noise2, 0., 1.));
    color = mix(color, vec3(1.), clamp(noiseOver, 0., 1.));
    color = mix(color, vec3(1.), .2+uHover*0.5);

    alpha *= uAlpha;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha * uAlpha;
}
{@}AboutText.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform vec3 uUIColor;
// uniform vec3 uColor0;
// uniform vec3 uColor1;
// uniform vec3 uColor2;
uniform float uAlpha;
// uniform vec3 uColor3;
uniform float uNoiseScale;
uniform float uNoiseSpeed;
uniform float uHover;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment
#require(range.glsl)
#require(msdf.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    vec2 screenUV = gl_FragCoord.xy / resolution;
    // float noise = cnoise(vec3(screenUV*uNoiseScale + .4324, time*uNoiseSpeed*0.3 + .345));
    // float noise2 = cnoise(vec3(screenUV*uNoiseScale*0.8, time*uNoiseSpeed*0.3));
    // float noiseOver = cnoise(vec3(screenUV*uNoiseScale*10., time*uNoiseSpeed*.7)) * uHover;

    vec3 color = uUIColor;
    // color = mix(color, uColor1, clamp(noise, 0., 1.));
    // color = mix(color, uColor2, clamp(noise2, 0., 1.));
    // color = mix(color, vec3(1.), clamp(noiseOver, 0., 1.));
    color = mix(color, vec3(1.0), uHover);
    //color = mix(color, vec3(1.), .4);

    alpha *= uAlpha;

    gl_FragColor.rgb = color;
    gl_FragColor.a = alpha * uAlpha;
}
{@}TestObjectShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;
varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

void main() {
    gl_FragColor = texture2D(tMap, vUv);
    gl_FragColor.a *= v_uAlpha;
}
{@}TestTextShader.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;

#!VARYINGS
varying vec2 vUv;
varying vec3 v_uColor;
varying float v_uAlpha;

#!SHADER: Vertex
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)

void main() {
    float alpha = msdf(tMap, vUv);

    gl_FragColor.rgb = v_uColor;
    gl_FragColor.a = alpha * v_uAlpha;
}
{@}HomeToListTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform float uTransition;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment
void main() {
    vec3 from = texture2D(tFrom, vUv).rgb;
    vec3 to = texture2D(tTo, vUv).rgb;

    vec3 color = mix(from, to, uTransition);

    gl_FragColor = vec4(color, 1.0);
}{@}HomeTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform sampler2D tDepth;
uniform sampler2D tGlitch;
uniform float uTransition;
uniform float uDistort;
uniform float uPhone;
uniform float uDirection;
uniform float uFar;
uniform float uGlitchScale;
uniform float uHomeHover;
uniform float uSpaceValue;
uniform vec3 uColor;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(depthvalue.fs)
#require(range.glsl)
#require(eases.glsl)
#require(simplenoise.glsl)
#require(rgbshift.fs)
#require(rgb2hsv.fs)
#require(blendmodes.glsl)


void main() {
    float depth = getDepthValue(tDepth, vUv, 0.1, uFar);
    //depth = 1.0-depth;

    //float middle = 0.2;
    //depth = crange(depth, 0.0, middle, 0.5, 0.0)+crange(depth, middle, 1.0, 0.5, 0.0);

    float transitionBounce = smoothstep(0.3, 0.7, uTransition) * smoothstep(0.75, 0.7, uTransition);
    //transitionBounce = mix(transitionBounce, 1.0, uSpaceValue);

    float padding = 0.1;
    float t = crange(uTransition, 0.0, 1.0, 0.0 - padding, 1.0 + padding);
    //t = mix(t, 0.3, uHomeHover);

    float transition = sineIn(crange(t, depth-padding, depth+padding, 0.0, 1.0));

    vec2 uv = vUv;
    float distort = mix(uDistort, uDistort*2.0, uSpaceValue);
    float curve = sineInOut(crange(transition, 0.0, 0.5, 0.0, 1.0) * crange(transition, 0.5, 1.0, 1.0, 0.0));
    float warp = uDistort * curve * texture2D(tGlitch, (vUv * uGlitchScale) + random(vec2(time))).r;
    warp *= 0.4+smoothstep(0.8, 1.0, 1.0-uHomeHover)*0.6;
    warp *= 1.0 + 1.0*smoothstep(0.7, 1.0, uSpaceValue);
    uv.x += warp * mix(0.03, 0.0, uSpaceValue) * uDirection; //crange(smoothstep(0.3, 0.7, uv.x), 0.0, 1.0, 1.0, -1.0)
    //uv.y += warp * 0.1*uSpaceValue;

    float progress = uTransition;
    vec3 from = getRGB(tFrom, uv, 0.1, warp*0.01+smoothstep(0.0, 1.0, warp)*transitionBounce*0.002).rgb;
    vec3 to = texture2D(tTo, uv).rgb;

//    float index = floor(uv.y * resolution.y * 0.5);
//    vec2 uv1 = uv;
//    float offset1 = random( vec2(index * 1.3, index * 2.4) ) * 0.1 * progress * pow(1.0-rgb2hsv(from).z, 2.0);
//    float offset2 = random( vec2(index * 1.3, index * 2.4) ) * 0.5 * progress * pow(rgb2hsv(from).z, 2.0);
//    uv1.x += offset2;
//    uv1.y += offset1;
//    vec3 from2 = getRGB(tFrom, uv1, time*0.1, progress*0.002).rgb;

    vec3 color = mix(from, to, transition);


    vec3 transColor = mix(uColor, vec3(1.0), mix(0.3, 0.6, uPhone));
    transColor = rgb2hsv(transColor);
    transColor.x += (1.0-smoothstep(-0.4, 0.7, uv.x)*smoothstep(1.4, 0.3, uv.x))*mix(0.2, 0.2, smoothstep(0.7, 1.0, uSpaceValue));
    float transZ = transColor.z;
    transColor.z *= smoothstep(-0.4, 0.2, uv.y)*smoothstep(1.4, 0.8, uv.y);
    transColor.z = mix(transColor.z, transZ, smoothstep(0.7, 1.0, uSpaceValue));
    transColor.z = mix(transColor.z, transZ, uPhone);
    transColor = hsv2rgb(transColor);

    color = mix(color, blendVividLight(color*1.5, transColor), smoothstep(0.1, 1.0, warp)*transitionBounce);
    color = mix(color, blendHardLight(color*1.5, transColor), smoothstep(0.4, 1.0, warp)*transitionBounce*0.8);

    gl_FragColor = vec4(color, 1.0);
}{@}ScreenInTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform sampler2D tRGB;
uniform float uTransition;
uniform float uNoiseStrength;
uniform float uNoiseScale;
uniform float uWarp;
uniform float uRGB;
uniform float uRGBScale;
uniform float uRGBPass;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

void main() {
    float dist = length(vUv - vec2(0.5));
    dist += cnoise(vUv * uNoiseScale) * 0.1 * uNoiseStrength * crange(uTransition, 0.0, 0.1, 0.0, 1.0);

    float fScale = mix(1.0, uWarp, crange(dist, 0.0, 0.5, 1.0, 0.0));

    vec2 fuv = scaleUV(vUv, vec2(fScale));

    vec3 from = texture2D(tFrom, fuv).rgb;
    vec3 to = texture2D(tTo, fuv).rgb;

    vec3 rgb = texture2D(tRGB, scaleUV(vUv * uRGBScale, vec2(uRGBPass))).rgb;
    vec3 blended = to * rgb;
    to = mix(to, blended, uRGB);

    float a = smoothstep(dist, dist+0.2, uTransition);
    vec3 color = mix(from, to, a);
    color = mix(color, from, crange(uTransition, 0.0, 0.45, 1.0, 0.0));

    gl_FragColor = vec4(color, 1.0);
}{@}ScreenOutTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform sampler2D tRGB;
uniform float uTransition;
uniform float uNoiseStrength;
uniform float uNoiseScale;
uniform float uWarp;
uniform float uRGBScale;
uniform float uRGBPass;
uniform float uRGB;
uniform float uRGBOutside;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

#require(range.glsl)
#require(transformUV.glsl)
#require(simplenoise.glsl)

void main() {
    float dist = length(vUv - vec2(0.5));
    dist += cnoise(vUv * uNoiseScale) * 0.1 * uNoiseStrength * crange(uTransition, 0.0, 0.1, 0.0, 1.0);

    float fScale = mix(1.0, uWarp, crange(dist, 0.0, 0.5, 1.0, 0.0));

    vec2 fuv = scaleUV(vUv, vec2(fScale));

    vec3 from = texture2D(tFrom, fuv).rgb;
    vec3 to = texture2D(tTo, fuv).rgb;

    vec3 rgb = texture2D(tRGB, scaleUV(vUv * uRGBScale, vec2(uRGBPass))).rgb;
    vec3 blended = from * rgb;
    vec3 blended2 = to * rgb;
    from = mix(from, blended, uRGB * crange(uTransition, 0.0, 0.1, 0.0, 1.0));
    to = mix(to, blended2, uRGB * crange(uTransition, 0.0, 0.1, 0.0, 1.0) * uRGBOutside);

    float a = 1.0 - smoothstep(dist, dist+0.2, 1.0 - crange(uTransition, 0.0, 1.0, 0.0, 1.2));
    vec3 color = mix(from, to, a);

    gl_FragColor = vec4(color, 1.0);
}{@}WorkTransition.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tFrom;
uniform sampler2D tTo;
uniform sampler2D tGlitch;
uniform sampler2D tGlitch2;
uniform float uTransition;
uniform float uDirection;
uniform float uGlitchScale;
uniform float uGlitchScale2;
uniform float uGlitchScroll;
uniform float uStretch;
uniform vec3 uNoise;

#!VARYINGS
varying vec2 vUv;

#!SHADER: Vertex
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}

#!SHADER: Fragment

vec3 glitch;

#require(range.glsl)
#require(simplenoise.glsl)
#require(conditionals.glsl)
#require(blendmodes.glsl)
#require(eases.glsl)
#require(transformUV.glsl)

vec2 barrel(vec2 uv, float strength) {
    #test Tests.transitionBarrel()
    vec2 st = uv - 0.5;
    float theta = atan(st.x, st.y);
    float radius = sqrt(dot(st, st));
    radius *= 1.0 + strength * (radius * radius);

    return 0.5 + radius * vec2(sin(theta), cos(theta));
    #endtest

    #test !Tests.transitionBarrel()
    return uv;
    #endtest
}

float getCrtNoise(vec2 uv) {
    return 0.1 * cnoise(vUv * 1.4 + time * 1.5) * uTransition;
}

float getTransitionUp(float padding) {
    float transition = crange(uTransition, 0.0, 1.0, -abs(padding), 1.0 + abs(padding));
    float t = crange(vUv.y, transition - padding, transition + padding, 0.0, 1.0);
    return 1.0 - t;
}

float getTransitionDown(float padding) {
    float transition = crange(uTransition, 0.0, 1.0, -abs(padding), 1.0 + abs(padding));
    float t = crange(1.0 - vUv.y, transition - padding, transition + padding, 0.0, 1.0);
    return 1.0 - t;
}

float getTransition(float padding) {
    return mix(getTransitionUp(padding), getTransitionDown(padding), when_lt(uDirection, 0.0));
}

float getCenterFromUp(float padding) {
    return 1.0 - crange(getTransitionUp(-padding), 0.5, 1.0, 0.0, 1.0);
}

float getCenterToUp(float padding) {
    return 1.0 - crange(getTransitionUp(padding), 0.5, 1.0, 0.0, 1.0);
}

float getCenterFromDown(float padding) {
    return 1.0 - crange(getTransitionDown(-padding), 0.5, 1.0, 0.0, 1.0);
}

float getCenterToDown(float padding) {
    return 1.0 - crange(getTransitionDown(padding), 0.5, 1.0, 0.0, 1.0);
}

float getCenter(float padding) {
    return mix(
    1.0 - (1.0 - getCenterFromUp(padding) +  1.0 - getCenterToUp(padding)),
    1.0 - (1.0 - getCenterFromDown(padding) + 1.0 - getCenterToDown(padding)),
    when_lt(uDirection, 0.0));
}

vec2 getFromUVUp() {
    vec2 uv = vUv;
    uv.x += cnoise((vUv.yy * uNoise.x) + (uNoise.z * time)) * uNoise.y * 0.1;

    uv = mix(vUv, uv, getCenterFromUp(0.2));
    uv.y -= crange(uTransition, 0.0, 1.0, 0.0, 0.5);

    #test Tests.transitionScale()
    uv = scaleUV(uv, vec2(1.0, crange(uTransition, 0.0, 1.0, 1.0, uStretch)));
    #endtest

    uv.y += glitch.r * 0.1 * sineOut(uTransition);

    return uv;
}

vec2 getFromUVDown() {
    vec2 uv = vUv;

    #test Tests.hdWorkTransition()
    uv.x += cnoise((vUv.yy * uNoise.x) + (uNoise.z * time)) * uNoise.y * 0.1;

    uv = mix(vUv, uv, getCenterFromDown(0.2));
    uv.y += crange(uTransition, 0.0, 1.0, 0.0, 0.5);

    #test Tests.transitionScale()
    uv = scaleUV(uv, vec2(1.0, crange(uTransition, 0.0, 1.0, 1.0, uStretch)));
    #endtest

    uv.y += glitch.r * 0.1 * sineOut(uTransition);
    #endtest


    return uv;
}


vec2 getToUVUp() {
    vec2 uv = vUv;

    #test Tests.hdWorkTransition()
    uv.x -= cnoise((vUv.yy * uNoise.x) + (uNoise.z * time)) * uNoise.y * 0.15;
    uv = mix(vUv, uv, getCenterToUp(0.2));
    uv.y += crange(uTransition, 0.0, 1.0, 0.5, 0.0);
    uv = scaleUV(uv, vec2(crange(uTransition * mix(crange(uv.y, 1.0, 0.0, 2.0, 1.0), 1.0, uTransition), 0.0, 1.0, uStretch, 1.0)));
    #endtest
    return uv;
}

vec2 getToUVDown() {
    vec2 uv = vUv;
    uv.x += cnoise((vUv.yy * uNoise.x) + (uNoise.z * time)) * uNoise.y * 0.15;

    uv = mix(vUv, uv, getCenterToDown(0.2));
    uv.y -= crange(uTransition, 0.0, 1.0, 0.5, 0.0);

    #test Tests.transitionScale()
    uv = scaleUV(uv, vec2(crange(uTransition * mix(crange(uv.y, 0.0, 1.0, 2.0, 1.0), 1.0, uTransition), 0.0, 1.0, uStretch, 1.0)));
    #endtest

    return uv;
}

vec2 getUVFrom() {
    return mix(getFromUVUp(), getFromUVDown(), when_lt(uDirection, 0.0));
}

vec2 getUVTo() {
    return barrel(mix(getToUVUp(), getToUVDown(), when_lt(uDirection, 0.0)), -4.0 * (1.0 - uTransition));
}

void applyGlitch(inout vec3 color) {
    vec2 uv = vUv * uGlitchScale;
    // uv.x += cnoise((vUv.yy * uNoise.x) + (uNoise.z * time)) * uNoise.y * 0.1;
    // uv.y -= time * uGlitchScroll;
    uv.y += getCrtNoise(vUv);

    vec3 glitch = blendOverlay(color, texture2D(tGlitch, uv).rgb);

    color = mix(color, glitch * 2.0, sineInOut(getCenter(0.4)));
}

void main() {
    #test Tests.addGlitchWorkTrasition()
    glitch = texture2D(tGlitch2, vUv * uGlitchScale2).rgb;
    #endtest

    vec3 from = texture2D(tFrom, getUVFrom()).rgb;
    from = from * 0.2 + from * 0.8 * (1.0 - uTransition);
    vec3 to = texture2D(tTo, getUVTo()).rgb;

    float blend = getTransition(0.1);
    vec3 color = mix(from, to, blend);

    #test Tests.addGlitchWorkTrasition()
    applyGlitch(color);
    #endtest

    gl_FragColor = vec4(color, 1.0);
}{@}WorkListText.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;
uniform float uStroke;
uniform float uPadding;
uniform float uHover;
uniform float uTransition;
uniform vec3 uBoundingMin;
uniform vec3 uBoundingMax;
uniform vec3 uTextColor;

#!VARYINGS
varying vec2 vUv;
varying vec2 vUv2;

#!SHADER: Vertex

#require(range.glsl)

vec2 getBoundingUV() {
    vec2 uv;
    uv.x = crange(position.x, uBoundingMin.x, uBoundingMax.x, 0.0, 1.0);
    uv.y = crange(position.y, uBoundingMin.y, uBoundingMax.y, 0.0, 1.0);
    return uv;
}

void main() {
    vUv = uv;
    vUv2 = getBoundingUV();
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)
#require(range.glsl)
#require(levelmask.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)

void main() {
    vec2 uv = vUv;

    float noise = cnoise(vec3(vUv*400.0, time));
    uv *= 1.0+noise*crange(uHover, 0.0, 0.5, 0.0, 1.0)*crange(uHover, 0.5, 1.0, 1.0, 0.0)*0.002;

    float fill = msdf(tMap, uv);
    float stroke = strokemsdf(tMap, uv, uStroke, uPadding * 0.1);

    float hover = crange(uHover, 0.0, 1.0, 0.05, 0.9);
    //hover *= smoothstep(vUv.x-0.1, vUv.x+0.1, uHover);

    float alpha = mix(stroke, fill, hover);
    alpha *= crange(uHover, 0.0, 1.0, 0.75, 1.0);

    if (uTransition < 2.0) {
        alpha *= animateLevels(texture2D(tMask, (gl_FragCoord.xy / resolution) * 3.0).r, uTransition);
    }

    vec3 color = vec3(uTextColor);

    float flicker = sin(time*10.0)*sin(time*20.0)*sin(time*4.0)*uHover;
    color = rgb2hsv(color);
    color.x += flicker*0.03;
    color.z += flicker*0.03;
    color = hsv2rgb(color);


    gl_FragColor.rgb = color;//vec3(getBoundingUV(), 1.0);
    gl_FragColor.a = alpha;
}{@}WorkListTextBatch.glsl{@}#!ATTRIBUTES

#!UNIFORMS
uniform sampler2D tMap;
uniform sampler2D tMask;

#!VARYINGS
varying vec2 vUv;
varying vec2 vUv2;
varying float v_uStroke;
varying float v_uPadding;
varying float v_uHover;
varying float v_uTransition;
varying vec3 v_uBoundingMin;
varying vec3 v_uBoundingMax;
varying vec3 v_uTextColor;

#!SHADER: Vertex

#require(range.glsl)

vec2 getBoundingUV() {
    vec2 uv;
    uv.x = crange(position.x, v_uBoundingMin.x, v_uBoundingMax.x, 0.0, 1.0);
    uv.y = crange(position.y, v_uBoundingMin.y, v_uBoundingMax.y, 0.0, 1.0);
    return uv;
}

void main() {
    vUv = uv;
    vUv2 = getBoundingUV();
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

#!SHADER: Fragment

#require(msdf.glsl)
#require(range.glsl)
#require(levelmask.glsl)
#require(simplenoise.glsl)
#require(rgb2hsv.fs)
#require(transformUV.glsl)

void main() {
    vec2 uv = vUv;

    float noise = cnoise(vec3(vUv*80.0, time));
    float noise2 = cnoise(vec3(vUv*10.0, time*0.5));
    float staticNoise = range(getNoise(vUv * 3., time*0.1), 0.0, 1.0, -1.0, 1.0);

    float hoverBounce = crange(v_uHover, 0.0, 0.5, 0.0, 1.0)*crange(v_uHover, 0.5, 1.0, 1.0, 0.0);
    uv *= 1.0+noise*hoverBounce*0.002;
    //uv = scaleUV(uv, vec2(1.0+hoverBounce*0.01));

    float fill = msdf(tMap, uv);
    float stroke = strokemsdf(tMap, uv, v_uStroke + hoverBounce*0.3, v_uPadding * 0.09);

    float hover = crange(v_uHover, 0.0, 1.0, 0.05, 0.9);
    //hover *= smoothstep(vUv.x-0.1, vUv.x+0.1, v_uHover);

    float alpha = mix(stroke, fill, hover);
    alpha *= crange(v_uHover, 0.0, 1.0, 0.9, 1.0);

    if (v_uTransition < 2.0) {
        alpha *= animateLevels(texture2D(tMask, (gl_FragCoord.xy / resolution) * 3.0).r, v_uTransition);
    }

    vec3 color = vec3(v_uTextColor);

    float flicker = sin(time*25.0)*sin(time*60.0)*sin(time*4.0)*sin(time*12.0)*v_uHover;
    color = rgb2hsv(color);
    color.x += flicker*0.05;
    color.z += -0.07+flicker*0.07;
    color = hsv2rgb(color);

    color += staticNoise*hoverBounce*1.2;

    color *= 1.0 + noise2*0.02;


    gl_FragColor.rgb = color;//vec3(getBoundingUV(), 1.0);
    gl_FragColor.a = alpha;
}
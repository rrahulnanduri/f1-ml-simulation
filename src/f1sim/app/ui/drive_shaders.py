"""GLSL for the driving hero (:mod:`f1sim.app.menu`): one light model for the cars, the floor
and the track.

**Units.** The softbox photos (``render_top.py --drive``) are the light falling on a white
diffuse car, exposed so that a white floor ``u_h0`` under the softbox reads ``u_white``
(linear). :func:`softbox_e` is the softbox's irradiance relative to that floor (a disc of
radius ``u_r0`` facing down: inverse square, both cosines), so everything here is in photo
units: a diffuse surface of albedo ``a`` shows ``a * u_white * e``; a reflection with BRDF ``f``
shows ``f * pi * u_white * e``.

**The car** (:data:`CAR_FS`), per pixel of its data maps:
- *Diffuse* light is photographed: the four photos around the light, blended. Dividing the
  blend by the unshadowed irradiance the photos' light gives (:func:`softbox_e` at
  ``u_light_grid``, the light clamped to the grid) leaves the visibility the photos saw:
  shadows and occlusion. The diffuse is then ``u_white * softbox_e(light) * visibility``, so
  beyond the grid the light keeps falling off with distance instead of switching off.
- *Reflections* are computed: GGX for the base material and a clear coat, with the softbox as
  a sphere light (the representative point, roughness widened by its size), so the highlight
  slides and keeps its soft shape. The same visibility shadows them.
- *Finishes*: the bodywork (the clear-coat weight in ``ao.g``) takes ``u_finish``: its colour
  scaled from the photographed black, and its own roughness, metal and coat.
- *Lamps*: the flashing trackside heads (``u_lamp_*``, small omni lights) light it the same
  way, without shadows.
- *At rest*: the ambient photo (a broad overhead light) and a sheen of it in the coat.

**The floor** is lit by :data:`FLOOR_LIGHT_FS` into a buffer smaller than the window (the
light and the shadows are soft; see :mod:`f1sim.app.ui.quality`), which :data:`FLOOR_FS` and
the track lines (:data:`TRACK_FS`) sample. It is darkened
under each car by its soft ground shadow at rest (the ambient photo's alpha, which takes only
the floor's own dim light), and gets the same light as the cars:
``u_floor_albedo * u_white * softbox_e`` with the floor's normal, darkened by the cars'
shadows. A floor point ``g``, the light ``L`` at height ``z`` and the car point ``p`` at height
``h`` that blocks it line up: ``p = (g (z - h) + L h) / z``. ``h`` comes from the car's height map
(one refinement from a typical height); the coverage there is read from a mip level that grows
with ``h / (z - h)`` times the softbox's size, so the shadow is sharp at the contact and soft
far from it.

Everything ends in the same display transform (:func:`display`): exposure, a filmic curve,
gamma 2.2.
"""

COMMON = """
const float PI = 3.14159265;
const int MAX_LAMPS = 8;
uniform float u_white;     // a white floor u_h0 under the softbox reads this, linear
uniform float u_h0;        // the softbox's height, metres
uniform float u_r0;        // its radius, metres
uniform float u_exposure;

// The softbox's irradiance relative to that white floor: v = light - point, n the normal
float softbox_e(vec3 v, vec3 n) {
    float d2 = max(dot(v, v), 1e-6);
    vec3 l = v * inversesqrt(d2);
    float r2 = u_r0 * u_r0;
    return (u_h0 * u_h0 + r2) / (d2 + r2) * max(dot(n, l), 0.0) * max(l.z, 0.0);
}

// A lamp head: a small omni light, in the same units (its colour carries its strength)
float lamp_e(vec3 v, vec3 n) {
    float d2 = max(dot(v, v), 1e-6);
    return (u_h0 * u_h0) / (d2 + 0.25) * max(dot(n, v * inversesqrt(d2)), 0.0);
}

vec3 decode(vec3 c) { return pow(c, vec3(2.2)); }
float decode1(float c) { return pow(c, 2.2); }

vec3 display(vec3 c) {
    c = max(c * u_exposure, 0.0);
    c = c * (2.51 * c + 0.03) / (c * (2.43 * c + 0.59) + 0.14);  // filmic curve
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}
"""

FLOOR_LIGHT = """
const int MAX_CARS = 6;
uniform float u_ppm;              // window pixels per metre
uniform vec3 u_light_px;          // the cursor light: window pixels and height (metres)
uniform float u_power;
uniform float u_floor_albedo;
uniform int u_n_lamps;
uniform vec3 u_lamp_px[MAX_LAMPS];   // window pixels and height (metres)
uniform vec3 u_lamp_col[MAX_LAMPS];
uniform int u_n_cars;
uniform vec2 u_car_anchor[MAX_CARS]; // each car's rear axle, window pixels
uniform vec2 u_car_rot[MAX_CARS];    // cos, sin of its turn on screen
uniform sampler2D u_cover;        // the body's albedo map: alpha = coverage
uniform sampler2D u_height;       // its surface map: b = height / u_height_scale
uniform sampler2D u_contact;      // its ambient photo: alpha = coverage + soft ground shadow
uniform float u_contact_edge;     // the ground shadow fades out this far inside the photo, m
uniform vec2 u_car_size_m;        // the photo, metres
uniform float u_axle_y;           // the rear axle, map metres down from the top
uniform float u_height_scale;
uniform float u_tex_ppm;          // the maps' pixels per metre
uniform float u_shadow_blur;      // the cast shadow's blur, in penumbra widths
uniform float u_shadow_strength;  // and how much light it takes

// A window point (metres) in car i's photo: uv, and whether it's on the photo
vec2 car_uv(vec2 p, int i) {
    vec2 d = p - u_car_anchor[i] / u_ppm;
    vec2 cs = u_car_rot[i];
    vec2 local = vec2(cs.x * d.x + cs.y * d.y, -cs.y * d.x + cs.x * d.y);  // x right, y up
    return vec2(local.x / u_car_size_m.x + 0.5, 1.0 - (u_axle_y - local.y) / u_car_size_m.y);
}

bool on_photo(vec2 uv) { return all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0))); }

// How much of the light car i blocks at floor point g (metres); L the light (metres), z its
// height
float car_shadow(vec2 g, vec2 L, float z, int i) {
    float h = 0.45;  // a typical body height, refined from the height map
    vec2 p = (g * (z - h) + L * h) / z;
    vec2 centre = u_car_anchor[i] / u_ppm;
    if (distance(p, centre) > 0.75 * u_car_size_m.y + 1.0) return 0.0;
    vec2 uv = car_uv(p, i);
    if (!on_photo(uv)) return 0.0;
    if (textureLod(u_cover, uv, 2.0).a > 0.5) {
        h = clamp(textureLod(u_height, uv, 2.0).b * u_height_scale, 0.05, 0.95 * z);
    }
    p = (g * (z - h) + L * h) / z;
    uv = car_uv(p, i);
    float blur_m = u_shadow_blur * u_r0 * h / (z - h);
    // Past the photo's edge the coverage fades over the blur (the photo is cut close to the
    // wing tips; stopping there would draw its rectangle)
    vec2 out_m = max(abs(uv - 0.5) - 0.5, 0.0) * u_car_size_m;
    float fade = 1.0 - smoothstep(0.0, blur_m + 0.02, length(out_m));
    if (fade <= 0.0) return 0.0;
    return u_shadow_strength * fade * textureLod(u_cover, uv, log2(max(blur_m * u_tex_ppm, 1.0))).a;
}

// The cars' soft ground shadows at rest (the broad overhead light's), at window pixel px:
// how much of the floor's ambient light they take
float contact_shadow(vec2 px) {
    vec2 g = px / u_ppm;
    float lit = 1.0;
    for (int i = 0; i < u_n_cars; i++) {
        if (distance(g, u_car_anchor[i] / u_ppm) > u_car_size_m.y) continue;
        vec2 uv = car_uv(g, i);
        if (!on_photo(uv)) continue;
        float cover = texture(u_cover, uv).a;
        float a = clamp((texture(u_contact, uv).a - cover) / max(1.0 - cover, 1e-4), 0.0, 1.0);
        vec2 m = vec2((uv.x - 0.5) * u_car_size_m.x, (1.0 - uv.y) * u_car_size_m.y);
        float edge = min(min(0.5 * u_car_size_m.x - abs(m.x), m.y), u_car_size_m.y - m.y);
        lit *= 1.0 - a * smoothstep(0.0, u_contact_edge, edge);
    }
    return 1.0 - lit;
}

// Light on the floor at window pixel px (radiance, before display)
vec3 floor_light(vec2 px) {
    vec2 g = px / u_ppm;
    vec2 L = u_light_px.xy / u_ppm;
    float z = u_light_px.z;
    vec3 up = vec3(0.0, 0.0, 1.0);
    float e = softbox_e(vec3(L - g, z), up) * u_power;
    if (e > 1e-4) {
        float vis = 1.0;
        for (int i = 0; i < u_n_cars; i++) vis *= 1.0 - car_shadow(g, L, z, i);
        e *= vis;
    }
    vec3 rad = vec3(e);
    for (int k = 0; k < u_n_lamps; k++) {
        vec3 v = vec3(u_lamp_px[k].xy / u_ppm - g, u_lamp_px[k].z);
        rad += lamp_e(v, up) * u_lamp_col[k];
    }
    return u_floor_albedo * u_white * rad;
}
"""

FLOOR_LIGHT_FS = (
    "#version 330\n"
    + COMMON
    + FLOOR_LIGHT
    + """
uniform vec2 u_size;   // the window, logical pixels
in vec2 v_uv;
out vec4 frag;
void main() {
    vec2 p = v_uv * u_size;
    frag = vec4(floor_light(p), contact_shadow(p));
}
"""
)

FLOOR_FS = (
    "#version 330\n"
    + COMMON
    + FLOOR_LIGHT
    + """
uniform sampler2D u_floor_light;  // FLOOR_LIGHT_FS's output: light, ground shadow
uniform vec2 u_size;
uniform float u_appear;
uniform vec3 u_bg;
uniform float u_lamp_glow_px;   // the lamp heads' glow radius, pixels
in vec2 v_uv;
out vec4 frag;
void main() {
    vec2 p = v_uv * u_size;
    float Ld = length(u_size);
    // A soft beam from beyond the top-left corner, angled down to the right
    vec2 o = vec2(-0.08 * u_size.x, 1.18 * u_size.y);
    vec2 dir = normalize(vec2(0.95, -1.0));
    vec2 d = p - o;
    float along = dot(d, dir);
    float across = abs(d.x * dir.y - d.y * dir.x);
    float width = 0.05 * Ld + 0.16 * max(along, 0.0);
    float beam = exp(-2.2 * (across / width) * (across / width))
        * smoothstep(0.0, 0.25 * Ld, along) * exp(-along / (0.85 * Ld));
    vec4 light = texture(u_floor_light, v_uv);
    vec3 bg = decode(u_bg + vec3(0.17 * beam * u_appear)) * (1.0 - light.a);
    // The floor, lit; the lamp heads glow
    vec3 lit = light.rgb;
    vec3 glow = vec3(0.0);
    for (int k = 0; k < u_n_lamps; k++) {
        float r = distance(p, u_lamp_px[k].xy) / u_lamp_glow_px;
        glow += u_lamp_col[k] * (exp(-r * r * 6.0) * 1.5 + 0.06 / (1.0 + 20.0 * r * r));
    }
    // Very faint grain so the gradients don't band
    float n = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    vec3 c = pow(bg + decode(display(lit + glow)), vec3(1.0 / 2.2));
    frag = vec4(c + n / 255.0, 1.0);
}
"""
)

TRACK_FS = (
    "#version 330\n"
    + COMMON
    + FLOOR_LIGHT
    + """
uniform sampler2D u_floor_light;  // FLOOR_LIGHT_FS's output: light, ground shadow
uniform vec2 u_size;
uniform vec2 u_fade;  // the track is at u_under left of x0, full right of x1 (pixels)
uniform float u_under;
uniform float u_line_lit;   // how strongly the lines pick up the floor light
in vec4 v_col;
in vec2 v_px;
out vec4 frag;
void main() {
    float a = v_col.a * mix(u_under, 1.0, smoothstep(u_fade.x, u_fade.y, v_px.x));
    vec4 light = texture(u_floor_light, v_px / u_size);
    a *= 1.0 - 0.8 * light.a;
    // Painted lines and kerbs reflect the light like the floor, only brighter
    vec3 lit = display(light.rgb * u_line_lit / u_floor_albedo);
    vec3 c = v_col.rgb * a + lit * v_col.rgb * min(1.0, 2.5 * a);
    frag = vec4(c, a);  // premultiplied
}
"""
)

TRACK_VS = """
#version 330
uniform vec2 u_car;     // the hero's rear axle, world metres
uniform vec2 u_rot;     // cos, sin of the rotation that puts the hero's heading up
uniform float u_ppm;
uniform vec2 u_anchor;  // the hero's rear axle on screen, window pixels
uniform vec2 u_size;
in vec2 in_pos;
in vec4 in_col;
out vec4 v_col;
out vec2 v_px;
void main() {
    vec2 d = in_pos - u_car;
    vec2 r = vec2(u_rot.x * d.x - u_rot.y * d.y, u_rot.y * d.x + u_rot.x * d.y);
    vec2 px = u_anchor + r * u_ppm;
    v_col = in_col;
    v_px = px;
    gl_Position = vec4(px / u_size * 2.0 - 1.0, 0.0, 1.0);
}
"""

CAR_VS = """
#version 330
uniform vec2 u_anchor;   // the car's rear axle, window pixels
uniform vec2 u_rot;      // cos, sin of the car's turn on screen (anticlockwise)
uniform float u_ppm;
uniform vec2 u_size_m;   // the photo, metres
uniform float u_axle_y;  // the rear axle, map metres down from the photo's top
uniform vec2 u_size;
in vec2 in_vert;         // 0..1 (the same as in_uv)
in vec2 in_uv;
out vec2 v_uv;
void main() {
    // The photo in metres from the rear axle: x right, y up the screen (towards the nose)
    vec2 m = vec2((in_uv.x - 0.5) * u_size_m.x, u_axle_y - (1.0 - in_uv.y) * u_size_m.y);
    vec2 r = vec2(u_rot.x * m.x - u_rot.y * m.y, u_rot.y * m.x + u_rot.x * m.y);
    v_uv = in_uv;
    gl_Position = vec4((u_anchor + r * u_ppm) / u_size * 2.0 - 1.0, 0.0, 1.0);
}
"""

CAR_FS = (
    "#version 330\n"
    + COMMON
    + """
uniform sampler2DArray u_body_light;  // per softbox position: light on the white car
uniform sampler2DArray u_wheel_light;
uniform sampler2D u_body_amb;   // at rest; alpha = coverage + soft ground shadow
uniform sampler2D u_body_alb;   // alpha = coverage
uniform sampler2D u_body_nrm;
uniform sampler2D u_body_srf;   // roughness, metallic, height / u_height_scale
uniform sampler2D u_body_ao;    // occlusion, clear-coat weight (= bodywork)
uniform sampler2D u_wheel_amb;
uniform sampler2D u_wheel_alb;
uniform sampler2D u_wheel_nrm;
uniform sampler2D u_wheel_srf;
uniform sampler2D u_wheel_ao;
uniform vec4 u_layers;
uniform vec4 u_weights;
uniform vec3 u_light;       // the cursor light in this car's frame: map metres, height
uniform vec3 u_light_grid;  // the same, clamped to the photo grid
uniform float u_power;
uniform float u_ambient_level;
uniform float u_ambient_gain;
uniform float u_amb_spec;
uniform float u_vis_lift;   // how much of the photographed self-shadowing is lifted
uniform float u_height_scale;
uniform float u_coat_rough;
uniform vec4 u_finish;      // bodywork: colour scale, roughness, metallic, clear coat
uniform int u_n_lamps;
uniform vec3 u_lamp_pos[MAX_LAMPS];  // in this car's frame: map metres, height
uniform vec3 u_lamp_col[MAX_LAMPS];
uniform float u_lamp_r;     // a lamp head's radius, metres
uniform vec2 u_size_m;
uniform vec4 u_wheels_m;    // front wheel centres (x0, y0, x1, y1), map metres
uniform float u_wheel_r;
uniform vec2 u_strip;       // the wheel strip: top row and height, map metres
uniform float u_steer;      // rad, + = left
uniform float u_wheel_gain;
in vec2 v_uv;
out vec4 frag;

float ggx(float ndh, float a) {
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float smith(float ndl, float ndv, float rough) {
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
}

// GGX reflection of a sphere light of radius r at v (from the point), per unit irradiance
// (the representative point; roughness widened by the light's size, energy kept)
vec3 sphere_spec(vec3 n, vec3 v, float r, float rough, vec3 F0) {
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 R = reflect(-V, n);
    vec3 c2r = dot(v, R) * R - v;
    vec3 l = normalize(v + c2r * clamp(r / max(length(c2r), 1e-4), 0.0, 1.0));
    float a = rough * rough;
    float aw = clamp(a + r / (2.0 * length(v)), 0.0, 1.0);
    vec3 H = normalize(l + V);
    float ndl = max(dot(n, l), 1e-3);
    float ndv = max(n.z, 1e-3);
    float ndh = max(dot(n, H), 0.0);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
    float D = ggx(ndh, a) * (a / aw) * (a / aw);
    return D * smith(ndl, ndv, rough) * F / (4.0 * ndl * ndv);
}

float blend_light(sampler2DArray t, vec2 uv) {
    return u_weights.x * decode1(texture(t, vec3(uv, u_layers.x)).r)
        + u_weights.y * decode1(texture(t, vec3(uv, u_layers.y)).r)
        + u_weights.z * decode1(texture(t, vec3(uv, u_layers.z)).r)
        + u_weights.w * decode1(texture(t, vec3(uv, u_layers.w)).r);
}

// One layer's pixel at uv, which shows at pos (map metres); turn = the layer's rotation
// (the front wheels' steering); gain brightens its colour. Premultiplied display colour.
vec4 shade(sampler2DArray light_t, sampler2D amb_t, sampler2D alb_t, sampler2D nrm_t,
           sampler2D srf_t, sampler2D ao_t, vec2 uv, vec2 pos, float turn, float gain) {
    vec4 alb = texture(alb_t, uv);
    if (alb.a <= 0.003) return vec4(0.0);
    vec4 aoc = texture(ao_t, uv);
    vec4 srf = texture(srf_t, uv);
    float paint = aoc.g;
    float occ = aoc.r;
    vec3 base = decode(alb.rgb) * mix(1.0, u_finish.x, paint) * gain;
    float rough = clamp(mix(srf.r, u_finish.y, paint), 0.04, 1.0);
    float metal = mix(srf.g, u_finish.z, paint);
    float coat = paint * u_finish.w;
    float h = srf.b * u_height_scale;
    vec3 n = normalize(texture(nrm_t, uv).xyz * 2.0 - 1.0);
    float cs = cos(turn), sn = sin(turn);
    n.xy = vec2(cs * n.x - sn * n.y, sn * n.x + cs * n.y);
    float ndv = max(n.z, 1e-3);
    vec3 F0 = mix(vec3(0.04), base, metal);
    float Fc = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    vec3 kd = (1.0 - metal) * base * (1.0 - coat * Fc);
    // Normal-map frame: x right, y towards the nose, z up
    vec3 p = vec3(pos.x, -pos.y, h);
    vec3 vL = vec3(u_light.x, -u_light.y, u_light.z) - p;
    vec3 vG = vec3(u_light_grid.x, -u_light_grid.y, u_light_grid.z) - p;
    float eL = softbox_e(vL, n);
    float eG = softbox_e(vG, n);
    // The photos' shadows and occlusion: their light over the unshadowed light they'd give
    float photo = blend_light(light_t, uv);
    float vis = clamp(photo / (u_white * eG + 1e-4), 0.0, 1.0);
    vis = mix(occ, vis, smoothstep(0.01, 0.08, eG));
    vis = mix(vis, 1.0, u_vis_lift);  // shading, not a shadow laid over the car
    vec3 spec = sphere_spec(n, vL, u_r0, rough, F0)
        + coat * sphere_spec(n, vL, u_r0, u_coat_rough, vec3(0.04));
    vec3 c = (kd + PI * spec) * u_white * eL * vis * u_power;
    // The flashing lamps
    for (int k = 0; k < u_n_lamps; k++) {
        vec3 v = vec3(u_lamp_pos[k].x, -u_lamp_pos[k].y, u_lamp_pos[k].z) - p;
        vec3 s = sphere_spec(n, v, u_lamp_r, rough, F0)
            + coat * sphere_spec(n, v, u_lamp_r, u_coat_rough, vec3(0.04));
        c += (kd + PI * s) * u_white * lamp_e(v, n) * occ * u_lamp_col[k];
    }
    // At rest: the broad overhead light, and its sheen in the coat on upward faces
    vec4 amb = texture(amb_t, uv);
    vec3 irr = decode(amb.rgb) / u_ambient_gain;
    c += kd * irr * u_ambient_level * occ;
    // A gloss black reflects the dark room on its flat tops and the overhead light only on
    // its curved shoulders: that's what shapes it (a reflection everywhere veils it)
    float rz = reflect(vec3(0.0, 0.0, -1.0), n).z;
    float sky = smoothstep(0.55, 0.8, rz) * (1.0 - smoothstep(0.9, 0.985, rz));
    vec3 Fs = F0 + (1.0 - F0) * pow(1.0 - ndv, 5.0);
    c += (Fs * mix(1.0, 0.35, rough) + coat * Fc) * sky * occ * u_amb_spec;
    return vec4(display(c) * alb.a, alb.a);
}

void main() {
    vec2 pos = vec2((v_uv.x - 0.5) * u_size_m.x, (1.0 - v_uv.y) * u_size_m.y);
    vec4 body = shade(u_body_light, u_body_amb, u_body_alb, u_body_nrm, u_body_srf, u_body_ao,
                      v_uv, pos, 0.0, 1.0);
    float car_a = body.a;
    // Front wheel: sample the straight wheel rotated about its centre by the steering angle
    vec2 c = pos.x < 0.0 ? u_wheels_m.xy : u_wheels_m.zw;
    vec2 d = pos - c;
    vec4 wheel = vec4(0.0);
    if (dot(d, d) < u_wheel_r * u_wheel_r) {
        float cs = cos(u_steer), sn = sin(u_steer);
        // Map y points down, so turning the wheel anticlockwise on screen (a left steer) is
        // a rotation by -steer here: sample the straight photo rotated back by +steer
        vec2 q = c + vec2(cs * d.x - sn * d.y, sn * d.x + cs * d.y);
        vec2 wuv = vec2(q.x / u_size_m.x + 0.5, 1.0 - (q.y - u_strip.x) / u_strip.y);
        if (all(greaterThan(wuv, vec2(0.0))) && all(lessThan(wuv, vec2(1.0)))) {
            wheel = shade(u_wheel_light, u_wheel_amb, u_wheel_alb, u_wheel_nrm, u_wheel_srf,
                          u_wheel_ao, wuv, pos, u_steer, u_wheel_gain);
        }
    }
    // The front wheels under the body (premultiplied); the floor pass drew the ground shadow
    frag = body + (1.0 - car_a) * wheel;
}
"""
)

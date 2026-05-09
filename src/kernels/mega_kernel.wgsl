struct Params {
    width: u32,
    height: u32,
    depth: u32,
    seed: u32,
    photon_radius: f32,
    iteration: u32,
    _pad0: f32,
    _pad1: f32,
};

struct Camera {
    origin: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    lower_left_corner: vec4<f32>,
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

struct SphereInstance {
    material_id: u32,
    scale: f32,
    transform: mat4x4<f32>,
};
struct SphereInstanceArray {
    contents: array<SphereInstance>,
};

struct GpuMaterial {
    color: vec4<f32>,
    roughness: f32,
    ior: f32,
    material_type: u32,
    pad1: f32,
};

struct Triangle {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
}

struct Hit {
    distance: f32,
    material_id: u32,
    pad2: f32,
    pad3: f32,
    location: vec3<f32>,
    normal: vec3<f32>,
};

struct BVHNode {
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    left_child: u32,
    right_child: u32,
    first_triangle: u32,
    n_triangles: u32,
};

struct Light {
    position: vec4<f32>,
    color: vec4<f32>,
    color_temp: f32,
    light_type: u32,
    pad1: f32,
    pad2: f32,
};

struct Vispoint {
    position: vec4<f32>,
    normal: vec4<f32>,
    wo: vec4<f32>,
    throughput: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var<storage, read_write> vispoints: array<Vispoint>;

@group(1) @binding(0) var<uniform> camera: Camera;

@group(2) @binding(0) var<storage, read> sphere_instances: SphereInstanceArray;
@group(3) @binding(0) var<storage, read> mesh_positions: array<vec3<f32>>;
@group(3) @binding(1) var<storage, read> mesh_indices: array<vec4<u32>>;
@group(4) @binding(0) var<storage, read> materials: array<GpuMaterial>;
@group(5) @binding(0) var<storage, read> bvh_nodes: array<BVHNode>;
@group(5) @binding(1) var<storage, read> bvh_triangle_indices: array<u32>;
@group(6) @binding(0) var<storage, read> scene_lights: array<Light>;

const EPS: f32 = 1e-5;
const VISIBLE_MIN: f32 = 380.0;
const VISIBLE_RANGE: f32 = 400.0;
const PI: f32 = 3.1415926535;
const INV_PI: f32 = 0.31830988618;
const DISPERSION_B: f32 = 0.004;
const K_PHOTONS: u32 = 4u;
const MAX_PHOTON_BOUNCES: u32 = 8u;
const PHOTON_CONE_COS: f32 = 0.707;

// ----- CIE spectral data (embedded, replaces cie_table buffer) -----
const CIE_X: array<f32, 81> = array<f32, 81>(
    0.001368000, 0.002236000, 0.004243000, 0.007650000, 0.01431000,
    0.02319000,  0.04351000,  0.07763000,  0.1343800,   0.2147700,
    0.2839000,   0.3285000,   0.3482800,   0.3480600,   0.3362000,
    0.3187000,   0.2908000,   0.2511000,   0.1953600,   0.1421000,
    0.09564000,  0.05795001,  0.03201000,  0.01470000,  0.004900000,
    0.002400000, 0.009300000, 0.02910000,  0.06327000,  0.1096000,
    0.1655000,   0.2257499,   0.2904000,   0.3597000,   0.4334499,
    0.5120501,   0.5945000,   0.6784000,   0.7621000,   0.8425000,
    0.9163000,   0.9786000,   1.0263000,   1.0567000,   1.0622000,
    1.0456000,   1.0026000,   0.9384000,   0.8544499,   0.7514000,
    0.6424000,   0.5419000,   0.4479000,   0.3608000,   0.2835000,
    0.2187000,   0.1649000,   0.1212000,   0.08740000,  0.06360000,
    0.04677000,  0.03290000,  0.02270000,  0.01584000,  0.01135916,
    0.008110916, 0.005790346, 0.004109457, 0.002899327, 0.002049190,
    0.001439971, 0.0009999493,0.0006900786,0.0004760213,0.0003323011,
    0.0002348261,0.0001661505,0.0001174130,0.00008307527,0.00005870652,
    0.00004150994,
);

const CIE_Y: array<f32, 81> = array<f32, 81>(
    0.00003900000, 0.00006400000, 0.0001200000, 0.0002170000, 0.0003960000,
    0.0006400000,  0.001210000,  0.002180000,  0.004000000,  0.007300000,
    0.01160000,   0.01684000,   0.02300000,   0.02980000,   0.03800000,
    0.04800000,   0.06000000,   0.07390000,   0.09098000,   0.1126000,
    0.1390200,    0.1693000,    0.2080200,    0.2586000,    0.3230000,
    0.4073000,    0.5030000,    0.6082000,    0.7100000,    0.7932000,
    0.8620000,    0.9148501,    0.9540000,    0.9803000,    0.9949501,
    1.0000000,    0.9950000,    0.9786000,    0.9520000,    0.9154000,
    0.8700000,    0.8163000,    0.7570000,    0.6949000,    0.6310000,
    0.5668000,    0.5030000,    0.4412000,    0.3810000,    0.3210000,
    0.2650000,    0.2170000,    0.1750000,    0.1382000,    0.1070000,
    0.08160000,   0.06100000,   0.04458000,   0.03200000,   0.02320000,
    0.01700000,   0.01192000,   0.008210000,  0.005723000,  0.004102000,
    0.002929000,  0.002091000,  0.001484000,  0.001047000,  0.0007400000,
    0.0005200000, 0.0003611000, 0.0002492000, 0.0001719000, 0.0001200000,
    0.00008480000,0.00006000000,0.00004240000,0.00003000000,0.00002120000,
    0.00001499000,
);

const CIE_Z: array<f32, 81> = array<f32, 81>(
    0.006450001, 0.01054999,  0.02005001,  0.03621000,  0.06785001,
    0.1102000,   0.2074000,   0.3713000,   0.6456000,   1.0390501,
    1.3856000,   1.6229600,   1.7470600,   1.7826000,   1.7721100,
    1.7441000,   1.6692000,   1.5281000,   1.2876400,   1.0419000,
    0.8129501,   0.6162000,   0.4651800,   0.3533000,   0.2720000,
    0.2123000,   0.1582000,   0.1117000,   0.07824999,  0.05725001,
    0.04216000,  0.02984000,  0.02030000,  0.01340000,  0.008749999,
    0.005749999, 0.003900000, 0.002749999, 0.002100000, 0.001800000,
    0.001650001, 0.001400000, 0.001100000, 0.0008000000,0.0006000000,
    0.0003400000,0.0002400000,0.0001900000,0.0001000000,0.00004999999,
    0.00003000000,0.00002000000,0.00001000000,0.000000000000,0.000000000000,
    0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
    0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
    0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
    0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
    0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,
    0.000000000000,
);

// ----- Spherical geometry helpers -----

fn cos_theta(w: vec3<f32>) -> f32 { return w.z; }
fn abs_cos_theta(w: vec3<f32>) -> f32 { return abs(w.z); }
fn cos2_theta(w: vec3<f32>) -> f32 { return w.z * w.z; }
fn tan2_theta(w: vec3<f32>) -> f32 {
    let c2 = cos2_theta(w);
    if (c2 < 1e-10) { return 1e20; }
    return (1.0 - c2) / c2;
}
fn same_hemisphere(wo: vec3<f32>, wi: vec3<f32>) -> bool {
    return wo.z * wi.z > 0.0;
}

// ----- Oren-Nayar diffuse BRDF -----

fn oren_nayar_f(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, sigma: f32) -> vec3<f32> {
    let ndotv = max(dot(n, wo), 0.0);
    let ndotl = max(dot(n, wi), 0.0);
    if (ndotv < 1e-6 || ndotl < 1e-6) { return vec3<f32>(0.0); }

    let sig2 = sigma * sigma;
    let A = 1.0 - 0.5 * sig2 / (sig2 + 0.33);
    let B = 0.45 * sig2 / (sig2 + 0.09);

    let sin2_v = max(0.0, 1.0 - ndotv * ndotv);
    let sin2_l = max(0.0, 1.0 - ndotl * ndotl);
    let sin_v = sqrt(sin2_v);
    let sin_l = sqrt(sin2_l);

    var cos_phi_diff: f32;
    if (sin_v > 1e-6 && sin_l > 1e-6) {
        let wo_t = wo - ndotv * n;
        let wi_t = wi - ndotl * n;
        cos_phi_diff = clamp(dot(wo_t, wi_t) / (sin_v * sin_l), -1.0, 1.0);
    } else {
        cos_phi_diff = 1.0;
    }

    let sin_alpha = max(sin_v, sin_l);
    let tan_beta = min(sin_v, sin_l) / max(ndotv, ndotl);

    return albedo * INV_PI * (A + B * max(0.0, cos_phi_diff) * sin_alpha * tan_beta);
}

// ----- Trowbridge-Reitz (GGX) microfacet functions -----

fn tr_d(wm: vec3<f32>, alpha: f32) -> f32 {
    let tan2 = tan2_theta(wm);
    if (tan2 > 1e20) { return 0.0; }
    let cos4 = cos2_theta(wm) * cos2_theta(wm);
    let e = tan2 / (alpha * alpha);
    return 1.0 / (PI * alpha * alpha * cos4 * (1.0 + e) * (1.0 + e));
}

fn tr_lambda(w: vec3<f32>, alpha: f32) -> f32 {
    let tan2 = tan2_theta(w);
    if (tan2 > 1e20) { return 0.0; }
    let a2 = alpha * alpha;
    return (sqrt(1.0 + a2 * tan2) - 1.0) * 0.5;
}

fn tr_g(wo: vec3<f32>, wi: vec3<f32>, alpha: f32) -> f32 {
    return 1.0 / (1.0 + tr_lambda(wo, alpha) + tr_lambda(wi, alpha));
}

fn roughness_to_alpha(roughness: f32) -> f32 { return sqrt(roughness); }
fn effectively_smooth(alpha: f32) -> bool { return alpha < 1e-3; }

fn tr_sample_wm(wo: vec3<f32>, u: vec2<f32>, alpha: f32) -> vec3<f32> {
    let wh = normalize(vec3<f32>(alpha * wo.x, alpha * wo.y, wo.z));
    var wh_adj = wh;
    if (wh_adj.z < 0.0) { wh_adj = -wh_adj; }

    var t1: vec3<f32>;
    if (abs(wh_adj.z) > 0.99999) { t1 = vec3<f32>(1.0, 0.0, 0.0); }
    else { t1 = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), wh_adj)); }
    let t2 = cross(wh_adj, t1);

    let r = sqrt(u.y);
    let phi_disk = 2.0 * PI * u.x;
    var p = vec2<f32>(r * cos(phi_disk), r * sin(phi_disk));

    let h = sqrt(max(0.0, 1.0 - p.x * p.x));
    p.y = mix(h, p.y, (1.0 + wh_adj.z) * 0.5);

    let pz = sqrt(max(0.0, 1.0 - p.x * p.x - p.y * p.y));
    let nh = p.x * t1 + p.y * t2 + pz * wh_adj;

    return normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-6)));
}

// ----- Cauchy spectral IOR -----

fn cauchy_ior(base_ior: f32, lambda_nm: f32) -> f32 {
    let lambda_um = lambda_nm * 1e-3;
    return base_ior + DISPERSION_B / (lambda_um * lambda_um);
}

// ----- Ray generation -----

fn get_ray(u: f32, v: f32) -> Ray {
    var ray: Ray;
    ray.origin = camera.origin.xyz;
    ray.direction = (camera.lower_left_corner
                    + camera.horizontal * u
                    + camera.vertical * v
                    - camera.origin).xyz;
    return ray;
}

// ----- Sphere intersection -----

fn hit_sphere(r: Ray, sphere: SphereInstance) -> Hit {
    let center = (sphere.transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let radius = sphere.scale;
    let oc: vec3<f32> = r.origin - center;
    let a: f32 = dot(r.direction, r.direction);
    let half_b: f32 = dot(oc, r.direction);
    let c: f32 = dot(oc, oc) - radius*radius;
    let discriminant: f32 = half_b * half_b - a*c;

    var hit: Hit;
    hit.distance = -1.0;
    hit.material_id = 0u;
    if (discriminant > 0.) {
        let hit_distance = (-half_b - sqrt(discriminant)) / a;
        hit.distance = hit_distance;
        hit.material_id = sphere.material_id;
        hit.location = r.origin + r.direction * hit_distance * 0.9999;
        hit.normal = normalize(hit.location - center);
    };
    return hit;
}

// ----- Triangle intersection -----

fn hit_triangle(r: Ray, triangle_index: u32) -> Hit {
    var hit: Hit;
    hit.distance = -1.;
    hit.material_id = 0u;

    let vertices = mesh_indices[triangle_index];
    let flt_eps = bitcast<f32>(0x1p-126f);
    let a_pos = mesh_positions[vertices[0]].xyz;
    let b_pos = mesh_positions[vertices[1]].xyz;
    let c_pos = mesh_positions[vertices[2]].xyz;
    let edge1 = b_pos - a_pos;
    let edge2 = c_pos - a_pos;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if (det > - flt_eps && det < flt_eps) { return hit; }

    let inv_det = 1. / det;
    let s = r.origin - a_pos;
    let u = inv_det * dot(s, ray_cross_e2);
    if (u < 0. || u > 1.) { return hit; }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(r.direction, s_cross_e1);
    if (v < 0. || u + v > 1.) { return hit; }

    let t = inv_det * dot(edge2, s_cross_e1);
    if (t > flt_eps) {
        hit.normal = normalize(cross(edge1, edge2));
        hit.location = r.origin + hit.normal*1e-5 + r.direction * t;
        hit.distance = t;
        hit.material_id = vertices.w;
        return hit;
    }
    return hit;
}

// ----- Closest hit -----

fn closest_sphere_hit(r: Ray) -> Hit {
    var num_instances: i32 = bitcast<i32>(arrayLength(&sphere_instances.contents));
    var current_hit: Hit;
    var best_hit: Hit;
    best_hit.distance = -10000000.0;
    for (var i: i32 = 0; i < num_instances; i=i+1) {
        current_hit = hit_sphere(r, sphere_instances.contents[i]);
        if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
            best_hit = current_hit;
        };
    }
    return best_hit;
}

// ----- BVH traversal -----

fn ray_aabb_intersect(r: Ray, bmin: vec4<f32>, bmax: vec4<f32>) -> bool {
    var tmin = 0.0;
    var tmax = 3.402823e+38;

    if (abs(r.direction.x) < 1.0e-20) {
        if (r.origin.x < bmin.x || r.origin.x > bmax.x) { return false; }
    } else {
        let inv = 1.0 / r.direction.x;
        var t0 = (bmin.x - r.origin.x) * inv;
        var t1 = (bmax.x - r.origin.x) * inv;
        if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
    }
    if (abs(r.direction.y) < 1.0e-20) {
        if (r.origin.y < bmin.y || r.origin.y > bmax.y) { return false; }
    } else {
        let inv = 1.0 / r.direction.y;
        var t0 = (bmin.y - r.origin.y) * inv;
        var t1 = (bmax.y - r.origin.y) * inv;
        if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
    }
    if (abs(r.direction.z) < 1.0e-20) {
        if (r.origin.z < bmin.z || r.origin.z > bmax.z) { return false; }
    } else {
        let inv = 1.0 / r.direction.z;
        var t0 = (bmin.z - r.origin.z) * inv;
        var t1 = (bmax.z - r.origin.z) * inv;
        if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
        tmin = max(tmin, t0);
        tmax = min(tmax, t1);
    }
    return tmax >= max(tmin, 0.0);
}

fn intersect_bvh(r: Ray) -> Hit {
    var best_hit: Hit;
    best_hit.distance = -10000000.0;

    let num_nodes = arrayLength(&bvh_nodes);
    if (num_nodes == 0u) { return best_hit; }

    var stack: array<u32, 64>;
    var sp: u32 = 0u;
    stack[sp] = 0u;
    sp = sp + 1u;

    while (sp > 0u) {
        sp = sp - 1u;
        let node_idx = stack[sp];
        let node = bvh_nodes[node_idx];
        if (!ray_aabb_intersect(r, node.bbox_min, node.bbox_max)) { continue; }
        if (node.n_triangles > 0u) {
            for (var i = 0u; i < node.n_triangles; i = i + 1u) {
                let tri_idx = bvh_triangle_indices[node.first_triangle + i];
                var current_hit = hit_triangle(r, tri_idx);
                if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
                    best_hit = current_hit;
                }
            }
        } else {
            stack[sp] = node.right_child;
            sp = sp + 1u;
            stack[sp] = node.left_child;
            sp = sp + 1u;
        }
    }
    return best_hit;
}

fn closest_triangle_hit(r: Ray) -> Hit { return intersect_bvh(r); }

// ----- Spectral functions -----

fn blackbody(lambda_nm: f32, temp: f32) -> f32 {
    let h = 6.62607015e-34;
    let c = 2.99792458e8;
    let k = 1.380649e-23;
    let c1 = 2.0 * h * c * c;
    let c2 = h * c / k;
    let l = lambda_nm * 1e-9;
    return c1 / (pow(l, 5.0) * (exp(c2 / (l * temp)) - 1.0)) * 1e-14;
}

fn cie_to_rgb(lambda_nm: f32) -> vec3<f32> {
    let t = (lambda_nm - VISIBLE_MIN) / 5.0;
    let i = u32(t);
    let f = t - f32(i);
    let a = min(i, 80u);
    let b = min(i + 1u, 80u);
    let x = mix(CIE_X[a], CIE_X[b], f);
    let y = mix(CIE_Y[a], CIE_Y[b], f);
    let z = mix(CIE_Z[a], CIE_Z[b], f);
    return vec3<f32>(
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    );
}

// ----- Shadow rays -----

fn hit_sphere_shadow(r: Ray, sphere: SphereInstance, t_max: f32) -> bool {
    let center = (sphere.transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let radius = sphere.scale;
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let half_b = dot(oc, r.direction);
    let c = dot(oc, oc) - radius * radius;
    let disc = half_b * half_b - a * c;
    if (disc <= 0.0) { return false; }
    let t = (-half_b - sqrt(disc)) / a;
    return t > 0.0 && t < t_max;
}

fn hit_triangle_shadow(r: Ray, triangle_index: u32, t_max: f32) -> bool {
    let vertices = mesh_indices[triangle_index];
    let a_pos = mesh_positions[vertices[0]].xyz;
    let b_pos = mesh_positions[vertices[1]].xyz;
    let c_pos = mesh_positions[vertices[2]].xyz;
    let flt_eps = bitcast<f32>(0x1p-126f);
    let edge1 = b_pos - a_pos;
    let edge2 = c_pos - a_pos;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);
    if (abs(det) < flt_eps) { return false; }

    let inv_det = 1.0 / det;
    let s = r.origin - a_pos;
    let u = inv_det * dot(s, ray_cross_e2);
    if (u < 0.0 || u > 1.0) { return false; }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(r.direction, s_cross_e1);
    if (v < 0.0 || u + v > 1.0) { return false; }

    let t = inv_det * dot(edge2, s_cross_e1);
    return t > flt_eps && t < t_max;
}

fn sphere_roots(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = origin - center;
    let a = dot(dir, dir);
    let half_b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = half_b * half_b - a * c;
    if (disc <= 0.0) { return vec2<f32>(-1.0, -1.0); }
    let sqrt_disc = sqrt(disc);
    return vec2<f32>((-half_b - sqrt_disc) / a, (-half_b + sqrt_disc) / a);
}

fn shadow_attenuation(r: Ray, t_max: f32, lambda_nm: f32) -> f32 {
    var atten = 1.0;
    let num_spheres = arrayLength(&sphere_instances.contents);
    for (var i = 0u; i < num_spheres; i = i + 1u) {
        let sphere = sphere_instances.contents[i];
        let center = (sphere.transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
        let ts = sphere_roots(r.origin, r.direction, center, sphere.scale);
        if (ts.y <= 0.0 || ts.x >= t_max) { continue; }

        let mat = materials[sphere.material_id];
        if (mat.material_type == 0u) { return 0.0; }

        let t_entry = max(ts.x, 0.0);
        let t_exit = min(ts.y, t_max);
        if (t_entry >= t_exit) { continue; }

        let p1 = r.origin + r.direction * t_entry;
        let n1 = normalize(p1 - center);
        let cos_1 = -dot(n1, r.direction);
        let R1 = fr_dielectric(cos_1, cauchy_ior(mat.ior, lambda_nm));

        let p2 = r.origin + r.direction * t_exit;
        let n2 = normalize(p2 - center);
        let cos_2 = -dot(n2, r.direction);
        let R2 = fr_dielectric(cos_2, cauchy_ior(mat.ior, lambda_nm));

        atten *= (1.0 - R1) * (1.0 - R2);
    }

    let num_nodes = arrayLength(&bvh_nodes);
    if (num_nodes > 0u) {
        var stack: array<u32, 64>;
        var sp: u32 = 0u;
        stack[sp] = 0u;
        sp = sp + 1u;
        while (sp > 0u) {
            sp = sp - 1u;
            let node_idx = stack[sp];
            let node = bvh_nodes[node_idx];
            if (!ray_aabb_intersect(r, node.bbox_min, node.bbox_max)) { continue; }
            if (node.n_triangles > 0u) {
                for (var ti = 0u; ti < node.n_triangles; ti = ti + 1u) {
                    if (hit_triangle_shadow(r, bvh_triangle_indices[node.first_triangle + ti], t_max)) { return 0.0; }
                }
            } else {
                stack[sp] = node.right_child;
                sp = sp + 1u;
                stack[sp] = node.left_child;
                sp = sp + 1u;
            }
        }
    }
    return atten;
}

// ----- Direct lighting -----

fn sample_direct_lighting(pos: vec3<f32>, norm: vec3<f32>, lambda_nm: f32, rng: ptr<function, u32>) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    let num_lights = arrayLength(&scene_lights);
    for (var i = 0u; i < num_lights; i = i + 1u) {
        let light = scene_lights[i];

        var spd: f32;
        if (light.color_temp > 0.0) { spd = blackbody(lambda_nm, light.color_temp); }
        else { spd = 1.0; }
        let cie_factor = cie_to_rgb(lambda_nm);
        let light_rgb = light.color.rgb * light.color.w * spd * cie_factor * VISIBLE_RANGE;

        if (light.light_type == 0u) {
            // Point light
            let to_light = light.position.xyz - pos;
            let dist = length(to_light);
            if (dist < EPS) { continue; }
            let light_dir = to_light / dist;
            let ndotl = dot(norm, light_dir);
            if (ndotl <= 0.0) { continue; }
            let shadow_ray = Ray(pos + norm * EPS, light_dir);
            let atten = shadow_attenuation(shadow_ray, dist - EPS, lambda_nm);
            if (atten <= 0.0) { continue; }
            result += light_rgb * ndotl * atten / (dist * dist);
        } else {
            // Square area light
            let hw = light.position.w;
            if (hw <= 0.0) { continue; }
            let u_sample = rand_2f(rng);
            let lp = sample_square_point(light, u_sample);
            let to_light = lp - pos;
            let dist = length(to_light);
            if (dist < EPS) { continue; }
            let light_dir = to_light / dist;
            let ndotl = dot(norm, light_dir);
            if (ndotl <= 0.0) { continue; }
            let l_normal = light_normal(light);
            let cos_light = max(0.0, dot(l_normal, -light_dir));
            if (cos_light <= 0.0) { continue; }
            let shadow_ray = Ray(pos + norm * EPS, light_dir);
            let atten = shadow_attenuation(shadow_ray, dist - EPS, lambda_nm);
            if (atten <= 0.0) { continue; }
            let pdf = 1.0 / max(4.0 * hw * hw, 1e-10);
            result += light_rgb * ndotl * cos_light * atten / (dist * dist * pdf);
        }
    }
    return result;
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (dir.y + 1.0);
    return vec3<f32>(0.0, 0.0, 0.0);
}

// ----- Fresnel and reflection/refraction -----

fn fr_dielectric(cos_theta_i: f32, eta: f32) -> f32 {
    var ct = clamp(cos_theta_i, -1.0, 1.0);
    var e = eta;
    if (ct < 0.0) { e = 1.0 / eta; ct = -ct; }
    let sin2_ti = 1.0 - ct * ct;
    let sin2_tt = sin2_ti / (e * e);
    if (sin2_tt >= 1.0) { return 1.0; }
    let ct_t = sqrt(1.0 - sin2_tt);
    let r_parl = (e * ct - ct_t) / (e * ct + ct_t);
    let r_perp = (ct - e * ct_t) / (ct + e * ct_t);
    return (r_parl * r_parl + r_perp * r_perp) * 0.5;
}

fn reflect_dir(wo: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return -wo + 2.0 * dot(wo, n) * n;
}

fn refract_dir(wo: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    var ct = dot(n, wo);
    var e = eta;
    var na = n;
    if (ct < 0.0) { e = 1.0 / eta; ct = -ct; na = -n; }
    let sin2_ti = max(0.0, 1.0 - ct * ct);
    let sin2_tt = sin2_ti / (e * e);
    if (sin2_tt >= 1.0) { return vec3<f32>(0.0); }
    let ct_t = sqrt(1.0 - sin2_tt);
    return -wo / e + (ct / e - ct_t) * na;
}

// ----- Random number generation -----

fn rand(rng: ptr<function, u32>) -> u32 {
    let oldstate: u32 = *rng;
    let res = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    *rng = *rng * 747796405u + 2891336453u;
    return (res >> 22u) ^ res;
}

fn rand_1f(rng: ptr<function, u32>) -> f32 {
    return f32(rand(rng)) * (1.0 / f32(0xFFFFFFFFu));
}

fn rand_2f(rng: ptr<function, u32>) -> vec2<f32> {
    return vec2<f32>(rand_1f(rng), rand_1f(rng));
}

fn rand_unit_vec(rng: ptr<function, u32>) -> vec3<f32> {
    let rand = rand_2f(rng);
    let theta = 2.0 * PI * rand.x;
    let phi = acos(1.0 - 2.0 * rand.y);
    return vec3<f32>(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

fn build_tangent_frame(n: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let alt = vec3<f32>(1.0, 0.0, 0.0);
    return select(normalize(cross(up, n)), alt, abs(n.y) > 0.99999);
}

fn light_normal(light: Light) -> vec3<f32> {
    let ny_sq = max(0.0, 1.0 - light.pad1 * light.pad1 - light.pad2 * light.pad2);
    return vec3<f32>(light.pad1, -sqrt(ny_sq), light.pad2);
}

fn sample_square_point(light: Light, u: vec2<f32>) -> vec3<f32> {
    let n = light_normal(light);
    let T = build_tangent_frame(n);
    let B = cross(n, T);
    let hw = light.position.w;
    let su = (u.x - 0.5) * 2.0 * hw;
    let sv = (u.y - 0.5) * 2.0 * hw;
    return light.position.xyz + su * T + sv * B;
}

fn sample_cosine_hemisphere_dir(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let u = rand_2f(rng);
    let theta = 2.0 * PI * u.x;
    let r = sqrt(u.y);
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - r * r));
    let T = build_tangent_frame(normal);
    let B = cross(normal, T);
    return x * T + y * B + z * normal;
}

fn sample_cone_toward(origin: vec3<f32>, targ: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let dir = normalize(targ - origin);
    let T = build_tangent_frame(dir);
    let B = cross(dir, T);

    let uc = rand_1f(rng);
    let u = rand_2f(rng);
    let cos_theta = 1.0 - uc * (1.0 - PHOTON_CONE_COS);
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * u.x;
    return sin_theta * cos(phi) * T + sin_theta * sin(phi) * B + cos_theta * dir;
}

// ----- Photon evaluation -----

fn evaluate_bsdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, mat: GpuMaterial, lambda_nm: f32) -> vec3<f32> {
    if (mat.material_type == 0u) {
        return oren_nayar_f(wo, wi, n, mat.color.rgb, mat.roughness);
    }
    let ndotv = dot(n, wo);
    let ndotl = dot(n, wi);
    if (ndotv * ndotl > 0.0) {
        // Reflection: evaluate microfacet BRDF
        let alpha = roughness_to_alpha(mat.roughness);
        let eta = cauchy_ior(mat.ior, lambda_nm);
        let wm = normalize(wi + wo);
        let R = fr_dielectric(dot(wo, wm), eta);
        let D = tr_d(wm, alpha);
        let G = tr_g(wo, wi, alpha);
        return vec3<f32>(D * G * R / max(4.0 * abs_cos_theta(wi) * abs_cos_theta(wo), 1e-10));
    }
    // Transmission: not evaluated (photons contribute mainly via reflection paths)
    return vec3<f32>(0.0);
}

fn trace_photon(rng: ptr<function, u32>, vis_pos: vec3<f32>, vis_norm: vec3<f32>,
                vis_wo: vec3<f32>, vis_mat: GpuMaterial, vis_throughput: vec3<f32>,
                rad: f32, lambda_nm: f32, light: Light) -> vec3<f32> {
    var contrib = vec3<f32>(0.0);
    let light_power = light.color.rgb * light.color.w;

    var rayon: Ray;
    var throughput: vec3<f32>;
    if (light.light_type == 0u) {
        let cone_factor = (1.0 - PHOTON_CONE_COS) * 0.5;
        throughput = light_power / f32(K_PHOTONS) * cone_factor;
        rayon = Ray(light.position.xyz, sample_cone_toward(light.position.xyz, vec3<f32>(0.0, 0.0, 0.0), rng));
    } else {
        let u_emit = rand_2f(rng);
        let lp = sample_square_point(light, u_emit);
        let l_norm = light_normal(light);
        let dir = sample_cosine_hemisphere_dir(l_norm, rng);
        throughput = light_power / f32(K_PHOTONS);
        rayon = Ray(lp + l_norm * EPS, dir);
    }
    var ray = rayon;

    for (var bounce: u32 = 0u; bounce < MAX_PHOTON_BOUNCES; bounce = bounce + 1u) {
        var hit = closest_sphere_hit(ray);
        let tri_hit = closest_triangle_hit(ray);
        if (tri_hit.distance > 0.0 && abs(tri_hit.distance) < abs(hit.distance)) { hit = tri_hit; }

        if (hit.distance < 0.0) { break; }

        let dist = length(hit.location - vis_pos);
        if (dist < rad) {
            let wi_photon = -ray.direction;
            let f = evaluate_bsdf(vis_wo, wi_photon, vis_norm, vis_mat, lambda_nm);
            let kernel = 1.0 - dist / rad;
            contrib += vis_throughput * f * throughput * kernel / max(PI * rad * rad, 1e-10);
        }

        let mat = materials[hit.material_id];
        let wo = -ray.direction;
        let normal = hit.normal;

        if (mat.material_type == 0u) {
            // Diffuse: cosine scatter
            let rn = rand_unit_vec(rng);
            let wi = normalize(normal + rn);
            let pdf = max(dot(normal, wi), 1e-10) * INV_PI;
            let f_diff = oren_nayar_f(normalize(wo), wi, normal, mat.color.rgb, mat.roughness);
            let cos_term = max(dot(normal, wi), 1e-10);
            throughput *= f_diff * cos_term / max(pdf, 1e-10);
            ray = Ray(hit.location + normal * EPS, wi);
        } else {
            // Dielectric
            let eta = cauchy_ior(mat.ior, lambda_nm);
            let alpha = roughness_to_alpha(mat.roughness);

            if (effectively_smooth(alpha)) {
                let cos_t = dot(wo, normal);
                let R = fr_dielectric(abs(cos_t), eta);
                if (rand_1f(rng) < R) {
                    let wi = reflect_dir(wo, normal);
                    ray = Ray(hit.location + normal * EPS, wi);
                } else {
                    let wi = refract_dir(wo, normal, eta);
                    if (length(wi) < 0.5) { break; }
                    let etap = select(eta, 1.0 / eta, cos_t < 0.0);
                    throughput /= (etap * etap);
                    ray = Ray(hit.location - normal * EPS, wi);
                }
            } else {
                // Rough GGX
                let T = build_tangent_frame(normal);
                let B = cross(normal, T);
                let wo_l = vec3<f32>(dot(wo, T), dot(wo, B), dot(wo, normal));
                let u_sample = rand_2f(rng);
                let wm = tr_sample_wm(wo_l, u_sample, alpha);

                let dot_wowm = abs(dot(wo_l, wm));
                let R = fr_dielectric(dot_wowm, eta);
                let Tns = 1.0 - R;

                if (rand_1f(rng) < R / max(R + Tns, 1e-10)) {
                    let wi_l = reflect_dir(wo_l, wm);
                    if (!same_hemisphere(wo_l, wi_l)) { break; }
                    let D = tr_d(wm, alpha);
                    let G = tr_g(wo_l, wi_l, alpha);
                    let bsdf = D * G * R / max(4.0 * abs_cos_theta(wi_l) * abs_cos_theta(wo_l), 1e-10);
                    let pdf = tr_lambda(wo_l, alpha) + 1.0;
                    let cos_term = abs_cos_theta(wi_l);
                    throughput *= bsdf * cos_term / max(pdf, 1e-10);
                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    ray = Ray(hit.location + normal * EPS, wi_w);
                } else {
                    let wi_l = refract_dir(wo_l, wm, eta);
                    if (length(wi_l) < 0.5 || same_hemisphere(wo_l, wi_l)) { break; }
                    let D = tr_d(wm, alpha);
                    let G = tr_g(wo_l, wi_l, alpha);
                    let ct_i = abs_cos_theta(wi_l);
                    let ct_o = abs_cos_theta(wo_l);
                    let denom = dot(wi_l, wm) + dot(wo_l, wm) / eta;
                    let bsdf = Tns * D * G * abs(dot(wi_l, wm) * dot(wo_l, wm) / max(ct_i * ct_o * denom * denom, 1e-10));
                    let pdf = tr_lambda(wo_l, alpha) + 1.0;
                    throughput *= bsdf * ct_i / max(pdf, 1e-10);
                    let etap = select(eta, 1.0 / eta, wo_l.z < 0.0);
                    throughput /= (etap * etap);
                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    ray = Ray(hit.location - normal * EPS, wi_w);
                }
            }
        }

        let prob = max(throughput.r, max(throughput.g, throughput.b));
        if (prob < 0.01) { break; }
        if (rand_1f(rng) > prob) { break; }
        throughput /= prob;
    }
    return contrib;
}

// ----- Main trace function -----

fn recursive_trace(r: Ray, rng: ptr<function, u32>, lambda_nm: f32, pixel_idx: u32) -> vec3<f32> {
    let max_depth: u32 = params.depth;

    var throughput: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var radiance: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var cur_ray: Ray = r;
    var vp_stored = false;

    for (var bounce: u32 = 0u; bounce < max_depth; bounce = bounce + 1u) {
        var best_hit = closest_sphere_hit(cur_ray);
        let triangle_hit = closest_triangle_hit(cur_ray);
        if (triangle_hit.distance > 0.0 && abs(triangle_hit.distance) < abs(best_hit.distance)) {
            best_hit = triangle_hit;
        }

        if (best_hit.distance < 0.0) {
            radiance += throughput * sky_color(cur_ray.direction);
            break;
        }

        let mat = materials[best_hit.material_id];
        let normal = best_hit.normal;
        let wo = -cur_ray.direction;

        if (mat.material_type == 0u) {
            let mat_color = mat.color.rgb;

            // Store vispoint at first diffuse hit
            if (!vp_stored && pixel_idx < params.width * params.height) {
                vispoints[pixel_idx] = Vispoint(
                    vec4<f32>(best_hit.location, 0.0),
                    vec4<f32>(normal, f32(best_hit.material_id)),
                    vec4<f32>(wo, 0.0),
                    vec4<f32>(throughput, 0.0),
                );
                vp_stored = true;
            }

            let direct = sample_direct_lighting(best_hit.location, normal, lambda_nm, rng);
            radiance += throughput * mat_color * direct;

            let rn = rand_unit_vec(rng);
            let wi = normalize(normal + rn);
            let pdf = max(dot(normal, wi), 1e-10) * INV_PI;
            let f_diff = oren_nayar_f(normalize(wo), wi, normal, mat_color, mat.roughness);
            let cos_term = max(dot(normal, wi), 1e-10);
            throughput *= f_diff * cos_term / max(pdf, 1e-10);
            cur_ray = Ray(best_hit.location + normal * EPS, wi);

        } else {
            let eta = mat.ior;
            let alpha = roughness_to_alpha(mat.roughness);

            if (effectively_smooth(alpha)) {
                let cos_theta = dot(wo, normal);
                let R = fr_dielectric(abs(cos_theta), eta);
                if (rand_1f(rng) < R) {
                    let wi = reflect_dir(wo, normal);
                    cur_ray = Ray(best_hit.location + normal * EPS, wi);
                } else {
                    let wi = refract_dir(wo, normal, eta);
                    if (length(wi) < 0.5) { break; }
                    let etap = select(eta, 1.0 / eta, cos_theta < 0.0);
                    throughput /= (etap * etap);
                    cur_ray = Ray(best_hit.location - normal * EPS, wi);
                }
            } else {
                let T = build_tangent_frame(normal);
                let B = cross(normal, T);
                let wo_l = vec3<f32>(dot(wo, T), dot(wo, B), dot(wo, normal));
                let u_sample = rand_2f(rng);
                let wm = tr_sample_wm(wo_l, u_sample, alpha);
                let dot_wowm = abs(dot(wo_l, wm));
                let R = fr_dielectric(dot_wowm, eta);
                let Tns = 1.0 - R;

                if (rand_1f(rng) < R / max(R + Tns, 1e-10)) {
                    let wi_l = reflect_dir(wo_l, wm);
                    if (!same_hemisphere(wo_l, wi_l)) { break; }
                    let D = tr_d(wm, alpha);
                    let G = tr_g(wo_l, wi_l, alpha);
                    let ct_i = abs_cos_theta(wi_l);
                    let ct_o = abs_cos_theta(wo_l);
                    let bsdf = D * G * R / max(4.0 * ct_i * ct_o, 1e-10);
                    let G1 = 1.0 / (1.0 + tr_lambda(wo_l, alpha));
                    let pdf_wm = (G1 / max(ct_o, 1e-10)) * D * dot_wowm;
                    let pdf = max(pdf_wm / max(4.0 * dot_wowm, 1e-10), 1e-10) * (R / max(R + Tns, 1e-10));
                    throughput *= bsdf * ct_i / max(pdf, 1e-10);
                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    cur_ray = Ray(best_hit.location + normal * EPS, wi_w);
                } else {
                    let wi_l = refract_dir(wo_l, wm, eta);
                    if (length(wi_l) < 0.5 || same_hemisphere(wo_l, wi_l)) { break; }
                    let D = tr_d(wm, alpha);
                    let G = tr_g(wo_l, wi_l, alpha);
                    let ct_i = abs_cos_theta(wi_l);
                    let ct_o = abs_cos_theta(wo_l);
                    let denom = dot(wi_l, wm) + dot(wo_l, wm) / eta;
                    let bsdf = Tns * D * G * abs(dot(wi_l, wm) * dot(wo_l, wm) / max(ct_i * ct_o * denom * denom, 1e-10));
                    let dwm_dwi = abs(dot(wi_l, wm)) / max(denom * denom, 1e-10);
                    let G1 = 1.0 / (1.0 + tr_lambda(wo_l, alpha));
                    let pdf = max((G1 / max(ct_o, 1e-10)) * D * dot_wowm * dwm_dwi * (Tns / max(R + Tns, 1e-10)), 1e-10);
                    throughput *= bsdf * ct_i / pdf;
                    let etap = select(eta, 1.0 / eta, wo_l.z < 0.0);
                    throughput /= (etap * etap);
                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    cur_ray = Ray(best_hit.location - normal * EPS, wi_w);
                }
            }
        }

        let prob = max(throughput.r, max(throughput.g, throughput.b));
        if (prob < 0.001) { break; }
        if (rand_1f(rng) > prob) { break; }
        throughput /= prob;
    }
    return radiance;
}

@compute @workgroup_size(8, 4, 1)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let pixel_idx = global_id.y * params.width + global_id.x;
    let pixel_coords = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    var rng: u32 = params.seed + 1203793u * global_id.x + 7u * global_id.y;
    let rand = rand_2f(&rng);
    let r = get_ray(pixel_coords.x + rand.x / f32(params.width), pixel_coords.y + rand.y / f32(params.height));

    let lambda = VISIBLE_MIN + rand_1f(&rng) * VISIBLE_RANGE;
    let cam_radiance = recursive_trace(r, &rng, lambda, pixel_idx);

    // Photon pass
    var photon_contrib = vec3<f32>(0.0);
    let num_lights = arrayLength(&scene_lights);
    if (pixel_idx < params.width * params.height && num_lights > 0u) {
        let vp = vispoints[pixel_idx];
        // Check if vispoint was stored (has non-zero position roughly)
        if (length(vp.position.xyz) > 0.001) {
            let vis_mat_id = u32(vp.normal.w);
            let vis_mat = materials[vis_mat_id];
            for (var k: u32 = 0u; k < K_PHOTONS; k = k + 1u) {
                let li = k % num_lights;
                let light = scene_lights[li];
                photon_contrib += trace_photon(&rng, vp.position.xyz, vp.normal.xyz,
                    vp.wo.xyz, vis_mat, vp.throughput.xyz,
                    params.photon_radius, lambda, light);
            }
        }
    }

    var pixel_color = vec4<f32>(cam_radiance + photon_contrib, 1.0);

    let prev = textureLoad(output_tex, vec2<i32>(global_id.xy));
    pixel_color = pixel_color + prev;
    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}

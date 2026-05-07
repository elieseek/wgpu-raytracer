struct Params {
    width: u32,
    height: u32,
    depth: u32,
    seed: u32,
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, read_write>;

@group(1) @binding(0) var<uniform> camera: Camera;

@group(2) @binding(0) var<storage, read> sphere_instances: SphereInstanceArray;
@group(3) @binding(0) var<storage, read> mesh_positions: array<vec3<f32>>;
@group(3) @binding(1) var<storage, read> mesh_indices: array<vec4<u32>>;
@group(4) @binding(0) var<storage, read> materials: array<GpuMaterial>;
@group(5) @binding(0) var<storage, read> bvh_nodes: array<BVHNode>;
@group(5) @binding(1) var<storage, read> bvh_triangle_indices: array<u32>;
@group(6) @binding(0) var<storage, read> scene_lights: array<Light>;
@group(6) @binding(1) var<storage, read> cie_table: array<vec4<f32>>;

const EPS: f32 = 1e-5;
const VISIBLE_MIN: f32 = 380.0;
const VISIBLE_RANGE: f32 = 400.0;
const PI: f32 = 3.1415926535;
const INV_PI: f32 = 0.31830988618;

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

fn roughness_to_alpha(roughness: f32) -> f32 {
    return sqrt(roughness);
}

fn effectively_smooth(alpha: f32) -> bool {
    return alpha < 1e-3;
}

fn tr_sample_wm(wo: vec3<f32>, u: vec2<f32>, alpha: f32) -> vec3<f32> {
    let wh = normalize(vec3<f32>(alpha * wo.x, alpha * wo.y, wo.z));
    var wh_adj = wh;
    if (wh_adj.z < 0.0) { wh_adj = -wh_adj; }

    var t1: vec3<f32>;
    if (abs(wh_adj.z) > 0.99999) {
        t1 = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        t1 = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), wh_adj));
    }
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
    let triangle = Triangle(
        mesh_positions[ vertices[0] ].xyz, mesh_positions[ vertices[1] ].xyz, mesh_positions[ vertices[2] ].xyz
    );
    let flt_eps = bitcast<f32>(0x1p-126f);
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if (det > - flt_eps && det < flt_eps) { return hit; }

    let inv_det = 1. / det;
    let s = r.origin - triangle.a;
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
        hit.material_id = mesh_indices[triangle_index].w;
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

fn closest_triangle_hit(r: Ray) -> Hit {
    return intersect_bvh(r);
}

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
    let va = cie_table[a];
    let vb = cie_table[b];
    return mix(va, vb, f).xyz;
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
    let triangle = Triangle(
        mesh_positions[vertices[0]].xyz,
        mesh_positions[vertices[1]].xyz,
        mesh_positions[vertices[2]].xyz
    );
    let flt_eps = bitcast<f32>(0x1p-126f);
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);
    if (abs(det) < flt_eps) { return false; }

    let inv_det = 1.0 / det;
    let s = r.origin - triangle.a;
    let u = inv_det * dot(s, ray_cross_e2);
    if (u < 0.0 || u > 1.0) { return false; }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(r.direction, s_cross_e1);
    if (v < 0.0 || u + v > 1.0) { return false; }

    let t = inv_det * dot(edge2, s_cross_e1);
    return t > flt_eps && t < t_max;
}

fn any_hit(r: Ray, t_max: f32) -> bool {
    let num_spheres = arrayLength(&sphere_instances.contents);
    for (var i = 0u; i < num_spheres; i = i + 1u) {
        if (hit_sphere_shadow(r, sphere_instances.contents[i], t_max)) { return true; }
    }
    let num_nodes = arrayLength(&bvh_nodes);
    if (num_nodes == 0u) { return false; }

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
                let tri_idx = bvh_triangle_indices[node.first_triangle + ti];
                if (hit_triangle_shadow(r, tri_idx, t_max)) { return true; }
            }
        } else {
            stack[sp] = node.right_child;
            sp = sp + 1u;
            stack[sp] = node.left_child;
            sp = sp + 1u;
        }
    }
    return false;
}

// ----- Direct lighting (Lambertian only) -----

fn sample_direct_lighting(pos: vec3<f32>, norm: vec3<f32>, lambda_nm: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    let num_lights = arrayLength(&scene_lights);
    for (var i = 0u; i < num_lights; i = i + 1u) {
        let light = scene_lights[i];
        let to_light = light.position.xyz - pos;
        let dist = length(to_light);
        if (dist < EPS) { continue; }
        let light_dir = to_light / dist;
        let ndotl = dot(norm, light_dir);
        if (ndotl <= 0.0) { continue; }

        let shadow_ray = Ray(pos + norm * EPS, light_dir);
        if (any_hit(shadow_ray, dist - EPS)) { continue; }

        var spd: f32;
        if (light.color_temp > 0.0) { spd = blackbody(lambda_nm, light.color_temp); }
        else { spd = 1.0; }

        let cie_factor = cie_to_rgb(lambda_nm);
        let light_rgb = light.color.rgb * light.color.w * spd * cie_factor * VISIBLE_RANGE;
        result += light_rgb * ndotl / (dist * dist);
    }
    return result;
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (dir.y + 1.0);
    return vec3<f32>(0.0, 0.0, 0.0);
}

// ----- Fresnel and reflection/refraction helpers -----

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
    let t = select(normalize(cross(up, n)), alt, abs(n.y) > 0.99999);
    return t;
}

// ----- Main trace function -----

fn recursive_trace(r: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    let max_depth: u32 = params.depth;
    let lambda = VISIBLE_MIN + rand_1f(rng) * VISIBLE_RANGE;

    var throughput: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var radiance: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var cur_ray: Ray = r;

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
        let wo = -cur_ray.direction; // from surface toward viewer

        if (mat.material_type == 0u) {
            // ---- DIFFUSE with Oren-Nayar roughness ----
            let mat_color = mat.color.rgb;
            let sigma = mat.roughness;

            // Direct lighting
            let direct = sample_direct_lighting(best_hit.location, normal, lambda);
            radiance += throughput * mat_color * direct;

            // Cosine-weighted hemisphere scatter
            let rn = rand_unit_vec(rng);
            let wi = normalize(normal + rn);

            let pdf = max(dot(normal, wi), 1e-10) * INV_PI;
            let f_diff = oren_nayar_f(normalize(wo), wi, normal, mat_color, sigma);
            let cos_term = max(dot(normal, wi), 1e-10);
            throughput *= f_diff * cos_term / max(pdf, 1e-10);

            cur_ray = Ray(best_hit.location + normal * EPS, wi);

            // Russian roulette
            let prob = max(throughput.r, max(throughput.g, throughput.b));
            if (prob < 0.001) { break; }
            if (rand_1f(rng) > prob) { break; }
            throughput /= prob;

        } else {
            // ---- DIELECTRIC (roughness via GGX microfacet) ----
            let eta = mat.ior;
            let alpha = roughness_to_alpha(mat.roughness);

            if (effectively_smooth(alpha)) {
                // ---- Smooth: perfect specular (Fresnel only) ----
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
                // ---- Rough: microfacet GGX ----
                let T = build_tangent_frame(normal);
                let B = cross(normal, T);
                let wo_l = vec3<f32>(dot(wo, T), dot(wo, B), dot(wo, normal));

                let u_sample = rand_2f(rng);
                let wm = tr_sample_wm(wo_l, u_sample, alpha);

                let dot_wo_wm = abs(dot(wo_l, wm));
                let R = fr_dielectric(dot_wo_wm, eta);
                let Tns = 1.0 - R;

                if (rand_1f(rng) < R / max(R + Tns, 1e-10)) {
                    // Reflection
                    let wi_l = reflect_dir(wo_l, wm);
                    if (!same_hemisphere(wo_l, wi_l)) { break; }

                    let D = tr_d(wm, alpha);
                    let G = tr_g(wo_l, wi_l, alpha);
                    let ct_i = abs_cos_theta(wi_l);
                    let ct_o = abs_cos_theta(wo_l);

                    let bsdf = D * G * R / max(4.0 * ct_i * ct_o, 1e-10);
                    let G1 = 1.0 / (1.0 + tr_lambda(wo_l, alpha));
                    let pdf_wm = (G1 / max(ct_o, 1e-10)) * D * dot_wo_wm;
                    let pdf = max(pdf_wm / max(4.0 * dot_wo_wm, 1e-10), 1e-10) * (R / max(R + Tns, 1e-10));

                    throughput *= bsdf * ct_i / max(pdf, 1e-10);

                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    cur_ray = Ray(best_hit.location + normal * EPS, wi_w);
                } else {
                    // Transmission
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
                    let pdf = max((G1 / max(ct_o, 1e-10)) * D * dot_wo_wm * dwm_dwi * (Tns / max(R + Tns, 1e-10)), 1e-10);

                    throughput *= bsdf * ct_i / pdf;
                    let etap = select(eta, 1.0 / eta, wo_l.z < 0.0);
                    throughput /= (etap * etap);

                    let wi_w = wi_l.x * T + wi_l.y * B + wi_l.z * normal;
                    cur_ray = Ray(best_hit.location - normal * EPS, wi_w);
                }
            }

            // Russian roulette
            let prob = max(throughput.r, max(throughput.g, throughput.b));
            if (prob < 0.001) { break; }
            if (rand_1f(rng) > prob) { break; }
            throughput /= prob;
        }
    }
    return radiance;
}

@compute @workgroup_size(8, 4, 1)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    var rng: u32 = params.seed + 1203793u * global_id.x + 7u * global_id.y;
    let rand = rand_2f(&rng);
    let r = get_ray(pixel_coords.x + rand.x / f32(params.width), pixel_coords.y + rand.y / f32(params.height));

    var pixel_color = vec4<f32>(recursive_trace(r, &rng), 1.0);
    let prev = textureLoad(output_tex, vec2<i32>(global_id.xy));
    pixel_color = pixel_color + prev;
    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}

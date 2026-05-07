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

struct LambertianArray {
    contents: array<vec3<f32>>,
};

struct Triangle {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
}

struct Hit {
    distance: f32,
    color: vec3<f32>,
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
@group(3) @binding(1) var<storage, read> mesh_indices: array<vec3<u32>>;
@group(4) @binding(0) var<storage, read> lambertians: LambertianArray;
@group(5) @binding(0) var<storage, read> bvh_nodes: array<BVHNode>;
@group(5) @binding(1) var<storage, read> bvh_triangle_indices: array<u32>;
@group(6) @binding(0) var<storage, read> scene_lights: array<Light>;
@group(6) @binding(1) var<storage, read> cie_table: array<vec4<f32>>;

const EPS: f32 = 1e-5;
const VISIBLE_MIN: f32 = 380.0;
const VISIBLE_RANGE: f32 = 400.0;

fn get_ray(u: f32, v: f32) -> Ray {
    var ray: Ray;
    ray.origin = camera.origin.xyz;
    ray.direction = (camera.lower_left_corner
                    + camera.horizontal * u
                    + camera.vertical * v
                    - camera.origin).xyz;
    return ray;
}

fn hit_sphere(r: Ray, sphere: SphereInstance) -> Hit {
    let transform_mat = sphere.transform;
    let color = lambertians.contents[sphere.material_id];
    let center = (transform_mat * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    let radius = sphere.scale;
    let oc: vec3<f32> = r.origin - center;
    let a: f32 = dot(r.direction, r.direction);
    let half_b: f32 = dot(oc, r.direction);
    let c: f32 = dot(oc, oc) - radius*radius;
    let discriminant: f32 = half_b * half_b - a*c;

    var hit: Hit;
    hit.distance = -1.0;
    hit.color = vec3<f32>(0., 0., 0.);
    if (discriminant > 0.) {
        let hit_distance = (-half_b - sqrt(discriminant)) / a;
        let hit_location = r.origin + r.direction * hit_distance * 0.9999;
        hit.distance = hit_distance;
        hit.color = color;
        hit.location = hit_location;
        hit.normal = normalize(hit_location - center);
    };

    return hit;
}

fn hit_triangle(r: Ray, triangle_index: u32) -> Hit {
    var hit: Hit;
    hit.distance = -1.;
    hit.color = vec3<f32>(0., 0., 0.);
    let vertices = mesh_indices[triangle_index];
    let triangle = Triangle(
        mesh_positions[ vertices[0] ].xyz, mesh_positions[ vertices[1] ].xyz, mesh_positions[ vertices[2] ].xyz
    );
    let epsilon = bitcast<f32>(0x1p-126f);
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    if (det > - epsilon && det < epsilon) {
        return hit;
    }

    let inv_det = 1. / det;
    let s = r.origin - triangle.a;
    let u = inv_det * dot(s, ray_cross_e2);

    if (u < 0. || u > 1.) {
        return hit;
    }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(r.direction, s_cross_e1);

    if (v < 0. || u + v > 1.) {
        return hit;
    }

    let t = inv_det * dot(edge2, s_cross_e1);

    if (t > epsilon) {
        let normal = cross(edge1, edge2);
        hit.normal = normalize(normal);
        if dot(hit.normal, r.direction) > 0. {
            hit.color = vec3<f32>(0., 0., 0.);
        }
        // Use first material for mesh (grey)
        hit.color = lambertians.contents[0];
        hit.location = r.origin + hit.normal*1e-5 +  r.direction * t;
        hit.distance = t;
        return hit;
    }
    return hit;
}

fn closest_sphere_hit(r: Ray) -> Hit {
    var num_instances: i32 = bitcast<i32>(arrayLength(&sphere_instances.contents));
    var current_hit: Hit;
    var best_hit: Hit;
    let t = 0.5 * (r.direction.y + 1.0);
    best_hit.distance = -10000000.0;
    best_hit.color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
    for (var i: i32 = 0; i < num_instances; i=i+1) {
        current_hit = hit_sphere(r, sphere_instances.contents[i]);
        if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
            best_hit = current_hit;
        };
    }
    return best_hit;
}

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
    let t = 0.5 * (r.direction.y + 1.0);
    best_hit.distance = -10000000.0;
    best_hit.color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);

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
                current_hit.color = current_hit.normal;
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

// ---------- Blackbody and CIE spectral functions ----------

fn blackbody(lambda_nm: f32, temp: f32) -> f32 {
    let h = 6.62607015e-34;
    let c = 2.99792458e8;
    let k = 1.380649e-23;
    let c1 = 2.0 * h * c * c;
    let c2 = h * c / k;
    let l = lambda_nm * 1e-9;
    let result = c1 / (pow(l, 5.0) * (exp(c2 / (l * temp)) - 1.0));
    return result * 1e-14;
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

// ---------- Shadow rays ----------

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
    let epsilon = bitcast<f32>(0x1p-126f);
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);
    if (abs(det) < epsilon) { return false; }

    let inv_det = 1.0 / det;
    let s = r.origin - triangle.a;
    let u = inv_det * dot(s, ray_cross_e2);
    if (u < 0.0 || u > 1.0) { return false; }

    let s_cross_e1 = cross(s, edge1);
    let v = inv_det * dot(r.direction, s_cross_e1);
    if (v < 0.0 || u + v > 1.0) { return false; }

    let t = inv_det * dot(edge2, s_cross_e1);
    return t > epsilon && t < t_max;
}

fn any_hit(r: Ray, t_max: f32) -> bool {
    // Test spheres
    let num_spheres = arrayLength(&sphere_instances.contents);
    for (var i = 0u; i < num_spheres; i = i + 1u) {
        if (hit_sphere_shadow(r, sphere_instances.contents[i], t_max)) {
            return true;
        }
    }

    // Test BVH
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
                if (hit_triangle_shadow(r, tri_idx, t_max)) {
                    return true;
                }
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

// ---------- Direct lighting ----------

fn sample_direct_lighting(hit_location: vec3<f32>, hit_normal: vec3<f32>, lambda_nm: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0, 0.0, 0.0);
    let num_lights = arrayLength(&scene_lights);
    for (var i = 0u; i < num_lights; i = i + 1u) {
        let light = scene_lights[i];
        let to_light = light.position.xyz - hit_location;
        let dist = length(to_light);
        if (dist < EPS) { continue; }
        let light_dir = to_light / dist;

        let ndotl = dot(hit_normal, light_dir);
        if (ndotl <= 0.0) { continue; }

        // Shadow ray
        let shadow_ray = Ray(hit_location + hit_normal * EPS, light_dir);
        if (any_hit(shadow_ray, dist - EPS)) { continue; }

        // Light SPD at this wavelength
        var spd: f32;
        if (light.color_temp > 0.0) {
            spd = blackbody(lambda_nm, light.color_temp);
        } else {
            spd = 1.0;
        }

        // CIE factor: unit spectral radiance at lambda → sRGB
        let cie_factor = cie_to_rgb(lambda_nm);

        // Light has RGB base color + intensity.  Multiply by cie_factor*VISIBLE_RANGE
        // to convert monochromatic contribution to sRGB with Monte Carlo PDF.
        let light_rgb = light.color.rgb * light.color.w * spd * cie_factor * VISIBLE_RANGE;

        // Lambertian diffuse, inverse-square falloff
        result += light_rgb * ndotl / (dist * dist);
    }
    return result;
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (dir.y + 1.0);
    return vec3<f32>(0.0, 0.0, 0.0); //(1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
}

fn rand(rng: ptr<function, u32>) -> u32 {
    let oldstate: u32 = *rng;
    let res = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    *rng = *rng * 747796405u + 2891336453u;
    return (res >> 22u) ^ res;
}

fn rand_1f(rng: ptr<function, u32>) -> f32 {
    let rand_res = rand(rng);
    return f32(rand_res) * (1.0 / f32(0xFFFFFFFFu));
}

fn rand_2f(rng: ptr<function, u32>) -> vec2<f32> {
    return vec2<f32>(rand_1f(rng), rand_1f(rng));
}

fn rand_unit_vec(rng: ptr<function, u32>) -> vec3<f32> {
    let pi = 3.1415926535;
    let rand = rand_2f(rng);
    let theta = 2.0 * pi * rand.x;
    let phi = acos(1.0 - 2.0 * rand.y);
    let x = sin(phi) * cos(theta);
    let y = sin(phi) * sin(theta);
    let z = cos(phi);
    return vec3<f32>(x, y, z);
}

fn recursive_trace(r: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    let max_depth: u32 = params.depth;

    // Sample hero wavelength for this path
    let lambda = VISIBLE_MIN + rand_1f(rng) * VISIBLE_RANGE;

    var throughput: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var radiance: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var cur_ray: Ray = r;

    for (var i: u32 = 0u; i < max_depth; i = i + 1u) {
        var best_hit = closest_sphere_hit(cur_ray);
        let triangle_hit = closest_triangle_hit(cur_ray);
        if (triangle_hit.distance > 0.0 && abs(triangle_hit.distance) < abs(best_hit.distance)) {
            best_hit = triangle_hit;
        }

        if (best_hit.distance < 0.0) {
            radiance += throughput * sky_color(cur_ray.direction);
            break;
        }

        // Direct lighting at hero wavelength
        let direct = sample_direct_lighting(best_hit.location, best_hit.normal, lambda);
        radiance += throughput * best_hit.color * direct;

        // Scatter (diffuse bounce)
        throughput *= best_hit.color;
        let rand = rand_unit_vec(rng);
        let new_direction = normalize(best_hit.normal + rand);
        cur_ray = Ray(best_hit.location, new_direction);

        // Russian roulette
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
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    var rng: u32 = params.seed + 1203793u * global_id.x + 7u * global_id.y;
    let rand = rand_2f(&rng);
    let r = get_ray(pixel_coords.x + rand.x / f32(params.width), pixel_coords.y + rand.y / f32(params.height));

    var pixel_color = vec4<f32>(recursive_trace(r, &rng), 1.0);

    let prev = textureLoad(output_tex, vec2<i32>(global_id.xy));
    pixel_color = pixel_color + prev;
    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}

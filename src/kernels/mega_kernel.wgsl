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

struct MeshPositions {
    positions: array<vec4<f32>>,
};
struct MeshIndices {
    indices: array<vec4<u32>>,
};

struct LambertianArray {
    contents: array<vec3<f32>>,
};

struct Sphere {
    radius: f32,
    position: vec3<f32>,
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, read_write>;

@group(1) @binding(0) var<uniform> camera: Camera;

@group(2) @binding(0) var<storage, read> sphere_instances: SphereInstanceArray;
@group(3) @binding(0) var<storage, read> mesh_positions: array<vec3<f32>>;
@group(3) @binding(1) var<storage, read> mesh_indices: array<vec3<u32>>;
@group(3) @binding(2) var<storage, read> mesh_normals: array<vec3<f32>>;
@group(3) @binding(3) var<storage, read> mesh_normal_indices: array<vec3<u32>>;
@group(4) @binding(0) var<storage, read> lambertians: LambertianArray;

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
        // prevent self-collisions by raising hit location off surface slightly
        let hit_location = r.origin + r.direction * hit_distance * 0.9999;
        hit.distance = hit_distance;
        hit.color = color;
        hit.location = hit_location;
        hit.normal = normalize(hit_location - center);
    };

    return hit;
}

// Möller–Trumbore intersection algorithm
fn hit_triangle(r: Ray, triangle_index: u32) -> Hit {
    let vertices = mesh_indices[triangle_index];
    let triangle = Triangle(
        mesh_positions[ vertices[0] ].xyz, mesh_positions[ vertices[1] ].xyz, mesh_positions[ vertices[2] ].xyz
    );
    let epsilon = bitcast<f32>(0x1p-126f);
    let edge1 = triangle.b - triangle.a;
    let edge2 = triangle.c - triangle.a;
    let ray_cross_e2 = cross(r.direction, edge2);
    let det = dot(edge1, ray_cross_e2);

    var hit: Hit;
    hit.distance = -1.;
    hit.color = vec3<f32>(1., 0., 0.);

    // Ray is parallel to triangle
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
        hit.location = r.origin + r.direction * t * 0.9999;
        hit.distance = t;
        let ni = mesh_normal_indices[triangle_index];
        hit.normal = normalize(cross(edge1, edge2));
        return hit;
    }
    return hit;
}

fn closest_sphere_hit(r: Ray) -> Hit {
    var transform_mat: mat4x4<f32>;
    var n: vec3<f32>;
    var sphere: SphereInstance;
    var color: vec3<f32>;
    var num_instances: i32 = bitcast<i32>(arrayLength(&sphere_instances.contents));
    var current_hit: Hit;
    var best_hit: Hit;
    let t = 0.5 * (r.direction.y + 1.0);
    best_hit.distance = -10000000.0;
    best_hit.color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
    for (var i: i32 = 0; i < num_instances; i=i+1) {
        sphere = sphere_instances.contents[i];
       
        current_hit = hit_sphere(r, sphere);
        if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
            best_hit = current_hit;
        }; 
    }

    return best_hit;
}

fn closest_triangle_hit(r: Ray) -> Hit {
    var color: vec3<f32>;
    var num_instances: u32 = bitcast<u32>(arrayLength(&mesh_indices));
    var current_hit: Hit;
    var best_hit: Hit;
    let t = 0.5 * (r.direction.y + 1.0);
    best_hit.distance = -10000000.0;
    best_hit.color = (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0) + t * vec3<f32>(0.5, 0.7, 1.0);
    for (var i: u32 = 0u; i < num_instances; i=i+1u) {
        current_hit = hit_triangle(r, i);
        current_hit.color = current_hit.normal;
        if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
            best_hit = current_hit;
        }; 
    }

    return best_hit;
}

fn rand(rng: ptr<function, u32>) -> u32 { //PCG RXS M XS 32/32
    let oldstate: u32 = *rng;
    let res = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    *rng = *rng * 747796405u + 2891336453u;
    return (res >> 22u) ^ res;
}

fn rand_1f(rng: ptr<function, u32>) -> f32 {
    let rand_res = rand(rng);
    let res = f32(rand_res) * (1.0 / f32(0xFFFFFFFFu));
    return res;
}

fn rand_2f(rng: ptr<function, u32>) -> vec2<f32> {
    let rand_res_1 = rand_1f(rng);
    let rand_res_2 = rand_1f(rng);
    let res = vec2<f32>(rand_res_1, rand_res_2);
    return res;
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
    var albedo: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    var cur_ray: Ray = r;
    for (var i: u32 = 0u; i < max_depth; i=i+1u) {
        var best_hit = closest_sphere_hit(cur_ray);
        let triangle_hit = closest_triangle_hit(cur_ray);
        if (triangle_hit.distance > 0.0 && abs(triangle_hit.distance) < abs(best_hit.distance)) {
            best_hit = triangle_hit;
        }

        albedo = albedo * best_hit.color;
        if (best_hit.distance < 0.0) {
            break;
        }
        let rand = rand_unit_vec(rng);
        let new_direction = normalize(best_hit.normal + rand);
        cur_ray = Ray(best_hit.location, new_direction);
    }
    return albedo;
}

@compute @workgroup_size(8, 4, 1)
fn cs_main(
    @builtin(global_invocation_id) global_id: vec3<u32>, 
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    var rng: u32 = params.seed + 1203793u * global_id.x + 7u * global_id.y;
    let rand = rand_2f(&rng);
    let rand_xy = rand;
    let r = get_ray(pixel_coords.x + rand_xy.x / f32(params.width), pixel_coords.y + rand_xy.y / f32(params.height));

    var pixel_color = vec4<f32>(recursive_trace(r, &rng), 1.0);

    let prev = textureLoad(output_tex, vec2<i32>(global_id.xy));

    pixel_color = pixel_color + prev;

    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}
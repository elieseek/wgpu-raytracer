let NUM_PIXEL_MAX = 3686400;// 2560 * 1440
let TWICE_PIXEL_MAX =  7372800; // 2 * 2560 * 1440

let RAY_TERMINATED = 0u;
let RAY_ACTIVE = 1u;
let RAY_HIT = 2u;
let RAY_INACTIVE = 3u;

struct Params {
    width: u32;
    height: u32;
    seed: u32;
};

struct SOAHit {
    location: vec3<f32>;
    normal: vec3<f32>;
    mat_id: u32;
};

struct StructureOfArrays {
    ray_gen: array<vec4<f32>, TWICE_PIXEL_MAX>;
    ray_hit: array<SOAHit, NUM_PIXEL_MAX>;
    accumulated_albedo: array<vec3<f32>, NUM_PIXEL_MAX>;
    ray_state: array<u32, NUM_PIXEL_MAX>;
};

struct QueueLen {
    ray_gen_len: atomic<u32>;
    ray_hit_len: atomic<u32>;
};

struct Camera {
    origin: vec4<f32>;
    horizontal: vec4<f32>;
    vertical: vec4<f32>;
    lower_left_corner: vec4<f32>;

};

struct Ray {
    origin: vec3<f32>;
    direction: vec3<f32>;
};

struct SphereInstance {
    material_id: u32;
    scale: f32;
    transform: mat4x4<f32>;
};
struct SphereInstanceArray {
    contents: array<SphereInstance>;
};

struct LambertianArray {
    contents: array<vec3<f32>>;
};

struct Sphere {
    radius: f32;
    position: vec3<f32>;
};
let unit_ball = Sphere(1.0, vec3<f32>(0.0, 0.0, 0.0));

struct Hit {
    distance: f32;
    color: vec3<f32>;
    location: vec3<f32>;
    normal: vec3<f32>;
    mat_id: u32;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var output_tex: texture_storage_2d<rgba32float, read_write>;

[[group(1), binding(0)]] var<uniform> camera: Camera;

[[group(2), binding(0)]] var<storage, read> sphere_instances: SphereInstanceArray;
[[group(3), binding(0)]] var<storage, read> lambertians: LambertianArray;

[[group(4), binding(0)]] var<storage, read_write> soa: StructureOfArrays;
[[group(4), binding(1)]] var<storage, read_write> queue_len: QueueLen;

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
    let center = (transform_mat * vec4<f32>(unit_ball.position, 1.0)).xyz;
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
        hit.normal = normalize(hit_location - center);
        hit.location = hit_location;// + 0.00001 * hit.normal;
        hit.mat_id = sphere.material_id;
    };

    return hit;
};

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
        if (current_hit.distance > 0.0 && (abs(current_hit.distance) < abs(best_hit.distance))) {
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

fn wf_reset(global_id: vec3<u32>) {
    let gid = global_id.x * params.height + global_id.y;
    soa.ray_state[gid] = RAY_TERMINATED;
    soa.accumulated_albedo[gid] = vec3<f32>(1.0);
}

fn wf_generate(global_id: vec3<u32>, rng: ptr<function, u32>) {
    let gid = global_id.x * params.height + global_id.y;
    if (global_id.x > params.width || global_id.y > params.height) {
        return;
    };

    if (soa.ray_state[gid] != RAY_TERMINATED) {
        return;
    };

    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    let rand = rand_2f(rng);
    let rand_xy = rand;
    let r = get_ray(pixel_coords.x + rand_xy.x / f32(params.width), pixel_coords.y + rand_xy.y / f32(params.height));

    soa.accumulated_albedo[gid] = vec3<f32>(1.0);
    soa.ray_gen[2u*gid] = vec4<f32>(r.origin, 0.0);
    soa.ray_gen[2u*gid + 1u] = vec4<f32>(r.direction, 0.0);
    soa.ray_state[gid] = RAY_ACTIVE;
}

fn wf_extend(global_id: vec3<u32>, rng: ptr<function, u32>) {
    let gid = global_id.x * params.height + global_id.y;
    if (soa.ray_state[gid] != RAY_ACTIVE) {
        return;
    };
    if (global_id.x > params.width || global_id.y > params.height) {
        return;
    };

    let r = Ray(soa.ray_gen[2u*gid].xyz, soa.ray_gen[2u*gid + 1u].xyz);

    let hit = closest_sphere_hit(r);

    if (hit.distance < 0.00) {
        soa.accumulated_albedo[gid] = soa.accumulated_albedo[gid] * hit.color;
        soa.ray_state[gid] = RAY_INACTIVE;
    } else {
        soa.ray_hit[gid] = SOAHit(hit.location, hit.normal, hit.mat_id);
        soa.ray_state[gid] = RAY_HIT;
    }
}

fn wf_shade(global_id: vec3<u32>, rng: ptr<function, u32>) {
    let gid = global_id.x * params.height + global_id.y;
    if (soa.ray_state[gid] != RAY_HIT) {
        return;
    };
    if (global_id.x > params.width || global_id.y > params.height) {
        return;
    };
    
    let hit: SOAHit = soa.ray_hit[gid];
    let albedo = lambertians.contents[hit.mat_id];
    soa.accumulated_albedo[gid] = soa.accumulated_albedo[gid] * albedo;
    let rand = rand_unit_vec(rng);
    let new_direction = normalize(hit.normal + rand); 
    soa.ray_gen[2u * gid] = vec4<f32>(hit.location, 0.0);
    soa.ray_gen[2u * gid + 1u] = vec4<f32>(new_direction, 0.0);
    soa.ray_state[gid] = RAY_ACTIVE;
}

fn wf_accumulate(global_id: vec3<u32>) {
    let gid = global_id.x * params.height + global_id.y;
    if (soa.ray_state[gid] != RAY_INACTIVE) {
        return;
    };
    if (global_id.x > params.width || global_id.y > params.height) {
        return;
    };
    
    var pixel_color = vec4<f32>(soa.accumulated_albedo[gid], 1.0);
    let prev = textureLoad(output_tex, vec2<i32>(global_id.xy));
    pixel_color = pixel_color + prev;

    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
    soa.ray_state[gid] = RAY_TERMINATED;
}

[[stage(compute), workgroup_size(8, 4, 1)]]
fn cs_main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>, 
    [[builtin(local_invocation_id)]] local_id: vec3<u32>
) {
    var rng = params.seed + 1203793u * global_id.x + 7u * global_id.y;
    let gid = global_id.x * params.height + global_id.y;
    wf_reset(global_id);
    for (var i = 0; i < 30; i = i+1) {
        wf_generate(global_id, &rng);
        wf_extend(global_id, &rng);
        wf_shade(global_id, &rng);
        wf_accumulate(global_id);
    }
    soa.ray_state[gid] = RAY_INACTIVE;
    wf_accumulate(global_id);
}
struct Params {
    width: u32;
    height: u32;
    seed: u32;
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
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var output_tex: texture_storage_2d<rgba8unorm, write>;

[[group(1), binding(0)]] var<uniform> camera: Camera;

[[group(2), binding(0)]] var<storage, read> sphere_instances: SphereInstanceArray;
[[group(3), binding(0)]] var<storage, read> lambertians: LambertianArray;

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
        hit.distance = (-half_b - sqrt(discriminant)) / a;
        hit.color = color;
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
    best_hit.distance = -10000000.0;
    best_hit.color = 0.5 * vec3<f32>(1.0, 1.0, 1.0);
    for (var i: i32 = 0; i < num_instances; i=i+1) {
        sphere = sphere_instances.contents[i];
       
        current_hit = hit_sphere(r, sphere);
        if (current_hit.distance > 0.0 && abs(current_hit.distance) < abs(best_hit.distance)) {
            best_hit = current_hit;
        }; 
    }

    return best_hit;
}

fn rand(input: f32) -> f32 {
    return fract(sin( f32(params.seed) / 10000000000.0 + dot(vec2<f32>(input, input), vec2<f32>(12.9898,78.233))) * 43758.5453);
}

[[stage(compute), workgroup_size(8, 4, 1)]]
fn cs_main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>, 
    [[builtin(local_invocation_id)]] local_id: vec3<u32>
) {
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    let r = get_ray(pixel_coords.x + rand(pixel_coords.x) / f32(params.width), pixel_coords.y + rand(pixel_coords.y) / f32(params.height));
    
    var best_hit: Hit;

    best_hit = closest_sphere_hit(r);

    let pixel_color = vec4<f32>(best_hit.color, 1.0);

    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}
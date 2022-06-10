struct Params {
    width: u32;
    height: u32;
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

fn hit_sphere(center: vec3<f32>, radius: f32, r: Ray) -> f32 {
    let oc: vec3<f32> = r.origin - center;
    let a: f32 = dot(r.direction, r.direction);
    let half_b: f32 = dot(oc, r.direction);
    let c: f32 = dot(oc, oc) - radius*radius;
    let discriminant: f32 = half_b * half_b - a*c;

    var result: f32 = -1.0;
    if (discriminant > 0.) {
        result = (-half_b - sqrt(discriminant)) / a;
    };

    return result;
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


[[stage(compute), workgroup_size(8, 4, 1)]]
fn cs_main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>, 
    [[builtin(local_invocation_id)]] local_id: vec3<u32>
) {
    var pixel_color = 0.5*vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    let r = get_ray(pixel_coords.x, pixel_coords.y);
    var transform_mat: mat4x4<f32>;
    var n: vec3<f32>;
    var sphere: SphereInstance;
    var color: vec3<f32>;
    var num_instances: i32 = bitcast<i32>(arrayLength(&sphere_instances.contents));
    for (var i: i32 = 0; i < num_instances; i=i+1) {
        sphere = sphere_instances.contents[i];
        transform_mat = sphere.transform;
        color = lambertians.contents[sphere.material_id];
        var sphere = Sphere( sphere.scale * unit_ball.radius, (transform_mat * vec4<f32>(unit_ball.position, 1.0)).xyz);
        var t = hit_sphere(sphere.position, sphere.radius, r);
        if (t > 0.0) {
            n = normalize(r.origin + t * r.direction - vec3<f32>(0.0, 0.0, -1.0));
            pixel_color = vec4<f32>(color, 1.0);
        }; 
    }

    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}
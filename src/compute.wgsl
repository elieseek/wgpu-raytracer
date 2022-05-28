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

fn get_ray(camera: Camera, u: f32, v: f32) -> Ray {
    var ray: Ray;
    ray.origin = camera.origin.xyz;
    ray.direction = (camera.lower_left_corner 
                    + camera.horizontal * u 
                    + camera.vertical * v
                    - camera.origin).xyz;
    return ray;
}

struct Sphere {
    origin: vec3<f32>;
    radius: f32;
};

fn hit_sphere(center: vec3<f32>, radius: f32, r: Ray) -> f32 {
    let oc: vec3<f32> = r.origin - center;
    let a: f32 = dot(r.direction, r.direction);
    let b: f32 = 2.0 * dot(oc, r.direction);
    let c: f32 = dot(oc, oc) - radius*radius;
    let discriminant: f32 = b*b - 4.*a*c;

    var result: f32 = -1.0;
    if (discriminant > 0.) {
        result = (-b - sqrt(discriminant)) / (2.0 * a);
    };

    return result;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var output_tex: texture_storage_2d<rgba8unorm, write>;

[[group(1), binding(0)]] var<uniform> camera: Camera;

[[stage(compute), workgroup_size(8, 4, 1)]]
fn cs_main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(local_invocation_id)]] local_id: vec3<u32>
) {
    var test_ball: Sphere;
    test_ball.origin = vec3<f32>(0.0, 0.0, -1.0);
    test_ball.radius = 0.5;
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));
    let r = get_ray(camera, pixel_coords.x, pixel_coords.y);

    var pixel_color = 0.5*vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let t = hit_sphere(test_ball.origin, test_ball.radius, r);
    var n: vec3<f32>;
    if (t > 0.0) {
        n = normalize(r.origin + t * r.direction - vec3<f32>(0.0, 0.0, -1.0));
        pixel_color = 0.5 * vec4<f32>(n.x + 1.0, n.y + 1.0, n.z + 1.0, 1.0);
    }; 



    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}
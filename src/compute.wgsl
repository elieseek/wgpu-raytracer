struct Params {
    width: u32;
    height: u32;
};

[[group(0), binding(0)]] var<uniform> params: Params;
[[group(0), binding(1)]] var output_tex: texture_storage_2d<rgba8unorm, write>;

[[stage(compute), workgroup_size(8, 4, 1)]]
fn cs_main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
    [[builtin(local_invocation_id)]] local_id: vec3<u32>
) {
    let pixel_coords: vec2<f32> = vec2<f32>(global_id.xy) / vec2<f32>(f32(params.width), f32(params.height));

    let pixel_color: vec4<f32> = vec4<f32>(pixel_coords.x, 1.0 - pixel_coords.y, 0.25, 1.0);

    textureStore(output_tex, vec2<i32>(global_id.xy), pixel_color);
}
struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

struct TonemapParams {
    key: f32,
    saturation: f32,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((in_vertex_index & 1u) ^ in_instance_index);
    let y = f32((in_vertex_index >> 1u) ^ in_instance_index);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.tex_coord = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var r_color: texture_2d<f32>;
@group(0) @binding(1) var r_sampler: sampler;
@group(0) @binding(2) var<uniform> tonemap_params: TonemapParams;

fn tonemap(col: vec3<f32>, key: f32, sat: f32) -> vec3<f32> {
    var c = col * key;
    c = c / (1.0 + c);
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(lum), c, sat);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(r_color, r_sampler, in.tex_coord);
    let avg = tex.rgb / max(tex.a, 1.0);
    let tm = tonemap(avg, tonemap_params.key, tonemap_params.saturation);
    return vec4<f32>(tm, 1.0);
}

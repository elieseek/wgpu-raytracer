#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TonemapUniform {
    pub key: f32,
    pub saturation: f32,
}

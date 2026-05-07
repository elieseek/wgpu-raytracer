#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub color_temp: f32,
    pub light_type: u32,
    _pad: [f32; 2],
}

impl GpuLight {
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, color_temp: f32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            color: [color[0], color[1], color[2], intensity],
            color_temp,
            light_type: 0,
            _pad: [0.0; 2],
        }
    }
}

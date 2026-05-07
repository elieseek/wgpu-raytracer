#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub color: [f32; 4],
    pub roughness: f32,
    pub ior: f32,
    pub material_type: u32,
    _pad: f32,
}

impl GpuMaterial {
    pub fn diffuse(color: [f32; 3]) -> Self {
        Self {
            color: [color[0], color[1], color[2], 0.0],
            roughness: 0.0,
            ior: 1.0,
            material_type: 0,
            _pad: 0.0,
        }
    }

    pub fn dielectric(ior: f32, roughness: f32) -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 0.0],
            roughness,
            ior,
            material_type: 1,
            _pad: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub color_temp: f32,
    pub light_type: u32,
    pub normal_x: f32,
    pub normal_z: f32,
}

impl GpuLight {
    #[allow(dead_code)]
    pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32, color_temp: f32) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            color: [color[0], color[1], color[2], intensity],
            color_temp,
            light_type: 0,
            normal_x: 0.0,
            normal_z: 0.0,
        }
    }

    pub fn square_area(
        center: [f32; 3],
        normal: [f32; 3],
        half_width: f32,
        color: [f32; 3],
        intensity: f32,
        color_temp: f32,
    ) -> Self {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        let (nx, ny, nz) = if len > 0.0 {
            (normal[0] / len, normal[1] / len, normal[2] / len)
        } else {
            (0.0, -1.0, 0.0)
        };
        // Ensure ny ≤ 0 (always faces downward). If ny > 0, flip the whole normal
        let (nx, nz) = if ny > 0.0 { (-nx, -nz) } else { (nx, nz) };
        Self {
            position: [center[0], center[1], center[2], half_width],
            color: [color[0], color[1], color[2], intensity],
            color_temp,
            light_type: 1,
            normal_x: nx,
            normal_z: nz,
        }
    }
}

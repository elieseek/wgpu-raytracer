use cgmath::Rotation3;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sphere {
    material_id: u32,
    scale: f32,
    _padding: [f32; 2],
    transform_matrix: [[f32; 4]; 4],
}

impl Sphere {
    pub fn new(
        material_id: u32,
        scale: f32,
        translation: cgmath::Vector3<f32>,
        rotation: cgmath::Deg<f32>,
    ) -> Self {
        let quaternion_rotation =
            cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_y(), rotation);
        let transform_matrix = cgmath::Matrix4::from_translation(translation)
            * cgmath::Matrix4::from(quaternion_rotation);

        Self {
            material_id,
            scale,
            _padding: [0.0, 0.0],
            transform_matrix: transform_matrix.into(),
        }
    }
}

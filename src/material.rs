#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Lambertian {
    color: [f32; 3],
    _padding: f32,
}

impl Lambertian {
    pub fn new<T: Into<[f32; 3]>>(color: T) -> Self {
        Self {
            color: color.into(),
            _padding: 0.0,
        }
    }
}

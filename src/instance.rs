use cgmath::Rotation3;
use tobj::{self, LoadOptions};

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

pub struct Mesh {
    pub positions: Vec<[f32; 4]>,
    pub normals: Vec<[f32; 4]>,
    pub indices: Vec<[u32; 4]>,
    pub normal_indices: Vec<[u32; 4]>,
}

impl Mesh {
    pub fn new() -> Mesh {
        Mesh {
            positions: vec![],
            normals: vec![],
            indices: vec![],
            normal_indices: vec![],
        }
    }

    pub async fn load_obj(&mut self, file_name: &str) {
        match tobj::load_obj(
            file_name,
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        ) {
            Ok((models, _)) => {
                for m in models {
                    let mesh = m.mesh;
                    println!(
                        "Loading model: {} ({} triangles)",
                        m.name,
                        mesh.indices.len()
                    );

                    let positions: Vec<[f32; 4]> = mesh
                        .positions
                        .chunks(3)
                        .map(|i| [i[0], i[1], i[2], 0.0])
                        .collect();
                    let normals: Vec<[f32; 4]> = mesh
                        .normals
                        .chunks(3)
                        .map(|i| [i[0], i[1], i[2], 0.0])
                        .collect();
                    let indices: Vec<[u32; 4]> = mesh
                        .indices
                        .chunks(3)
                        .map(|i| [i[0], i[1], i[2], 0])
                        .collect();
                    let normal_indices: Vec<[u32; 4]> = mesh
                        .indices
                        .chunks(3)
                        .map(|i| [i[0], i[1], i[2], 0])
                        .collect();

                    for p in positions {
                        self.positions.push(p)
                    }
                    for n in normals {
                        self.normals.push(n)
                    }
                    for i in indices {
                        self.indices.push(i)
                    }
                    for ni in normal_indices {
                        self.normal_indices.push(ni)
                    }
                }
            }
            Err(e) => {
                eprint!("Failed to load {:?} due to {:?}", file_name, e);
            }
        }
    }
}


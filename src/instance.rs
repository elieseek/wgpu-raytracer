use std::rc::Rc;

use cgmath::{Rotation3, Point3, Vector3, ElementWise};
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

    //Todo: handle panics gracefully
    fn get_triangle(self, index: usize) -> Triangle {
        let indices = self.indices[index];
        let p1 = *self.positions.get(indices[0] as usize).unwrap();
        let p2 = *self.positions.get(indices[1] as usize).unwrap();
        let p3 = *self.positions.get(indices[2] as usize).unwrap();
        
        Triangle {
            p1: Point3::new(p1[0], p1[1], p1[2]),
            p2: Point3::new(p2[0], p2[1], p2[2]),
            p3: Point3::new(p3[0], p3[1], p3[2]),
        }
    }
}

struct Triangle {
    p1: Point3<f32>,
    p2: Point3<f32>,
    p3: Point3<f32>,
}

impl Triangle {
    fn get_bounds(&self) -> AABB {
        let min_point = [0, 1, 2].map(|i| [self.p1[i], self.p2[i], self.p3[i]].into_iter().reduce(f32::min).unwrap());
        let max_point = [0, 1, 2].map(|i| [self.p1[i], self.p2[i], self.p3[i]].into_iter().reduce(f32::max).unwrap());

        AABB {
            min_point: Point3::new(min_point[0], min_point[1], min_point[2]),
            max_point: Point3::new(max_point[0], max_point[1], max_point[2]),
        }
    }
}

#[derive(Clone, Copy)]
struct AABB {
    min_point: Point3<f32>,
    max_point: Point3<f32>,
}

impl AABB {
    fn union(box1: AABB, box2: AABB) -> AABB {
        let min_point = [0,1,2].map(|i| [box1.min_point[i], box2.min_point[i]].into_iter().reduce(f32::min).unwrap());
        let max_point = [0,1,2].map(|i| [box1.max_point[i], box2.max_point[i]].into_iter().reduce(f32::max).unwrap());

        AABB {
            min_point: Point3::new(min_point[0], min_point[1], min_point[2]),
            max_point: Point3::new(max_point[0], max_point[1], max_point[2]),
        }
    }

    fn centroid(self) -> Point3<f32> {
        (0.5 * self.min_point).add_element_wise(0.5 * self.max_point)
    }

    fn surface_area(self) -> f32 {
        let d = self.max_point - self.min_point;
        2. * (d.x * d.y + d.x * d.z + d.y * d.z)
    }


}

struct BVH {
    max_prims_in_node: isize,
    primitives: Vec<Triangle>,
    linear_bvh_node: bool,
}

impl BVH {
    fn new(&self, primitives: Vec<Triangle>) -> BVH {
        // Initialise BVHBuildNode for primitive range
        todo!()
    }
    
    fn get_bounds_array(primitives: Vec<Triangle>) -> Vec<BVHPrimitive> {
        let mut bounds_array = vec![];
        for i in 0..primitives.len() {
            bounds_array.push(
                BVHPrimitive {
                    index: i,
                    aabb: primitives[i].get_bounds(),
                }
                 
            );
        };

        bounds_array
    }
}

struct BVHPrimitive {
    index: usize,
    aabb: AABB,
}

struct BVHBuildNode {
    aabb: AABB,
    left: Option<BVHBuildNodeRef>,
    right: Option<BVHBuildNodeRef>,
    split_axis: i64,
    first_prim_offset: isize,
    n_primitives: isize,   
}

type BVHBuildNodeRef = Rc<BVHBuildNode>;


impl BVHBuildNode {
    fn new(self, first: isize, n: isize, aabb: &AABB) -> BVHBuildNode {
        BVHBuildNode {
            aabb: *aabb,
            left: None,
            right: None,
            split_axis: 0,
            first_prim_offset: first,
            n_primitives: n,
        }
    }
    fn init_interior(&mut self, axis: i64, c_0: &BVHBuildNodeRef, c_1: &BVHBuildNodeRef) {
        self.split_axis = axis;
        self.left = Some(Rc::clone(&c_0));
        self.right = Some(Rc::clone(&c_1));
        self.aabb = AABB::union(c_0.aabb, c_1.aabb);
        self.n_primitives = 0;
    }
}
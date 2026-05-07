use bytemuck::Zeroable;
use cgmath::{Rotation3, Point3, ElementWise};
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
    pub indices: Vec<[u32; 4]>,
    pub material_id: u32,
}

impl Mesh {
    pub fn new() -> Mesh {
        Mesh {
            positions: vec![],
            indices: vec![],
            material_id: 0,
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
                    let indices: Vec<[u32; 4]> = mesh
                        .indices
                        .chunks(3)
                        .map(|i| [i[0], i[1], i[2], self.material_id])
                        .collect();

                    for p in positions {
                        self.positions.push(p)
                    }
                    for i in indices {
                        self.indices.push(i)
                    }
                }
            }
            Err(e) => {
                eprint!("Failed to load {:?} due to {:?}", file_name, e);
            }
        }
    }

    pub fn get_triangle(&self, index: usize) -> Option<Triangle> {
        let indices = self.indices.get(index)?;
        let p1 = self.positions.get(indices[0] as usize)?;
        let p2 = self.positions.get(indices[1] as usize)?;
        let p3 = self.positions.get(indices[2] as usize)?;

        Some(Triangle {
            p1: Point3::new(p1[0], p1[1], p1[2]),
            p2: Point3::new(p2[0], p2[1], p2[2]),
            p3: Point3::new(p3[0], p3[1], p3[2]),
        })
    }

    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }
}

#[derive(Clone, Copy)]
pub struct Triangle {
    pub p1: Point3<f32>,
    pub p2: Point3<f32>,
    pub p3: Point3<f32>,
}

impl Triangle {
    pub fn get_bounds(&self) -> AABB {
        let min_point = [0, 1, 2].map(|i| [self.p1[i], self.p2[i], self.p3[i]].into_iter().reduce(f32::min).unwrap());
        let max_point = [0, 1, 2].map(|i| [self.p1[i], self.p2[i], self.p3[i]].into_iter().reduce(f32::max).unwrap());

        AABB {
            min_point: Point3::new(min_point[0], min_point[1], min_point[2]),
            max_point: Point3::new(max_point[0], max_point[1], max_point[2]),
        }
    }
}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub struct AABB {
    pub min_point: Point3<f32>,
    pub max_point: Point3<f32>,
}

impl AABB {
    pub fn union(box1: AABB, box2: AABB) -> AABB {
        let min_point = [0,1,2].map(|i| [box1.min_point[i], box2.min_point[i]].into_iter().reduce(f32::min).unwrap());
        let max_point = [0,1,2].map(|i| [box1.max_point[i], box2.max_point[i]].into_iter().reduce(f32::max).unwrap());

        AABB {
            min_point: Point3::new(min_point[0], min_point[1], min_point[2]),
            max_point: Point3::new(max_point[0], max_point[1], max_point[2]),
        }
    }

    pub fn centroid(self) -> Point3<f32> {
        (0.5 * self.min_point).add_element_wise(0.5 * self.max_point)
    }

    pub fn longest_axis(&self) -> usize {
        let d = self.max_point - self.min_point;
        if d.x >= d.y && d.x >= d.z { 0 }
        else if d.y >= d.z { 1 }
        else { 2 }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBVHNode {
    pub bbox_min: [f32; 4],
    pub bbox_max: [f32; 4],
    pub left_child: u32,
    pub right_child: u32,
    pub first_triangle: u32,
    pub n_triangles: u32,
}

struct BVHPrimitive {
    index: usize,
    aabb: AABB,
}

#[allow(clippy::upper_case_acronyms)]
pub struct BVH {
    pub nodes: Vec<GpuBVHNode>,
    pub triangle_indices: Vec<u32>,
}

impl BVH {
    pub fn build(mesh: &Mesh, max_prims_in_node: usize) -> Self {
        let max_prims = max_prims_in_node.max(1);
        let n = mesh.num_triangles();

        if n == 0 {
            return BVH {
                nodes: vec![GpuBVHNode::zeroed()],
                triangle_indices: vec![0u32],
            };
        }

        let mut primitives: Vec<BVHPrimitive> = (0..n)
            .filter_map(|i| {
                let tri = mesh.get_triangle(i)?;
                Some(BVHPrimitive {
                    index: i,
                    aabb: tri.get_bounds(),
                })
            })
            .collect();

        if primitives.is_empty() {
            return BVH {
                nodes: vec![GpuBVHNode::zeroed()],
                triangle_indices: vec![0u32],
            };
        }

        let mut bvh = BVH {
            nodes: Vec::new(),
            triangle_indices: Vec::new(),
        };

        bvh.build_recursive(&mut primitives, max_prims);

        bvh
    }

    fn build_recursive(&mut self, primitives: &mut [BVHPrimitive], max_prims: usize) -> u32 {
        let node_idx = self.nodes.len() as u32;
        self.nodes.push(GpuBVHNode::zeroed());

        let bounds = Self::primitive_bounds(primitives);
        let n = primitives.len();

        if n <= max_prims {
            let first = self.triangle_indices.len() as u32;
            for prim in primitives.iter() {
                self.triangle_indices.push(prim.index as u32);
            }
            self.nodes[node_idx as usize] = GpuBVHNode {
                bbox_min: [bounds.min_point.x, bounds.min_point.y, bounds.min_point.z, 0.0],
                bbox_max: [bounds.max_point.x, bounds.max_point.y, bounds.max_point.z, 0.0],
                left_child: 0,
                right_child: 0,
                first_triangle: first,
                n_triangles: n as u32,
            };
            return node_idx;
        }

        let cb = Self::centroid_bounds(primitives);
        let dim = cb.longest_axis();

        primitives.sort_by(|a, b| {
            a.aabb.centroid()[dim]
                .partial_cmp(&b.aabb.centroid()[dim])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = n / 2;
        let (left, right) = primitives.split_at_mut(mid);
        let left_child = self.build_recursive(left, max_prims);
        let right_child = self.build_recursive(right, max_prims);

        self.nodes[node_idx as usize] = GpuBVHNode {
            bbox_min: [bounds.min_point.x, bounds.min_point.y, bounds.min_point.z, 0.0],
            bbox_max: [bounds.max_point.x, bounds.max_point.y, bounds.max_point.z, 0.0],
            left_child,
            right_child,
            first_triangle: 0,
            n_triangles: 0,
        };

        node_idx
    }

    fn primitive_bounds(primitives: &[BVHPrimitive]) -> AABB {
        let mut result = primitives[0].aabb;
        for p in &primitives[1..] {
            result = AABB::union(result, p.aabb);
        }
        result
    }

    fn centroid_bounds(primitives: &[BVHPrimitive]) -> AABB {
        let mut min_pt = [f32::MAX; 3];
        let mut max_pt = [f32::MIN; 3];
        for p in primitives {
            let c = p.aabb.centroid();
            min_pt[0] = min_pt[0].min(c.x);
            min_pt[1] = min_pt[1].min(c.y);
            min_pt[2] = min_pt[2].min(c.z);
            max_pt[0] = max_pt[0].max(c.x);
            max_pt[1] = max_pt[1].max(c.y);
            max_pt[2] = max_pt[2].max(c.z);
        }
        AABB {
            min_point: Point3::new(min_pt[0], min_pt[1], min_pt[2]),
            max_point: Point3::new(max_pt[0], max_pt[1], max_pt[2]),
        }
    }
}

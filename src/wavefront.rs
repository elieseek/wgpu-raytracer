use std::mem;

use wgpu::{util::DeviceExt, BufferUsages};

use crate::{camera::CameraUniform, Scene};

const CONFIG_SIZE: u64 =
    (mem::size_of::<u32>() + mem::size_of::<u32>() + mem::size_of::<u32>()) as u64;

const NUM_PIXELS_MAX: usize = 2560 * 1440;

pub struct ComputePass {
    pub config_buffer: wgpu::Buffer,
    pub config_data: ConfigData,
    pub wavefront_pipeline: wgpu::ComputePipeline,
    pub global_bind_group: wgpu::BindGroup,
    pub global_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub soa_buffer: wgpu::Buffer,
    pub soa_bind_group: wgpu::BindGroup,
    pub soa_bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePass {
    pub fn new(
        device: &wgpu::Device,
        size: &winit::dpi::PhysicalSize<u32>,
        output_view: &wgpu::TextureView,
        camera_uniform: &CameraUniform,
        scene: &Scene,
    ) -> Self {
        let seed = rand::random();

        // Global Binding
        let config_data = ConfigData {
            width: size.width,
            height: size.height,
            seed,
        };

        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Buffer"),
            size: CONFIG_SIZE,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("global_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadWrite,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });

        // Camera Binding
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::cast_slice(&[*camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Structure of Arrays Binding
        // 0.0_f32 has the same binary representation as 0u32
        let structure_of_arrays = vec![0.0_f32; (8 + 8 + 4 + 1) * NUM_PIXELS_MAX];
        let queue_len = [0_u32; 2];

        let soa_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("soa_buffer"),
            contents: bytemuck::cast_slice(&structure_of_arrays),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let queue_len_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("queue_len_buffer"),
            contents: bytemuck::cast_slice(&queue_len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let soa_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("soa_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let soa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soa_bind_group"),
            layout: &soa_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: soa_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: queue_len_buffer.as_entire_binding(),
                },
            ],
        });
        let wavefront_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("wavefront_module"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels/wavefront.wgsl").into()),
        });

        let wavefront_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("wavefront_pipeline_layout"),
                bind_group_layouts: &[
                    &global_bind_group_layout,
                    &camera_bind_group_layout,
                    &scene.sphere_bind_group_layout,
                    &scene.material_bind_group_layout,
                    &soa_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let wavefront_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wavefront_module"),
            layout: Some(&wavefront_pipeline_layout),
            module: &wavefront_module,
            entry_point: "cs_main",
        });

        Self {
            config_buffer,
            config_data,
            wavefront_pipeline,
            global_bind_group,
            global_bind_group_layout,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            soa_buffer,
            soa_bind_group,
            soa_bind_group_layout,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        size: &winit::dpi::PhysicalSize<u32>,
        scene: &Scene,
    ) -> Result<(), wgpu::SurfaceError> {
        self.config_data.seed = rand::random();
        let config_host = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&self.config_data),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&config_host, 0, &self.config_buffer, 0, CONFIG_SIZE);

        let mut wavefront = encoder.begin_compute_pass(&Default::default());
        wavefront.set_pipeline(&self.wavefront_pipeline);
        wavefront.set_bind_group(0, &self.global_bind_group, &[]);
        wavefront.set_bind_group(1, &self.camera_bind_group, &[]);
        wavefront.set_bind_group(2, &scene.sphere_bind_group, &[]);
        wavefront.set_bind_group(3, &scene.material_bind_group, &[]);
        wavefront.set_bind_group(4, &self.soa_bind_group, &[]);
        wavefront.dispatch(size.width / 8, size.height / 4, 1);

        Ok(())
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        new_size: &winit::dpi::PhysicalSize<u32>,
        output_view: &wgpu::TextureView,
    ) {
        self.config_data = ConfigData {
            width: new_size.width,
            height: new_size.height,
            seed: rand::random(),
        };
        self.global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });
    }

    pub fn update(&mut self, queue: &wgpu::Queue, camera_uniform: CameraUniform) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConfigData {
    width: u32,
    height: u32,
    seed: u32,
}

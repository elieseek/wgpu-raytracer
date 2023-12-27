use std::mem;

use wgpu::{util::DeviceExt, BufferUsages};

use crate::{camera::CameraUniform, Scene};

// Config data: width, height, depth, seed
const CONFIG_SIZE: u64 =
    (mem::size_of::<u32>() + mem::size_of::<u32>() + mem::size_of::<u32>() + mem::size_of::<u32>())
        as u64;

pub const DEFAULT_DEPTH: u32 = 30;

pub struct ComputePass {
    pub config_buffer: wgpu::Buffer,
    pub config_data: ConfigData,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub preview_next_frame: bool,
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
        let config_data = ConfigData {
            width: size.width,
            height: size.height,
            depth: DEFAULT_DEPTH,
            seed,
        };
        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Buffer"),
            size: CONFIG_SIZE,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels/mega_kernel.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
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

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &bind_group_layout,
                    &camera_bind_group_layout,
                    &scene.sphere_bind_group_layout,
                    &scene.mesh_bind_group_layout,
                    &scene.material_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "cs_main",
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
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

        Self {
            config_buffer,
            config_data,
            pipeline,
            bind_group,
            bind_group_layout,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            preview_next_frame: false,
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
        self.config_data.depth = DEFAULT_DEPTH;
        if self.preview_next_frame {
            self.config_data.depth = 1;
            self.preview_next_frame = false;
        }

        let config_host = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&self.config_data),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&config_host, 0, &self.config_buffer, 0, CONFIG_SIZE);

        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.camera_bind_group, &[]);
        compute_pass.set_bind_group(2, &scene.sphere_bind_group, &[]);
        compute_pass.set_bind_group(3, &scene.mesh_bind_group, &[]);
        compute_pass.set_bind_group(4, &scene.material_bind_group, &[]);

        compute_pass.dispatch_workgroups(size.width / 8, size.height / 4, 1);
        Ok(())
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        new_size: &winit::dpi::PhysicalSize<u32>,
        output_view: &wgpu::TextureView,
    ) {
        self.preview_next_frame = true;
        self.config_data = ConfigData {
            width: new_size.width,
            height: new_size.height,
            depth: DEFAULT_DEPTH,
            seed: rand::random(),
        };
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
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
        self.preview_next_frame = true;
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
    depth: u32,
    seed: u32,
}

use wgpu::util::DeviceExt;

pub struct RenderPass {
    pub render_pipeline: wgpu::RenderPipeline,
    pub copy_bind_group: wgpu::BindGroup,
    pub copy_bind_group_layout: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
    sample: f32,
    sample_buffer: wgpu::Buffer,
    sample_bind_group: wgpu::BindGroup,
}

impl RenderPass {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        source_view: &wgpu::TextureView,
    ) -> Self {
        let copy_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("copy.wgsl").into()),
        });

        let sample: f32 = 0.0;

        let sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sample_buffer"),
            contents: bytemuck::cast_slice(&[sample]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sample_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sample_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let sample_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sample_bind_group"),
            layout: &sample_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sample_buffer.as_entire_binding(),
            }],
        });

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("copy_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&copy_bind_group_layout, &sample_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &copy_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &copy_shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_bind_group"),
            layout: &copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            render_pipeline,
            copy_bind_group,
            copy_bind_group_layout,
            sampler,
            sample,
            sample_buffer,
            sample_bind_group,
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.copy_bind_group, &[]);
        render_pass.set_bind_group(1, &self.sample_bind_group, &[]);
        render_pass.draw(0..3, 0..2);
        self.sample += 1.0;
        Ok(())
    }

    pub fn resize(&mut self, device: &wgpu::Device, source_view: &wgpu::TextureView) {
        self.copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_bind_group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    pub fn reset_sample(&mut self) {
        self.sample = 0.0;
    }

    pub fn update(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.sample_buffer, 0, bytemuck::cast_slice(&[self.sample]));
    }
}

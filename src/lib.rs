use std::{sync::Arc, time::Instant};
use wgpu::{util::DeviceExt, Extent3d};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use blit::RenderPass;
use mega_kernel::ComputePass;
use instance::{Mesh, BVH};
use spectrum::generate_cie_to_rgb_table;
use light::GpuLight;

mod blit;
mod camera;
mod instance;
mod light;
mod material;
mod mega_kernel;
mod spectrum;
// mod wavefront;

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct App {
    state: Option<State>,
    last_frame: Option<Instant>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("Raytracer")
                    .with_inner_size(winit::dpi::PhysicalSize::new(1600_u32, 900_u32)),
            )
            .unwrap();
        let window = Arc::new(window);
        window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        window.set_cursor_visible(false);

        self.state = Some(pollster::block_on(State::new(window)));
        self.last_frame = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if window_id != state.window().id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let duration = self
                    .last_frame
                    .replace(now)
                    .map(|last_frame| now.duration_since(last_frame).as_micros())
                    .unwrap_or_default();
                state.update(duration);
                state.render();
            }
            event => state.input_window_event(&event),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.input_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window().request_redraw();
        }
    }
}

struct State {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Arc<Window>,
    window_focused: bool,
    compute_texture: wgpu::Texture,
    compute_view: wgpu::TextureView,
    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_controller: camera::CameraController,
    scene: Scene,
    compute_pass: ComputePass,
    render_pass: RenderPass,
    clear_flag: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::CLEAR_TEXTURE,
                    required_limits: wgpu::Limits {
                        max_bind_groups: 7,
                        max_storage_buffer_binding_size: 512 * 1024 * 1024,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        // let format = surface.get_preferred_format(&adapter).unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = if surface_caps
            .present_modes
            .contains(&wgpu::PresentMode::Immediate)
        {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let compute_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compute_texture"),
            size: Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let compute_view = compute_texture.create_view(&Default::default());

        let camera = camera::Camera::new(
            (0.0, 0.0, 0.0).into(),
            (0.0, 0.0, 1.0).into(),
            cgmath::Vector3::unit_y(),
            75.0,
            16.0 / 9.0,
        );

        let camera_uniform = camera.get_uniform();
        let camera_controller = camera::CameraController::new(5e-6);

        let mat0 = material::GpuMaterial::diffuse([0.8, 0.8, 0.8]);
        let mat1 = material::GpuMaterial::diffuse([0.2, 0.85, 0.2]);
        let mat2 = material::GpuMaterial::dielectric(1.5, 0.0);

        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material_buffer"),
            contents: bytemuck::cast_slice(&[mat0, mat1, mat2]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sphere1: instance::Sphere =
            instance::Sphere::new(0, 1.0, cgmath::vec3(0.0, 1.0, -1.0), cgmath::Deg(0.0));
        let sphere2 =
            instance::Sphere::new(1, 1000.0, cgmath::vec3(0.0, -1000.0, 0.0), cgmath::Deg(0.0));
        let sphere3 = instance::Sphere::new(2, 1.0, cgmath::vec3(0.0, 1.0, 1.0), cgmath::Deg(0.0));

        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere_buffer"),
            contents: bytemuck::cast_slice(&[sphere1, sphere2, sphere3]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sphere_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sphere_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let sphere_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sphere_bind_group"),
            layout: &sphere_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sphere_buffer.as_entire_binding(),
            }],
        });

        let mut obj_model = Mesh::new();
        obj_model.material_id = 2; // grey diffuse
        obj_model.load_obj("res/monkey.obj").await;

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("position_buffer"),
            contents: bytemuck::cast_slice(&obj_model.positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&obj_model.indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mesh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let mesh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_bind_group"),
            layout: &mesh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material_bind_group"),
            layout: &material_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });

        let bvh = BVH::build(&obj_model, 2);

        let bvh_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_node_buffer"),
            contents: bytemuck::cast_slice(&bvh.nodes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bvh_triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh_triangle_buffer"),
            contents: bytemuck::cast_slice(&bvh.triangle_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bvh_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bvh_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bvh_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bvh_bind_group"),
            layout: &bvh_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bvh_node_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bvh_triangle_buffer.as_entire_binding(),
                },
            ],
        });

        let light1 = GpuLight::point([0.0, 20.0, -6.0], [1.0, 1.0, 1.0], 10.0, 5500.0);
        let light2 = GpuLight::point([0.0, 30.0, 5.0], [1.0, 0.8, 0.5], 10.0, 3000.0);

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light_buffer"),
            contents: bytemuck::cast_slice(&[light1, light2]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let cie_data = generate_cie_to_rgb_table();
        let cie_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cie_buffer"),
            contents: bytemuck::cast_slice(&cie_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("light_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bind_group"),
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cie_buffer.as_entire_binding(),
                },
            ],
        });

        let scene = Scene {
            sphere_bind_group_layout,
            sphere_bind_group,
            mesh_bind_group_layout,
            mesh_bind_group,
            material_bind_group_layout,
            material_bind_group,
            bvh_bind_group_layout,
            bvh_bind_group,
            light_bind_group_layout,
            light_bind_group,
        };

        let compute_pass = ComputePass::new(&device, &size, &compute_view, &camera_uniform, &scene);
        let render_pass = RenderPass::new(&device, surface_format, &compute_view);
        let clear_flag = false;

        Self {
            instance,
            surface,
            device,
            queue,
            config,
            size,
            window,
            window_focused: true,
            compute_texture,
            compute_view,
            camera,
            camera_uniform,
            camera_controller,
            scene,
            compute_pass,
            render_pass,
            clear_flag,
        }
    }

    fn window(&self) -> &Window {
        &self.window
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.recreate_surface();
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("Surface validation error while acquiring the next frame");
                return;
            }
        };

        let mut encoder = self.device.create_command_encoder(&Default::default());
        if self.clear_flag {
            encoder.clear_texture(
                &self.compute_texture,
                &wgpu::ImageSubresourceRange {
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                },
            );
            self.clear_flag = false;
        }

        self.compute_pass
            .render(&self.device, &mut encoder, &self.size, &self.scene);

        self.render_pass.render(
            &mut encoder,
            &frame.texture.create_view(&Default::default()),
        );

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn recreate_surface(&mut self) {
        self.surface = self.instance.create_surface(self.window.clone()).unwrap();
        self.surface.configure(&self.device, &self.config);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.height > 0 && new_size.width > 0 {
            self.size = new_size;
            self.config.height = new_size.height;
            self.config.width = new_size.width;
            self.surface.configure(&self.device, &self.config);

            // We need to recreate the comput and output texture on resize
            self.compute_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("compute_texture"),
                size: Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            self.compute_view = self.compute_texture.create_view(&Default::default());

            self.compute_pass
                .resize(&self.device, &new_size, &self.compute_view);
            self.render_pass.resize(&self.device, &self.compute_view);
        }
    }

    fn input_device_event(&mut self, event: &DeviceEvent) -> bool {
        if self.window_focused {
            self.camera_controller.process_events(event);
        }
        false
    }

    fn input_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                self.window_focused = false;
                self.window
                    .set_cursor_grab(CursorGrabMode::None)
                    .unwrap();
                self.window.set_cursor_visible(true);
            }
            WindowEvent::MouseInput {
                state: winit::event::ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.window_focused = true;
                self.window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .unwrap();
                self.window.set_cursor_visible(false);
            }
            WindowEvent::Resized(physical_size) => {
                self.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.resize(self.window.inner_size());
            }
            _ => {}
        }
    }

    fn update(&mut self, duration: u128) {
        let was_updated = self
            .camera_controller
            .update_camera(&mut self.camera, duration);
        if was_updated {
            self.clear_flag = true;
            self.camera_uniform = self.camera.get_uniform();
            self.compute_pass.update(&self.queue, self.camera_uniform);
        }
    }
}

pub struct Scene {
    sphere_bind_group_layout: wgpu::BindGroupLayout,
    sphere_bind_group: wgpu::BindGroup,
    mesh_bind_group_layout: wgpu::BindGroupLayout,
    mesh_bind_group: wgpu::BindGroup,
    material_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_group: wgpu::BindGroup,
    bvh_bind_group_layout: wgpu::BindGroupLayout,
    bvh_bind_group: wgpu::BindGroup,
    pub light_bind_group_layout: wgpu::BindGroupLayout,
    pub light_bind_group: wgpu::BindGroup,
}

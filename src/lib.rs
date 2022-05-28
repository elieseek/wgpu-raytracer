use wgpu::Extent3d;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use compute_pass::ComputePass;
use render_pass::RenderPass;

mod camera;
mod compute_pass;
mod render_pass;

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    window.set_title("Raytracer");
    window.set_inner_size(winit::dpi::PhysicalSize::new(1600_u32, 900_u32));
    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    let mut state = State::new(&window).await;

    let mut window_focused = true;

    event_loop.run(move |event, _, control_flow| match event {
        Event::DeviceEvent {
            device_id: _,
            event,
        } if window_focused => {
            state.input(&event);
        }
        Event::WindowEvent {
            window_id,
            ref event,
        } if window_id == window.id() => {
            match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => {
                    window_focused = false;
                    window.set_cursor_grab(false).unwrap();
                    window.set_cursor_visible(true);
                }
                WindowEvent::MouseInput {
                    state: winit::event::ElementState::Pressed,
                    button: winit::event::MouseButton::Left,
                    ..
                } => {
                    window_focused = true;
                    window.set_cursor_grab(true).unwrap();
                    window.set_cursor_visible(false);
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    // new_inner_size is &&mut so we have to dereference it twice
                    state.resize(**new_inner_size);
                }
                _ => {}
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // Redraw requested will only trigger once, unless we manually
            // request it.
            window.request_redraw();
        }
        _ => {}
    });
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    camera: camera::Camera,
    camera_uniform: camera::CameraUniform,
    camera_controller: camera::CameraController,
    compute_pass: ComputePass,
    render_pass: RenderPass,
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(&window) };
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
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();
        let format = surface.get_preferred_format(&adapter).unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("compute_output_texture"),
            size: Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let camera = camera::Camera::new(
            (0.0, 0.0, 0.0).into(),
            (0.0, 0.0, -1.0).into(),
            cgmath::Vector3::unit_y(),
            70.0,
            16.0 / 9.0,
        );

        let camera_uniform = camera.get_uniform();
        let camera_controller = camera::CameraController::new(0.05);

        let output_view = output.create_view(&Default::default());

        let compute_pass = ComputePass::new(&device, &size, &output_view, &camera_uniform);
        let render_pass = RenderPass::new(&device, format, &output_view);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            output,
            output_view,
            camera,
            camera_uniform,
            camera_controller,
            compute_pass,
            render_pass,
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;

        let mut encoder = self.device.create_command_encoder(&Default::default());

        self.compute_pass
            .render(&self.device, &mut encoder, &self.size)
            .unwrap();

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_pass.render(&mut encoder, &view).unwrap();

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.height > 0 && new_size.width > 0 {
            self.size = new_size;
            self.config.height = new_size.height;
            self.config.width = new_size.width;
            self.surface.configure(&self.device, &self.config);

            // We need to recreate the output texture on resize
            self.output = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("compute_output_texture"),
                size: Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            self.output_view = self.output.create_view(&Default::default());

            self.compute_pass
                .resize(&self.device, &new_size, &self.output_view);
            self.render_pass.resize(&self.device, &self.output_view);
        }
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform = self.camera.get_uniform();
        self.compute_pass.update(&self.queue, self.camera_uniform);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Dimensions {
    width: u32,
    height: u32,
}

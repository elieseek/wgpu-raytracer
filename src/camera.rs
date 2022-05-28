use std::f32::consts::PI;

use cgmath::prelude::*;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

pub struct Camera {
    pub origin: cgmath::Point3<f32>,
    pub horizontal: cgmath::Vector3<f32>,
    pub vertical: cgmath::Vector3<f32>,
    pub lower_left_corner: cgmath::Point3<f32>,
}

impl Camera {
    pub fn new(
        look_from: cgmath::Vector3<f32>,
        look_at: cgmath::Vector3<f32>,
        v_up: cgmath::Vector3<f32>,
        vfov: f32,
        aspect_ratio: f32,
    ) -> Self {
        let theta = vfov * PI / 180.;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = v_up.cross(w).normalize();
        let v = w.cross(u);

        let origin = look_from;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left_corner = origin - 0.5 * horizontal - 0.5 * vertical - w;
        Camera {
            origin: cgmath::point3(origin.x, origin.y, origin.z),
            horizontal: cgmath::vec3(horizontal.x, horizontal.y, horizontal.z),
            vertical: cgmath::vec3(vertical.x, vertical.y, vertical.z),
            lower_left_corner: cgmath::point3(
                lower_left_corner.x,
                lower_left_corner.y,
                lower_left_corner.z,
            ),
        }
    }

    pub fn get_uniform(&self) -> CameraUniform {
        CameraUniform {
            origin: [self.origin.x, self.origin.y, self.origin.z, 0.0],
            horizontal: [self.horizontal.x, self.horizontal.y, self.horizontal.z, 0.0],
            vertical: [self.vertical.x, self.vertical.y, self.vertical.z, 0.0],
            lower_left_corner: [
                self.lower_left_corner.x,
                self.lower_left_corner.y,
                self.lower_left_corner.z,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub origin: [f32; 4],
    pub horizontal: [f32; 4],
    pub vertical: [f32; 4],
    pub lower_left_corner: [f32; 4],
}

pub struct CameraController {
    move_speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_mouse_dragged: bool,
    click_position: winit::dpi::PhysicalPosition<f32>,
    mouse_position: winit::dpi::PhysicalPosition<f32>,
}

impl CameraController {
    pub fn new(move_speed: f32) -> Self {
        Self {
            move_speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_mouse_dragged: false,
            click_position: winit::dpi::PhysicalPosition::default(),
            mouse_position: winit::dpi::PhysicalPosition::default(),
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_position = (*position).cast();
                true
            }

            WindowEvent::MouseInput {
                button: winit::event::MouseButton::Left,
                state,
                ..
            } => match state {
                winit::event::ElementState::Pressed => {
                    self.click_position = self.mouse_position;
                    self.is_mouse_dragged = true;
                    true
                }
                winit::event::ElementState::Released => {
                    self.is_mouse_dragged = false;
                    true
                }
            },
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.vertical.cross(camera.horizontal);
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        let right_norm = camera.horizontal.normalize();

        // Project forward direction onto xz plane
        if self.is_forward_pressed && forward_mag > self.move_speed {
            camera.origin += forward_norm * self.move_speed;
            camera.lower_left_corner += forward_norm * self.move_speed;
        }
        if self.is_backward_pressed {
            camera.origin -= forward_norm * self.move_speed;
            camera.lower_left_corner -= forward_norm * self.move_speed;
        }
        if self.is_right_pressed {
            camera.origin += right_norm * self.move_speed;
            camera.lower_left_corner += right_norm * self.move_speed;
        }
        if self.is_left_pressed {
            camera.origin -= right_norm * self.move_speed;
            camera.lower_left_corner -= right_norm * self.move_speed;
        }

        if self.is_mouse_dragged {
            let mouse_move = cgmath::vec2(
                self.mouse_position.x - self.click_position.x,
                self.click_position.y - self.mouse_position.y,
            );

            let mouse_move_mag = mouse_move.magnitude();
            if mouse_move_mag > 0.001 {
                let mouse_move_norm = mouse_move.normalize();

                // Our mouse moves on a plane with coordinates x: camera.up, y: right_norm
                let rotation_direction =
                    camera.vertical * mouse_move_norm.y + right_norm * mouse_move_norm.x;
                let rotation_axis = forward_norm.cross(rotation_direction).normalize();
                let rotation = cgmath::Quaternion::from_axis_angle(
                    rotation_axis,
                    cgmath::Rad(0.0001 * mouse_move_mag * std::f32::consts::FRAC_2_PI),
                );
                camera.horizontal = rotation.rotate_vector(camera.horizontal);
                camera.vertical = rotation.rotate_vector(camera.vertical);

                // We need to rotate the lower left corner about the
                // origin, so apply translations.
                camera.lower_left_corner = rotation
                    .rotate_point(camera.lower_left_corner - camera.origin.to_vec())
                    + camera.origin.to_vec();
            }
        }
    }
}

use std::f32::consts::PI;

use cgmath::prelude::*;
use winit::event::{DeviceEvent, ElementState, VirtualKeyCode};

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
    default_speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_mouse_dragged: bool,
    is_speed_boost: bool,
    mouse_delta: cgmath::Vector2<f32>,
}

impl CameraController {
    pub fn new(default_speed: f32) -> Self {
        Self {
            default_speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_mouse_dragged: false,
            is_up_pressed: false,
            is_down_pressed: false,
            is_speed_boost: false,
            mouse_delta: cgmath::vec2(0.0, 0.0),
        }
    }

    pub fn process_events(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Key(keyboard_input) => {
                let is_pressed = keyboard_input.state == ElementState::Pressed;
                match keyboard_input.virtual_keycode.unwrap() {
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
                    VirtualKeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LControl => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift => {
                        self.is_speed_boost = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_delta = cgmath::vec2(delta.0 as f32, delta.1 as f32);
                self.is_mouse_dragged = true;
                true
            }
            _ => {
                self.is_mouse_dragged = false;
                false
            }
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.vertical.cross(camera.horizontal);
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        let right_norm = camera.horizontal.normalize();

        let move_speed = if self.is_speed_boost {
            2. * self.default_speed
        } else {
            self.default_speed
        };

        // Project forward direction onto xz plane
        if self.is_forward_pressed && forward_mag > move_speed {
            camera.origin += forward_norm * move_speed;
            camera.lower_left_corner += forward_norm * move_speed;
        }
        if self.is_backward_pressed {
            camera.origin -= forward_norm * move_speed;
            camera.lower_left_corner -= forward_norm * move_speed;
        }
        if self.is_right_pressed {
            camera.origin += right_norm * move_speed;
            camera.lower_left_corner += right_norm * move_speed;
        }
        if self.is_left_pressed {
            camera.origin -= right_norm * move_speed;
            camera.lower_left_corner -= right_norm * move_speed;
        }
        if self.is_up_pressed {
            camera.origin -= cgmath::Vector3::unit_y() * move_speed;
            camera.lower_left_corner -= cgmath::Vector3::unit_y() * move_speed;
        }
        if self.is_down_pressed {
            camera.origin += cgmath::Vector3::unit_y() * move_speed;
            camera.lower_left_corner += cgmath::Vector3::unit_y() * move_speed;
        }

        if self.is_mouse_dragged {
            let mouse_move_mag = self.mouse_delta.magnitude();
            if mouse_move_mag > 0.0001 {
                // Our vertical rotation depends on camera.horizontal, so we apply horizontal
                // rotations first.
                let horizontal_rotation = cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_y(),
                    cgmath::Rad(-0.01 * self.mouse_delta.x * std::f32::consts::FRAC_2_PI),
                );
                camera.horizontal = horizontal_rotation.rotate_vector(camera.horizontal);
                camera.vertical = horizontal_rotation.rotate_vector(camera.vertical);
                // We need to rotate the lower left corner about the
                // origin, so apply translations.
                camera.lower_left_corner = horizontal_rotation
                    .rotate_point(camera.lower_left_corner - camera.origin.to_vec())
                    + camera.origin.to_vec();

                let vertical_rotation = cgmath::Quaternion::from_axis_angle(
                    camera.horizontal.normalize(),
                    cgmath::Rad(0.01 * self.mouse_delta.y * std::f32::consts::FRAC_2_PI),
                );
                camera.vertical = vertical_rotation.rotate_vector(camera.vertical);

                camera.lower_left_corner = vertical_rotation
                    .rotate_point(camera.lower_left_corner - camera.origin.to_vec())
                    + camera.origin.to_vec();

                self.mouse_delta = cgmath::Vector2::zero();
                self.is_mouse_dragged = false;
            }
        }
    }
}

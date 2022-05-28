use wgpu_raytracer::run;

fn main() {
    pollster::block_on(run());
}

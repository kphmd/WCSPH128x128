import taichi as ti
import numpy as np
from particle_system import ParticleSystem
from wcsph import WCSPHSolver

ti.init(arch=ti.cpu)

# Use GPU for higher peformance if available
# ti.init(arch=ti.gpu, device_memory_GB=1, packed=True)


if __name__ == "__main__":
    # ps = ParticleSystem((512, 512))
    ps = ParticleSystem((128, 128))

    ps.add_cube(lower_corner=[6, 2],
                cube_size=[3.0, 6.0],
                # velocity=[-5.0, -10.0],
                velocity=[0.0, 0.0],
                density=1000.0,
                color=0x956333,
                material=1)

    ps.add_cube(lower_corner=[3, 1],
                cube_size=[3.0, 5.0],
                # velocity=[0.0, -20.0],
                velocity=[0.0, 0.0],
                density=1000.0,
                color=0x956333,
                material=1)

    wcsph_solver = WCSPHSolver(ps)
    gui = ti.GUI('WCSPH',ps.res,background_color=0xFFFFFF)
    gui.fps_limit = 20
    while gui.running:
        for i in range(5):
            wcsph_solver.step()
        particle_info = ps.dump()
        gui.circles(particle_info['position'] * ps.screen_to_world_ratio / ps.res[0],
                    radius=ps.particle_radius / 1.5 * ps.screen_to_world_ratio,
                    color=0x0000ff)
        gui.show()

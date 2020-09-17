import taichi as ti
import random
ti.init(arch=ti.gpu)

dim = 2
n_particles = 15000
n_grid = 1024
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-3
use_apic = False

g = 9.8

FLUID = 0
AIR = 1
SOLID = 2

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
grid_type = ti.field(dtype=ti.i32, shape=(n_grid,n_grid))


@ti.kernel
def P2G():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * v[p]
                grid_m[base + offset] += weight

@ti.kernel
def normalize():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]

@ti.kernel
def G2P():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                new_v += weight * grid_v[base + ti.Vector([i, j])]

        x[p] = x[p] + v[p] * dt
        v[p] = new_v

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.6 + 0.2]
        v[i] = [0,0]
    for i in range(grid_type):
        grid_type

@ti.kernel
def apply_gravity(dt: ti.f32):
    for i, j in grid_v:
        grid_v[i, j] += [0, -g * dt]

init()
gui = ti.GUI("PIC v.s. APIC", (512, 512))
for frame in range(200000):
    #for s in range(5):
    grid_v.fill([0, 0])
    grid_m.fill(0)
    apply_gravity(dt)
    P2G()
    normalize()
    G2P()
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
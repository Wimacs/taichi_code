import taichi as ti
import random
ti.init(arch=ti.gpu)

dim = 2
n_particles = 30000
n_grid = 512
dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 1.0e-2
inv_dt = 1/dt
rho = 100

g = 10

FLUID = 0
AIR = 1
SOLID = 2

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_v_new = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
grid_type = ti.field(dtype=ti.i32, shape=(n_grid,n_grid))

velocity_divs = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
velocity_divs_new = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
pressure = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
pressure_new = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

@ti.func
def clamp_pos(pos):
    return ti.Vector([max(min(0.95, pos[0]), 0.05), max(min(0.95, pos[1]), 0.05)])

@ti.kernel
def swap(vf: ti.template(),vf_new: ti.template()):
    for i, j in vf:
        vf[i, j] = vf_new[i, j]

@ti.kernel
def divergence():
    ti.cache_read_only(grid_v)
    for i, j in grid_v:
        vl = grid_v[i - 1, j][0]
        vr = grid_v[i + 1, j][0]
        vb = grid_v[i, j - 1][1]
        vt = grid_v[i, j + 1][1]
        vc = grid_v[i, j]
        if i <= 10:
            vl = 0
        if i >= n_grid - 10:
            vr = 0
        if j <= 10:
            vb = 0
        if j >= n_grid - 10:
            vt = 0
        velocity_divs[i, j] = (vr - vl + vt - vb) * 0.5 
        #print(velocity_divs[i, j])

@ti.kernel
def pressure_projection():
    ti.cache_read_only(pressure)
    for i, j in pressure:
        pl = pressure[i - 1, j]
        pr = pressure[i + 1, j]
        pb = pressure[i, j - 1]
        pt = pressure[i, j + 1]
        div = velocity_divs[i, j]
        # if i <= 10:
        #     pl = 0
        # if i >= n_grid - 10:
        #     pr = 0
        # if j <= 10:
        #     pb = 0
        # if j >= n_grid - 10:
        #     pt = 0
        pressure[i, j] = (pl + pr + pb + pt - div) * 0.25


        
@ti.kernel
def enforce_boundary():
    for i, j in grid_v:
        if i < 10 and grid_v[i, j][0] < 0:          grid_v[i, j][0] = 0 # Boundary conditions
        if i > n_grid - 10 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
        if j < 10 and grid_v[i, j][1] < 0:          grid_v[i, j][1] = 0
        if j > n_grid - 10 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

@ti.kernel
def subtract():
    ti.cache_read_only(pressure)
    scale = 0.5 
    for i, j in grid_v:
        pl = pressure[i - 1, j]
        pr = pressure[i + 1, j]
        pb = pressure[i, j - 1]
        pt = pressure[i, j + 1]
        grid_v[i, j] -= scale * ti.Vector([pr - pl, pt - pb])
        #if (0.5 * inv_dx * ti.Vector([pr - pl, pt - pb]))[0] != 0 or (0.5 * inv_dx * ti.Vector([pr - pl, pt - pb]))[1] !=0 :
            #print(inv_dx * ti.Vector([pr - pl, pt - pb]))

@ti.kernel
def P2G():
    # #grid_type[None] = AIR
    # for i in (range(n_grid)):
    #     for j in (range(n_grid)):
    #         if i == 0 or i == n_grid - 1 or j == 0 or j == n_grid - 1:
    #             grid_type = SOLID
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        affine = C[p]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                dpos = (offset.cast(float) - fx)* dx
                grid_v[base + offset] += weight * (v[p] + affine @ dpos)
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
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v[p] = new_v
        x[p] = x[p] + v[p] * dt
        #print(new_v)
        C[p] = new_C

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [(ti.random() * 0.6 + 0.2), (ti.random() * 0.6 + 0.01)]
        v[i] = [0,0]
    # grid_type[None] = AIR
    # for i in (range(n_grid)):
    #     for j in (range(n_grid)):
    #         if i == 0 or i == n_grid - 1 or j == 0 or j == n_grid - 1:
    #             grid_type[i, j] = SOLID
    #         else:    
    #             if 0.2< i < 0.8 and 0.2< j < 0.8:
    #                 grid_type[i, j] = FLUID
    #             else:
    #                 grid_type[i, j] = AIR
            

@ti.kernel
def apply_gravity(dt: ti.f32):
    for i, j in grid_v:
        if 10 < i < n_grid-10 and 10 < j < n_grid-10 and grid_m[i, j] > 0:
            grid_v[i, j] += [0, -g * dt]

init()
gui = ti.GUI("PIC", (512, 512))
for frame in range(200000):
    grid_v.fill([0, 0])
    grid_m.fill(0)    
    velocity_divs.fill(0)
    #pressure.fill(0)
    #pressure_new.fill(0)
    
    P2G()
    
    normalize()
    apply_gravity(dt)
    enforce_boundary()
    divergence()
    
    for iter in range(600):
        pressure_projection()
        #swap(pressure,pressure_new)
        #pressure_new.fill(0)
        #print(pressure)
    
    subtract()
    
    
    
    enforce_boundary()
    G2P()
    #gui.set_image(grid_v.to_numpy() * 0.1 + 0.5)
    gui.set_image(pressure.to_numpy() * 0.1  + 0.5)
    #gui.set_image(velocity_divs.to_numpy() * 0.6 + 0.5)
    #gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.show()
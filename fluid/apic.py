import taichi as ti
from MGPCGSolver import MGPCGSolver
import numpy as np
#from utils import ColorMap, vec2, vec3, clamp
import utils
import random
import time

ti.init(arch=ti.gpu)
grid_res = 256
m = grid_res
n = grid_res
w = 10
h = 10 * n / m
dt = 0.01
grid_dx = w / m
inv_dx = 1 / grid_dx
grid_dy = h / n
inv_dy = 1 / grid_dy
g = -9.8
screen_res = (800, 800 * m//n)
rho = 1000
gui = ti.GUI("watersim2D", screen_res)


# __________________init__________________________
#grid
grid_type = ti.field(dtype=ti.i32, shape=(m,n))
grid_u = ti.field(dtype=ti.i32, shape=(m,n))
grid_v = ti.field(dtype=ti.i32, shape=(m,n))
grid_u_last = ti.field(dtype=ti.i32, shape=(m,n))
grid_v_last = ti.field(dtype=ti.i32, shape=(m,n))
grid_p = ti.field(dtype=ti.f32, shape=(m,n))
grid_m = ti.field(dtype=ti.f32, shape=(m,n))

#particles
particle_position = ti.Vector.field(2, dtype=ti.f32, shape=m*n)
particle_velocity = ti.Vector.field(2, dtype=ti.f32, shape=m*n)

#solver
mg_level = 4
smoothing = 2
bottom_smoothing = 10
solver = MGPCGSolver(m,n,grid_u,grid_v,grid_type,multigrid_level=mg_level,pre_and_post_smoothing=smoothing,bottom_smoothing=bottom_smoothing)

def init():
    @ti.kernel
    def init_dambreak(x: ti.f32, y: ti.f32):
        xn = int(x / grid_dx)
        yn = int(y / grid_dy)
        for i, j in grid_type:
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                grid_type[i, j] = utils.SOLID  # boundary
            else:
                if i <= xn and j <= yn:
                    grid_type[i, j] = utils.FLUID
                else:
                    grid_type[i, j] = utils.AIR
        
        for i in particle_position:
            particle_position[i] = [ti.random()/w * x, ti.random()/h *y]
            particle_velocity[i] = [0, 0]
    
    @ti.kernel
    def init_field():
        for i, j in grid_u:
            grid_u[i, j] = 0.0
            grid_u_last[i, j] = 0.0

        for i, j in grid_v:
            grid_v[i, j] = 0.0
            grid_v_last[i, j] = 0.0

        for i, j in grid_p:
            grid_p[i, j] = 0.0

        for i, j in grid_m:
            grid_m[i, j] = 0.0
    
    init_dambreak(4, 4)
    init_field()

def render():
    np_particles = particle_position.to_numpy()
    bg_color = 0xFFFFFF
    particle_color = 0xFF0000
    particle_radius = 1.2
    gui.clear(bg_color)
    gui.circles(np_particles,radius=particle_radius,color=particle_color)
    gui.show()

# __________________solve__________________________
@ti.kernel
def apply_gravity(dt: ti.f32):
    for i, j in grid_v:
        grid_v[i, j] += g * dt

@ti.kernel
def boundary():
    for i, j in ti.ndrange(m, n):
        if grid_type[i, j] == utils.SOLID:
            grid_u = 0.0
    for i, j in ti.ndrange(m, n):
        if grid_type[i, j] == utils.SOLID:
            grid_v = 0.0

@ti.kernel
def solve_pressure(dt: ti.f32):
    A = dt / (rho * grid_dx * grid_dx)
    b = 1 / grid_dx
    solver.system_init(A, b)
    solver.solve(1000)
    grid_p.copy_from(solver.p)

@ti.kernel
def apply_pressure(dt: ti.f32):
    scale = dt / (rho * grid_dx)
    for i, j in ti.ndrange(m, n):
        if grid_type[i - 1, j] == utils.FLUID or grid_type[i, j] == utils.FLUID:
            if grid_type[i - 1, j] == utils.SOLID or grid_type[i, j] == utils.SOLID:
                grid_u[i, j] = 0
            else:
                grid_u[i, j] -= scale * (grid_p[i, j] - grid_p[i - 1, j])

        if grid_type[i, j - 1] == utils.FLUID or grid_type[i, j] == utils.FLUID:
            if grid_type[i, j - 1] == utils.SOLID or grid_type[i, j] == utils.SOLID:
                grid_v[i, j] = 0
            else:
                grid_v[i, j] -= scale * (grid_p[i, j] - grid_p[i, j - 1])

@ti.kernel
def update_particle(dt: ti.f32):
    for i in particle_position:
        cur_pos = particle_position[i]
        cur_v = particle_velocity[i]
        cur_pos += cur_v * dt
        #boundary
        if cur_pos[0] <= 0:
            cur_pos[0] = 0
            cur_v[0] = 0
        if cur_pos[0] >= w:
            cur_pos[0] = w 
            cur_v[0] = 0
        if cur_pos[1] <= 0:
            cur_pos[1] = 0
            cur_v[1] = 0
        if cur_pos[1] >= h:
            cur_pos[1] = h
            cur_v[1] = 0
        particle_position[p] = cur_pos
        particle_velocity[p] = cur_v

@ti.kernel
def update_grid_type():
    for i,j in grid_type:
        if grid_type[i, j] != utils.SOLID:
            grid_type[i ,j] = utils.AIR
        for i in particle_position:
            pos = particle_position[i]
            idx = ti.cast(ti.floor(pos / ti.Vector([grid_dx, grid_dy])), ti.i32)
            if grid_type[idx[0], idx[1]] != utils.SOLID:
                grid_type[idx] = utils.FLUID

@ti.kernel
def P2G():
    for p in particle_position:
        base = (particle_position[p] * inv_dx - 0.5).cast(int)
        fx = particle_position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * particle_velocity[p]
                grid_m[base + offset] += weight

@ti.kernel
def G2P():
    for p in particle_position:
        base = (particle_position[p] * inv_dx - 0.5).cast(int)
        fx = particle_position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                weight = w[i][0] * w[j][1]
                new_v += weight * grid_v[base + ti.Vector([i,j])]
        grid_v = new_v
init()
while True:
    render()
    G2P()
    apply_gravity(dt)
    boundary()
    P2G()
    
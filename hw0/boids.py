import taichi as ti
import numpy as np
import math

ti.init(arch = ti.gpu)

dt = 1

vis_radias = 40.0
max_speed = 4.0

boids_size = 2000
boids_color = 255
boids_radias = 1
resx = 800
resy = 600
boids_pos = ti.Vector(2, dt=ti.f32, shape=boids_size)
boids_velocity = ti.Vector(2, dt=ti.f32, shape=boids_size)
img = ti.Vector(3,ti.i32, shape=(resx, resy))


@ti.kernel
def init():
    for x in range(0,boids_size):
        boids_pos[x] = ti.Vector([ti.random(ti.f32)*resx, ti.random(ti.f32)*resy])
        boids_velocity[x] = ti.Vector([max_speed*(ti.random(ti.f32)-0.5), max_speed*(ti.random(ti.f32)-0.5)])

@ti.kernel
def render():
    for i,j in ti.ndrange((0,resx),(0,resy)):
        img[i, j] = ti.Vector([255 - boids_color,255 - boids_color,255 - boids_color])
    for i in range(boids_size):
        for x in range(ti.cast(boids_pos[i][0],ti.i32) - boids_radias, ti.cast(boids_pos[i][0],ti.i32) + boids_radias):
            for y in range(ti.cast(boids_pos[i][1],ti.i32) - boids_radias,ti.cast(boids_pos[i][1],ti.i32) + boids_radias):
                img[x, y] = ti.Vector([boids_velocity[i][1]*255,boids_velocity[i].norm()*255,boids_velocity[i][0]*255])

@ti.kernel
def update_pos():
    for x in range(boids_size):
        boids_pos[x] = boids_pos[x] + dt * boids_velocity[x];
        if (boids_pos[x][0] > resx): boids_pos[x][0] = 1
        elif (boids_pos[x][1] > resy): boids_pos[x][1] = 1
        elif (boids_pos[x][0] < 0): boids_pos[x][0] = resx-1
        elif (boids_pos[x][1] < 0): boids_pos[x][1] = resy-1

@ti.kernel
def update_by_rules():
    for i in range(boids_size):
        avoid = ti.Vector([0,0])
        cnt=0
        avoid = ti.Vector([0.,0.])
        follow = ti.Vector([0.,0.])
        middle = ti.Vector([0.,0.])
        for j in range(boids_size):
            if i!=j:
                dis = boids_pos[i] - boids_pos[j]
                if dis.norm() < vis_radias:
                    cnt += 1
                    boids_velocity[i] += dis.normalized()/dis.norm()*max_speed
                    follow += boids_velocity[j]
                    middle = middle + boids_pos[j]
        if cnt != 0:
            middle = middle/cnt - boids_pos[i]            
            boids_velocity[i] += (middle + (follow/cnt)).normalized()
            if boids_velocity[i].norm() > max_speed:
                boids_velocity[i] = boids_velocity[i].normalized() *max_speed

init()
gui = ti.GUI('Boids', (resx, resy))
while True:
    update_by_rules()
    update_pos()
    render()
    gui.set_image(img.to_numpy().astype(np.uint8))
    gui.show()
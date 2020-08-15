import taichi as ti
import math
import time
import numpy as np
import math

ti.init(arch=ti.cpu)

real = ti.f32
dim = 2
n_vertices_x = 24
n_vertices_y = 2
vertices_mass = 1
w = 1 / vertices_mass
n_vertices = n_vertices_x * n_vertices_y
dt = 3e-4
dx = 0.025
fext = [0, -6000]
iteration = 20
p_cons_n = n_vertices * 2 - n_vertices_x - n_vertices_y
d_cons_n = 2 * (n_vertices - n_vertices_x - n_vertices_y + 1)
p_d = dx
d_d = 1.41421 * dx
k = 1
constraint = ti.Vector(4, dt=real, shape=(4*n_vertices-3*n_vertices_x-3*n_vertices_y+2))


tank_size_x = 2
tank_size_y = 2
tank_n = tank_size_x * tank_size_y
tank_dx = 0.04
p_tank_cons_n = tank_n * 2 - tank_size_x - tank_size_x
d_tank_cons_n = 2 * (tank_n - tank_size_x - tank_size_x + 1)
tank_constraint = ti.Vector(4, dt=real, shape=(4*tank_n-3*tank_size_x-3*tank_size_y+2))



x = ti.Vector(dim, dt=real, shape=n_vertices)
v = ti.Vector(dim, dt=real, shape=n_vertices)
p = ti.Vector(dim, dt=real, shape=n_vertices)

x1 = ti.Vector(dim, dt=real, shape=n_vertices)
v1 = ti.Vector(dim, dt=real, shape=n_vertices)
p1 = ti.Vector(dim, dt=real, shape=n_vertices)

xt = ti.Vector(dim, dt=real, shape=n_vertices)
vt = ti.Vector(dim, dt=real, shape=n_vertices)
pt = ti.Vector(dim, dt=real, shape=n_vertices)

xt1 = ti.Vector(dim, dt=real, shape=n_vertices)
vt1 = ti.Vector(dim, dt=real, shape=n_vertices)
pt1 = ti.Vector(dim, dt=real, shape=n_vertices)


mesh = lambda i, j: i * n_vertices_y + j

for i in range(n_vertices_x):
    for j in range(n_vertices_y):
        t = mesh(i, j)
        x[t] = [0.1 + i * dx , 0.3 + j * dx  ]
        v[t] = [0, 0]
        x1[t] = [0.3 + i * dx , 0.6 + j * dx  ]
        v1[t] = [0, 0]

for i in range(tank_size_x):
    for j in range(tank_size_y):
        t = mesh(i, j)
        xt[t] = [0.2 + i * tank_dx, 0.5 + j * tank_dx]
        vt[t] = [0, 0]
        xt1[t] = [0.65 + i * tank_dx, 0.8 + j * tank_dx]
        vt1[t] = [0, 0]


gui = ti.GUI("PBD", (640, 640), background_color=0xF8F8FF)

@ti.kernel
def prediction():
    for i in range(n_vertices_x):
        for j in range(n_vertices_y):
            if (not(i == n_vertices_x - 1)) and (not(i == 0)):
                v[mesh(i, j)] = v[mesh(i, j)] + dt * w * ti.Vector(fext)
                p[mesh(i, j)] = x[mesh(i, j)] + dt * v[mesh(i, j)]
                v1[mesh(i, j)] = v1[mesh(i, j)] + dt * w * ti.Vector(fext)
                p1[mesh(i, j)] = x1[mesh(i, j)] + dt * v1[mesh(i, j)]
            else:
                v[mesh(i, j)] = ti.Vector([0, 0])
                p[mesh(i, j)] = x[mesh(i, j)]
                v1[mesh(i, j)] = ti.Vector([0, 0])
                p1[mesh(i, j)] = x1[mesh(i, j)]

    for i in range(tank_size_x):
        for j in range(tank_size_y):
            vt[mesh(i, j)] = vt[mesh(i, j)] + dt * w * ti.Vector(fext)
            vt[mesh(i, j)] = vt[mesh(i, j)] * 0.99
            pt[mesh(i, j)] = xt[mesh(i, j)] + dt * vt[mesh(i, j)]
            vt1[mesh(i, j)] = vt1[mesh(i, j)] + dt * w * ti.Vector(fext)
            vt1[mesh(i, j)] = vt1[mesh(i, j)] * 0.99
            pt1[mesh(i, j)] = xt1[mesh(i, j)] + dt * vt1[mesh(i, j)]


@ti.kernel
def gen_constraint():
    cnt = 0
    for i in range(n_vertices_x):
        for j in range(n_vertices_y-1):
            constraint[cnt] = [i, j, i, j+1]
            cnt += 1
    for i in range(n_vertices_x-1):
        for j in range(n_vertices_y):
            constraint[cnt] = [i, j, i+1, j]
            cnt += 1
    for i in range(n_vertices_x-1):
        for j in range(n_vertices_y-1):
            constraint[cnt] = [i, j, i+1, j+1]
            cnt += 1
    for i in range(1,n_vertices_x):
        for j in range(0,n_vertices_y-1):
            constraint[cnt] = [i, j, i-1, j+1]
            cnt += 1

    cnt = 0
    for i in range(tank_size_x):
        for j in range(tank_size_y-1):
            tank_constraint[cnt] = [i, j, i, j+1]
            cnt += 1
    for i in range(tank_size_x-1):
        for j in range(tank_size_y):
            tank_constraint[cnt] = [i, j, i+1, j]
            cnt += 1
    for i in range(tank_size_x-1):
        for j in range(tank_size_y-1):
            tank_constraint[cnt] = [i, j, i+1, j+1]
            cnt += 1
    for i in range(1,tank_size_x):
        for j in range(0,tank_size_y-1):
            tank_constraint[cnt] = [i, j, i-1, j+1]
            cnt += 1


@ti.kernel
def stretching_constraint_projection():
    for iter in range(30):
        for i in range(p_cons_n):
            cur_p_0 = [constraint[i][0], constraint[i][1]]
            cur_p_1 = [constraint[i][2], constraint[i][3]]
            cur_dis = p[mesh(cur_p_0[0],cur_p_0[1])] - p[mesh(cur_p_1[0],cur_p_1[1])]
            cur_dis1 = p1[mesh(cur_p_0[0],cur_p_0[1])] - p1[mesh(cur_p_1[0],cur_p_1[1])]
            delta = cur_dis.norm() - p_d
            delta2 = cur_dis1.norm() - p_d
            p[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis/cur_dis.norm() * delta * k
            p[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis/cur_dis.norm() * delta * k
            if (delta2 > 0):
                p1[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k
                p1[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k


        for i in range(p_cons_n+1, p_cons_n + d_cons_n):
            cur_p_0 = [constraint[i][0], constraint[i][1]]
            cur_p_1 = [constraint[i][2], constraint[i][3]]
            cur_dis = p[mesh(cur_p_0[0],cur_p_0[1])] - p[mesh(cur_p_1[0],cur_p_1[1])]
            cur_dis1 = p1[mesh(cur_p_0[0],cur_p_0[1])] - p1[mesh(cur_p_1[0],cur_p_1[1])]
            delta = cur_dis.norm() - d_d
            delta2 = cur_dis1.norm() - d_d

            p[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis/cur_dis.norm() * delta * k
            p[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis/cur_dis.norm() * delta * k
            if (delta2 > 0):
                p1[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k
                p1[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k


        for i in range(p_tank_cons_n):
            cur_p_0 = [tank_constraint[i][0], tank_constraint[i][1]]
            cur_p_1 = [tank_constraint[i][2], tank_constraint[i][3]]
            cur_dis = pt[mesh(cur_p_0[0],cur_p_0[1])] - pt[mesh(cur_p_1[0],cur_p_1[1])]
            cur_dis1 = pt1[mesh(cur_p_0[0],cur_p_0[1])] - pt1[mesh(cur_p_1[0],cur_p_1[1])]
            delta = cur_dis.norm() - tank_dx
            delta2 = cur_dis1.norm() - tank_dx
            pt[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis/cur_dis.norm() * delta * k
            pt[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis/cur_dis.norm() * delta * k
            pt1[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k
            pt1[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k

        for i in range(p_tank_cons_n+1, p_tank_cons_n + d_tank_cons_n):
            cur_p_0 = [tank_constraint[i][0], tank_constraint[i][1]]
            cur_p_1 = [tank_constraint[i][2], tank_constraint[i][3]]
            cur_dis = pt[mesh(cur_p_0[0],cur_p_0[1])] - pt[mesh(cur_p_1[0],cur_p_1[1])]
            cur_dis1 = pt1[mesh(cur_p_0[0],cur_p_0[1])] - pt1[mesh(cur_p_1[0],cur_p_1[1])]
            delta = cur_dis.norm() - tank_dx * 1.41421
            delta2 = cur_dis1.norm() - tank_dx * 1.41421
            pt[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis/cur_dis.norm() * delta * k
            pt[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis/cur_dis.norm() * delta * k
            pt1[mesh(cur_p_0[0],cur_p_0[1])] -= 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k
            pt1[mesh(cur_p_1[0],cur_p_1[1])] += 0.5 * cur_dis1/cur_dis1.norm() * delta2 * k

        for t in range(tank_n):
            for i in range(n_vertices_x-1):
                #A lb p[mesh(i,1)] B lt p[mesh(i,0)] C rt p[mesh(i+1,0)] D rb p[mesh(i+1,1)]
                a = (p[mesh(i,0)][0] - p[mesh(i,1)][0])*(pt[t][1] - p[mesh(i,1)][1]) - (p[mesh(i,0)][1] - p[mesh(i,1)][1])*(pt[t][0] - p[mesh(i,1)][0])
                b = (p[mesh(i+1,0)][0] - p[mesh(i,0)][0])*(pt[t][1] - p[mesh(i,0)][1]) - (p[mesh(i+1,0)][1] - p[mesh(i,0)][1])*(pt[t][0] - p[mesh(i,0)][0])
                c = (p[mesh(i+1,1)][0] - p[mesh(i+1,0)][0])*(pt[t][1] - p[mesh(i+1,0)][1]) - (p[mesh(i+1,1)][1] - p[mesh(i+1,0)][1])*(pt[t][0] - p[mesh(i+1,0)][0])
                d = (p[mesh(i,1)][0] - p[mesh(i+1,1)][0])*(pt[t][1] - p[mesh(i+1,1)][1]) - (p[mesh(i,1)][1] - p[mesh(i+1,1)][1]) * (pt[t][0] - p[mesh(i+1,1)][0])
                if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                    if (abs((p[mesh(i,0)][1] + p[mesh(i+1,0)][1])/2 - pt[t][1])) < (abs((p[mesh(i,1)][1] + p[mesh(i+1,1)][1])/2 - pt[t][1])):
                        ax = p[mesh(i,0)][0]
                        ay = p[mesh(i,0)][1]
                        bx = p[mesh(i+1,0)][0]
                        by = p[mesh(i+1,0)][1]
                        cx = pt[t][0]
                        cy = pt[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p[mesh(i,0)][1] - p[mesh(i+1,0)][1])/(p[mesh(i,0)][0] - p[mesh(i+1,0)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt[t] = pt[t] + dir2
                        p[mesh(i,0)]  =  p[mesh(i,0)] - dir2
                        p[mesh(i+1,0)] =  p[mesh(i+1,0)] - dir2
                    else:
                        ax = p[mesh(i,1)][0]
                        ay = p[mesh(i,1)][1]
                        bx = p[mesh(i+1,1)][0]
                        by = p[mesh(i+1,1)][1]
                        cx = pt[t][0]
                        cy = pt[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p[mesh(i,1)][1] - p[mesh(i+1,1)][1])/(p[mesh(i,1)][0] - p[mesh(i+1,1)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt[t] = pt[t] + dir2
                        p[mesh(i,0)]  =  p[mesh(i,0)] - dir2
                        p[mesh(i+1,0)] =  p[mesh(i+1,0)] - dir2
                        #print(dis)
            
            for i in range(n_vertices_x-1):
                a = (p1[mesh(i,0)][0] - p1[mesh(i,1)][0])*(pt[t][1] - p1[mesh(i,1)][1]) - (p1[mesh(i,0)][1] - p1[mesh(i,1)][1])*(pt[t][0] - p1[mesh(i,1)][0])
                b = (p1[mesh(i+1,0)][0] - p1[mesh(i,0)][0])*(pt[t][1] - p1[mesh(i,0)][1]) - (p1[mesh(i+1,0)][1] - p1[mesh(i,0)][1])*(pt[t][0] - p1[mesh(i,0)][0])
                c = (p1[mesh(i+1,1)][0] - p1[mesh(i+1,0)][0])*(pt[t][1] - p1[mesh(i+1,0)][1]) - (p1[mesh(i+1,1)][1] - p1[mesh(i+1,0)][1])*(pt[t][0] - p1[mesh(i+1,0)][0])
                d = (p1[mesh(i,1)][0] - p1[mesh(i+1,1)][0])*(pt[t][1] - p1[mesh(i+1,1)][1]) - (p1[mesh(i,1)][1] - p1[mesh(i+1,1)][1]) * (pt[t][0] - p1[mesh(i+1,1)][0])
                if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                    if (abs((p[mesh(i,0)][1] + p[mesh(i+1,0)][1])/2 - pt[t][1])) < (abs((p[mesh(i,1)][1] + p[mesh(i+1,1)][1])/2 - pt[t][1])):
                        ax = p1[mesh(i,0)][0]
                        ay = p1[mesh(i,0)][1]
                        bx = p1[mesh(i+1,0)][0]
                        by = p1[mesh(i+1,0)][1]
                        cx = pt[t][0]
                        cy = pt[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p1[mesh(i,0)][1] - p1[mesh(i+1,0)][1])/(p1[mesh(i,0)][0] - p1[mesh(i+1,0)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt[t] = pt[t] + dir2
                        p1[mesh(i,0)]  =  p1[mesh(i,0)] - dir2
                        p1[mesh(i+1,0)] =  p1[mesh(i+1,0)] - dir2
                    else:
                        ax = p1[mesh(i,1)][0]
                        ay = p1[mesh(i,1)][1]
                        bx = p1[mesh(i+1,1)][0]
                        by = p1[mesh(i+1,1)][1]
                        cx = pt[t][0]
                        cy = pt[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p1[mesh(i,1)][1] - p1[mesh(i+1,1)][1])/(p1[mesh(i,1)][0] - p1[mesh(i+1,1)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt[t] = pt[t] + dir2
                        p1[mesh(i,0)]  =  p1[mesh(i,0)] - dir2
                        p1[mesh(i+1,0)] =  p1[mesh(i+1,0)] - dir2

            for i in range(n_vertices_x-1):
                    #A lb p[mesh(i,1)] B lt p[mesh(i,0)] C rt p[mesh(i+1,0)] D rb p[mesh(i+1,1)]
                a = (p1[mesh(i,0)][0] - p1[mesh(i,1)][0])*(pt1[t][1] - p1[mesh(i,1)][1]) - (p1[mesh(i,0)][1] - p1[mesh(i,1)][1])*(pt1[t][0] - p1[mesh(i,1)][0])
                b = (p1[mesh(i+1,0)][0] - p1[mesh(i,0)][0])*(pt1[t][1] - p1[mesh(i,0)][1]) - (p1[mesh(i+1,0)][1] - p1[mesh(i,0)][1])*(pt1[t][0] - p1[mesh(i,0)][0])
                c = (p1[mesh(i+1,1)][0] - p1[mesh(i+1,0)][0])*(pt1[t][1] - p1[mesh(i+1,0)][1]) - (p1[mesh(i+1,1)][1] - p1[mesh(i+1,0)][1])*(pt1[t][0] - p1[mesh(i+1,0)][0])
                d = (p1[mesh(i,1)][0] - p1[mesh(i+1,1)][0])*(pt1[t][1] - p1[mesh(i+1,1)][1]) - (p1[mesh(i,1)][1] - p1[mesh(i+1,1)][1]) * (pt1[t][0] - p1[mesh(i+1,1)][0])
                if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                    if (abs((p[mesh(i,0)][1] + p[mesh(i+1,0)][1])/2 - pt1[t][1])) < (abs((p[mesh(i,1)][1] + p[mesh(i+1,1)][1])/2 - pt1[t][1])):
                        ax = p1[mesh(i,0)][0]
                        ay = p1[mesh(i,0)][1]
                        bx = p1[mesh(i+1,0)][0]
                        by = p1[mesh(i+1,0)][1]
                        cx = pt1[t][0]
                        cy = pt1[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p1[mesh(i,0)][1] - p1[mesh(i+1,0)][1])/(p1[mesh(i,0)][0] - p1[mesh(i+1,0)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt1[t] = pt1[t] + dir2
                        p1[mesh(i,0)]  =  p1[mesh(i,0)] - dir2
                        p1[mesh(i+1,0)] =  p1[mesh(i+1,0)] - dir2
                    else:
                        ax = p1[mesh(i,1)][0]
                        ay = p1[mesh(i,1)][1]
                        bx = p1[mesh(i+1,1)][0]
                        by = p1[mesh(i+1,1)][1]
                        cx = pt1[t][0]
                        cy = pt1[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p1[mesh(i,1)][1] - p1[mesh(i+1,1)][1])/(p1[mesh(i,1)][0] - p1[mesh(i+1,1)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt1[t] = pt1[t] + dir2
                        p1[mesh(i,0)]  =  p1[mesh(i,0)] - dir2
                        p1[mesh(i+1,0)] =  p1[mesh(i+1,0)] - dir2
            
            for i in range(n_vertices_x-1):
                a = (p[mesh(i,0)][0] - p[mesh(i,1)][0])*(pt1[t][1] - p[mesh(i,1)][1]) - (p[mesh(i,0)][1] - p[mesh(i,1)][1])*(pt1[t][0] - p[mesh(i,1)][0])
                b = (p[mesh(i+1,0)][0] - p[mesh(i,0)][0])*(pt1[t][1] - p[mesh(i,0)][1]) - (p[mesh(i+1,0)][1] - p[mesh(i,0)][1])*(pt1[t][0] - p[mesh(i,0)][0])
                c = (p[mesh(i+1,1)][0] - p[mesh(i+1,0)][0])*(pt1[t][1] - p[mesh(i+1,0)][1]) - (p[mesh(i+1,1)][1] - p[mesh(i+1,0)][1])*(pt1[t][0] - p[mesh(i+1,0)][0])
                d = (p[mesh(i,1)][0] - p[mesh(i+1,1)][0])*(pt1[t][1] - p[mesh(i+1,1)][1]) - (p[mesh(i,1)][1] - p[mesh(i+1,1)][1]) * (pt1[t][0] - p[mesh(i+1,1)][0])
                if((a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0 and c < 0 and d < 0)):
                    if (abs((p[mesh(i,0)][1] + p[mesh(i+1,0)][1])/2 - pt1[t][1])) < (abs((p[mesh(i,1)][1] + p[mesh(i+1,1)][1])/2 - pt1[t][1])):
                        ax = p[mesh(i,0)][0]
                        ay = p[mesh(i,0)][1]
                        bx = p[mesh(i+1,0)][0]
                        by = p[mesh(i+1,0)][1]
                        cx = pt1[t][0]
                        cy = pt1[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p[mesh(i,0)][1] - p[mesh(i+1,0)][1])/(p[mesh(i,0)][0] - p[mesh(i+1,0)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt1[t] = pt1[t] + dir2
                        p[mesh(i,0)]  =  p[mesh(i,0)] - dir2
                        p[mesh(i+1,0)] =  p[mesh(i+1,0)] - dir2
                    else:
                        ax = p[mesh(i,1)][0]
                        ay = p[mesh(i,1)][1]
                        bx = p[mesh(i+1,1)][0]
                        by = p[mesh(i+1,1)][1]
                        cx = pt1[t][0]
                        cy = pt1[t][1]
                        dis = (cy-by)*(cy-by) + (cx-bx)*(cx-bx) - (((bx-ax)*(bx-cx) + (by-ay)*(by-cy)) * ((bx-ax)*(bx-cx) + (by-ay)*(by-cy))) / ((ay-by)*(ay-by) + (ax-bx)*(ax-bx))
                        if (dis <= 0):
                            dis = 0
                        else:
                            dis = ti.sqrt(dis)
                        dir = [-((p[mesh(i,1)][1] - p[mesh(i+1,1)][1])/(p[mesh(i,1)][0] - p[mesh(i+1,1)][0])), 1]
                        fuck = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        shit = ti.sqrt((dir[0]*dir[0])+(dir[1]*dir[1]))
                        dir2 = [(dir[0]/fuck)*dis, (dir[1]/shit)*dis]
                        if (fuck==0):
                            dir2[0]=0
                        if (shit==0):
                            dir2[1]=0
                        pt1[t] = pt1[t] + dir2
                        p[mesh(i,0)]  =  p[mesh(i,0)] - dir2
                        p[mesh(i+1,0)] =  p[mesh(i+1,0)] - dir2




@ti.kernel
def update():
    for i in range(n_vertices_x):
        for j in range(n_vertices_y):
            if (not(i == n_vertices_x - 1)) and (not(i == 0)):
                v[mesh(i, j)] = (p[mesh(i, j)] - x[mesh(i ,j)]) / dt
                x[mesh(i, j)] = p[mesh(i, j)]
                v1[mesh(i, j)] = (p1[mesh(i, j)] - x1[mesh(i ,j)]) / dt
                x1[mesh(i, j)] = p1[mesh(i, j)]
            else:
                v[mesh(i, j)] = ti.Vector([0, 0])
                v1[mesh(i, j)] = ti.Vector([0, 0])

    for i in range(tank_size_x):
        for j in range(tank_size_y):
                vt[mesh(i, j)] = (pt[mesh(i, j)] - xt[mesh(i ,j)]) / dt
                xt[mesh(i, j)] = pt[mesh(i, j)]
                vt1[mesh(i, j)] = (pt1[mesh(i, j)] - xt1[mesh(i ,j)]) / dt
                xt1[mesh(i, j)] = pt1[mesh(i, j)]


result_dir = "./results"
video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
while True:
    prediction()
    gen_constraint()
    stretching_constraint_projection()
    update()
    #time.sleep(5)
    node_x = x.to_numpy()
    node_x1 = x1.to_numpy()
    # gui.circles(node_x, radius=2, color=0xDC143C)
    # gui.circles(node_x1, radius=2, color=0xDC143C)
    # node_xt = xt.to_numpy()
    # gui.circles(node_xt, radius=2, color=0xDC143C)

    # while gui.get_event(ti.GUI.PRESS):
    #     pass
    # if gui.is_pressed(ti.GUI.LMB):
    #    if np.linalg.norm(gui.get_cursor_pos() - node_x1[mesh(n_vertices_x//2, 1)]) < 0.15 :
        
        # elif np.linalg.norm(gui.get_cursor_pos() - node_x[mesh(0, n_vertices_y - 1)]) < 0.1 :
        #     x[mesh(0, n_vertices_y - 1)] = gui.get_cursor_pos()
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'a':
            for i in range(tank_size_x):
                for j in range(tank_size_y):
                    vt1[mesh(i,j)] = [-18,0]
        elif gui.event.key == 'd':
            for i in range(tank_size_x):
                for j in range(tank_size_y):
                    vt1[mesh(i,j)] = [18,0]
        elif gui.event.key == ' ':
            for i in range(tank_size_x):
                for j in range(tank_size_y):
                    vt1[mesh(i,j)] = [0,30]
            

    for i in range(n_vertices_x):
        for j in range(n_vertices_y-1):
            gui.line(x[mesh(i,j)],x[mesh(i,j+1)],radius=1.5,color=0x00BFFF)
    for i in range(n_vertices_x-1):
        for j in range(n_vertices_y):
            gui.line(x[mesh(i,j)],x[mesh(i+1,j)],radius=1.5,color=0x00BFFF)
    for i in range(n_vertices_x-1):
        for j in range(n_vertices_y-1):
            gui.line(x[mesh(i,j)],x[mesh(i+1,j+1)],radius=1.0,color=0x00BFFF)
    for i in range(1,n_vertices_x):
        for j in range(0,n_vertices_y-1):
            gui.line(x[mesh(i,j)],x[mesh(i-1,j+1)],radius=1.0,color=0x00BFFF)




    # pixel = gui.get_image()
    # video_manager.write_frame(pixel)
    # video_manager.make_video(gif=True, mp4=True)

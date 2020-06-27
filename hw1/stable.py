import taichi as ti
import numpy as np
import time

res = 700
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.02
p_jacobi_iters = 100
f_strength = 30000.0
dye_decay = 0.9
eps = 0.002
debug = False

assert res > 2

ti.init(arch=ti.opengl)

_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
_new_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
velocity_divs = ti.var(dt=ti.f32, shape=(res, res))
velocity_curls = ti.var(dt=ti.f32, shape=(res, res))
_pressures = ti.var(dt=ti.f32, shape=(res, res))
_new_pressures = ti.var(dt=ti.f32, shape=(res, res))
color_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_new_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))

Mac = False
rk = 2
MacClip = False

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

@ti.func
def vec(x, y):
    return ti.Vector([x, y])


@ti.func
def sample(qf, u, v):
    i, j = int(u), int(v)
    # Nearest
    i = max(0, min(res - 1, i))
    j = max(0, min(res - 1, j))
    return qf[i, j]

@ti.func
def sample_min(x, p):
    #print(p)
    I = ti.cast(ti.floor(p+1),ti.i32)
    return min(sample(x,I[0],I[1]),  sample(x,I[0]+0.5,I[1]), sample(x,I[0],I[1]+1), sample(x,I[0]+1,I[1]+1))

@ti.func
def sample_max(x, p):
    #print("cur p=",p)
    I = ti.cast(ti.floor(p+1),ti.i32)
    return max(sample(x,I[0],I[1]),  sample(x,I[0]+0.5,I[1]), sample(x,I[0],I[1]+1), sample(x,I[0]+1,I[1]+1))



@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def clamp(p):
    for d in ti.static(range(p.n)):
        p[d] = min(1 - 1e-4 - dx + 0.5 * dx, max(p[d], 0.5 * dx))
    return p

@ti.func
def bilerp(vf, u, v):
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = int(s), int(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu + 0.5, iv + 0.5)
    b = sample(vf, iu + 1.5, iv + 0.5)
    c = sample(vf, iu + 0.5, iv + 1.5)
    d = sample(vf, iu + 1.5, iv + 1.5)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def semi_lagrangian(vf: ti.template(), qf: ti.template(), new_qf: ti.template(),_dt):
    for i,j in vf:
        if ti.static(rk == 1) :
            coord = ti.Vector([i, j]) + 0.5 - _dt * vf[i, j]
            new_qf[i, j] = bilerp(qf, coord[0], coord[1])
        elif ti.static(rk == 2) :
            coord_mid = ti.Vector([i, j]) + 0.5 - _dt * 0.5 * vf[i,j]
            coord = ti.Vector([i,j])+ 0.5 - _dt * bilerp(vf, coord_mid[0], coord_mid[1])
            new_qf[i, j] = bilerp(qf, coord[0], coord[1])

_new_aux_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
_new_aux_dyes = ti.Vector(3, dt=ti.f32, shape=(res, res))


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), is_dyes : ti.template()):
    if (ti.static(Mac)):
        semi_lagrangian(vf,qf,new_qf,dt)
        if (ti.static(is_dyes)) :
            semi_lagrangian(vf,new_qf,_new_aux_dyes,-dt)
        else :
            semi_lagrangian(vf,new_qf,_new_aux_velocities,-dt)
        #for i in range(res):
            #for j in range(res):
        for i,j in vf:
            #if (i>j):
                # print(i,j)
            if (ti.static(is_dyes)) :
                new_qf[i, j] = new_qf[i, j] + 0.5 * (qf[i, j] - _new_aux_dyes[i ,j])
            else :
                new_qf[i, j] = new_qf[i, j] + 0.5 * (qf[i, j] - _new_aux_velocities[i ,j])
            if (ti.static(MacClip)) :
                coord_mid = ti.Vector([i, j]) + 0.5 - dt * 0.5 * vf[i,j]
                coord = ti.Vector([i,j])+ 0.5 - dt * bilerp(vf, coord_mid[0], coord_mid[1])
                min_val = sample_min(qf,coord)
                max_val = sample_max(qf,coord)
                #print(min_val,max_val)
                if (new_qf[i, j].norm() - sample(qf,coord[0],coord[1]).norm() > eps) :
                    new_qf[i, j] = bilerp(qf,coord[0],coord[1])
                
    else:
        semi_lagrangian(vf,qf,new_qf,dt);

force_radius = res / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res / 15.0)**2
f_strength_dt = f_strength * dt

@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template()):
    for i, j in vf:
            omx, omy = 0.1*res, 0.5*res
            mdir = ti.Vector([1., 0.])
            dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
            d2 = dx * dx + dy * dy
            # dv = F * dt
            factor = ti.exp(-d2 * inv_force_radius)
            momentum = mdir * f_strength_dt * factor
            v = vf[i, j]
            vf[i, j] = v + momentum
            # add dye
            dc = dyef[i, j]
            dc += ti.exp(-d2 * inv_dye_denom) * ti.Vector(
                [1.0, 1.0, 1.0])
            dc *= dye_decay
            dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)[0]
        vr = sample(vf, i + 1, j)[0]
        vb = sample(vf, i, j - 1)[1]
        vt = sample(vf, i, j + 1)[1]
        vc = sample(vf, i, j)
        if i == 0:
            vl = -vc[0]
        if i == res - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == res - 1:
            vt = -vc[1]
        velocity_divs[i, j] = (vr - vl + vt - vb) * half_inv_dx

@ti.kernel
def curl(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)[1]
        vr = -sample(vf, i + 1, j)[1]
        vb = -sample(vf, i, j - 1)[0]
        vt = sample(vf, i, j + 1)[0]
        vc = sample(vf, i, j)
        if i == 0:
            vl = vc[1]
        if i == res - 1:
            vr = -vc[1]
        if j == 0:
            vb = -vc[0]
        if j == res - 1:
            vt = vc[0]
        velocity_curls[i, j] = (vr + vl + vt + vb) * half_inv_dx


p_alpha = -dx * dx


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        v = sample(vf, i, j)
        v = v - half_inv_dx * ti.Vector([pr - pl, pt - pb])
        vf[i, j] = v


@ti.kernel
def fill_color_v2(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])


@ti.kernel
def fill_color_v3(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])


@ti.kernel
def fill_color_s(sf: ti.template()):
    for i, j in sf:
        s = abs(sf[i, j])
        color_buffer[i, j] = ti.Vector([s/10, 0,255-s/10])


def step(cnt):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt,False)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt,True)
    velocities_pair.swap()
    dyes_pair.swap()

    if (cnt < 6) :
        apply_impulse(velocities_pair.cur, dyes_pair.cur)

    divergence(velocities_pair.cur)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    curl(velocities_pair.cur)
    fill_color_v3(dyes_pair.cur)
    #fill_color_s(velocity_curls)
    #fill_color_v2(velocities_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

def reset():
    velocities_pair.cur.fill(ti.Vector([0, 0]))
    pressures_pair.cur.fill(0.0)
    dyes_pair.cur.fill(ti.Vector([0, 0, 0]))
    color_buffer.fill(ti.Vector([0, 0, 0]))


def main():
    result_dir = './rk=2'
    video_manger = ti.VideoManager(output_dir=result_dir, framerate=30, automatic_build=False)

    global debug
    gui = ti.GUI('Stable-Fluid', (res, res))
    paused = False
    cnt = 0
    while True:
        cnt += 1
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 'p':
                paused = not paused
            elif e.key == 'd':
                debug = not debug
        if (cnt >= 1000):
            break
        if not paused:
            step(cnt)

        
        img = color_buffer.to_numpy()
        video_manger.write_frame(img)
        gui.set_image(img)
        gui.show()
    
    video_manger.make_video(gif=True, mp4=True)


if __name__ == '__main__':
    main()
import taichi as ti
import numpy as np
import time

res = 600
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.03
p_jacobi_iters = 60
f_strength = 10000.0
dye_decay = 1
debug = False

assert res > 2

ti.init(arch=ti.gpu)

_velocities_x = ti.Vector(dt=ti.f32, shape=(res+1, res))
_new_velocities_x = ti.Vector(dt=ti.f32, shape=(res+1, res))
_velocities_y = ti.Vector(dt=ti.f32, shape=(res , res +1))
_new_velocities_y = ti.Vector(dt=ti.f32, shape=(res , res +1))

velocity_divs = ti.var(dt=ti.f32, shape=(res, res))
_pressures = ti.var(dt=ti.f32, shape=(res, res))
_new_pressures = ti.var(dt=ti.f32, shape=(res, res))
color_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_new_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair_x = TexPair(_velocities_x, _new_velocities_x)
velocities_pair_y = TexPair(_velocities_y, _new_velocities_y)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def sample(qf, u, v):
    i, j = int(u), int(v)
    # Nearest
    i = max(0, min(res - 1, i))
    j = max(0, min(res - 1, j))
    return qf[i, j]
    


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


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


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        coord = ti.Vector([i, j]) + 0.5 - dt * vf[i, j]
        new_qf[i, j] = bilerp(qf, coord[0], coord[1])



force_radius = res / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res / 15.0)**2
f_strength_dt = f_strength * dt


@ti.kernel
def apply_impulse(vf_x, vf_y, dyef: ti.template(),
                  imp_data: ti.ext_arr()):
    for i in vf_x:
        for j in vf_y:
            omx, omy = imp_data[2], imp_data[3]
            mdir = ti.Vector([imp_data[0], imp_data[1]])
            dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
            d2 = dx * dx + dy * dy
            # dv = F * dt
            factor = ti.exp(-d2 * inv_force_radius)
            momentum = mdir * f_strength_dt * factor
            vf_x[i] = vf_x[i] + momentum[0]
            vf_y[j] = vf_y[j] + momentum[1]

            # add dye
            dc = dyef[i, j]
            if mdir.norm() > 0.5:
                dc += ti.exp(-d2 * inv_dye_denom) * ti.Vector(
                    [imp_data[4], imp_data[5], imp_data[6]])
            dc *= dye_decay
            dyef[i, j] = dc


@ti.kernel
def divergence(vf_x, vf_y):
    for i in vf_x:
        for j in vf_y:
            vl = sample(vf_x, i, j)[0]
            vr = sample(vf_x, i + 1, j)[0]
            vb = sample(vf_y, i, j)[1]
            vt = sample(vf_y, i, j + 1)[1]
            velocity_divs[i, j] = (vr - vl + vt - vb) * half_inv_dx


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
def subtract_gradient(vf_x, vf_y, pf: ti.template()):
    for i in vf_x:
        for j in vf_y:
            pl = sample(pf, i - 1, j)
            pr = sample(pf, i + 1, j)
            pb = sample(pf, i, j - 1)
            pt = sample(pf, i, j + 1)
            v_x = sample(vf_x, i, j)
            v_y = sample(vf_y, i, j)
            vf_x[i] = v_x - half_inv_dx * (pr - pl)
            vf_y[j] = v_y - half_inv_dx * (pt - pb)


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
        color_buffer[i, j] = ti.Vector([s, s * 0.25, 0.2])


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair_x.swap()
    velocities_pair_y.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    fill_color_v3(dyes_pair.cur)
    #fill_color_s(velocity_divs)
    #fill_color_v2(velocities_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f'divergence={div_s}')


def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.array([0] * 8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = vec2_npf32(gui.get_cursor_pos()) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(ti.Vector([0, 0]))
    pressures_pair.cur.fill(0.0)
    dyes_pair.cur.fill(ti.Vector([0, 0, 0]))
    color_buffer.fill(ti.Vector([0, 0, 0]))


def main():
    global debug
    gui = ti.GUI('Stable-Fluid', (res, res))
    md_gen = MouseDataGen()
    paused = False
    while True:
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

        if not paused:
            mouse_data = md_gen(gui)
            step(mouse_data)

        img = color_buffer.to_numpy()
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
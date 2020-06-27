import taichi as ti
import numpy as np
import math

# CUDA 在生成随机数时比 OpenGL 慢很多
ti.init(arch=[ti.opengl, ti.metal])

# 画布
nx = 800
ny = 800
screen = ti.Vector(3, dt=ti.f32, shape=(nx, ny))

# 相机参数
# 0.6.8版本有kernel里不能将临时Vector变量相加的bug，这里使用零维张量
lookfrom = ti.Vector(3, dt=ti.f32, shape=())
lookat = ti.Vector(3, dt=ti.f32, shape=())
up = ti.Vector(3, dt=ti.f32, shape=())

fov = 60  # 视角
aspect = ny / nx  # 长宽比

cam_lower_left_corner = ti.Vector(3, dt=ti.f32, shape=())
cam_horizontal = ti.Vector(3, dt=ti.f32, shape=())
cam_vertical = ti.Vector(3, dt=ti.f32, shape=())
cam_origin = ti.Vector(3, dt=ti.f32, shape=())

# 球体列表，[0]-[2]为圆心，[3]为半径
sphere_num = 9
sphere_origin_list = ti.Vector(3, dt=ti.f32, shape=(sphere_num))  # 球心
sphere_radius_list = ti.var(dt=ti.f32, shape=(sphere_num))  # 半径
sphere_material_list = ti.var(dt=ti.i32,
                              shape=(sphere_num))  # 材质，1=漫反射，2=金属，3=光源
sphere_material_color_list = ti.Vector(3, dt=ti.f32,
                                       shape=(sphere_num))  # 材质颜色
sphere_metal_fuzz_list = ti.var(dt=ti.f32, shape=(sphere_num))  # 金属光泽度

sphere_origin_list[0] = [0, -0.2, -1.5]
sphere_radius_list[0] = 0.3
sphere_material_list[0] = 1
sphere_material_color_list[0] = [0.8, 0.3, 0.3]

sphere_origin_list[1] = [-0.8, 0.2, -1]
sphere_radius_list[1] = 0.7
sphere_material_list[1] = 2
sphere_material_color_list[1] = [0.6, 0.8, 0.8]
sphere_metal_fuzz_list[1] = 0.0

sphere_origin_list[2] = [0.7, 0, -0.5]
sphere_radius_list[2] = 0.5
sphere_material_list[2] = 2
sphere_material_color_list[2] = [0.8, 0.6, 0.2]
sphere_metal_fuzz_list[2] = 0.2

# 光源
sphere_origin_list[3] = [0, 5.4, -1]
sphere_radius_list[3] = 3
sphere_material_list[3] = 3
sphere_material_color_list[3] = [10, 10, 10]

# 地面
sphere_origin_list[4] = [0, -100.5, -1]
sphere_radius_list[4] = 100
sphere_material_list[4] = 1
sphere_material_color_list[4] = [0.8, 0.8, 0.8]

# 屋顶
sphere_origin_list[5] = [0, 102.5, -1]
sphere_radius_list[5] = 100
sphere_material_list[5] = 1
sphere_material_color_list[5] = [0.8, 0.8, 0.8]

# 背
sphere_origin_list[6] = [0, 1, 101]
sphere_radius_list[6] = 100
sphere_material_list[6] = 1
sphere_material_color_list[6] = [0.8, 0.8, 0.8]

# 右
sphere_origin_list[7] = [-101.5, 0, -1]
sphere_radius_list[7] = 100
sphere_material_list[7] = 1
sphere_material_color_list[7] = [0.6, 0.0, 0.0]

# 左
sphere_origin_list[8] = [101.5, 0, -1]
sphere_radius_list[8] = 100
sphere_material_list[8] = 1
sphere_material_color_list[8] = [0.0, 0.6, 0.0]

lookfrom[None] = [0.0, 1.0, -5.0]


# 归一化
@ti.func
def normalization(v):
    return v.normalized()


# 生成相机参数
@ti.kernel
def generate_cam_parameter():
    # lookfrom[None] = [0.0, 1.0, -5.0]
    lookat[None] = [0.0, 1.0, -1.0]
    up[None] = [0.0, 1.0, 0.0]
    theta = fov * (math.pi / 180.0)
    half_height = ti.tan(theta / 2.0)
    half_width = aspect * half_height
    cam_origin[None] = lookfrom[None]
    w = normalization(lookfrom[None] - lookat[None])
    u = normalization(up[None].cross(w))
    v = w.cross(u)
    cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
    cam_lower_left_corner[
        None] = cam_origin - half_width * u - half_height * v - w
    cam_horizontal[None] = 2 * half_width * u
    cam_vertical[None] = 2 * half_height * v


@ti.func
def random_in_unit_sphere():
    p = 2.0 * ti.Vector([ti.random(), ti.random(),
                         ti.random()]) - ti.Vector([1, 1, 1])
    while (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] >= 1.0):
        p = 2.0 * ti.Vector(
            [ti.random(), ti.random(), ti.random()]) - ti.Vector([1, 1, 1])
    return p


# 生成光线
@ti.func
def cam_get_ray(u, v):
    return cam_origin, cam_lower_left_corner[
        None] + u * cam_horizontal[None] + v * cam_vertical[None] - cam_origin


@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n


@ti.func
def scatter(ray_origin, ray_direction, hit_p, hit_normal, hit_material,
            hit_fuzz):
    scattered_ray_origin = hit_p
    scattered_ray_direction = ti.Vector([0.0, 0.0, 0.0])
    flag = False
    if hit_material == 1:  # 漫反射
        target = hit_p + hit_normal + random_in_unit_sphere()
        scattered_ray_direction = target - hit_p
        flag = True
    elif hit_material == 2:  # 金属
        reflected = reflect(normalization(ray_direction), hit_normal)
        scattered_ray_direction = reflected + hit_fuzz * random_in_unit_sphere(
        )  # 后一项为光泽度
        flag = scattered_ray_direction.dot(hit_normal) > 0
    else:
        pass
    return flag, scattered_ray_origin, scattered_ray_direction


@ti.func
def hit_sphere(sphere_center, sphere_radius, ray_origin, ray_direction, t_min,
               t_max):
    oc = ray_origin - sphere_center
    a = ray_direction.dot(ray_direction)
    b = oc.dot(ray_direction)
    c = oc.dot(oc) - sphere_radius * sphere_radius
    discriminant = b * b - a * c

    hit_flag = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    if discriminant > 0.0:
        temp = (-b - ti.sqrt(b * b - a * c)) / a
        if temp < t_max and temp > t_min:
            hit_t = temp
            hit_p = ray_origin + hit_t * ray_direction
            hit_normal = (hit_p - sphere_center) / sphere_radius
            hit_flag = True
        if hit_flag == False:
            temp = (-b + ti.sqrt(b * b - a * c)) / a
            if temp < t_max and temp > t_min:
                hit_t = temp
                hit_p = ray_origin + hit_t * ray_direction
                hit_normal = (hit_p - sphere_center) / sphere_radius
                hit_flag = True
    return hit_flag, hit_t, hit_p, hit_normal


@ti.func
def hit_all_spheres(ray_origin, ray_direction, t_min, t_max):
    hit_anything = False
    hit_t = 0.0
    hit_p = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 0.0, 0.0])
    hit_material = 1
    hit_material_color = ti.Vector([0.0, 0.0, 0.0])
    hit_fuzz = 0.0
    closest_so_far = t_max
    for i in range(sphere_num):
        hit_flag, temp_hit_t, temp_hit_p, temp_hit_normal = \
            hit_sphere(sphere_origin_list[i], sphere_radius_list[i], ray_origin, ray_direction, t_min, closest_so_far)
        if hit_flag:
            hit_anything = True
            closest_so_far = temp_hit_t
            hit_t = temp_hit_t
            hit_p = temp_hit_p
            hit_normal = temp_hit_normal
            hit_material = sphere_material_list[i]
            hit_material_color = sphere_material_color_list[i]
            hit_fuzz = sphere_metal_fuzz_list[i]
    return hit_anything, hit_t, hit_p, hit_normal, hit_material, hit_material_color, hit_fuzz


@ti.func
def color(ray_origin, ray_direction):
    col = ti.Vector([0.0, 0.0, 0.0])
    coefficient = ti.Vector([1.0, 1.0, 1.0])
    for i in range(10):
        hit_flag, hit_t, hit_p, hit_normal, hit_material, hit_material_color, hit_fuzz = \
            hit_all_spheres(ray_origin, ray_direction, 0.001, 10e9)
        if hit_flag:
            if hit_material == 3:  # 光源
                col = coefficient * hit_material_color
                break

            flag, ray_origin, ray_direction = \
                scatter(ray_origin, ray_direction, hit_p, hit_normal, hit_material, hit_fuzz)
            if flag:
                coefficient *= hit_material_color  # 衰减
            else:
                break
        else:
            unit_direction = normalization(ray_direction)
            t = 0.5 * (unit_direction.y + 1.0)
            col = (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector(
                [0.5, 0.7, 1.0])
            col *= coefficient
            break
    return col


@ti.kernel
def draw():
    for i, j in screen:
        u = (i + ti.random()) / nx  # random是为抗锯齿
        v = (j + ti.random()) / ny
        col = ti.Vector.zero(ti.f32, 3)
        for t in ti.static(range(4)):
            ray_origin, ray_direction = cam_get_ray(u, v)
            col += color(ray_origin, ray_direction)
        screen[i, j] += col / 4

gui = ti.GUI("screen", (nx, ny))

while True:
    generate_cam_parameter()
    screen.fill(0)
    for count in range(1, 10000):
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.LMB:
                x, y = gui.get_cursor_pos()
                lookfrom[None][0] = x * 4 - 2
                lookfrom[None][1] = y * 2 - 1
                break
            elif gui.event.key == ti.GUI.ESCAPE:
                exit()
        draw()
        gui.set_image(np.sqrt(screen.to_numpy() / count))
        gui.show()
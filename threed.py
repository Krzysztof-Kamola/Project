import taichi as ti
from datetime import datetime
import numpy as np
import math
import random

ti.init(arch=ti.gpu)

camera_active = False

#bounds

left_bound = -30.0
right_bound = 30.0
bottom_bound = 0.0
top_bound = 10.0
back_bound = 30.0
front_bound = -30.0

#paricle initial space

p_left_bound = -20.0
p_right_bound = 20.0
p_bottom_bound = 5.0
p_top_bound = 10.0
p_back_bound = 10.0
p_front_bound = -10.0

diameter = 0.7

num_paticles_x = int(np.abs(p_left_bound - p_right_bound) / diameter)
num_paticles_y = int(np.abs(p_bottom_bound - p_top_bound) / diameter)
num_paticles_z = int(np.abs(p_front_bound - p_back_bound) / diameter)

total_particles = int(num_paticles_x*num_paticles_y*num_paticles_z)

dim = 3
x = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
v = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
f = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
rho = ti.field(dtype=float,shape=(total_particles))
p = ti.field(dtype=float,shape=(total_particles))

gravity = ti.Vector([0.0, -9.8,0.0]) * 300

rest_dens = 300
gas_const = 2000
h = 2
hsq = h*h
mass = 2.5
visc = 20
DT = 0.0007

POLY6 = 4 / (np.pi * np.power(h,8))
SPIKY_GRAD = -9.8 / (np.pi * np.power(h,5))
VISC_LAP = 40 / (np.pi * np.power(h,5))

eps = h
bound_damping = -0.5

last_frame = datetime.now()
this_frame = datetime.now()
dt = this_frame - last_frame

@ti.func
def initialize_mass_points():
    for i in range(0,num_paticles_x):
        for j in range(0,num_paticles_y):
            for k in range(0,num_paticles_z):
                index = (k* num_paticles_x * num_paticles_y) + (j * num_paticles_x) + i
                x[index] = [(i*diameter + p_left_bound) + ti.random() / 10,
                 j*diameter + p_bottom_bound,
                 k*diameter + p_front_bound + ti.random() / 10]
                v[index] = [0,0,0]
                f[index] = [0,0,0]
                rho[index] = 0.0
                p[index] = 0.0


@ti.func
def ComputeDensityPressure():
    for i in range(0,total_particles):
        rho[i]= 0.0
        for j in range(0,total_particles):
            rij = ti.Vector([0.0,0.0,0.0])
            rij = x[j] - x[i]
            r2 = ti.math.dot(rij,rij)
            if(r2 < hsq):
                rho[i] += mass * POLY6 * ti.math.pow(hsq - r2,3.0)
        p[i] = gas_const * (rho[i] - rest_dens)
        

@ti.func
def ComputeForces():
    for i in range(0,total_particles):
        fpress = ti.Vector([0.0,0.0,0.0])
        fvisc = ti.Vector([0.0,0.0,0.0])
        for j in range(0,total_particles):
            if i != j:
                rij = x[j] - x[i]
                r = ti.math.length(rij)
                if r < h:
                    fpress += -ti.math.normalize(rij) * mass * (p[i] + p[j]) / (2.0 * rho[j]) * SPIKY_GRAD * ti.math.pow(h-r,3.0)
                    fvisc += visc * mass * (v[j] - v[i]) / rho[j] * VISC_LAP * (h-r)
        fgrav = gravity * mass / rho[i]
        f[i] = fgrav + fvisc + fpress

@ti.func
def integrate():
    for i in range(0,total_particles):
        v[i] += DT * f[i]
        x[i] += DT * v[i]

        if(x[i][0] < left_bound):
            v[i][0] *= bound_damping
            x[i][0] = left_bound
        if(x[i][0] > right_bound):
            v[i][0] *= bound_damping
            x[i][0] = right_bound
        if(x[i][1] < 0.0):
            v[i][1] *= bound_damping
            x[i][1] = bottom_bound
        if(x[i][2] > back_bound):
            v[i][2] *= bound_damping
            x[i][2] = back_bound
        if(x[i][2] < front_bound):
            v[i][2] *= bound_damping
            x[i][2] = front_bound
        # if(x[i][1] > top_bound):
        #     v[i][1] *= bound_damping
        #     x[i][1] = top_bound
               

@ti.kernel
def step():
    for i in ti.grouped(x):
        #v[i] += gravity * dt
        v[i] = v[i] + gravity
    for i in ti.grouped(x):
        x[i] = x[i] + (v[i]*0.0001)
        #x[i] = x[i] + ti.Vector([0,-0.001,0])



def keyboard_handling():
    global camera_active
    if window.get_event(tag=ti.ui.PRESS):
        if window.event.key == ' ':
            camera_active = not camera_active
        if window.event.key == 'r':
            init()
            



window = ti.ui.Window("Taichi sim on GGUI", (1024, (1024)),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()


@ti.kernel
def update():
    ComputeDensityPressure()
    ComputeForces()
    integrate()

@ti.kernel
def init():
    initialize_mass_points()

# for i in range(num_paticles):
#         print(x[i])

camera.position(10,0, 100)
camera.lookat(0.0, 0.0, 0)


init()
while window.running:
    this_frame = datetime.now()
    dt = this_frame - last_frame
    keyboard_handling()
    #activate_camera()
    if camera_active:
        camera.track_user_inputs(window,movement_speed=1)
    scene.set_camera(camera)
    scene.point_light(pos=(10, 10, 20), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    update()
    scene.particles(x, radius=0.3 * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
    last_frame = datetime.now()
    #print(dt)

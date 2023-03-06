import taichi as ti
from datetime import datetime
import numpy as np
import math
import random



ti.init(arch=ti.gpu)

num_paticles_x = 20
num_paticles_y = 20
num_paticles_z = 20

total_particles = num_paticles_x*num_paticles_y
radius = 0.5
x = ti.Vector.field(2,dtype=float,shape=(total_particles,))
v = ti.Vector.field(2,dtype=float,shape=(total_particles,))
f = ti.Vector.field(2,dtype=float,shape=(total_particles,))
rho = ti.field(dtype=float,shape=(total_particles))
p = ti.field(dtype=float,shape=(total_particles))

gravity = ti.Vector([0.0, -9.8]) * 300

rest_dens = 300
gas_const = 2000 * 2
h = 2
hsq = h*h
mass = 2.5
visc = 200
DT = 0.0007

POLY6 = 4 / (np.pi * np.power(h,8))
SPIKY_GRAD = -10 / (np.pi * np.power(h,5))
VISC_LAP = 40 / (np.pi * np.power(h,5))

eps = h
bound_damping = -0.5

last_frame = datetime.now()
this_frame = datetime.now()
dt = this_frame - last_frame

def initialize_mass_points():
    
    for i in range(0,num_paticles_x):
        for j in range(0,num_paticles_y):
            #print("i:",i,", x:",i % num_paticles_x,", y:",np.round_(i / num_paticles_y))
            index = (i*num_paticles_x)+j
            x[index] = [((i) * 2 - 15) + (random.randint(-100,100)/10000),(j) * 2]
            v[index] = [0,0]
            f[index] = [0,0]
            rho[index] = 0.0
            p[index] = 0.0


@ti.func
def ComputeDensityPressure():
    for i in range(0,total_particles):
        rho[i]= 0.0
        for j in range(0,total_particles):
            rij = ti.Vector([0.0,0.0])
            rij = x[j] - x[i]
            r2 = ti.math.dot(rij,rij)
            if(r2 < hsq):
                rho[i] += mass * POLY6 * ti.math.pow(hsq - r2,3.0)
        p[i] = gas_const * (rho[i] - rest_dens)
        

@ti.func
def ComputeForces():
    for i in range(0,total_particles):
        fpress = ti.Vector([0.0,0.0])
        fvisc = ti.Vector([0.0,0.0])
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
    left_bound = -30.0
    right_bound = 30.0
    bottom_bound = 0.0
    top_bound = 10.0
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



window = ti.ui.Window("Taichi sim on GGUI", (1024, (1024)),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

camera_active = False

def activate_camera():
    global camera_active
    if window.get_event(tag=ti.ui.PRESS):
        if window.event.key == ' ':
            camera_active = not camera_active

@ti.kernel
def update():
    ComputeDensityPressure()
    ComputeForces()
    integrate()

initialize_mass_points()
# for i in range(num_paticles):
#         print(x[i])

camera.position(10,0, 100)
camera.lookat(0.0, 0.0, 0)

while window.running:
    this_frame = datetime.now()
    dt = this_frame - last_frame

    activate_camera()
    if camera_active:
        camera.track_user_inputs(window,movement_speed=1)
    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    #step()
    update()
    #ren()
    #print(particles2[1].p)
    #print("x:",x[10][0],"y:",x[10][1])
    scene.particles(x, radius=radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()
    last_frame = datetime.now()
    #print(dt)
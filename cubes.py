import taichi as ti
from datetime import datetime
import numpy as np
import math
import random
import tables

ti.init(arch=ti.gpu)

camera_active = False


#bounds

left_bound = -20.0
right_bound = 20.0
bottom_bound = 0.0
top_bound = 20.0
back_bound = 20.0
front_bound = -20.0

box_width = int(np.abs(left_bound - right_bound))
box_height = int(np.abs(bottom_bound - top_bound))
box_depth = int(np.abs(front_bound - back_bound))

#particle initial space

p_left_bound = -30.0
p_right_bound = 30.0
p_bottom_bound = 5.0
p_top_bound = 10.0
p_back_bound = 10.0
p_front_bound = -10.0
dim = 3
class Cube:
    def __init__(self):
        self.corners = ti.Vector.field(dim,dtype=float,shape=(8,))


diameter = 0.8

num_paticles_x = int(np.abs(p_left_bound - p_right_bound) / diameter)
num_paticles_y = int(np.abs(p_bottom_bound - p_top_bound) / diameter)
num_paticles_z = int(np.abs(p_front_bound - p_back_bound) / diameter)

total_particles = int(num_paticles_x*num_paticles_y*num_paticles_z)

x = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
v = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
f = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
rho = ti.field(dtype=float,shape=(total_particles))
p = ti.field(dtype=float,shape=(total_particles))

box_ratio = 10

num_points_x = int(box_width * (1/box_ratio)) + 1
num_points_y = int(box_height * (1/box_ratio)) + 1
num_points_z = int(box_depth * (1/box_ratio)) + 1

num_points = int(num_points_x * num_points_y * num_points_z)

print(num_points)

scalar_field_x = ti.Vector.field(dim,dtype=float,shape=(num_points,))
scalar_field = ti.field(dtype=float,shape=num_points)

corners = ti.Vector.field(dim,dtype=float,shape=(8,))

corners[0] = [left_bound,top_bound,back_bound]
corners[1] = [left_bound,top_bound,front_bound]
corners[2] = [left_bound,bottom_bound,back_bound]
corners[3] = [left_bound,bottom_bound,front_bound]
corners[4] = [right_bound,top_bound,back_bound] 
corners[5] = [right_bound,top_bound,front_bound]
corners[6] = [right_bound,bottom_bound,back_bound]
corners[7] = [right_bound,bottom_bound,front_bound]


num_cubes = (num_points_x-1)*(num_points_y-1)*(num_points_z-1)

triangles = ti.Vector.field(3,dtype=float,shape=(num_cubes*5*3,))

cubes = ti.Struct.field({
    "corner1": ti.math.vec3,
    "corner2": ti.math.vec3,
    "corner3": ti.math.vec3,
    "corner4": ti.math.vec3,
    "corner5": ti.math.vec3,
    "corner6": ti.math.vec3,
    "corner7": ti.math.vec3,
    "corner8": ti.math.vec3,
    "val1": int,
    "val2": int,
    "val3": int,
    "val4": int,
    "val5": int,
    "val6": int,
    "val7": int,
    "val8": int
},shape=(num_cubes,))


cube_indexes = ti.field(dtype=int,shape=num_cubes)
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

@ti.kernel
def init_scalar_field():
    for i in range(0,num_points_x):
        for j in range(0,num_points_y):
            for k in range(0,num_points_z):
                index = (k* num_points_x * num_points_y) + (j * num_points_x) + i
                scalar_field_x[index] = [i * box_ratio + left_bound,
                 j * box_ratio + bottom_bound,
                 k * box_ratio + front_bound]
                index += 1

@ti.kernel
def init_cubes():
    for i in range(0,num_points_x-1):
        for j in range(0,num_points_y-1):
            for k in range(0,num_points_z-1):
                index = int((k* (num_points_x-1) * (num_points_y-1)) + (j * (num_points_x-1)) + i)
                index2 = int((k* num_points_x * num_points_y) + (j * num_points_x) + i)
                cubes[index].corner1 = scalar_field_x[index2]
                cubes[index].val1 = index2
                index2 = int((k* num_points_x * num_points_y) + (j * num_points_x) + (i+1))
                cubes[index].corner2 = scalar_field_x[index2]
                cubes[index].val2 = index2
                index2 = int(((k+1)* num_points_x * num_points_y) + ((j) * num_points_x) + (i+1))
                cubes[index].corner3 = scalar_field_x[index2]
                cubes[index].val3 = index2
                index2 = int(((k+1)* num_points_x * num_points_y) + ((j) * num_points_x) + (i))
                cubes[index].corner4 = scalar_field_x[index2]
                cubes[index].val4 = index2
                index2 = int(((k)* num_points_x * num_points_y) + ((j+1) * num_points_x) + i)
                cubes[index].corner5 = scalar_field_x[index2]
                cubes[index].val5 = index2
                index2 = int(((k)* num_points_x * num_points_y) + ((j+1) * num_points_x) + (i+1))
                cubes[index].corner6 = scalar_field_x[index2]
                cubes[index].val6 = index2
                index2 = int(((k+1)* num_points_x * num_points_y) + ((j+1) * num_points_x) + (i+1))
                cubes[index].corner7 = scalar_field_x[index2]
                cubes[index].val7 = index2
                index2 = int(((k+1) * num_points_x * num_points_y) + ((j+1) * num_points_x) + (i))
                cubes[index].corner8 = scalar_field_x[index2]
                cubes[index].val8 = index2

@ti.func
def update_field():
    for i in range(0,num_points):
        scalar_field[i] = 0.0
        for j in range(0,total_particles):
            rij = scalar_field_x[i] - x[j]
            r2 = ti.math.dot(rij,rij)
            if r2 < 25.0:
                scalar_field[i] += 1.0

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
            x[i][0] = left_bound
        if(x[i][1] < 0.0):
            v[i][1] *= bound_damping
            x[i][1] = bottom_bound
        if(x[i][2] > back_bound):
            v[i][2] *= bound_damping
            x[i][2] = back_bound
        if(x[i][2] < front_bound):
            v[i][2] *= bound_damping
            x[i][2] = front_bound
        if(x[i][1] > top_bound):
            v[i][1] *= bound_damping
            x[i][1] = top_bound

vertlist = ti.Vector.field(3,dtype=float,shape=(12,))

isolevel = 20.0
@ti.func
def calc_cube_index():
    for index in range(0,num_cubes):
        cubeindex = 0
        if scalar_field[cubes[index].val1] > isolevel: cubeindex |= 1
        if scalar_field[cubes[index].val2] > isolevel: cubeindex |= 2
        if scalar_field[cubes[index].val3] > isolevel: cubeindex |= 4
        if scalar_field[cubes[index].val4] > isolevel: cubeindex |= 8
        if scalar_field[cubes[index].val5] > isolevel: cubeindex |= 16
        if scalar_field[cubes[index].val6] > isolevel: cubeindex |= 32
        if scalar_field[cubes[index].val7] > isolevel: cubeindex |= 64
        if scalar_field[cubes[index].val8] > isolevel: cubeindex |= 128
        cube_indexes[index] = cubeindex

def create_iso_surface(index):
    for i in range(0,12):
        vertlist[i] = [0.0,0.0,0.0]
    cubeindex = 0
    if scalar_field[cubes[index].val1] > isolevel: cubeindex |= 1
    if scalar_field[cubes[index].val2] > isolevel: cubeindex |= 2
    if scalar_field[cubes[index].val3] > isolevel: cubeindex |= 4
    if scalar_field[cubes[index].val4] > isolevel: cubeindex |= 8
    if scalar_field[cubes[index].val5] > isolevel: cubeindex |= 16
    if scalar_field[cubes[index].val6] > isolevel: cubeindex |= 32
    if scalar_field[cubes[index].val7] > isolevel: cubeindex |= 64
    if scalar_field[cubes[index].val8] > isolevel: cubeindex |= 128

    if tables.edgeTable[cubeindex] == 0:
        return
    if tables.edgeTable[cubeindex] & 1:
        vertlist[0] = VertexInterp(isolevel,cubes[index].corner1,cubes[index].corner2,scalar_field[cubes[index].val1],scalar_field[cubes[index].val2])
    if tables.edgeTable[cubeindex] & 2:
        vertlist[1] = VertexInterp(isolevel,cubes[index].corner2,cubes[index].corner3,scalar_field[cubes[index].val2],scalar_field[cubes[index].val3])
    if tables.edgeTable[cubeindex] & 4:
        vertlist[2] = VertexInterp(isolevel,cubes[index].corner3,cubes[index].corner4,scalar_field[cubes[index].val3],scalar_field[cubes[index].val4])
    if tables.edgeTable[cubeindex] & 8:
        vertlist[3] = VertexInterp(isolevel,cubes[index].corner4,cubes[index].corner1,scalar_field[cubes[index].val4],scalar_field[cubes[index].val1])
    if tables.edgeTable[cubeindex] & 16:
        vertlist[4] = VertexInterp(isolevel,cubes[index].corner5,cubes[index].corner6,scalar_field[cubes[index].val5],scalar_field[cubes[index].val6])
    if tables.edgeTable[cubeindex] & 32:
        vertlist[5] = VertexInterp(isolevel,cubes[index].corner6,cubes[index].corner7,scalar_field[cubes[index].val6],scalar_field[cubes[index].val7])
    if tables.edgeTable[cubeindex] & 64:
        vertlist[6] = VertexInterp(isolevel,cubes[index].corner7,cubes[index].corner8,scalar_field[cubes[index].val7],scalar_field[cubes[index].val8])
    if tables.edgeTable[cubeindex] & 128:
        vertlist[7] = VertexInterp(isolevel,cubes[index].corner8,cubes[index].corner5,scalar_field[cubes[index].val8],scalar_field[cubes[index].val5])
    if tables.edgeTable[cubeindex] & 256:
        vertlist[8] = VertexInterp(isolevel,cubes[index].corner1,cubes[index].corner5,scalar_field[cubes[index].val1],scalar_field[cubes[index].val5])
    if tables.edgeTable[cubeindex] & 512:
        vertlist[9] = VertexInterp(isolevel,cubes[index].corner2,cubes[index].corner6,scalar_field[cubes[index].val2],scalar_field[cubes[index].val6])
    if tables.edgeTable[cubeindex] & 1024:
        vertlist[10] = VertexInterp(isolevel,cubes[index].corner3,cubes[index].corner7,scalar_field[cubes[index].val3],scalar_field[cubes[index].val7])
    if tables.edgeTable[cubeindex] & 2048:
        vertlist[11] = VertexInterp(isolevel,cubes[index].corner4,cubes[index].corner8,scalar_field[cubes[index].val4],scalar_field[cubes[index].val8])
    i = 0
    while tables.triTable[cubeindex][i] != -1:
        triangles[index * 15 + (i + 0)] = vertlist[tables.triTable[cubeindex][i]]
        triangles[index * 15 + (i + 1)] = vertlist[tables.triTable[cubeindex][i+1]]
        triangles[index * 15 + (i + 2)] = vertlist[tables.triTable[cubeindex][i+2]]
        i += 3
    
    # while tables.triTable[cube_indexes[index]][i] != -1:
    #     triangles[numtriag].
def VertexInterp(isolevel,p1,p2,valp1,valp2):
    mu = 0.0
    p = ti.Vector([0.0,0.0,0.0])

    if ti.abs(isolevel-valp1) < 0.00001:
        return p1
    if ti.abs(isolevel-valp2) < 0.00001:
        return p2
    if ti.abs(valp1 - valp2) < 0.00001:
        return p1

    mu = (isolevel - valp1) / (valp2 - valp1)
    p.x = p1.x + mu * (p2.x - p1.x)
    p.y = p1.y + mu * (p2.y - p1.y)
    p.z = p1.z + mu * (p2.z - p1.z)

    #print(p)
    return p

def marchingcubes():
    for i in range(0,num_cubes):
        create_iso_surface(i)

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
    update_field()
    #calc_cube_index()
    #create_iso_surface()

@ti.kernel
def init():
    initialize_mass_points()
@ti.kernel
def clear_triangles():
    for i in range(0,num_cubes*15):
        triangles[i] = [0.0,0.0,0.0]
    

camera.position(10,0, 100)
camera.lookat(0.0, 0.0, 0.0)

def ren():
    scene.particles(scalar_field_x, radius=0.1, color=(0.1, 0.6, 0.1))
    scene.particles(corners, radius=0.3, color=(0.5, 0.2, 0.2))
    scene.mesh(triangles, color=(0.5, 0.2, 0.2))
    #scene.mesh(cube, color=(0.5, 0.2, 0.2))
    scene.particles(x, radius=0.3 * 0.95, color=(0.2, 0.2, 0.8))
    #scene.particles(cubes, radius=0.3 * 0.95, color=(0.2, 0.2, 0.8))
    clear_triangles()
    canvas.scene(scene)
    window.show()

init_scalar_field()
init_cubes()
init()

cube = ti.Vector.field(3,dtype=float,shape=(12,))

ff = 340
cube[0] = cubes[ff].corner1
cube[1] = cubes[ff].corner2
cube[2] = cubes[ff].corner3
cube[3] = cubes[ff].corner1
cube[4] = cubes[ff].corner3
cube[5] = cubes[ff].corner4
cube[6] = cubes[ff].corner5
cube[7] = cubes[ff].corner6
cube[8] = cubes[ff].corner7
cube[9] = cubes[ff].corner5
cube[10] = cubes[ff].corner7
cube[11] = cubes[ff].corner8




# triangles[0] = [20,20,0]
# triangles[1] = [30,20,0]
# triangles[2] = [10,30,0]
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
    marchingcubes()
    ren()

    #print(scalar_field[1])
    #scene.particles(scalar_field, radius=0.3 * 0.95, color=(0.5, 0.42, 0.8))
    last_frame = datetime.now()
    #print(dt)

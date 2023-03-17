import taichi as ti
from datetime import datetime
import numpy as np
import math
import random
import tables

ti.init(arch=ti.gpu)

camera_active = False

bound_damping = -0.5

cell_size = 2.51
cell_recpr = 1.0 / cell_size

window = ti.ui.Window("Taichi sim on GGUI", (1024, 800),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

camera.position(-0.18,52.7, 167.5)
camera.lookat(-0.2, 52.0, 166.5)

dim = 3
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0

#bounds

left_bound = -15.0
right_bound = 15.0
bottom_bound = 0.0
top_bound = 40.0
back_bound = 15.0
front_bound = -15.0

box_width = int(np.abs(left_bound - right_bound))
box_height = int(np.abs(bottom_bound - top_bound))
box_depth = int(np.abs(front_bound - back_bound))

#particle initial space

p_left_bound = -10.0
p_right_bound = 10.0
p_bottom_bound = 5.0
p_top_bound = 45.0
p_back_bound = 10.0
p_front_bound = -10.0

h = 1.1
diameter = h

num_paticles_x = int(np.abs(p_left_bound - p_right_bound) / diameter)
num_paticles_y = int(np.abs(p_bottom_bound - p_top_bound) / diameter)
num_paticles_z = int(np.abs(p_front_bound - p_back_bound) / diameter)

total_particles = int(num_paticles_x*num_paticles_y*num_paticles_z)

print("Total number of particles: ", total_particles)

v = ti.Vector.field(dim,dtype=float,shape=(total_particles,))


mass = 1.0
rho = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.1
corrK = 0.01
gravity = ti.Vector([0.0, -9.8, 0.0])
neighbor_radius = h * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)

grid_size = (box_width,box_height,box_depth)

ti.root.dense(ti.i, total_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, total_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, total_particles).place(lambdas, position_deltas)

@ti.func
def poly6_value(s,h):
    result = 0.0
    if s > 0 and s < h:
        x = (ti.pow(h,2) - ti.math.pow(s,2)) / ti.pow(h,3)
        poly6_factor = 315.0/(64.0*ti.math.pi*ti.pow(h,9))
        result = poly6_factor * ti.pow(x,3)
    return result

@ti.func
def spiky_gradient(r,h):
    result = ti.Vector([0.0,0.0,0.0])
    len = r.norm()
    if len > 0 and len < h:
        x = (h - len) / (ti.pow(h,3))
        fact = spiky_grad_factor * ti.pow(x,2)
        result = r * fact / len
    return result

@ti.func
def compute_scorr(pos_ij):
    x = poly6_value(pos_ij.norm(),h) / poly6_value(corr_deltaQ_coeff * h,h)
    x = ti.pow(x,4)
    result = -(corrK) * x
    return result

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1] and 0 <= c[2] and c[2] < grid_size[2]

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
                positions[index] = [(i*diameter + p_left_bound) + ti.random() / 10,
                 j*diameter + p_bottom_bound,
                 k*diameter + p_front_bound + ti.random() / 10]
                v[index] = [0.0,0.0,0.0]

def keyboard_handling():
    global camera_active
    if window.get_event(tag=ti.ui.PRESS):
        if window.event.key == ' ':
            camera_active = not camera_active
        if window.event.key == 'r':
            init()
            
@ti.func
def boundary_check(pos): 
    if(pos[0] < left_bound):
        pos[0] = left_bound
    if(pos[0] > right_bound):
        pos[0] = right_bound
    if(pos[1] < bottom_bound):
        pos[1] = bottom_bound
    if(pos[1] > top_bound):
        pos[1] = top_bound
    if(pos[2] > back_bound):
        pos[2] = back_bound
    if(pos[2] < front_bound):
        pos[2] = front_bound
    return pos

@ti.kernel
def prolouge():
    # save old particle positions
    for i in positions:
        old_positions[i] = positions[i]

    # apply external forces to particle and update positions
    for i in positions:
        pos = positions[i]
        vel = velocities[i]
        vel += gravity * time_delta
        pos += vel * time_delta
        positions[i] = boundary_check(pos)
        #positions[i] = pos

    # clear the particles stored neighbours and num particles in cell
    for i in ti.grouped(grid_num_particles):
        grid_num_particles[i] = 0
    for i in ti.grouped(particle_neighbors):
        particle_neighbors[i] = -1

    # set particle to appropriate cell
    # for i in positions:
    #     cell = get_cell(position_deltas[i])
    #     offs = ti.atomic_add(grid_num_particles[cell], 1)
    #     grid2particles[cell, offs] = i
    
    # for i in positions:
    #     p_i = positions[i]
    #     cell = get_cell(p_i)
    #     for offs in ti.static(ti.grouped(ti.ndrange((-1,2),(-1,2),(-1,2)))):
    #         cell_to_check = cell + offs

    # find a particels neighbours
    for i in positions:
        n_i = 0
        for j in range(0,total_particles):
            if i != j and n_i < max_num_neighbors and (positions[i] - positions[j]).norm() < neighbor_radius:
                particle_neighbors[i,n_i] = j
                n_i += 1
        particle_num_neighbors[i] = n_i

@ti.kernel
def substep():
    for i in positions:
        p_i = positions[i]
        grad_i = ti.Vector([0.0,0.0,0.0])
        sum_grad_sqr = 0.0
        density_const = 0.0

        for j in range(particle_num_neighbors[i]):
            j_i = particle_neighbors[i,j]
            if j_i < 0:
                break
            p_j = positions[j_i]
            pos_ij = p_i - p_j
            grad_j = spiky_gradient(pos_ij,h)
            grad_i += grad_j
            sum_grad_sqr += grad_j.dot(grad_j)
            density_const += poly6_value(pos_ij.norm(), h)

        density_const = (mass * density_const/rho) - 1.0

        sum_grad_sqr += grad_i.dot(grad_i)
        lambdas[i] = -(density_const)/ (sum_grad_sqr + lambda_epsilon)
    
    for i in positions:
        p_i = positions[i]
        lambda_i = lambdas[i]
        pos_delta_i  =ti.Vector([0.0,0.0,0.0])
        for j in range(particle_num_neighbors[i]):
            j_i = particle_neighbors[i,j]
            if j_i < 0:
                break
            p_j = positions[j_i]
            lambda_j = lambdas[j_i]
            pos_ij = p_i - p_j
            scorr = compute_scorr(pos_ij)
            pos_delta_i += (lambda_i + lambda_j + scorr) * \
                spiky_gradient(pos_ij,h)
        
        pos_delta_i /= rho
        position_deltas[i] = pos_delta_i
    
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def epilouge():
    for i in positions:
        p_i = positions[i]
        positions[i] = boundary_check(p_i)
        #positions[i] = p_i
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i])/time_delta

numIters = 7
def PBF():
    prolouge()
    for i in range(numIters):
        substep()
    epilouge()

def ren():
    # scene.particles(scalar_field_x, radius=0.1, color=(0.1, 0.6, 0.1))
    # scene.particles(corners, radius=0.3, color=(0.5, 0.2, 0.2))
    # scene.mesh(triangles, color=(0.5, 0.2, 0.2))
    #scene.mesh(cube, color=(0.5, 0.2, 0.2))
    scene.particles(positions, radius=0.2, color=(0.2,0.2,0.8))
    #scene.particles(cubes, radius=0.3 * 0.95, color=(0.2, 0.2, 0.8))
    #clear_triangles()
    canvas.scene(scene)
    window.show()

@ti.kernel
def init():
    initialize_mass_points()

init()
print("Finished intializing")

camera.position(-0.18,52.7, 167.5)
camera.lookat(0.0, 0.0, 0.0)
while window.running:
    keyboard_handling()
    if camera_active:
        camera.track_user_inputs(window,movement_speed=1)
    scene.set_camera(camera)
    scene.point_light(pos=(10, 10, 20), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    PBF()
    ren()
    #print(b[10])
    #marchingcubes()
    #print(boundary[10])
    #print(scalar_field[1])
    #scene.particles(scalar_field, radius=0.3 * 0.95, color=(0.5, 0.42, 0.8))
    last_frame = datetime.now()
    #print(dt)

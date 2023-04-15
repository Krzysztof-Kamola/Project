import taichi as ti
import taichi_glsl as ts
from datetime import datetime
import numpy as np
import math
import random
import tables

ti.init(arch=ti.gpu)

camera_active = False

bound_damping = -0.5

window = ti.ui.Window("Taichi sim on GGUI", (1024, 800),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 0.6))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()

camera.position(-0.18,52.7, 167.5)
camera.lookat(-0.2, 52.0, 166.5)

dim = 3
bg_color = 0x112f41
particle_color = ti.Vector([0.2,0.2,0.8])
surface_color = ti.Vector([0.8,0.2,0.2])
diffuse_color = ti.Vector([1.0,1.0,1.0])
max_num_particles_per_cell = 100
max_num_neighbors = 100
dt = 1.0 / 60.0
epsilon = 1e-5
particle_radius = 0.25

#bounds

left_bound = 0.0
right_bound = 15.0
bottom_bound = 0.0
top_bound = 40.0
back_bound = 15.0
front_bound = 0.0

origin = ti.Vector([left_bound, bottom_bound, front_bound])

box_width = int(np.abs(left_bound - right_bound))
box_height = int(np.abs(bottom_bound - top_bound))
box_depth = int(np.abs(front_bound - back_bound))

#particle initial space

p_left_bound =  0.0
p_right_bound = 10.0
p_bottom_bound = 0.0
p_top_bound = 10.0
p_back_bound = 10.0
p_front_bound = 0.0

h = 1.1
diameter = h * 0.6

num_paticles_x = int(np.abs(p_left_bound - p_right_bound) / diameter)
num_paticles_y = int(np.abs(p_bottom_bound - p_top_bound) / diameter)
num_paticles_z = int(np.abs(p_front_bound - p_back_bound) / diameter)

total_particles = int(num_paticles_x*num_paticles_y*num_paticles_z)

print("Total number of particles: ", total_particles)

neighbor_radius = h * 1.1
cell_size = 2 * neighbor_radius

num_cells_x = int(np.abs(left_bound - right_bound) / cell_size)
num_cells_y = int(np.abs(bottom_bound - top_bound) / cell_size)
num_cells_z = int(np.abs(front_bound - back_bound) / cell_size)

num_cells = int(num_cells_x*num_cells_y*num_cells_z)

print("Number of cells: ", num_cells)

v = ti.Vector.field(dim,dtype=float,shape=(total_particles,))

mass = 1.0
rho = 1.0
lambda_epsilon = 100.0
num_iters = 10
corr_deltaQ_coeff = 0.1
corrK = 0.01
gravity = ti.Vector([0.0, -9.8, 0.0])
neighbor_radius = h * 1.1
vorticity_eps = 0.01
viscocity_const = 0.1
rV = h # volume radius of a fluid particle
density_threshold = 0.02 # Used for find surface particles

k_ta = 100.0
k_wc = 100.0

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
max_diffuse_particles = total_particles * 100

# Simulation data
diffuse_particles = ti.Struct.field({
    "pos": ti.math.vec3,
    "vel": ti.math.vec3,
    "lifespan": float,
    "active": int,
  }, shape=(max_diffuse_particles,))

diffuse_particles_copy = ti.Struct.field({
    "pos": ti.math.vec3,
    "vel": ti.math.vec3,
    "lifespan": float,
    "active": int,
  }, shape=(max_diffuse_particles,))

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)

grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)

lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
surface = ti.field(int,shape=total_particles)
cover_vector = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
colour = ti.Vector.field(dim,dtype=float,shape=(total_particles,))
scorr_list = ti.field(dtype=float,shape=total_particles)
density = ti.field(float,total_particles)
vorticity = ti.Vector.field(dim,dtype=float, shape=total_particles)
surface_normals = ti.Vector.field(dim,dtype=float, shape=total_particles)
densities = ti.field(dtype=float)

#grid_cells_x = ti.Vector.field(dim,dtype=float,shape=(num_cells,))
num_diffuse_particles = ti.field(dtype=int, shape=total_particles)
diffuse_positions = ti.Vector.field(dim,float)
diffuse_velocities = ti.Vector.field(dim,float)
diffuse_lifetime = ti.field(dtype=float)
diffuse_active = ti.field(dtype=int)
diffuse_count = ti.field(dtype=int,shape=1)

#grid_num_diffuse = ti.field(dtype=int)
diffuse_neighbors = ti.field(dtype=int)
diffuse_num_neighbors = ti.field(dtype=int)
#grid2diffuse = ti.field(int)

grid_size = (box_width,box_height,box_depth)

ti.root.dense(ti.i, total_particles).place(old_positions, positions, velocities, densities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, total_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, total_particles).place(lambdas, position_deltas)
ti.root.dense(ti.i, max_diffuse_particles).place(diffuse_positions,diffuse_velocities,diffuse_lifetime,diffuse_active)
nb_dnode = ti.root.dense(ti.i, max_diffuse_particles)
nb_dnode.place(diffuse_num_neighbors)
nb_dnode.dense(ti.j, max_num_neighbors).place(diffuse_neighbors)

@ti.func
def poly6_value(s,h):
    result = 0.0
    if s > 0 and s < h:
        x = ti.pow((ti.pow(h,2) - ti.pow(s,2)) / ti.pow(h,3),3)
        poly6_factor = 315.0/(64.0*ti.math.pi*ti.pow(h,9))
        result = poly6_factor * x

        # x = ti.pow((ti.pow(h,2) - ti.math.pow(s,2)),3)
        # #poly6_factor = 315.0/(64.0*ti.math.pi*ti.pow(h,9))
        # result = poly6_factor * x
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
    k = 0.001
    delta_q = 0.2 * h
    n = 4
    x = poly6_value(pos_ij.norm(),h) / poly6_value(delta_q,h)
    x = ti.pow(x,n)
    result = -(k) * x
    return result

@ti.func
def phi(I,t_min,t_max):
    return (ti.min(I, t_max) - ti.min(I, t_min)) / (t_max - t_min)

@ti.func
def weighting(x_ij, h):
    return 1 - x_ij.norm() / h if x_ij.norm() <= h else 0

@ti.func
def is_in_grid(c):
    return left_bound <= c[0] and c[0] < right_bound and bottom_bound <= c[1] and c[1] < top_bound and front_bound <= c[2] and c[2] < back_bound

# @ti.func
# def get_cell(pos):
#     x_i = ti.floor(pos[0]/cell_size)
#     y_i = ti.floor(pos[1]/cell_size)
#     z_i = ti.floor(pos[2]/cell_size)
#     index = int((x_i*num_cells_y + y_i) * num_cells_z + z_i)
#     return index



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
                velocities[index] = [0.0,0.0,0.0]


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

@ti.func
def get_cell(pos):
    return int(ti.floor(pos / cell_size))

@ti.func
def set_grid():
    grid_num_particles.fill(0)
    particle_neighbors.fill(-1)
    particle_num_neighbors.fill(0)

    for i in positions:
        cell = get_cell(positions[i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = i

    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


    # grid_num_particles.fill(0)
    # particle_neighbors.fill(-1)
    # particle_num_neighbors.fill(0)

    # count = 0
    # cell = ti.Vector([0,0,0])
    # d_cell = ti.Vector([0,0,0])
    # nb_i = 0
    # d_nb_i = 0
    # pos_i = ti.Vector([0.0,0.0,0.0])
    # d_pos_i = ti.Vector([0.0,0.0,0.0])

    # if diffuse_count[0] > total_particles:
    #     count = diffuse_count[0]
    # else:
    #     count = total_particles

    # for i in range(count):
    #     if i < total_particles:
    #         cell = get_cell(positions[i])
    #         offs = ti.atomic_add(grid_num_particles[cell], 1)
    #         grid2particles[cell, offs] = i

    #     if i < diffuse_count[0]:
    #         cell = get_cell(diffuse_particles.pos[i])
    #         offs = ti.atomic_add(grid_num_diffuse[cell], 1)
    #         grid2diffuse[cell, offs] = i

    # for p_i in range(count):
    #     if p_i < total_particles:
    #         pos_i = positions[p_i]
    #         cell = get_cell(pos_i)
    #         nb_i = 0

    #     if p_i < diffuse_count[0]:
    #         d_pos_i = diffuse_particles.pos[p_i]
    #         d_cell = get_cell(d_pos_i)
    #         d_nb_i = 0

    #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
    #         if p_i < total_particles:
    #             cell_to_check = cell + offs
    #             if is_in_grid(cell_to_check):
    #                 for j in range(grid_num_particles[cell_to_check]):
    #                     p_j = grid2particles[cell_to_check, j]
    #                     if nb_i < max_num_neighbors and (
    #                             pos_i - positions[p_j]).norm() < neighbor_radius:
    #                         particle_neighbors[p_i, nb_i] = p_j
    #                         nb_i += 1
    #         particle_num_neighbors[p_i] = nb_i

    #         if p_i < diffuse_count[0]:
    #             d_cell_to_check = d_cell + offs
    #             if is_in_grid(d_cell_to_check):
    #                 for j in range(grid_num_particles[d_cell_to_check]):
    #                     p_j = grid2particles[d_cell_to_check, j]
    #                     if d_nb_i < max_num_neighbors and (
    #                             d_pos_i - positions[p_j]).norm() < neighbor_radius:
    #                         diffuse_neighbors[p_i, d_nb_i] = p_j
    #                         d_nb_i += 1
    #         diffuse_num_neighbors[p_i] = d_nb_i

@ti.func
def init_cells():
    for i in range(0,num_cells_x-1):
        for j in range(0,num_cells_y-1):
            for k in range(0,num_cells_z-1):
                index = int((k* (num_cells_x-1) * (num_cells_y-1)) + (j * (num_cells_x-1)) + i)
                #index2 = int((k* num_points_x * num_points_y) + (j * num_points_x) + i)
                grid_cells_x[index] = [(i*grid_size + p_left_bound),
                 j*grid_size + p_bottom_bound,
                 k*grid_size + p_front_bound]
                
@ti.func
def save_old_positions():
    for i in positions:
        old_positions[i] = positions[i]

@ti.func
def apply_forces():
    for i in positions:
        pos = positions[i]
        vel = velocities[i]
        vel += gravity * dt
        pos += vel * dt
        positions[i] = boundary_check(pos)

@ti.kernel
def PBF_first_step():
    save_old_positions()
    apply_forces()

    # Note grid optimisation doesnt work properly when dt is too small
    set_grid()

    # find a particels neighbours
    # for i in positions:
    #     n_i = 0
    #     for j in range(0,total_particles):
    #         if i != j and n_i < max_num_neighbors and (positions[i] - positions[j]).norm() < neighbor_radius:
    #             rij = positions[i] - positions[j]
    #             cover_vector[i] += ti.math.normalize(rij)
    #             particle_neighbors[i,n_i] = j
    #             n_i += 1
    #     particle_num_neighbors[i] = n_i

@ti.kernel
def substep():
    for i in positions:
        p_i = positions[i]
        grad_i = ti.Vector([0.0,0.0,0.0])
        sum_grad_sqr = 0.0
        density = 0.0

        for j in range(particle_num_neighbors[i]):
            j_i = particle_neighbors[i,j]
            if j_i < 0:
                break
            p_j = positions[j_i]
            pos_ij = p_i - p_j
            grad_j = spiky_gradient(pos_ij,h)
            grad_i += grad_j
            sum_grad_sqr += grad_j.dot(grad_j)
            density += poly6_value(pos_ij.norm(), h)

        density_constraint = (mass * density/rho) - 1.0

        sum_grad_sqr += grad_i.dot(grad_i)
        lambdas[i] = -(density_constraint)/ (sum_grad_sqr + lambda_epsilon)
    
    for i in positions:
        p_i = positions[i]
        lambda_i = lambdas[i]
        pos_delta_i = ti.Vector([0.0,0.0,0.0])
        scorr_list[i] = 0.0
        for j in range(particle_num_neighbors[i]):
            j_i = particle_neighbors[i,j]
            if j_i < 0:
                break
            p_j = positions[j_i]
            lambda_j = lambdas[j_i]
            pos_ij = p_i - p_j
            scorr = compute_scorr(pos_ij)
            scorr_list[i] += scorr
            pos_delta_i += (lambda_i + lambda_j + scorr) * \
                spiky_gradient(pos_ij,h)
        
        pos_delta_i /= rho
        position_deltas[i] = pos_delta_i
    
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def vorticity_confinement():
    for i in positions:
        pos_i = positions[i]
        vorticity[i] = pos_i * 0.0
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            vorticity[i] += mass * (velocities[p_j] - velocities[i]).cross(spiky_gradient(pos_ji, h))

    for i in positions:
        pos_i = positions[i]
        loc_vec_i = ti.Vector([0.0,0.0,0.0])
        for j in range(particle_num_neighbors[i]):
            p_j = particle_neighbors[i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            loc_vec_i += mass * vorticity[p_j].norm() * spiky_gradient(pos_ji, h)
        vorticity_i = vorticity[i]
        # loc_vec_i += mass * omega_i.norm() * spiky_gradient(pos_i * 0.0, h) / (epsilon + density[i])
        loc_vec_i = loc_vec_i / (epsilon + loc_vec_i.norm())
        velocities[i] += (vorticity_eps * loc_vec_i.cross(vorticity_i))/mass * dt

@ti.kernel
def viscocity():
    for i in positions:
        p_i = positions[i]
        v_i = velocities[i]
        v_delta_i = ti.Vector([0.0,0.0,0.0])

        for j in range(0,particle_num_neighbors[i]):
            j_i = particle_neighbors[i,j]
            if j_i >= 0:
                p_j = positions[j_i]
                v_j = velocities[j_i]
                p_ij = p_i - p_j
                v_ji = v_j - v_i
                v_delta_i += v_ji * poly6_value(p_ij.norm(),h)
        
        velocities[i] += viscocity_const * v_delta_i

@ti.kernel
def compute_density():
    for p_i in positions:
        pos_i = positions[p_i]
        density = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            density += poly6_value(pos_ji.norm(), h)

        density = (mass * density / rho) - 1.0
        densities[p_i] = density

@ti.kernel
def find_surface_particles():
    for p_i in positions:
        density = densities[p_i]
        if (density) < density_threshold:
            surface[p_i] = 1
        else:
            surface[p_i] = 0


@ti.kernel
def compute_surface_normals():
    for p_i in positions:
        if surface[p_i] == 1:  # Only compute normals for surface particles
            pos_i = positions[p_i]
            normal = ti.Vector([0.0, 0.0, 0.0])

            for j in range(particle_num_neighbors[p_i]):
                p_j = particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - positions[p_j]
                normal += spiky_gradient(pos_ji, h)

            surface_normals[p_i] = normal.normalized()
        else:
            surface_normals[p_i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def epilouge():
    for i in positions:
        p_i = positions[i]
        positions[i] = boundary_check(p_i)
                
    for i in positions:
        if surface[i] == 0:
            colour[i] = particle_color
        else:
            colour[i] = surface_color
        velocities[i] = (positions[i] - old_positions[i])/dt

numIters = 10
def PBF():
    PBF_first_step()
    for i in range(numIters):
        substep()
    epilouge()
    vorticity_confinement()
    viscocity()
    compute_density()
    find_surface_particles()
    compute_surface_normals()


# Diffuse particle code
# Based on the paper:
# "Unified Spray, Foam and Bubbles for Particle-Based Fluids" by Markus Ihmsen et al.
@ti.kernel
def compute_num_diffuse_particles():
    for p_i in positions:
        if surface[p_i] == 1:
            v_diff_i = 0.0
            kappa_i = 0.0
            for j in range(particle_num_neighbors[p_i]):
                p_j = particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                if surface[p_j] == 1:
                    pos_ij = positions[p_i] - positions[p_j]
                    pos_ji = positions[p_j] - positions[p_i]
                    vel_ij = velocities[p_i] - velocities[p_j]
                    vel_n = vel_ij.normalized()
                    pos_ij_n = pos_ij.normalized()
                    pos_ji_n = pos_ji.normalized()
                    # Eq 2
                    v_diff_i += vel_ij.norm() * (1 - vel_n.dot(pos_ij_n)) * weighting(pos_ij,h)
                    # Eq 4
                    kappa_ij = (1 - surface_normals[p_i].dot(surface_normals[p_j])) * weighting(pos_ij, h)
                    # Eq 6
                    if pos_ji_n.dot(surface_normals[p_i]) < 0:
                        kappa_i += kappa_ij

            # Eq 1 for each potential 
            I_ta = phi(v_diff_i, 2, 8)
            v_i_n = velocities[p_i].normalized()
            # Eq 7
            delta_vn_i = 0 if v_i_n.dot(surface_normals[p_i]) < 0.6 else 1

            I_wc = phi(kappa_i * delta_vn_i, 5, 20)
            E_k_i = 0.5 * mass * velocities[p_i].dot(velocities[p_i])
            I_k = phi(E_k_i, 5, 50)

            # Eq 8
            num_diffuse_particles[p_i] = int(I_k * ((k_ta * I_ta) + (k_wc * I_wc)) * dt)

@ti.kernel
def gen_diffuse_particles():
    for p_i in positions:
        for i in range(num_diffuse_particles[p_i]):
            if diffuse_count[0] < max_diffuse_particles:
                index = ti.atomic_add(diffuse_count[0], 1)
                X_r = ti.random()
                X_theta = ti.random()
                X_h = ti.random()
                vel = velocities[p_i]
                pos = positions[p_i]
                # Compute the radial distance, azimuth, and height within the cylinder
                r = rV * ti.sqrt(X_r)
                theta = X_theta * 2 * ti.pi
                h = X_h * (vel * dt).norm()

                # Compute the orthogonal basis for the reference plane
                e1_prime = ti.Vector([0, 0, 0])
                e2_prime = ti.Vector([0, 0, 0])
                if vel.norm() > 0:
                    e1_prime = ti.Vector([vel[1], -vel[0], 0]).normalized()
                    e2_prime = vel.cross(e1_prime).normalized()

                # Compute the position and velocity of the diffuse particle
                diffuse_particles[index].pos = pos + (r * ti.cos(theta) * e1_prime) + (r * ti.sin(theta) * e2_prime) + (h * vel.normalized())
                diffuse_particles[index].vel = (r * ti.cos(theta) * e1_prime) + (r * ti.sin(theta) * e2_prime) + vel
                diffuse_particles[index].lifespan = 10.0
                diffuse_particles[index].active = 1

@ti.kernel
def dissolution():
    for i in range(diffuse_count[0]):
        diffuse_particles[i].lifespan -= dt
        if diffuse_particles.lifespan[i] <= 0:
            diffuse_particles[i].pos = ti.Vector([0.0, 0.0, 0.0])
            diffuse_particles[i].vel = ti.Vector([0.0, 0.0, 0.0])
            diffuse_particles[i].lifespan = 0.0
            diffuse_particles[i].active = 0
    
@ti.kernel
def squash_array():
    count = 0
    for i in range(diffuse_count[0]):
        if diffuse_particles[i].active == 0:
            diffuse_particles[i] = diffuse_particles_copy[diffuse_count[0]+1]
            ti.atomic_add(count,1)
    diffuse_count[0] -= count

@ti.func
def find_diffuse_neighbours():
    # grid_num_diffuse.fill(0)
    # diffuse_neighbors.fill(-1)
    # diffuse_num_neighbors.fill(0)

    # for i in range(diffuse_count[0]):
    #     cell = get_cell(diffuse_positions[i])
    #     offs = ti.atomic_add(grid_num_diffuse[cell], 1)
    #     grid2diffuse[cell, offs] = i

    # for p_i in range(diffuse_count[0]):
    #     pos_i = diffuse_positions[p_i]
    #     cell = get_cell(pos_i)
    #     nb_i = 0
    #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
    #         cell_to_check = cell + offs
    #         if is_in_grid(cell_to_check):
    #             for j in range(grid_num_particles[cell_to_check]):
    #                 p_j = grid2particles[cell_to_check, j]
    #                 if nb_i < max_num_neighbors and (
    #                         pos_i - positions[p_j]).norm() < neighbor_radius:
    #                     diffuse_neighbors[p_i, nb_i] = p_j
    #                     nb_i += 1
    #     diffuse_num_neighbors[p_i] = nb_i


    # grid_num_diffuse.fill(0)
    # diffuse_neighbors.fill(-1)
    # diffuse_num_neighbors.fill(0)

    # for p_i in range(diffuse_count[0]):
    #     pos_i = diffuse_particles.pos[p_i]
    #     cell = get_cell(pos_i)
    #     nb_i = 0
    #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
    #         cell_to_check = cell + offs
    #         if is_in_grid(cell_to_check):
    #             for j in range(grid_num_particles[cell_to_check]):
    #                 p_j = grid2particles[cell_to_check, j]
    #                 if nb_i < max_num_neighbors and p_j != p_i and (
    #                         pos_i - positions[p_j]).norm() < neighbor_radius:
    #                     diffuse_neighbors[p_i, nb_i] = p_j
    #                     nb_i += 1
    #     diffuse_num_neighbors[p_i] = nb_i

    diffuse_neighbors.fill(-1)
    diffuse_num_neighbors.fill(0)

    for p_i in range(diffuse_count[0]):
        pos_i = diffuse_particles.pos[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for p_j in range(total_particles):
            if nb_i < max_num_neighbors and (
                    pos_i - positions[p_j]).norm() < neighbor_radius:
                diffuse_neighbors[p_i, nb_i] = p_j
                nb_i += 1
        diffuse_num_neighbors[p_i] = nb_i
                                

@ti.kernel
def advect_particles():
    for i in range(max_diffuse_particles):
        diffuse_num_neighbors[i] = 0
        for j in range(max_num_neighbors):
            diffuse_neighbors[i,j] = -1

    for p_i in range(max_diffuse_particles):
        if(diffuse_particles[p_i].active == 1):
            pos_i = diffuse_particles[p_i].pos
            nb_i = 0
            for p_j in range(total_particles):
                if nb_i < max_num_neighbors and (
                        pos_i - positions[p_j]).norm() < neighbor_radius:
                    diffuse_neighbors[p_i, nb_i] = p_j
                    nb_i += 1
            diffuse_num_neighbors[p_i] = nb_i
            if nb_i > 20:
                pass
            elif nb_i < 10:
                pass
            else:
                pass


    

    for i in range(diffuse_count[0]):
        # Advection for spray particles. Simple ballistic motion
        if True:
            diffuse_particles[i].vel += gravity * dt
            diffuse_particles[i].pos += diffuse_particles[i].vel * dt
            diffuse_particles[i].pos = boundary_check(diffuse_particles[i].pos)
        else:
            sum_fluid_vel = ti.Vector([0.0, 0.0, 0.0])
            for j in range(diffuse_num_neighbors[i]):
                j = diffuse_neighbors[i,j]
                pos_i = diffuse_particles[i].pos
                pos_j = positions[j]
                vel_j = velocities[j]

                avg_fluid_vel += vel_j * K(pos_i - pos_j, h)
                denom += K(pos_i - pos_j, h)
            avg_fluid_vel /= denom

            # Advection for foam particles
            if False:
                # Calculate velocity of foam. Average velocity of surrounding fluid particles
                # v_new = (Sum (fluid vel(t + dt) * W(diffuse pos - fluid pos, h)) /
                # (Sum (W(diffuse pos - fluid pos, h)))

                diffuse_particles[i].pos = diffuse_particles[i].pos + (dt * avg_fluid_vel)

            # Advection for bubbles
            if False:
                # find velocity of bubble
                # diffuse vel + dt * (-buoyancy*gravity + drag*((v_new - bubble vel)/dt))
                v_bub = diffuse_particles[i].vel + (dt * (-buoyancy * gravity + drag * ((avg_fluid_vel - diffuse_particles[i].vel) / dt)))
                diffuse_particles[i].pos = diffuse_particles[i].pos + (dt * v_bub)
                pass
    
    num_diffuse_particles.fill(0)

def dissolution_full():
    dissolution()
    diffuse_particles_copy.pos.copy_from(diffuse_particles.pos)
    diffuse_particles_copy.vel.copy_from(diffuse_particles.vel)
    diffuse_particles_copy.lifespan.copy_from(diffuse_particles.lifespan)
    diffuse_particles_copy.active.copy_from(diffuse_particles.active)
    squash_array()

def diffuse():
    advect_particles()
    compute_num_diffuse_particles()
    gen_diffuse_particles()
    dissolution_full()

def ren():
    diffuse_positions.copy_from(diffuse_particles.pos)
    #scene.particles(grid_cells_x, radius=0.5, color=(0.1, 0.6, 0.1))
    # scene.particles(corners, radius=0.3, color=(0.5, 0.2, 0.2))
    # scene.mesh(triangles, color=(0.5, 0.2, 0.2))
    #scene.mesh(cube, color=(0.5, 0.2, 0.2))
    #ti.surfaceMaterials(scene, triangles, material)


    scene.particles(positions, radius=particle_radius, per_vertex_color=colour)

    diffuse_positions.copy_from(diffuse_particles.pos)
    scene.particles(diffuse_positions, radius=particle_radius*0.5, color=(1.0,1.0,1.0),index_count=diffuse_count[0])


    #scene.particles(cubes, radius=0.3 * 0.95, color=(0.2, 0.2, 0.8))
    #clear_triangles()
    canvas.scene(scene)
    window.show()

def simulation():
    PBF()
    diffuse()

def cam():
    if camera_active:
        camera.track_user_inputs(window,movement_speed=1)
    scene.set_camera(camera)
    scene.point_light(pos=(10, 10, 20), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

@ti.kernel
def init():
    initialize_mass_points()
    #init_cells()

init()
print("Finished intializing")
frame = 0
camera.position(-0.18,42.7, 100.5)
camera.lookat(0.0, 0.0, 0.0)
while window.running:
    keyboard_handling()
    cam()

    simulation()

    # print("Num diffuse:",diffuse_count[0])

    ren()

    if frame % 20 == 0:
        print("Frame:", frame)
        num = diffuse_num_neighbors.to_numpy()
        max_,min_,avg= np.max(num), np.min(num), np.mean(num)
        print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
        print("Num diffuse:",diffuse_count[0])

    frame += 1
    # print(diffuse_num_neighbors.to_numpy())

    #marchingcubes()
    #print(boundary[10])
    #print(scalar_field[1])
    #scene.particles(scalar_field, radius=0.3 * 0.95, color=(0.5, 0.42, 0.8))
    #print(dt)

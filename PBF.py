import taichi as ti
import taichi_glsl as ts
from datetime import datetime
import numpy as np
import math
import random
import tables

# ti.init(arch=ti.gpu)

camera_active = False

@ti.data_oriented
class Simulator:
    dim = 3
    particle_color = ti.Vector([0.2,0.2,0.8])
    surface_color = ti.Vector([0.8,0.2,0.2])
    diffuse_color = ti.Vector([1.0,1.0,1.0])
    bound_damping = -0.5
    
    def __init__(self,max_num_particles_per_cell=100,max_num_neighbors=100,dt=1.0/60.0,iters=10,
                 epsilon=1e-3,particle_radius=0.25,left_bound=0.0,right_bound=20.0,bottom_bound=0.0,top_bound=40.0,back_bound=20.0,front_bound=0.0,
                 h=1.1,rho=1.0,corr_deltaQ_coeff=0.1,corrK=0.01,vorticity_eps=0.01,viscocity_const=0.1,
                 density_threshold=0.02,drag=0.4,buoyancy=0.9,k_ta=100,k_wc=100,max_particles=50000,max_diffuse_particles=1000,life_time=1.0):
        
        self.max_num_particles_per_cell = max_num_particles_per_cell
        self.max_num_neighbors = max_num_neighbors
        self.dt = dt # Change in time
        self.epsilon = epsilon # Relaxation parameter
        self.particle_radius = particle_radius
        self.iters = iters # Number of iterations
        self.max_particles = max_particles
        self.total_particles = ti.field(int,shape=1)
        self.total_particles[0] = 0
        
        #bounds

        self.left_bound = left_bound
        self.right_bound = right_bound
        self.bottom_bound = bottom_bound
        self.top_bound = top_bound
        self.back_bound = back_bound
        self.front_bound = front_bound

        self.origin = ti.Vector([self.left_bound, self.bottom_bound, self.front_bound])

        self.box_width = int(np.abs(self.left_bound - self.right_bound))
        self.box_height = int(np.abs(self.bottom_bound - self.top_bound))
        self.box_depth = int(np.abs(self.front_bound - self.back_bound))

        self.h = h # Particle influence radius
        self.diameter = self.h * 0.6
        self.neighbor_radius = self.h * 1.1
        self.cell_size = 2 * self.neighbor_radius
        
        self.mass = 1.0
        self.rho = rho # desnity
        
        #parameters for s_corr used for surface tension
        self.corr_deltaQ_coeff = corr_deltaQ_coeff
        self.corrK = corrK
        
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.vorticity_eps = vorticity_eps
        self.viscocity_const = viscocity_const
        self.density_threshold = density_threshold # Used for find surface particles
        self.rV = self.h # volume radius of a fluid particle
        
        # Bubble parameters
        self.drag = drag
        self.buoyancy = buoyancy

        self.k_ta = k_ta # number of diffuse particles generated per second
        self.k_wc = k_wc # number of diffuse particles generated at wave crests per second
        self.life_time = life_time

        self.poly6_factor = 315.0 / 64.0 / math.pi
        self.spiky_grad_factor = -45.0 / math.pi
        self.max_diffuse_particles = max_diffuse_particles

        # Simulation data
        self.diffuse_particles = ti.Struct.field({
            "pos": ti.math.vec3,
            "vel": ti.math.vec3,
            "lifespan": float,
            "active": int,
            "type": int,
        }, shape=(self.max_diffuse_particles,))

        self.diffuse_particles_copy = ti.Struct.field({
            "pos": ti.math.vec3,
            "vel": ti.math.vec3,
            "lifespan": float,
            "active": int,
            "type": int,
        }, shape=(self.max_diffuse_particles,))

        self.old_positions = ti.Vector.field(self.dim, float)
        self.positions = ti.Vector.field(self.dim, float)
        self.velocities = ti.Vector.field(self.dim, float)

        self.grid_num_particles = ti.field(int)
        self.grid2particles = ti.field(int)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)

        self.lambdas = ti.field(float)
        self.position_deltas = ti.Vector.field(self.dim, float)
        self.surface = ti.field(int,shape=self.max_particles)
        #self.cover_vector = ti.Vector.field(self.dim,dtype=float,shape=(self.max_particles,))
        self.colour = ti.Vector.field(self.dim,dtype=float,shape=(self.max_particles,))
        self.scorr_list = ti.field(dtype=float,shape=self.max_particles)
        self.density = ti.field(float,self.max_particles)
        self.vorticity = ti.Vector.field(self.dim,dtype=float, shape=self.max_particles)
        self.surface_normals = ti.Vector.field(self.dim,dtype=float, shape=self.max_particles)
        self.densities = ti.field(dtype=float)

        #grid_cells_x = ti.Vector.field(dim,dtype=float,shape=(num_cells,))
        self.num_diffuse_particles = ti.field(dtype=int, shape=self.max_particles)
        self.diffuse_positions = ti.Vector.field(self.dim,float)
        self.diffuse_velocities = ti.Vector.field(self.dim,float)
        self.diffuse_lifetime = ti.field(dtype=float)
        self.diffuse_active = ti.field(dtype=int)
        self.diffuse_count = ti.field(dtype=int,shape=1)

        #grid_num_diffuse = ti.field(dtype=int)
        self.diffuse_neighbors = ti.field(dtype=int)
        self.diffuse_num_neighbors = ti.field(dtype=int)
        #grid2diffuse = ti.field(int)

        self.cell_size = 2*self.neighbor_radius

        def round_up(f, s, c):
            return (math.floor(f * (1/c) / s) + 1) * s

        print("Box size:",self.box_width,self.box_height,self.box_depth)
        self.grid_size = (round_up(self.box_width,1,self.cell_size),round_up(self.box_height,1,self.cell_size),round_up(self.box_depth,1,self.cell_size))

        ti.root.dense(ti.i, self.max_particles).place(self.old_positions, self.positions, self.velocities, self.densities)
        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l, max_num_particles_per_cell).place(self.grid2particles)
        nb_node = ti.root.dense(ti.i, self.max_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.max_particles).place(self.lambdas, self.position_deltas)
        ti.root.dense(ti.i, self.max_diffuse_particles).place(self.diffuse_positions,self.diffuse_velocities,self.diffuse_lifetime,self.diffuse_active)
        nb_dnode = ti.root.dense(ti.i, self.max_diffuse_particles)
        nb_dnode.place(self.diffuse_num_neighbors)
        nb_dnode.dense(ti.j, self.max_num_neighbors).place(self.diffuse_neighbors)

    @ti.func
    def poly6_value(self,s,h):
        result = 0.0
        if s > 0 and s < h:
            x = ti.pow((ti.pow(h,2) - ti.pow(s,2)) / ti.pow(h,3),3)
            poly6_factor = 315.0/(64.0*ti.math.pi*ti.pow(h,9))
            result = poly6_factor * x
        return result

    @ti.func
    def spiky_gradient(self,r,h):
        result = ti.Vector([0.0,0.0,0.0])
        len = r.norm()
        if len > 0 and len < h:
            x = (h - len) / (ti.pow(h,3))
            fact = self.spiky_grad_factor * ti.pow(x,2)
            result = r * fact / len
        return result

    @ti.func
    def compute_scorr(self,pos_ij):
        k = 0.001
        delta_q = 0.2 * self.h
        n = 4
        x = self.poly6_value(pos_ij.norm(),self.h) / self.poly6_value(delta_q,self.h)
        x = ti.pow(x,n)
        result = -(k) * x
        return result

    @ti.func
    def phi(self,I,t_min,t_max):
        return (ti.min(I, t_max) - ti.min(I, t_min)) / (t_max - t_min)

    @ti.func
    def weighting(self,x_ij, h):
        return 1 - x_ij.norm() / h if x_ij.norm() <= h else 0

    @ti.func
    def cubic_spline(self,x, h):
        q = x.norm() / h
        coeff = 1 / (ti.pi * h * h * h)
        result = 0.0
        if q <= 1 and q >= 0:
            result = coeff * (1 - (1.5 * q * q) + (0.75 * q * q * q))
        elif q <= 2 and q >= 1:
            result = coeff * 0.25 * ti.pow(2 - q,3)
        return result

    @ti.func
    def is_in_grid(self,c):
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]

    @ti.kernel
    def spawn_fluid(self,origin: ti.types.vector(3,float),width: float,height: float,depth: float,spacing: float):
        
        num_particles_x = int(width / spacing)
        num_particles_y = int(height / spacing)
        num_particles_z = int(depth / spacing)

        num_particles = num_particles_x * num_particles_y * num_particles_z

        if self.total_particles[0] + num_particles < self.max_particles:
            for i in range(0,num_particles_x):
                for j in range(0,num_particles_y):
                    for k in range(0,num_particles_z):
                        index = self.total_particles[0] + (k* num_particles_x * num_particles_y) + (j * num_particles_x) + i
                        self.positions[index] = [(i*spacing + origin[0]) + ti.random() / 10,
                        j*spacing + origin[1],
                        k*spacing + origin[2] + ti.random() / 10]
                        self.velocities[index] = [0.0,0.0,0.0]
            self.total_particles[0] += num_particles
                
    @ti.func
    def boundary_check(self,pos): 
        if(pos[0] < self.left_bound):
            pos[0] = self.left_bound
        if(pos[0] > self.right_bound):
            pos[0] = self.right_bound
        if(pos[1] < self.bottom_bound):
            pos[1] = self.bottom_bound
        if(pos[1] > self.top_bound):
            pos[1] = self.top_bound
        if(pos[2] > self.back_bound):
            pos[2] = self.back_bound
        if(pos[2] < self.front_bound):
            pos[2] = self.front_bound
        return pos

    # PBF simulation based on the paper Position-Based-Fluids by Macklin
    # Solution also based of the 2D implementation by Ye Kuang
    @ti.func
    def get_cell(self,pos):
        return int(ti.floor(pos / self.cell_size))

    # find particle neighbours
    @ti.func
    def set_grid(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.particle_num_neighbors.fill(0)

        for i in range(self.total_particles[0]):
            cell = self.get_cell(self.positions[i])
            offs = ti.atomic_add(self.grid_num_particles[cell], 1)
            self.grid2particles[cell, offs] = i

        for p_i in range(self.total_particles[0]):
            pos_i = self.positions[p_i]
            cell = self.get_cell(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        p_j = self.grid2particles[cell_to_check, j]
                        if nb_i < self.max_num_neighbors and p_j != p_i and (
                                pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            self.particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i
                    
    @ti.func
    def save_old_positions(self):
        for i in range(self.total_particles[0]):
            self.old_positions[i] = self.positions[i]

    @ti.func
    def apply_forces(self):
        for i in range(self.total_particles[0]):
            pos = self.positions[i]
            vel = self.velocities[i]
            vel += self.gravity * self.dt
            pos += vel * self.dt
            self.positions[i] = self.boundary_check(pos)

    @ti.kernel
    def PBF_first_step(self):
        self.save_old_positions()
        self.apply_forces()
        self.set_grid()

    @ti.kernel
    def substep(self):
        for i in range(self.total_particles[0]):
            p_i = self.positions[i]
            grad_i = ti.Vector([0.0,0.0,0.0])
            sum_grad_sqr = 0.0
            density = 0.0

            for j in range(self.particle_num_neighbors[i]):
                j_i = self.particle_neighbors[i,j]
                if j_i < 0:
                    break
                p_j = self.positions[j_i]
                pos_ij = p_i - p_j
                grad_j = self.spiky_gradient(pos_ij,self.h)
                grad_i += grad_j
                sum_grad_sqr += grad_j.dot(grad_j)
                # Eq. 2
                density += self.poly6_value(pos_ij.norm(), self.h)

            # Eq. 1
            density_constraint = (self.mass * density/self.rho) - 1.0
            sum_grad_sqr += grad_i.dot(grad_i)
            # Eq. 11
            self.lambdas[i] = -(density_constraint)/ (sum_grad_sqr + self.epsilon)
        
        for i in range(self.total_particles[0]):
            p_i = self.positions[i]
            lambda_i = self.lambdas[i]
            pos_delta_i = ti.Vector([0.0,0.0,0.0])
            self.scorr_list[i] = 0.0
            for j in range(self.particle_num_neighbors[i]):
                j_i = self.particle_neighbors[i,j]
                if j_i < 0:
                    break
                p_j = self.positions[j_i]
                lambda_j = self.lambdas[j_i]
                pos_ij = p_i - p_j
                scorr = self.compute_scorr(pos_ij)
                # Eq. 14
                self.scorr_list[i] += scorr
                pos_delta_i += (lambda_i + lambda_j + scorr) * \
                    self.spiky_gradient(pos_ij,self.h)
            
            pos_delta_i /= self.rho
            self.position_deltas[i] = pos_delta_i
        
        for i in range(self.total_particles[0]):
            self.positions[i] += self.position_deltas[i]

    @ti.kernel
    def vorticity_confinement(self):
        for i in range(self.total_particles[0]):
            pos_i = self.positions[i]
            self.vorticity[i] = pos_i * 0.0
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                # Eq. 15
                self.vorticity[i] += self.mass * (self.velocities[p_j] - self.velocities[i]).cross(self.spiky_gradient(pos_ji, self.h))

        for i in range(self.total_particles[0]):
            pos_i = self.positions[i]
            loc_vec_i = ti.Vector([0.0,0.0,0.0])
            for j in range(self.particle_num_neighbors[i]):
                p_j = self.particle_neighbors[i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                loc_vec_i += self.mass * self.vorticity[p_j].norm() * self.spiky_gradient(pos_ji, self.h)
            vorticity_i = self.vorticity[i]
            loc_vec_i = loc_vec_i / (self.epsilon + loc_vec_i.norm())
            # Eq. 16
            self.velocities[i] += (self.vorticity_eps * loc_vec_i.cross(vorticity_i))/self.mass * self.dt

    @ti.kernel
    def viscocity(self):
        # Eq. 17
        for i in range(self.total_particles[0]):
            p_i = self.positions[i]
            v_i = self.velocities[i]
            v_delta_i = ti.Vector([0.0,0.0,0.0])

            for j in range(0,self.particle_num_neighbors[i]):
                j_i = self.particle_neighbors[i,j]
                if j_i >= 0:
                    p_j = self.positions[j_i]
                    v_j = self.velocities[j_i]
                    p_ij = p_i - p_j
                    v_ji = v_j - v_i
                    v_delta_i += v_ji * self.poly6_value(p_ij.norm(),self.h)
            
            self.velocities[i] += self.viscocity_const * v_delta_i

    @ti.kernel
    def compute_density(self):
        for p_i in range(self.total_particles[0]):
            pos_i = self.positions[p_i]
            density = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                density += self.poly6_value(pos_ji.norm(), self.h)

            density = (self.mass * density / self.rho) - 1.0
            self.densities[p_i] = density

    @ti.kernel
    def find_surface_particles(self):
        for p_i in range(self.total_particles[0]):
            density = self.densities[p_i]
            if (density) < self.density_threshold:
                self.surface[p_i] = 1
            else:
                self.surface[p_i] = 0


    @ti.kernel
    def compute_surface_normals(self):
        for p_i in range(self.total_particles[0]):
            if self.surface[p_i] == 1:  # Only compute normals for surface particles
                pos_i = self.positions[p_i]
                normal = ti.Vector([0.0, 0.0, 0.0])

                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if p_j < 0:
                        break
                    pos_ji = pos_i - self.positions[p_j]
                    normal += self.mass * self.densities[p_j] * self.cubic_spline(pos_ji, self.h)

                self.surface_normals[p_i] = normal.normalized()
            else:
                self.surface_normals[p_i] = ti.Vector([0.0, 1.0, 0.0])

    @ti.kernel
    def final_step(self):
        for i in range(self.total_particles[0]):
            p_i = self.positions[i]
            self.positions[i] = self.boundary_check(p_i)
                    
        for i in range(self.total_particles[0]):
            if self.surface[i] == 0:
                self.colour[i] = self.particle_color
            else:
                self.colour[i] = self.surface_color
            self.velocities[i] = (self.positions[i] - self.old_positions[i])/self.dt

    def PBF(self):
        self.PBF_first_step()
        for i in range(self.iters):
            self.substep()
        self.final_step()
        self.vorticity_confinement()
        self.viscocity()
        self.compute_density()
        self.find_surface_particles()
        self.compute_surface_normals()

    # Diffuse particle code
    # Based on the paper:
    # "Unified Spray, Foam and Bubbles for Particle-Based Fluids" by Markus Ihmsen et al.
    @ti.kernel
    def compute_num_diffuse_particles(self):
        for p_i in range(self.total_particles[0]):
            if self.surface[p_i] == 1:
                v_diff_i = 0.0
                kappa_i = 0.0
                for j in range(self.particle_num_neighbors[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if p_j < 0 or self.surface[p_j] == 0:
                        break
                    pos_ij = self.positions[p_i] - self.positions[p_j]
                    pos_ji = self.positions[p_j] - self.positions[p_i]
                    vel_ij = self.velocities[p_i] - self.velocities[p_j]
                    vel_n = vel_ij.normalized()
                    pos_ij_n = pos_ij.normalized()
                    pos_ji_n = pos_ji.normalized()
                    # Eq 2
                    v_diff_i += vel_ij.norm() * (1 - vel_n.dot(pos_ij_n)) * self.weighting(pos_ij,self.h)
                    # Eq 4
                    kappa_ij = (1 - self.surface_normals[p_i].dot(self.surface_normals[p_j])) * self.weighting(pos_ij, self.h)
                    # Eq 6
                    if pos_ji_n.dot(self.surface_normals[p_i]) < 0:
                        kappa_i += kappa_ij

                # Eq 1 for each potential 
                I_ta = self.phi(v_diff_i, 2, 8)
                v_i_n = self.velocities[p_i].normalized()
                # Eq 7
                delta_vn_i = 0 if v_i_n.dot(self.surface_normals[p_i]) < 0.6 else 1

                I_wc = self.phi(kappa_i * delta_vn_i, 5, 20)
                E_k_i = 0.5 * self.mass * self.velocities[p_i].dot(self.velocities[p_i])
                I_k = self.phi(E_k_i, 5, 50)

                # Eq 8 
                self.num_diffuse_particles[p_i] = int(I_k * ((self.k_ta * I_ta) + (self.k_wc * I_wc)) * self.dt)

    @ti.kernel
    def gen_diffuse_particles(self):
        for p_i in range(self.total_particles[0]):
            for i in range(self.num_diffuse_particles[p_i]):
                if self.diffuse_count[0] < self.max_diffuse_particles:
                    index = ti.atomic_add(self.diffuse_count[0], 1)
                    X_r = ti.random()
                    X_theta = ti.random()
                    X_h = ti.random()
                    vel = self.velocities[p_i]
                    pos = self.positions[p_i]
                    # Compute the radial distance, azimuth, and height within the cylinder
                    r = self.rV * ti.sqrt(X_r)
                    theta = X_theta * 2 * ti.pi
                    h = X_h * (vel * self.dt).norm()

                    # Compute the orthogonal basis for the reference plane
                    e1_prime = ti.Vector([0, 0, 0])
                    e2_prime = ti.Vector([0, 0, 0])
                    if vel.norm() > 0:
                        e1_prime = ti.Vector([vel[1], -vel[0], 0]).normalized()
                        e2_prime = vel.cross(e1_prime).normalized()

                    # Compute the position and velocity of the diffuse particle
                    self.diffuse_particles[index].pos = self.boundary_check(pos + (r * ti.cos(theta) * e1_prime) + (r * ti.sin(theta) * e2_prime) + (h * vel.normalized()))
                    self.diffuse_particles[index].vel = (r * ti.cos(theta) * e1_prime) + (r * ti.sin(theta) * e2_prime) + vel
                    self.diffuse_particles[index].lifespan = self.life_time
                    self.diffuse_particles[index].active = 1

    @ti.kernel
    def dissolution(self):
        # Despawn diffuse particles
        for i in range(self.diffuse_count[0]):
            if self.diffuse_particles[i].active == 1 and self.diffuse_particles[i].type == 2:
                self.diffuse_particles[i].lifespan -= self.dt
                if self.diffuse_particles.lifespan[i] <= 0:
                    self.diffuse_particles[i].pos = ti.Vector([0.0, 0.0, 0.0])
                    self.diffuse_particles[i].vel = ti.Vector([0.0, 0.0, 0.0])
                    self.diffuse_particles[i].lifespan = 0.0
                    self.diffuse_particles[i].active = 0
        
    @ti.kernel
    def squash_array(self):
        count = 0
        for i in range(self.diffuse_count[0]):
            if self.diffuse_particles[i].active == 0:
                self.diffuse_particles[i] = self.diffuse_particles_copy[self.diffuse_count[0]+1]
                ti.atomic_add(count,1)
        self.diffuse_count[0] -= count

    @ti.func
    def find_diffuse_neighbours(self):
        for i in range(self.max_diffuse_particles):
            self.diffuse_num_neighbors[i] = 0
            for j in range(self.max_num_neighbors):
                self.diffuse_neighbors[i,j] = -1

        for p_i in range(self.max_diffuse_particles):
            if(self.diffuse_particles[p_i].active == 1):
                pos_i = self.diffuse_particles[p_i].pos
                cell = self.get_cell(pos_i)
                nb_i = 0
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and (
                                    pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                                self.diffuse_neighbors[p_i, nb_i] = p_j
                                nb_i += 1
                self.diffuse_num_neighbors[p_i] = nb_i

    @ti.func
    def compute_density_diffuse(self):
        for p_i in range(self.max_diffuse_particles):
            pos_i = self.diffuse_particles[p_i].pos
            density = 0.0

            for j in range(self.diffuse_num_neighbors[p_i]):
                p_j = self.diffuse_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.positions[p_j]
                density += self.poly6_value(pos_ji.norm(), self.h)

            density = (self.mass * density / self.rho) - 1.0
            # Classify diffuse particles
            if density < self.density_threshold - 0.01:
                self.diffuse_particles[p_i].type = 1
            elif density > 0.02:
                self.diffuse_particles[p_i].type = 3
            else:
                self.diffuse_particles[p_i].type = 2


    @ti.kernel
    def advect_particles(self):
        self.find_diffuse_neighbours()
        self.compute_density_diffuse()

        for i in range(self.max_diffuse_particles):
            if self.diffuse_particles[i].active == 1:
                # Advection for spray particles. Simple ballistic motion
                if self.diffuse_particles[i].type == 1:
                    self.diffuse_particles[i].vel += self.gravity * self.dt
                    self.diffuse_particles[i].pos += self.diffuse_particles[i].vel * self.dt
                    self.diffuse_particles[i].pos = self.boundary_check(self.diffuse_particles[i].pos)
                else:
                    avg_fluid_vel = ti.Vector([0.0, 0.0, 0.0])
                    denom = 0.0
                    for p_j in range(self.diffuse_num_neighbors[i]):
                        j = self.diffuse_neighbors[i,p_j]
                        pos_i = self.diffuse_particles[i].pos
                        pos_j = self.positions[j]
                        vel_j = self.velocities[j]

                        avg_fluid_vel += vel_j * self.cubic_spline(pos_i - pos_j, self.h)
                        denom += self.cubic_spline(pos_i - pos_j, self.h)
                    if denom > 0.0:
                        avg_fluid_vel /= denom
                    else:
                        avg_fluid_vel = ti.Vector([0.0, 0.0, 0.0])

                    # Advection for foam particles
                    if self.diffuse_particles[i].type == 2:
                        # Calculate velocity of foam. Average velocity of surrounding fluid particles
                        # v_new = (Sum (fluid vel(t + dt) * W(diffuse pos - fluid pos, h)) /
                        # (Sum (W(diffuse pos - fluid pos, h)))
                        self.diffuse_particles[i].vel = avg_fluid_vel
                        self.diffuse_particles[i].pos = self.boundary_check(self.diffuse_particles[i].pos + (self.dt * avg_fluid_vel))

                    # Advection for bubbles
                    if self.diffuse_particles[i].type == 3:
                        # find velocity of bubble
                        # diffuse vel + dt * (-buoyancy*gravity + drag*((v_new - bubble vel)/dt))
                        drag = self.drag
                        buoyancy = self.buoyancy
                        # v_bub = diffuse_particles[i].vel + (((buoyancy * ti.Vector([0.0,9.8,0.0])) + (drag * ((avg_fluid_vel - diffuse_particles[i].vel) / dt))))
                        self.diffuse_particles[i].vel = self.diffuse_particles[i].vel + self.dt * (buoyancy * ti.Vector([0.0, 9.8, 0.0]) + (drag * ((avg_fluid_vel - self.diffuse_particles[i].vel) / self.dt)))
                        self.diffuse_particles[i].pos = self.boundary_check(self.diffuse_particles[i].pos + (self.dt * self.diffuse_particles[i].vel))
            
        self.num_diffuse_particles.fill(0)

    def dissolution_full(self):
        self.dissolution()
        self.diffuse_particles_copy.pos.copy_from(self.diffuse_particles.pos)
        self.diffuse_particles_copy.vel.copy_from(self.diffuse_particles.vel)
        self.diffuse_particles_copy.lifespan.copy_from(self.diffuse_particles.lifespan)
        self.diffuse_particles_copy.active.copy_from(self.diffuse_particles.active)
        self.diffuse_particles_copy.type.copy_from(self.diffuse_particles.type)
        self.squash_array()

    def diffuse(self):
        self.compute_num_diffuse_particles()
        self.gen_diffuse_particles()
        self.advect_particles()
        self.dissolution_full()
    

    def simulation(self):
        self.PBF()
        self.diffuse()
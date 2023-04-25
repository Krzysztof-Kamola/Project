import taichi as ti
import taichi_glsl as ts
from datetime import datetime
import numpy as np
import math
import random
import tables


@ti.data_oriented
class MarchingCubes:
    dim = 3
    particle_color = ti.Vector([0.2,0.2,0.8])
    surface_color = ti.Vector([0.8,0.2,0.2])
    diffuse_color = ti.Vector([1.0,1.0,1.0])
    bound_damping = -0.5
    def __init__(self,particles,num_particles,diffuse_particles,num_diffuse,h,box_size,isolevel):
        self.particles = ti.Vector.field(self.dim, dtype=ti.f32, shape=(num_particles,))
        self.diffuse_particles = ti.Vector.field(self.dim, dtype=ti.f32, shape=(num_diffuse,))
        self.particles.copy_from(particles)
        self.diffuse_particles.copy_from(diffuse_particles)
        self.num_particles = num_particles
        self.num_diffuse = num_diffuse
        self.cube_size = h/2
        self.box_size = box_size
        self.isolevel = isolevel

        self.grid_num_particles = ti.field(int)
        self.grid_num_diffuse = ti.field(int)
        self.cube_index = ti.field(int)
        self.cube_triangles = ti.Vector.field(self.dim,float)
        self.num_triangles = ti.field(int)
        self.vert_lists = ti.Vector.field(self.dim,float)
        
        self.edge_table = ti.field(int,shape=tables.edgeTable.shape)
        self.tri_table = ti.field(int,shape=tables.triTable.shape)

        def round_up(f, s, c):
            return (math.floor(f * (1/c) / s) + 1) * s

        print("Number of cells:",(round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size)))
        self.grid_size = (round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size))
        numcells = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.triangles = ti.Vector.field(self.dim,float,shape=(numcells*15,))

        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles,self.grid_num_diffuse,self.cube_index,self.num_triangles)
        grid_snode.dense(ti.l, 15).place(self.cube_triangles)
        grid_snode.dense(ti.l, 12).place(self.vert_lists)
        
        self.edge_table.from_numpy(tables.edgeTable)
        self.tri_table.from_numpy(tables.triTable)

        self.total_triangles = ti.field(int,shape = 1)
        self.total_triangles[0] = 0

    
    @ti.func
    def is_in_grid(self,c):
        return 0.0 <= c[0] and c[0] < self.box_size[0] and 0.0 <= c[1] and c[1] < self.box_size[1] and 0.0 <= c[2] and c[2] < self.box_size[2]

    @ti.func
    def get_cell(self,pos):
        return int(ti.floor(pos / self.cube_size))
    
    @ti.func
    def get_pos(self,cell):
        return cell * self.cube_size

    @ti.func
    def update_grid(self):
        self.grid_num_particles.fill(0)
        self.grid_num_diffuse.fill(0)

        for i in range(self.num_particles):
            cell = self.get_cell(self.particles[i])
            self.grid_num_particles[cell] += 1

        # for i in range(self.num_diffuse):
        #     cell = self.get_cell(self.diffuse_particles[i])
        #     offs = ti.atomic_add(self.grid_num_diffuse[cell], 1)

    @ti.func
    def VertexInterp(self,isolevel,p1,p2,valp1,valp2):
        mu = 0.0
        p = ti.Vector([0.0,0.0,0.0])
        result = ti.Vector([0.0,0.0,0.0])
        result = p1

        if ti.abs(isolevel-valp1) < 0.00001:
            result = p1
        elif ti.abs(isolevel-valp2) < 0.00001:
            result = p2
        elif ti.abs(valp1 - valp2) < 0.00001:
            result = p1
        else:
            mu = (isolevel - valp1) / (valp2 - valp1)
            p.x = p1.x + mu * (p2.x - p1.x)
            p.y = p1.y + mu * (p2.y - p1.y)
            p.z = p1.z + mu * (p2.z - p1.z)
            result = p

        return result

    @ti.func
    def compute_indicies(self):
         for cell in ti.grouped(self.grid_num_particles):
            cubeindex = 0
            cell_to_check = cell
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 1
            cell_to_check = cell + ti.Vector([1,0,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 2
            cell_to_check = cell + ti.Vector([1,0,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 4
            cell_to_check = cell + ti.Vector([0,0,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 8
            cell_to_check = cell + ti.Vector([0,1,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 16
            cell_to_check = cell + ti.Vector([1,1,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 32
            cell_to_check = cell + ti.Vector([1,1,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 64
            cell_to_check = cell + ti.Vector([0,1,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 128
            
            self.cube_index[cell] = cubeindex

    @ti.func
    def create_iso_surface(self,cell: ti.types.vector(3,int)):
        cubeindex =  self.cube_index[cell]
        for i in range(0,12):
            self.vert_lists[cell,i] = [0.0,0.0,0.0]
        if self.edge_table[cubeindex] == 0:
            pass
        if self.edge_table[cubeindex] & 1:
            cell1 = cell
            cell2 = cell + ti.Vector([1,0,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 2:
            cell1 = cell + ti.Vector([1,0,0])
            cell2 = cell + ti.Vector([1,0,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,1] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 4:
            cell1 = cell + ti.Vector([1,0,1])
            cell2 = cell + ti.Vector([0,0,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,2] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 8:
            cell1 = cell + ti.Vector([0,0,1])
            cell2 = cell + ti.Vector([0,0,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,3] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 16:
            cell1 = cell + ti.Vector([0,1,0])
            cell2 = cell + ti.Vector([1,1,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,4] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 32:
            cell1 = cell + ti.Vector([1,1,0])
            cell2 = cell + ti.Vector([1,1,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,5] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 64:
            cell1 = cell + ti.Vector([1,1,1])
            cell2 = cell + ti.Vector([0,1,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,6] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 128:
            cell1 = cell + ti.Vector([0,1,1])
            cell2 = cell + ti.Vector([0,1,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,7] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 256:
            cell1 = cell
            cell2 = cell + ti.Vector([0,1,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,8] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 512:
            cell1 = cell + ti.Vector([1,0,0])
            cell2 = cell + ti.Vector([1,1,0])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,9] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 1024:
            cell1 = cell + ti.Vector([1,0,1])
            cell2 = cell + ti.Vector([1,1,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,10] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        if self.edge_table[cubeindex] & 2048:
            cell1 = cell + ti.Vector([0,0,1])
            cell2 = cell + ti.Vector([0,1,1])
            cell1_val = self.grid_num_particles[cell1]
            cell2_val = self.grid_num_particles[cell2]
            cell1_pos = self.get_pos(cell1)
            cell2_pos = self.get_pos(cell2)
            self.vert_lists[cell,11] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
        
        i = 0
        num_triangles = 0
        index = (cell[2] * self.grid_size[0] * self.grid_size[1]) + (cell[1] * self.grid_size[0]) + cell[0]
        while self.tri_table[(cubeindex,i)] != -1:
            self.triangles[index * 15 + (i + 0)] = self.vert_lists[cell,self.tri_table[(cubeindex,i)]]
            self.triangles[index * 15 + (i + 1)] = self.vert_lists[cell,self.tri_table[(cubeindex,i+1)]]
            self.triangles[index * 15 + (i + 2)] = self.vert_lists[cell,self.tri_table[(cubeindex,i+2)]]
            # self.cube_triangles[cell,i] = self.vertlist[self.tri_table[(cubeindex,i)]]
            # self.cube_triangles[cell,i+1] = self.vertlist[self.tri_table[(cubeindex,i+1)]]
            # self.cube_triangles[cell,i+2] = self.vertlist[self.tri_table[(cubeindex,i+2)]]
            num_triangles += 3
            i += 3
        self.num_triangles[cell] = num_triangles

    @ti.kernel
    def marching_cubes(self):
        self.update_grid()
        self.compute_indicies()
        for cell in ti.grouped(self.grid_num_particles):
            self.create_iso_surface(cell)


    @ti.kernel
    def create_mesh(self):
        total_triangles = 0
        for cell in ti.grouped(self.grid_num_particles):
            for i in range(self.num_triangles[cell]):
                self.triangles[total_triangles] = self.cube_triangles[cell,i]
                total_triangles += 1
        self.total_triangles[0] = total_triangles
        #self.total_triangles[0] = total_triangles


# @ti.kernel
# def marchingcubes(self):
#     for i in range(0,num_cubes):
#         self.create_iso_surface(i)
        

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
        

        self.edge_table = ti.field(int,shape=tables.edgeTable.shape)
        self.tri_table = ti.field(int,shape=tables.triTable.shape)

        self.vertlist = ti.Vector.field(3,dtype=float,shape=(12,))

        def round_up(f, s, c):
            return (math.floor(f * (1/c) / s) + 1) * s

        print("Number of cells:",(round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size)))
        self.grid_size = (round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size))
        numcells = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.triangles = ti.Vector.field(self.dim,float,shape=(numcells*15,))

        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles,self.grid_num_diffuse,self.cube_index,self.num_triangles)
        grid_snode.dense(ti.l, 15).place(self.cube_triangles)
        
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

    @ti.kernel
    def update_grid(self):
        self.grid_num_particles.fill(0)
        self.grid_num_diffuse.fill(0)

        for i in range(self.num_particles):
            cell = self.get_cell(self.particles[i])
            self.grid_num_particles[cell] += 1

        for i in range(self.num_diffuse):
            cell = self.get_cell(self.diffuse_particles[i])
            offs = ti.atomic_add(self.grid_num_diffuse[cell], 1)

    @ti.func
    def VertexInterp(self,isolevel,p1,p2,valp1,valp2):
        mu = 0.0
        p = ti.Vector([0.0,0.0,0.0])
        result = ti.Vector([0.0,0.0,0.0])

        if ti.abs(isolevel-valp1) < 0.00001:
            reuslt = p1
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

    @ti.kernel
    def compute_indicies(self):
        for i in range(0,12):
            self.vertlist[i] = [0.0,0.0,0.0]

        cubeindex = 0
        
        for cell in ti.grouped(self.grid_num_particles):
            i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    if self.grid_num_particles[cell_to_check] > self.isolevel:
                        self.cube_index[cell] |= int(1 << i)
                i += 1

    @ti.kernel
    def create_iso_surface(self):
        for cell in ti.grouped(self.grid_num_particles):
            cubeindex = 0
            for i in range(0,12):
                self.vertlist[i] = [0.0,0.0,0.0]
            # for offs in ti.static(ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2)))):
            #     cell_to_check = cell + offs
            #     if self.is_in_grid(cell_to_check):
            #         if self.grid_num_particles[cell_to_check] > self.isolevel:
            #             self.cube_index[cell] |= int(1 << i)
            #     i += 1
            cubeindex = 0
            cell_to_check = cell + ti.Vector([0,0,0])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 1
            cell_to_check = cell + ti.Vector([1,0,0])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 2
            cell_to_check = cell + ti.Vector([1,0,1])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 4
            cell_to_check = cell + ti.Vector([0,0,1])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 8
            cell_to_check = cell + ti.Vector([0,1,0])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 16
            cell_to_check = cell + ti.Vector([1,1,0])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 32
            cell_to_check = cell + ti.Vector([1,1,1])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 64
            cell_to_check = cell + ti.Vector([0,1,1])
            if self.grid_num_particles[cell_to_check] and self.is_in_grid(cell_to_check) > self.isolevel: cubeindex |= 128
        
            if self.edge_table[cubeindex] == 0:
                pass
            if self.edge_table[cubeindex] & 1:
                cell1 = cell
                cell2 = cell + ti.Vector([1,0,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 2:
                cell1 = cell + ti.Vector([1,0,0])
                cell2 = cell + ti.Vector([1,0,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 4:
                cell1 = cell + ti.Vector([1,0,1])
                cell2 = cell + ti.Vector([0,0,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 8:
                cell1 = cell + ti.Vector([0,0,1])
                cell2 = cell + ti.Vector([0,0,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 16:
                cell1 = cell + ti.Vector([0,1,0])
                cell2 = cell + ti.Vector([1,1,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 32:
                cell1 = cell + ti.Vector([1,1,0])
                cell2 = cell + ti.Vector([1,1,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 64:
                cell1 = cell + ti.Vector([1,1,1])
                cell2 = cell + ti.Vector([0,1,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 128:
                cell1 = cell + ti.Vector([0,1,1])
                cell2 = cell + ti.Vector([0,1,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 256:
                cell1 = cell
                cell2 = cell + ti.Vector([0,1,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 512:
                cell1 = cell + ti.Vector([1,0,0])
                cell2 = cell + ti.Vector([1,1,0])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 1024:
                cell1 = cell + ti.Vector([1,0,1])
                cell2 = cell + ti.Vector([1,1,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            if self.edge_table[cubeindex] & 2048:
                cell1 = cell + ti.Vector([0,0,1])
                cell2 = cell + ti.Vector([0,1,1])
                cell1_val = self.grid_num_particles[cell1]
                cell2_val = self.grid_num_particles[cell2]
                cell1_pos = self.get_pos(cell1)
                cell2_pos = self.get_pos(cell2)
                self.vertlist[0] = self.VertexInterp(self.isolevel,cell1_pos,cell2_pos,cell1_val,cell2_val)
            
            i = 0
            num_triangles = 0
            while self.tri_table[(cubeindex,i)] != -1:
                self.triangles[0] = self.vertlist[self.tri_table[(cubeindex,i)]]
                self.triangles[1] = self.vertlist[self.tri_table[(cubeindex,i+1)]]
                self.triangles[2] = self.vertlist[self.tri_table[(cubeindex,i+2)]]
                num_triangles += 1
                i += 3
            self.num_triangles[cell] = num_triangles

    @ti.kernel
    def create_mesh(self):
        total_triangles = 0
        for cell in ti.grouped(self.grid_num_particles):
            for i in range(self.num_triangles[cell]):
                self.triangles[total_triangles] = self.cube_triangles[cell,i*3]
                self.triangles[total_triangles+1] = self.cube_triangles[cell,i*3 + 1]
                self.triangles[total_triangles+2] = self.cube_triangles[cell,i*3 + 2]
                total_triangles += 3
        self.total_triangles[0] = total_triangles


# @ti.kernel
# def marchingcubes(self):
#     for i in range(0,num_cubes):
#         self.create_iso_surface(i)
        

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
    def __init__(self,particles,num_particles,diffuse_particles,num_diffuse,h,box_size,isolevel,total_particles,total_diffuse):
        self.particles = ti.Vector.field(self.dim, dtype=ti.f32, shape=(num_particles,))
        self.diffuse_particles = ti.Vector.field(self.dim, dtype=ti.f32, shape=(num_diffuse,))
        self.particles.copy_from(particles)
        self.diffuse_particles.copy_from(diffuse_particles)
        self.num_particles = num_particles
        self.num_diffuse = num_diffuse
        self.cube_size = h*0.5
        self.check_size = h
        self.box_size = box_size
        self.isolevel = isolevel
        self.diffuse_level = 2

        self.particle_color = ti.Vector([0.2,0.2,0.8])
        self.surface_color = ti.Vector([0.8,0.2,0.2])
        self.diffuse_color = ti.Vector([1.0,1.0,1.0])

        self.grid_num_particles = ti.field(int)
        self.grid_num_diffuse = ti.field(int)
        self.cube_index = ti.field(int)
        self.cube_triangles = ti.Vector.field(self.dim,float)
        self.num_triangles = ti.field(int)
        self.vert_lists = ti.Vector.field(self.dim,float)
        self.vert_colour_lists = ti.Vector.field(self.dim,float)
        
        self.edge_table = ti.field(int,shape=tables.edgeTable.shape)
        self.tri_table = ti.field(int,shape=tables.triTable.shape)

        def round_up(f, s, c):
            return (math.floor(f * (1/c) / s) + 1) * s

        print("Number of cells:",(round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size)))
        self.grid_size = (round_up(self.box_size[0],1,self.cube_size),round_up(self.box_size[1],1,self.cube_size),round_up(self.box_size[2],1,self.cube_size))
        numcells = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.max_triangles = numcells * 15
        self.triangles = ti.Vector.field(self.dim,float,shape=(self.max_triangles,))
        self.colour = ti.Vector.field(self.dim,float,shape=(self.max_triangles,))

        grid_snode = ti.root.dense(ti.ijk, self.grid_size)
        grid_snode.place(self.grid_num_particles,self.grid_num_diffuse,self.cube_index,self.num_triangles)
        grid_snode.dense(ti.l, 15).place(self.cube_triangles)
        grid_snode.dense(ti.l, 12).place(self.vert_lists,self.vert_colour_lists)
        
        self.edge_table.from_numpy(tables.edgeTable)
        self.tri_table.from_numpy(tables.triTable)

        self.total_triangles = ti.field(int,shape = 1)
        self.total_triangles[0] = 0
        self.total_particles = ti.field(int,shape = 1)
        self.total_particles[0] = total_particles
        self.total_diffuse = ti.field(int,shape = 1)
        self.total_diffuse[0] = total_diffuse

    
    @ti.func
    def is_in_grid(self,c):
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]

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
        # Get number of particle in a cell 
        for i in range(self.total_particles[0]):
            cell = self.get_cell(self.particles[i])
            for offs in ti.static(ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    self.grid_num_particles[cell_to_check] += 1

        for i in range(self.total_diffuse[0]):
            cell = self.get_cell(self.diffuse_particles[i])
            for offs in ti.static(ti.grouped(ti.ndrange((0, 2), (0, 2), (0, 2)))):
                cell_to_check = cell + offs
                if self.is_in_grid(cell_to_check):
                    self.grid_num_diffuse[cell_to_check] += 1
        
        # Handeling bounds
        for cell in ti.grouped(self.grid_num_particles):
            if cell[0] == self.grid_size[0]:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0
            if cell[0] == 0:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0
            if cell[1] == self.grid_size[1]:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0
            if cell[1] == 0:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0
            if cell[2] == self.grid_size[2]:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0
            if cell[2] == 0:
                self.grid_num_particles[cell] = 0
                self.grid_num_diffuse[cell] = 0

    @ti.func
    def VertexInterp(self,isolevel,p1,p2,valp1,valp2):
        mu = 0.0
        p = ti.Vector([0.0,0.0,0.0])
        result = ti.Vector([0.0,0.0,0.0])

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
    def ColourInterp(self,isolevel,p_1,p_2,d_1,d_2):
        mu = 0.0
        p1 = ti.Vector([0.0,0.0,0.0])

        num_diffuse = d_1 + d_2
        num_partilce = p_1 + p_2

        if num_diffuse == 0:
            p1 = self.particle_color
        elif num_partilce == 0:
            p1 = self.diffuse_color
        else:
            if num_diffuse > num_partilce:
                mu = num_partilce/num_diffuse
            else:
                mu = num_diffuse/num_partilce

            p1.x = self.diffuse_color.x + mu*(self.particle_color.x - self.diffuse_color.x)
            p1.y = self.diffuse_color.y + mu*(self.particle_color.y - self.diffuse_color.y)
            p1.z = self.diffuse_color.z + mu*(self.particle_color.z - self.diffuse_color.z)

        return p1

    @ti.func
    def ColourInterp2(self,iso,p1,d1):
        mu = 0.0
        result = ti.Vector([0.0,0.0,0.0])

        if d1 < iso:
            result = self.particle_color
        elif p1 == 0:
            result = self.diffuse_color
        else:
            if d1 > p1:
                mu = p1/d1
            else:
                mu = d1/p1
            result.x = self.diffuse_color.x + mu*(self.particle_color.x - self.diffuse_color.x)
            result.y = self.diffuse_color.y + mu*(self.particle_color.y - self.diffuse_color.y)
            result.z = self.diffuse_color.z + mu*(self.particle_color.z - self.diffuse_color.z)

        return result

    @ti.func
    def compute_indicies(self):
        # Find which corners of the cube are active
         self.cube_index.fill(0)
         for cell in ti.grouped(self.grid_num_particles):
            cubeindex = 0
            cell_to_check = cell
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 1
            cell_to_check = cell + ti.Vector([1,0,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 2
            cell_to_check = cell + ti.Vector([1,0,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 4
            cell_to_check = cell + ti.Vector([0,0,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 8
            cell_to_check = cell + ti.Vector([0,1,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 16
            cell_to_check = cell + ti.Vector([1,1,0])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check] >= self.isolevel): cubeindex |= 32
            cell_to_check = cell + ti.Vector([1,1,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 64
            cell_to_check = cell + ti.Vector([0,1,1])
            if self.is_in_grid(cell_to_check) and (self.grid_num_particles[cell_to_check]  >= self.isolevel): cubeindex |= 128
            
            self.cube_index[cell] = cubeindex

    @ti.func
    def create_iso_surface(self,cell: ti.types.vector(3,int)):
        cubeindex =  self.cube_index[cell]
        for i in range(0,12):
            self.vert_lists[cell,i] = [0.0,0.0,0.0]
        
        #Update list of verticies based on look-up table
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
            # Set traingles
            self.triangles[index * 15 + (i + 0)] = self.vert_lists[cell,self.tri_table[(cubeindex,i)]]
            self.triangles[index * 15 + (i + 1)] = self.vert_lists[cell,self.tri_table[(cubeindex,i+1)]]
            self.triangles[index * 15 + (i + 2)] = self.vert_lists[cell,self.tri_table[(cubeindex,i+2)]]

            #Set colours at each vertex
            cell2 = self.get_cell(self.vert_lists[cell,self.tri_table[(cubeindex,i)]])
            self.colour[index * 15 + (i + 0)] = self.ColourInterp2(self.diffuse_level,self.grid_num_particles[cell2],self.grid_num_diffuse[cell2])
            cell2 = self.get_cell(self.vert_lists[cell,self.tri_table[(cubeindex,i+1)]])
            self.colour[index * 15 + (i + 1)] = self.ColourInterp2(self.diffuse_level,self.grid_num_particles[cell2],self.grid_num_diffuse[cell2])
            cell2 = self.get_cell(self.vert_lists[cell,self.tri_table[(cubeindex,i+2)]])
            self.colour[index * 15 + (i + 2)] = self.ColourInterp2(self.diffuse_level,self.grid_num_particles[cell2],self.grid_num_diffuse[cell2])

            num_triangles += 3
            i += 3
        self.num_triangles[cell] = num_triangles

    @ti.func
    def clear(self):
        for cell in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[cell] = 0
            self.grid_num_diffuse[cell] = 0
            self.num_triangles[cell] = 0
        for i in range(self.max_triangles):
            self.triangles[i] = ti.Vector([0.0,0.0,0.0])
            self.colour[i] = ti.Vector([0.0,0.0,0.0])
        self.total_triangles[0] = 0

    @ti.kernel
    def marching_cubes(self):
        self.clear()
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

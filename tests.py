import taichi as ti
from PBF import Simulator
from MarchingCubes import MarchingCubes
import time
import numpy as np

ti.init(arch=ti.gpu)

camera_active = False

window = ti.ui.Window("Taichi sim on GGUI", (1024, 800),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 0.6))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
gui = window.get_gui()
video_manager = ti.tools.VideoManager(output_dir="./output", framerate=60, automatic_build=False)

def cam():
    if camera_active:
        camera.track_user_inputs(window,movement_speed=1)
    scene.set_camera(camera)
    scene.point_light(pos=(10, 500, 20), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

def keyboard_handling():
        global camera_active
        if window.get_event(tag=ti.ui.PRESS):
            if window.event.key == ' ':
                camera_active = not camera_active

def dam_break_num_particles(origin,right,top,back,spacing,right_b,top_b,back_b):
    frame = 0
    camera.position(33.70615112,17.7,99.73799397)
    camera.lookat(33.6708612 ,17.72402054,98.73890557)
    
    # camera.position(58.60418869,27.03372005,209.89571828)
    # camera.lookat(58.56113275,26.9852619,208.8978215)
    
    # video_manager = ti.tools.VideoManager(output_dir="./num_100000", framerate=60, automatic_build=False)

    # Sim = Simulator(max_particles=50000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=50.0,top_bound=40.0,back_bound=25.0)
    # Sim.spawn_fluid((1.0,1.0,1.0),20.0,20.0,25.0,0.7)
    # Sim = Simulator(dt=1/60,max_particles=100000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=60.0,top_bound=40.0,back_bound=25.0)
    Sim = Simulator(dt=1/60,max_particles=100000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=right_b,top_bound=top_b,back_bound=back_b)
    Sim.spawn_fluid(origin,right,top,back,spacing)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 600
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start
        #Sim.diffuse()
        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, color=(0.2,0.2,0.8), index_count=Sim.total_particles[0])
        canvas.scene(scene)
        window.show()
        # img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        rendering_end = time.time()
        
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            print(Sim.diffuse_count[0])
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])
            print(camera.curr_lookat)
            print(camera.curr_position)
        frame += 1
    print("Number of particles:", Sim.total_particles[0])
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    
    print("\n")

def dam_break_dt_test(dt):
    frame = 0
    camera.position(33.70615112,17.7,99.73799397)
    camera.lookat(33.6708612 ,17.72402054,98.73890557)

    Sim = Simulator(dt=dt,max_particles=55000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=60.0,top_bound=40.0,back_bound=25.0)
    Sim.spawn_fluid((1.0,1.0,1.0),25.0,20.0,25.0,0.7)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = int(10 * (1/dt))
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start
        #Sim.diffuse()
        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])
        canvas.scene(scene)
        window.show()
        rendering_end = time.time()
        # img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            print(Sim.diffuse_count[0])
            print(Sim.calc_volume())
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])

        frame += 1
    print("Number of particles:", Sim.total_particles[0])
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print("\n")

def dam_break1():
    frame = 0
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    # Sim = Simulator(max_particles=50000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=50.0,top_bound=40.0,back_bound=25.0)
    # Sim.spawn_fluid((1.0,1.0,1.0),20.0,20.0,25.0,0.7)
    Sim = Simulator(dt=1/240,max_particles=55000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=70.0,top_bound=40.0,back_bound=25.0)
    Sim.spawn_fluid((1.0,1.0,1.0),30.0,22.0,25.0,0.7)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 2400
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start
        #Sim.diffuse()
        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])
        canvas.scene(scene)
        window.show()
        rendering_end = time.time()
        # img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            print(Sim.diffuse_count[0])
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])

        frame += 1
    print("Number of particles:", Sim.total_particles[0])
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print(camera.position())
    print("\n")

def dam_break2():
    frame = 0
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    Sim = Simulator(dt=1/60.0,max_particles=50000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=40.0,top_bound=30.0,back_bound=25.0)
    Sim.spawn_fluid((1.0,1.0,1.0),10.0,10.0,25.0,0.7)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 600
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start

        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])

        canvas.scene(scene)
        # img = window.get_image_buffer_as_numpy()
        # video_manager.write_frame(img)
        window.show()
        rendering_end = time.time()
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            print(Sim.diffuse_count[0])
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])

        frame += 1
    print("Number of particles:", Sim.total_particles[0])
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print("\n")
    # video_manager.make_video(gif=False, mp4=True)


def dam_break_diffuse():
    frame = 0
    # camera.position(-0.18,42.7, 100.5)
    # camera.lookat(0.0, 0.0, 0.0)
    camera.position(36.31793349,16.99370678,136.5374458)
    camera.lookat(36.28462543,17.04270997,135.53920272)
    # camera.position(58.60418869,27.03372005,209.89571828)
    # camera.lookat(58.56113275,26.9852619,208.8978215)
    #video_manager = ti.tools.VideoManager(output_dir="./dam_break_diffuse", framerate=60, automatic_build=False)


    Sim = Simulator(dt=1/60.0,max_particles=100000,max_num_neighbors=200,max_num_particles_per_cell=400,max_diffuse_particles=500000,right_bound=60.0,top_bound=60.0,back_bound=50.0,k_wc=100,k_ta=100)
    Sim.spawn_fluid((1.0,1.0,1.0),20.0,25.0,50.0,0.7)
    print(Sim.total_particles[0])
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 700
    diffuse_time = 0.0
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start

        diffuse_start = time.time()
        Sim.diffuse()
        diffuse_end = time.time()
        
        diffuse_time += diffuse_end - diffuse_start

        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, color=(0.2,0.2,0.8), index_count=Sim.total_particles[0])

        Sim.diffuse_positions.copy_from(Sim.diffuse_particles.pos)
        scene.particles(Sim.diffuse_positions, radius=Sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=Sim.diffuse_count[0])

        canvas.scene(scene)
        #img = window.get_image_buffer_as_numpy()
        #video_manager.write_frame(img)
        
        window.show()
        rendering_end = time.time()
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            print(Sim.diffuse_count[0])
            print(camera.curr_lookat)
            print(camera.curr_position)
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])

        frame += 1
    print("Number of particles:", Sim.total_particles[0])
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print("Total diffuse time:", diffuse_time)
    print("\n")
    # video_manager.make_video(gif=False, mp4=True)



def rain_drop():
    frame = 0
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    video_manager = ti.tools.VideoManager(output_dir="./surface_tension", framerate=60, automatic_build=False)
    
    Sim = Simulator()
    Sim.spawn_fluid((1.0,1.0,1.0),10.0,10.0,19.0,0.7)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 700
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start

        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, color=(0.2,0.2,0.8), index_count=Sim.total_particles[0])

        Sim.diffuse_positions.copy_from(Sim.diffuse_particles.pos)
        scene.particles(Sim.diffuse_positions, radius=Sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=0)

        canvas.scene(scene)
        window.show()
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)
        rendering_end = time.time()
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
        if frame == 300:
            Sim.spawn_fluid((5.0,20.0,5.0),5.0,5.0,5.0,0.7)
        frame += 1
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print("\n")

def marching_cubes_test():
    sim = Simulator(dt=1/60.0,max_particles=50000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=40.0,top_bound=30.0,back_bound=25.0)
    sim.spawn_fluid((1.0,1.0,1.0),20.0,20.0,25.0,0.7)
    sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
    cubes = MarchingCubes(sim.positions,sim.max_particles,sim.diffuse_positions,sim.max_diffuse_particles,1.1,(sim.right_bound,sim.top_bound,sim.back_bound),1,sim.total_particles[0],sim.diffuse_count[0])
    cubes.marching_cubes()
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)
    while window.running:
        keyboard_handling()
        cam()
        sim.PBF()

        cubes.particles.copy_from(sim.positions)
        sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
        cubes.diffuse_particles.copy_from(sim.diffuse_positions)
        cubes.marching_cubes()
        scene.mesh(cubes.triangles,per_vertex_color=cubes.colour)
        canvas.scene(scene)
        window.show()

def marching_cubes_diffuse_test():
    sim = Simulator(dt=1/60.0,max_particles=100000,max_num_neighbors=150,max_num_particles_per_cell=400,max_diffuse_particles=100000,right_bound=70.0,top_bound=40.0,back_bound=25.0)
    sim.spawn_fluid((1.0,1.0,1.0),50.0,20.0,25.0,0.7)
    print(sim.total_particles[0])
    cubes = MarchingCubes(sim.positions,sim.max_particles,sim.diffuse_positions,sim.max_diffuse_particles,1.1,(sim.right_bound,sim.top_bound,sim.back_bound),1,sim.total_particles[0],sim.diffuse_count[0])
    cubes.marching_cubes()
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)
    cubes_time = 0.0
    num_frames = 700
    rendering_time = 0.0
    frame = 0
    for i in range(num_frames):
        keyboard_handling()
        cam()
        sim.PBF()
        sim.diffuse()
        cubes.particles.copy_from(sim.positions)
        sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
        cubes_start = time.time()
        cubes.diffuse_particles.copy_from(sim.diffuse_positions)
        cubes.total_diffuse[0] = sim.diffuse_count[0]
        cubes.marching_cubes()
        cubes_end = time.time()
        cubes_time = cubes_end - cubes_start
        
        # scene.particles(sim.positions, radius=sim.particle_radius, per_vertex_color=sim.colour, index_count=sim.total_particles[0])
        sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
        scene.particles(sim.diffuse_positions, radius=sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=sim.diffuse_count[0])
        scene.mesh(cubes.triangles,per_vertex_color=cubes.colour)
        canvas.scene(scene)
        window.show()
        
        if frame % 20 == 0:
            print("Frame:", frame)
            print("cubes time:", cubes_end - cubes_start)
        frame += 1
    print(sim.total_particles[0])
    print("Total cubes time:", cubes_time)
    print("Average cubes framerate:", num_frames / cubes_time)
    print("\n")


def test_num_particles():
    #dam_break_num_particles((1.0,1.0,1.0),10,10,25,0.7,30.0,30.0,25.0)
    #dam_break_num_particles((1.0,1.0,1.0),20,10,25,0.7,30.0,30.0,25.0)
    dam_break_num_particles((1.0,1.0,1.0),20,20,25,0.7,40.0,30.0,25.0)
    #dam_break_num_particles((1.0,1.0,1.0),35,20,25,0.7,60.0,50.0,25.0)
    #dam_break_num_particles((1.0,1.0,1.0),50,30,25,0.7,80.0,50.0,25.0)
    #dam_break_num_particles((1.0,1.0,1.0),75,37,100,0.65,100.0,50.0,100.0)

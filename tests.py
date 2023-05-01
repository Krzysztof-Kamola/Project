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
    scene.point_light(pos=(10, 60, 20), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

def keyboard_handling():
        global camera_active
        if window.get_event(tag=ti.ui.PRESS):
            if window.event.key == ' ':
                camera_active = not camera_active

def dam_break_num_particles(origin,right,top,back,spacing):
    frame = 0
    camera.position(33.70615112,17.7,99.73799397)
    camera.lookat(33.6708612 ,17.72402054,98.73890557)

    # Sim = Simulator(max_particles=50000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=50.0,top_bound=40.0,back_bound=25.0)
    # Sim.spawn_fluid((1.0,1.0,1.0),20.0,20.0,25.0,0.7)
    Sim = Simulator(dt=1/60,max_particles=55000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=10,right_bound=60.0,top_bound=40.0,back_bound=25.0)
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
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    Sim = Simulator(dt=1/60.0,max_particles=20000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=50000*10,right_bound=30.0,top_bound=20.0,back_bound=25.0)
    Sim.spawn_fluid((1.0,1.0,1.0),10.0,20.0,10.0,0.7)
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

        Sim.diffuse()

        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])

        Sim.diffuse_positions.copy_from(Sim.diffuse_particles.pos)
        scene.particles(Sim.diffuse_positions, radius=Sim.particle_radius*0.5, color=(1.0,1.0,1.0))

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



def rain_drop():
    frame = 0
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    Sim = Simulator()
    #Sim.init()
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
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])

        Sim.diffuse_positions.copy_from(Sim.diffuse_particles.pos)
        scene.particles(Sim.diffuse_positions, radius=Sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=0)

        canvas.scene(scene)
        window.show()
        rendering_end = time.time()
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])
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
        #scene.particles(sim.positions, radius=sim.particle_radius, per_vertex_color=sim.colour, index_count=sim.total_particles[0])
        scene.mesh(cubes.triangles,per_vertex_color=cubes.colour)
        canvas.scene(scene)
        window.show()

def marching_cubes_diffuse_test():
    sim = Simulator(dt=1/60.0,max_particles=20000,max_num_neighbors=150,max_num_particles_per_cell=300,max_diffuse_particles=50000*10,right_bound=30.0,top_bound=20.0,back_bound=25.0)
    sim.spawn_fluid((1.0,1.0,1.0),10.0,20.0,25.0,0.7)
    cubes = MarchingCubes(sim.positions,sim.max_particles,sim.diffuse_positions,sim.max_diffuse_particles,1.1,(sim.right_bound,sim.top_bound,sim.back_bound),1,sim.total_particles[0],sim.diffuse_count[0])
    cubes.marching_cubes()
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)
    while window.running:
        keyboard_handling()
        cam()
        sim.PBF()
        sim.diffuse()
        cubes.particles.copy_from(sim.positions)
        sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
        cubes.diffuse_particles.copy_from(sim.diffuse_positions)
        cubes.total_diffuse[0] = sim.diffuse_count[0]
        cubes.marching_cubes()
        # scene.particles(sim.positions, radius=sim.particle_radius, per_vertex_color=sim.colour, index_count=sim.total_particles[0])
        sim.diffuse_positions.copy_from(sim.diffuse_particles.pos)
        scene.particles(sim.diffuse_positions, radius=sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=sim.diffuse_count[0])
        scene.mesh(cubes.triangles,per_vertex_color=cubes.colour)
        canvas.scene(scene)
        window.show()

# dam_break_diffuse()

#marching_cubes_diffuse_test()

# dam_break1()

def test_num_particles():
    dam_break_num_particles((1.0,1.0,1.0),10,10,25,0.7)
    #dam_break_num_particles((1.0,1.0,1.0),20,10,25,0.7)
    #dam_break_num_particles((1.0,1.0,1.0),20,20,25,0.7)
    #dam_break_num_particles((1.0,1.0,1.0),35,20,25,0.7)

def test_dt():
    dam_break_dt_test(1/15)
    # dam_break_dt_test(1/30)
    # dam_break_dt_test(1/60)
    # dam_break_dt_test(1/120)

def test_vorticity():
    frame = 0
    camera.position(-0.18,42.7, 100.5)
    camera.lookat(0.0, 0.0, 0.0)

    Sim = Simulator(right_bound=30,back_bound=30,top_bound=40)
    #Sim.init()
    Sim.spawn_fluid((5.0,5.0,5.0),20.0,10.0,20.0,0.7)
    print(Sim.total_particles[0])
    print("Finished intializing")
    PBF_time = 0.0
    rendering_time = 0.0
    num_frames = 800
    for i in range(num_frames):
        keyboard_handling()
        cam()
        PBF_start = time.time()
        Sim.PBF_no_vorticity()
        PBF_end = time.time()
        PBF_time += PBF_end - PBF_start

        rendering_start = time.time()
        scene.particles(Sim.positions, radius=Sim.particle_radius, per_vertex_color=Sim.colour, index_count=Sim.total_particles[0])

        # Sim.diffuse_positions.copy_from(Sim.diffuse_particles.pos)
        # scene.particles(Sim.diffuse_positions, radius=Sim.particle_radius*0.5, color=(1.0,1.0,1.0),index_count=0)

        canvas.scene(scene)
        window.show()
        rendering_end = time.time()
        rendering_time += rendering_end - rendering_start

        if frame % 20 == 0:
            print("Frame:", frame)
            print("PBF time:", PBF_end - PBF_start)
            # num = densities.to_numpy()
            # max_,min_,avg= np.max(num), np.min(num), np.mean(num)
            # print("Max neighbours:",max_, "Min neighbours:", min_, "avg :", avg)
            # print("Num diffuse:",diffuse_count[0])
        if frame == 400:
            Sim.spawn_fluid((10.0,20.5,10.0),10.0,10.0,10.0,0.7)
        frame += 1
    print("Total PBF time:", PBF_time)
    print("Average PBF framerate:", num_frames / PBF_time)
    print("\nTotal rendering time:", rendering_time)
    print("Average rendering framerate:", num_frames / rendering_time)
    print("\n")

# Project

In order to be able to run this code you will have to create a virtual enviroment and pip install the following:
- numpy
- taichi
- taichi-glsl

In the tests,py folder there are a few tests that can be ran straight away.
The basic order of creating your own scenario is:
- initialising taichi
- setting up camera parameters
- creating a simulator and marching cube object
- spawn some water
- in a loop run sim.PBF() followed by sim.diffuse()
- copy sim.diffuse_particles to sim.particle positions
- display the particles using taichis scene,canvas and window objects
- if you want to see the mesh then instead of rendering particles copy the particle positions from the simulator to the marching cubes object as well as the particle counts.
- Then run cubes.marching_cubes and use cubes.triangles scene.mesh

These steps are all shown in the tests copying them and changing parameters should work.

If the program doesnt work change parameters to have a smaller boundary or to have a smaller amount particles.

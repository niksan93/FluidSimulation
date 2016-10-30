/*
	Header file for CUDA kernel calculations on GPU.
*/

#include "face.h"
#include "particle.h"
#include <windows.h>
#include <glm/glm.hpp>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
using namespace std;

#define threadsPerBlock 128							// the amount of thread in block
#define H 1.2f										// smoothing length for smoothing kernel
#define H_POW_6 (float)(H*H*H*H*H*H)				// smoothing length in 6 degree, for spiky kernel
#define H_POW_9 (float)(H*H*H*H*H*H*H*H*H)			// smoothing length 9 degree, for poly6 kernel
#define PI 3.14159265f								// Pi
#define NEIGHBOR_MAX_NUM 20							// the maximum amount of neighbors of a particle
#define EPSILON 0.01f								// epsilon
#define K 0.1f										// small positive constant
#define N_POW 4										// the degree of s_corr calculation
#define DELTA_Q 0.2f								// koefficient of H for s_corr calculation
#define C 0.01f										// parameter for calculating viscosity
#define REST_DENSITY 10.0f							// density in a rested state
#define SOLVER_ITERATIONS 3							// the amount of iterations
#define DAMBREAK 1									// dam break simulation parameter
#define CUBEFALL 2									// cube fall simulation parameter
#define PYRAMIDFALL 3								// pyramid fall simulation parameter
#define MUGFALL 4									// mug fall simulation parameter

// sizes of a container for particles
#define BOX_X 41							
#define BOX_Y 55
#define BOX_Z 36

// host method for calling from renderScene, updates the particles' positions
void update(float4* pos, float delta_time, int particlesNumber, float move_wall, float move_wall_y, float move_wall_z, bool applyVorticity, float objMaxX, float objMaxY, float objMaxZ, float objMinX, float objMinZ, float objMinY, vector<float>* vec);
// host method for calling from renderScene, generates particles
void initialize(float4* pos, int n, int particlesNumber, float current_Box_x, float current_Box_y, float current_Box_z, std::vector<facePlane>objFaces, int appEvent, vector<float>* vec);
// allocs memory on GPU
void mallocCUDA(int particlesNumber);
// host method for calling from renderScene, generates additional particles
void initializeAdditionalParticles(float4* pos, int n, int particlesNumber, float current_Box_x, float current_Box_y, float current_Box_z);
// free memory on GPU
void freeCUDA();
// host method containing device methods searching for particle neighbors
void findNeighbors(int particlesNumber);
// host method updating the object on the scene
void updateObject(std::vector<facePlane>objFaces);

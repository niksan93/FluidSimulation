/*
	CUDA file for kernel calculations on GPU.
*/

#include "kernel.h"

particle* particles;	// particles array
facePlane* objPlanes;	// array for storing face planes of an object
int* neighbors;			// neighbots array
int* num_neighbors;		// array storing the amount of neighbots of each particle
int* grid_ind;			// array storing cells in which particles are located in the grid
int* grid;				// indicies of particles, which are first in each cell
bool applyVorticity_old = true;
int objFaceNum = 0;	// the amount of face planes of an object

// external forces on X and Z axes
float inForceX = 0.0f, inForceZ = 0.0f;

// grid size
int gridSize = 2 * (BOX_X + 2) * 2 * (BOX_Y + 2) * 2 * (BOX_Z + 2);

#include <iostream>
#include <sstream>

#define DBOUT( s )            \
{                             \
	std::wostringstream os_;    \
	os_ << s << "\t";                   \
	OutputDebugStringW(os_.str().c_str());  \
}

// +++++++++++++++++++++++++++++++++++++++
// GRID methods
// +++++++++++++++++++++++++++++++++++++++

// clear grid
__global__ void clearGrid(int* grid, int gridSize) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < gridSize) {
		grid[index] = -1;
	}
}

// finds the index of cells, in which each particle is located in the grid
__global__ void findParticleCell(particle* particles, int particlesNumber, int* grid_ind) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		int x, y, z;
		x = (int)(2.0f * (particles[index].predicted_p.x + 2));
		y = (int)(2.0f * (particles[index].predicted_p.y + 2));
		z = (int)(2.0f * (particles[index].predicted_p.z + 2));

		int grid_index = (BOX_X + 2) * 2 * (BOX_Y + 2) * 2 * z + y * 2 * (BOX_X + 2) + x;
		grid_ind[index] = grid_index;
	}
}

// finds the first particles in a cell and stores its number
__global__ void findFirstParticlesInCell(int* grid_ind, int* grid, int gridSize, int particlesNumber) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		
		if (index == 0) {	// if its the first particle, store its index in the grid
			grid[grid_ind[index]] = index;
		}
		else {				// check, if the cell is located for the first time, store it
			if (grid_ind[index] != grid_ind[index - 1]) {
				if (grid_ind[index] >= 0 && grid_ind[index] < gridSize)
					grid[grid_ind[index]] = index;
			}
		}
	}
}

// finds the closest neighbors of the particle, <= NEIGHBORS_MAX_NUM
__global__ void findNearestNeighbors(int* grid, int* grid_ind, particle* particles, int particlesNumber, int* neighbors, int* num_neighbors, int gridSize) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		int x, y, z;
		x = (int)(2.0f * (particles[index].predicted_p.x + 2));
		y = (int)(2.0f * (particles[index].predicted_p.y + 2));
		z = (int)(2.0f * (particles[index].predicted_p.z + 2));

		int neighborsNumber = 0;
		int grid_index;

		glm::vec4 another_p, p = particles[index].predicted_p;
		float max, r;
		int max_index, beginCell, currentCell;

		// check the particles around
		for (int i = int(-H) * 2 + z; i <= int(H) * 2 + z; i++){
			for (int j = int(-H) * 2 + y; j <= int(H) * 2 + y; j++){
				for (int k = int(-H) * 2 + x; k <= int(H) * 2 + x; k++){
					grid_index = (BOX_X + 2) * 2 * (BOX_Y + 2) * 2 * i + j * 2 * (BOX_X + 2) + k;

					if (grid_index >= gridSize || grid_index < 0){
						continue;
					}

					beginCell = grid[grid_index];

					if (beginCell < 0) continue;

					currentCell = beginCell;
					while (currentCell < particlesNumber && grid_ind[beginCell] == grid_ind[currentCell]){
						if (currentCell == index){
							++currentCell;
							continue;
						}
						another_p = particles[currentCell].predicted_p;
						r = glm::length(p - another_p);

						if (neighborsNumber < NEIGHBOR_MAX_NUM){
							if (r < H){
								neighbors[index * NEIGHBOR_MAX_NUM + neighborsNumber] = currentCell;
								++neighborsNumber;
							}
						}
						else{
							max = glm::length(p - particles[neighbors[index * NEIGHBOR_MAX_NUM]].predicted_p);
							max_index = 0;
							for (int m = 1; m < neighborsNumber; m++){
								float d = glm::length(p - particles[neighbors[index * NEIGHBOR_MAX_NUM + m]].predicted_p);
								if (d > max){
									max = d;
									max_index = m;
								}
							}

							if (r < max && r < H){
								neighbors[index * NEIGHBOR_MAX_NUM + max_index] = currentCell;
							}
						}

						++currentCell;
					}
				}
			}
		}
		num_neighbors[index] = neighborsNumber;
	}
}

// +++++++++++++++++++++++++++++++++++++++
// SOLVER methods
// +++++++++++++++++++++++++++++++++++++++

// calculates poly6Kernel. r = distance between two particles
__device__ float poly6Kernel(float r) {
	float result;
	float temp = (H*H - r*r)*(H*H - r*r)*(H*H - r*r);
	result = 315.0f / (64.0f * PI*H_POW_9) * temp;
	return result;
}

// calculates gradient spikyKernel according to the formula:
// -45*(h-r)^2/(Pi*h^6)
//	r = distance between two particles
__device__ glm::vec3 spikyKernelGradient(particle p_i, particle p_j) {
	glm::vec3 r = glm::vec3(p_i.predicted_p - p_j.predicted_p);
	float temp = (H - glm::length(r))*(H - glm::length(r));
	float gradient_magnitude = -45.0f * temp / (PI*H_POW_6);
	float div = (glm::length(r));
	return gradient_magnitude * 1.0f / div * r;
}

// calculates the constraint of density C_i for i-th particle
__device__ float calculateCi(int index, particle* particles, int particlesNumber, int* neighbors, int* num_neighbors) {
	float density = 0.0f;
	for (int j = 0; j < num_neighbors[index]; ++j) {
		density += poly6Kernel(glm::length(particles[index].predicted_p - particles[neighbors[index * NEIGHBOR_MAX_NUM + j]].predicted_p));
	}
	return (density / float(REST_DENSITY)) - 1.0f;
}

// calculates gradient on the constraint C_i corresonding to the particle k,
// k = j
__device__ glm::vec3 calculateGradientCi(particle particle_i, particle particle_k) {
	glm::vec3 grad(0.0f);
	grad = -spikyKernelGradient(particle_i, particle_k);
	return 1.0f / float(REST_DENSITY) * grad;
}

// calculates gradient on the constraint C_i corresonding to the particle k,
// k = i
__device__ glm::vec3 calculateGradientCiPart_i(int index, particle* particles, int particlesNumber, int* neighbors, int* num_neighbors) {
	glm::vec3 sum_grad(0.0f);
	for (int i = 0; i < num_neighbors[index]; ++i) {
		sum_grad += spikyKernelGradient(particles[index], particles[neighbors[index * NEIGHBOR_MAX_NUM + i]]);
	}
	return 1.0f / float(REST_DENSITY) * sum_grad;
}

// calculates parameter of artificial density s_corr
__device__ float calculate_s_corr(particle particle_i, particle particle_j) {
	float s_corr, up_kernel,	// numerator
		down_kernel,	// denominator
		temp;
	up_kernel = poly6Kernel(glm::length(particle_i.predicted_p - particle_j.predicted_p));
	down_kernel = poly6Kernel(DELTA_Q * H);
	if (down_kernel < 0.000000001f)
		return 0.0f;
	temp = up_kernel / down_kernel;
	s_corr = - 1.0f * K * pow(temp, N_POW);
	return s_corr;
}

// updates lambda of all particles
__global__ void updateLambdas(particle* particles, int particlesNumber, int* neighbors, int* num_neighbors) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		float lambda;
		float numerator,		// numerator
			denominator = 0.0f,	// denominator
			grad_Ci = 0.0f;
		numerator = -1.0f * calculateCi(index, particles, particlesNumber, neighbors, num_neighbors);
		for (int k = 0; k < num_neighbors[index]; ++k) {
			grad_Ci += glm::length(calculateGradientCi(particles[index], particles[neighbors[index * NEIGHBOR_MAX_NUM + k]]));
			denominator += (grad_Ci * grad_Ci);
		}
		grad_Ci = glm::length(calculateGradientCiPart_i(index, particles, particlesNumber, neighbors, num_neighbors));
		denominator += (grad_Ci * grad_Ci);
		lambda = numerator / (denominator + EPSILON);

		particles[index].lambda = lambda;
	}
}

// calculates the change of positions of particles corresponding to each other
__global__ void calculatePositions(particle* particles, int particlesNumber, int* neighbors, int* num_neighbors) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		glm::vec3 delta(0.0f);
		float s_corr;
		for (int j = 0; j < num_neighbors[index]; ++j) {
			s_corr = calculate_s_corr(particles[index], particles[neighbors[index * NEIGHBOR_MAX_NUM + j]]);/* */
			float lambda_sum = particles[index].lambda + particles[neighbors[index * NEIGHBOR_MAX_NUM + j]].lambda + s_corr;
			delta += lambda_sum * spikyKernelGradient(particles[index], particles[neighbors[index * NEIGHBOR_MAX_NUM + j]]);
		}
		delta = 1.0f / REST_DENSITY * delta;
		particles[index].delta_position = glm::vec4(delta, 0.0f);
	}
	
}

// applies external forces to particles
__global__ void apply_forces(particle* particles, int particlesNumber, float delta_time) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		particle p = particles[index];
		p.velocity += p.external_forces * delta_time;
		p.predicted_p = p.position + glm::vec4(p.velocity, 0.0f) * delta_time;
		particles[index] = p;
	}
}

// updates particles' positions
__global__ void updatePositions(particle* particles, int particlesNumber) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		particles[index].position = particles[index].predicted_p;
	}
}

// updates the predicted positions by delta_position
__global__ void updatePredictedPositions(particle* particles, int particlesNumber) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		particles[index].predicted_p += particles[index].delta_position;
	}
}

// initializes particles in the default state, corresponding to the type of simulation
__global__ void initializeParticles(particle* particles, int n, float current_Box_x, float current_Box_y, float current_Box_z, float forceX, float forceZ, int appEvent) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < n * n * n) {
		int x = blockIdx.x % n;
		int y = blockIdx.x / n;

		float gravity_force = -9.8f;

		particle p = particles[index];

		if (appEvent == DAMBREAK) {
			p.position = glm::vec4((n*0.4f) / 100.0f + x*0.4f, (n*0.4f) / 50.0f + y*0.4f, (n*0.4f) / 100.0f + threadIdx.x * 0.4f, 1.0f);
		}
		if (appEvent == CUBEFALL) {
			p.position = glm::vec4((current_Box_x - n*0.4f) / 2.0f + x*0.4f, (current_Box_y - n*0.4f) / 2.0f + y*0.4f, (current_Box_z - n*0.4f) / 2.0f + threadIdx.x * 0.4f, 1.0f);
		}
		if (appEvent == PYRAMIDFALL) {
			p.position = glm::vec4((current_Box_x - n*0.4f) / 2.0f + x*0.4f, (current_Box_y - n*0.4f) / 2.0f + y*0.4f, (current_Box_z - n*0.4f) / 2.0f + threadIdx.x * 0.4f, 1.0f);
		}
		if (appEvent == MUGFALL) {
			p.position = glm::vec4((current_Box_x - n*0.4f) / 2.0f + x*0.4f, (current_Box_y - n*0.4f) / 2.0f + y*0.4f, (current_Box_z - n*0.4f) / 2.0f + threadIdx.x * 0.4f, 1.0f);
		}

		p.external_forces = glm::vec3(forceX, gravity_force, forceZ);
		p.velocity = glm::vec3(0.0f);
		p.lambda = 0.0f;
		p.delta_position = glm::vec4(0.0f);
		p.vorticity = glm::vec3(0.0f);
		particles[index] = p;
	}
}

// initializes the additional cube of particles in the default state
__global__ void initializeAddParticles(particle* particles, int n, int particlesNumber, float current_Box_x, float current_Box_y, float current_Box_z) {
	int index = particlesNumber - (blockDim.x*blockIdx.x + threadIdx.x) - 1;
	if (index < particlesNumber) {
		int x = blockIdx.x % n;
		int y = blockIdx.x / n;

		float gravity_force = -9.8f;

		particle p = particles[index];
		p.position = glm::vec4((float)(current_Box_x - n*0.4f) / 2.0f + x*0.4f, (float)current_Box_y - 3.0f - y*0.4f, (float)(current_Box_z - n*0.4f) / 2.0f + threadIdx.x * 0.4f, 1.0f);
		p.external_forces = glm::vec3(0.0f, gravity_force, 0.0f);
		p.velocity = glm::vec3(0.0f);
		p.lambda = 0.0f;
		p.delta_position = glm::vec4(0.0f);
		p.vorticity = glm::vec3(0.0f);
		particles[index] = p;
	}
}

// updates velocity vector of calculating delta p 
__global__ void updateVelocity(particle* particles, int particlesNumber, float delta_time)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber){
		particles[index].velocity = glm::vec3((1.0f / delta_time)*(particles[index].predicted_p - particles[index].position));
	}
}

// transfers data from the array of particles to VBO
__global__ void updateVBO(float4* positions, int particlesNumber, particle* particles) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		float x, y, z;
		x = particles[index].position.x*1.0f;
		y = particles[index].position.y*1.0f;
		z = particles[index].position.z*1.0f;
		positions[index] = make_float4(x, y, z, 1.0f);
	}
}

// performs collision detection with a static object and updates the velocity vector
__global__ void objectCollision(particle* particles, int particlesNumber, facePlane* faces, int facesNum, float objMaxX, float objMaxY, float objMaxZ, float objMinX, float objMinZ, float objMinY) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		glm::vec4 currentPos = particles[index].position;
		glm::vec4 predictedPos = particles[index].predicted_p + particles[index].delta_position;
		if ((predictedPos.x > objMinX) && (predictedPos.x < objMaxX) && (predictedPos.y < objMaxY) && (predictedPos.z < objMaxZ) && (predictedPos.z > objMinZ) && (predictedPos.y > objMinY)) {
			for (int i = 0; i < facesNum; i++) {
				glm::vec3 intersectionPoint;
				glm::vec3 planeVertex1 = faces[i].coord1;
				glm::vec3 planeVertex2 = faces[i].coord2;
				glm::vec3 planeVertex3 = faces[i].coord3;
				glm::vec3 planeNormal = faces[i].normal;
				glm::vec3 origin; origin.x = currentPos.x; origin.y = currentPos.y; origin.z = currentPos.z;
				glm::vec3 direction; direction.x = predictedPos.x; direction.y = predictedPos.y; direction.z = predictedPos.z;
				glm::vec3 vx = direction - origin;
				glm::normalize(vx);

				glm::vec3 E1 = planeVertex2 - planeVertex1;
				glm::vec3 E2 = planeVertex3 - planeVertex1;
				glm::vec3 T = origin - planeVertex1;
				glm::vec3 P = glm::cross(vx, E2);
				glm::vec3 Q = glm::cross(T, E1);
				float dotPE1 = 1 / glm::dot(P, E1);

				float t;
				/*if (denom != 0) {
					t = -(d + planeNormal.x * currentPos.x + planeNormal.y * currentPos.y + planeNormal.z * currentPos.z) / denom;
					intersectionPoint.x = origin.x + t * vx.x;
					intersectionPoint.y = origin.y + t * vx.y;
					intersectionPoint.z = origin.z + t * vx.z;
				}
				else
					continue;*/

				 t = dotPE1 * glm::dot(Q, E2);
				intersectionPoint.x = origin.x + t * vx.x;
				intersectionPoint.y = origin.y + t * vx.y;
				intersectionPoint.z = origin.z + t * vx.z;
				float u = dotPE1 * glm::dot(P, T);
				float v = dotPE1 * glm::dot(Q, vx);
				float t1 = 1 - u - v;

				glm::vec3 Sv = glm::cross(planeVertex2 - planeVertex1, planeVertex3 - planeVertex1);
				float S = glm::sqrt(Sv.x * Sv.x + Sv.y * Sv.y + Sv.z * Sv.z);
				glm::vec3 uv = glm::cross(planeVertex2 - intersectionPoint, planeVertex1 - intersectionPoint);
				u = glm::sqrt(uv.x * uv.x + uv.y * uv.y + uv.z * uv.z);
				glm::vec3 vv = glm::cross(planeVertex2 - intersectionPoint, planeVertex3 - intersectionPoint);
				v = glm::sqrt(vv.x * vv.x + vv.y * vv.y + vv.z * vv.z);
				glm::vec3 t1v = glm::cross(planeVertex3 - intersectionPoint, planeVertex1 - intersectionPoint);
				t1 = glm::sqrt(t1v.x * t1v.x + t1v.y * t1v.y + t1v.z * t1v.z);

				if (glm::abs(u + v + t1 - S) < 0.001f) {
					//line equation intersection
					intersectionPoint.x = origin.x + t * vx.x;
					intersectionPoint.y = origin.y + t * vx.y;
					intersectionPoint.z = origin.z + t * vx.z;

					if ((((origin.x <= intersectionPoint.x) && (intersectionPoint.x <= direction.x)) || (((direction.x <= intersectionPoint.x) && (intersectionPoint.x <= origin.x)))) &&
						(((origin.y <= intersectionPoint.y) && (intersectionPoint.y <= direction.y)) || (((direction.y <= intersectionPoint.y) && (intersectionPoint.y <= origin.y)))) &&
						(((origin.z <= intersectionPoint.z) && (intersectionPoint.z <= direction.z)) || (((direction.z <= intersectionPoint.z) && (intersectionPoint.z <= origin.z))))) {
						particles[index].predicted_p = glm::vec4(intersectionPoint, 1.0f) + glm::vec4(planeNormal, 1.0f) * 0.1f;
						particles[index].predicted_p -= particles[index].delta_position;
						glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, planeNormal)) * planeNormal);
						particles[index].velocity = changed_direction;
					}
				}
			}
		}
	}
}

// performs collision detection with the boundaries of the container and updates the velocity vector
__global__ void performCollisionResponse(particle* particles, int particlesNumber, float move_wall_x, float move_wall_y, float move_wall_z, facePlane* faces, int facesNum) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		if (particles[index].predicted_p.y < 0.1f)
		{
			particles[index].predicted_p.y = 0.1001f;
			glm::vec3 floor_normal(0.0f, 1.0f, 0.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}
		if (particles[index].predicted_p.y > BOX_Y + move_wall_y - 0.1f)
		{
			particles[index].predicted_p.y = BOX_Y + move_wall_y - 0.1f - 0.0001f;
			glm::vec3 floor_normal(0.0f, -1.0f, 0.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}
		
		if (particles[index].predicted_p.x < 0.1f)
		{
			particles[index].predicted_p.x = 0.1001f;
			glm::vec3 floor_normal(1.0f, 0.0f, 0.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}

		if (particles[index].predicted_p.x > BOX_X + move_wall_x - 0.1f)
		{
			particles[index].predicted_p.x = BOX_X + move_wall_x - 0.1f - 0.0001f;
			glm::vec3 floor_normal(-1.0f, 0.0f, 0.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}
		
		if (particles[index].predicted_p.z > BOX_Z + move_wall_z - 0.1f)
		{
			particles[index].predicted_p.z = BOX_Z + move_wall_z - 0.1f - 0.0001f;
			glm::vec3 floor_normal(0.0f, 0.0f, -1.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}

		if (particles[index].predicted_p.z < 0.1f)
		{
			particles[index].predicted_p.z = 0.1001f;
			glm::vec3 floor_normal(0.0f, 0.0f, 1.0f);
			glm::vec3 changed_direction = particles[index].velocity - glm::vec3(2.0f * (glm::dot(particles[index].velocity, floor_normal)) * floor_normal);
			particles[index].velocity = changed_direction;
		}
	}
}

// adds viscosity
__global__ void applyViscosity(particle* particles, int particlesNumber, int* neighbors, int* num_neighbors) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	glm::vec3 new_vel(0.0f);
	if (index < particlesNumber) {
		int neigh_num = num_neighbors[index];
		particle p_i = particles[index], p_j;
		for (int j = 0; j < neigh_num; ++j) {
			p_j = particles[neighbors[NEIGHBOR_MAX_NUM * index + j]];
			glm::vec3 vel_i_j = p_j.velocity - p_i.velocity;
			new_vel += vel_i_j * poly6Kernel(glm::length(p_i.predicted_p - p_j.predicted_p));
		}
		new_vel = p_i.velocity + C * new_vel;
		particles[index].velocity = new_vel;
	}
}

// calculates vorticity in the positions of particles
__global__ void calculateVorticity(particle* particles, int particlesNumber, int* neighbors, int* num_neighbors){
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		glm::vec3 vorticity(0.0f);
		particle p_i = particles[index], p_j;
		glm::vec3 v_i = p_i.velocity,
			v_j;

		for (int j = 0; j < num_neighbors[index]; ++j) {
			p_j = particles[neighbors[NEIGHBOR_MAX_NUM * index + j]];
			v_j = p_j.velocity;
			glm::vec3 v_ij = v_j - v_i;
			vorticity += glm::cross(v_ij, spikyKernelGradient(p_i, p_j));
		}
		particles[index].vorticity = vorticity;
	}
}

// adds vorticity
__global__ void applyVorticityConfinement(particle* particles, int particlesNumber, float forceX, float forceZ) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		glm::vec3 vort_i = particles[index].vorticity;

		glm::vec3 N = glm::vec3(glm::length(vort_i)) / glm::length(vort_i);
		glm::vec3 vorticity_force =	0.25f*glm::cross(N, vort_i);
		particles[index].external_forces = glm::vec3(forceX, -9.8f, forceZ) + vorticity_force;
	}
}
// sets external forces on default
__global__ void setDefaultExternalForces(particle* particles, int particlesNumber) {
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if (index < particlesNumber) {
		particles[index].external_forces = glm::vec3(0.0f, -9.8f, 0.0f);
	}
}

void showArray(int* arr, int matSize) {
	DBOUT("\r");
	for (int i = 0; i < matSize; i++) {
		DBOUT(arr[i]);
	}
	DBOUT("ARRAYEND\r");
}

// radix sort by key.
// source - valye array, dest - destination value array
// sourceP - key array, destP - destination key array
void radix(int byte, int len, int *source, int *dest, particle* sourceP, particle* destP)
{
	int count[256];
	int index[256];
	memset(count, 0, sizeof (count));
	for (int i = 0; i<len; i++) { 
		count[((source[i]) >> (byte * 8)) & 0xff]++; 
	}
	index[0] = 0;
	for (int i = 1; i<256; i++) { 
		index[i] = index[i - 1] + count[i - 1]; 
	}
	for (int i = 0; i<len; i++) { 
		destP[index[((source[i]) >> (byte * 8)) & 0xff]] = sourceP[i];
		dest[index[((source[i]) >> (byte * 8)) & 0xff]++] = source[i]; 
	}
}


void printApproxMS(vector<float> vec) {
	int size = vec.size();
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += vec[i];
	}
	sum /= size;
	DBOUT(sum);
}



// finds particle neighbors
void findNeighbors(int particlesNumber, vector<float>*vec) {
	dim3 blocks = dim3((BOX_X + 2) * 2 * (BOX_Y + 2) * 2);
	dim3 threads = dim3(2 * (BOX_Z + 2));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	clearGrid << <blocks, threads >> > (grid, gridSize);

	blocks = dim3((particlesNumber + threadsPerBlock - 1) / threadsPerBlock);
	threads = dim3(threadsPerBlock);
	cudaEventRecord(start);
	findParticleCell << <blocks, threads >> > (particles, particlesNumber, grid_ind);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[2].push_back(milliseconds);

	cudaEventRecord(start);
	particle* hostParticles = new particle[particlesNumber];
	int* hostGrid_ind = new int[particlesNumber];
	
	cudaMemcpy(hostParticles, particles, particlesNumber*sizeof(particle), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostGrid_ind, grid_ind, particlesNumber*sizeof(int), cudaMemcpyDeviceToHost);
	
	// creating supporting host array for sorting purposes
	particle* newHostParticles;
	int* newHostGrid_ind;
	newHostParticles = new particle[particlesNumber];
	newHostGrid_ind = new int[particlesNumber];

	radix(0, particlesNumber, hostGrid_ind, newHostGrid_ind, hostParticles, newHostParticles);
	radix(1, particlesNumber, newHostGrid_ind, hostGrid_ind, newHostParticles, hostParticles);
	radix(2, particlesNumber, hostGrid_ind, newHostGrid_ind, hostParticles, newHostParticles);
	radix(3, particlesNumber, newHostGrid_ind, hostGrid_ind, newHostParticles, hostParticles);

	cudaMemcpy(particles, hostParticles, particlesNumber*sizeof(particle), cudaMemcpyHostToDevice);
	cudaMemcpy(grid_ind, hostGrid_ind, particlesNumber*sizeof(int), cudaMemcpyHostToDevice);

	delete[]newHostGrid_ind;
	delete[]newHostParticles;
	delete[]hostParticles;
	delete[]hostGrid_ind;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[3].push_back(milliseconds);

	cudaEventRecord(start);
	findFirstParticlesInCell << <blocks, threads >> >(grid_ind, grid, gridSize, particlesNumber);

	findNearestNeighbors << <blocks, threads >> >(grid, grid_ind, particles, particlesNumber, neighbors, num_neighbors, gridSize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[4].push_back(milliseconds);
}


// +++++++++++++++++++++++++++++++++++++++
// CUDA methods
// +++++++++++++++++++++++++++++++++++++++

// allocates memory on GPU
void mallocCUDA(int particlesNumber) {
	cudaMalloc((void**)&particles, particlesNumber * sizeof(particle));

	cudaMalloc((void**)&neighbors, NEIGHBOR_MAX_NUM * particlesNumber * sizeof(int));
	cudaMalloc((void**)&num_neighbors, particlesNumber * sizeof(int));

	cudaMalloc((void**)&grid_ind, particlesNumber * sizeof(int));
	cudaMalloc((void**)&grid, gridSize * sizeof(int));
}

// host method for calling from renderScene, updates the particles' positions
void update(float4* pos, float delta_time, int particlesNumber, float move_wall_x, float move_wall_y, float move_wall_z, bool applyVorticity, float objMaxX, float objMaxY, float objMaxZ, float objMinX, float objMinZ, float objMinY, vector<float>*vec) {
	dim3 blocks = dim3((particlesNumber + threadsPerBlock - 1)/ threadsPerBlock);
	dim3 threads = dim3(threadsPerBlock);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	
	if (!applyVorticity) {
		if (applyVorticity_old)
		{
			setDefaultExternalForces << <blocks, threads >> > (particles, particlesNumber);
			applyVorticity_old = false;
		}
	}
	cudaEventRecord(start);
	apply_forces << <blocks, threads >> > (particles, particlesNumber, delta_time);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[1].push_back(milliseconds);
	
	findNeighbors(particlesNumber, vec);
	
	cudaEventRecord(start);
	for (int i = 0; i < SOLVER_ITERATIONS; ++i) {
		updateLambdas << <blocks, threads >> > (particles, particlesNumber, neighbors, num_neighbors);
		calculatePositions << <blocks, threads >> > (particles, particlesNumber, neighbors, num_neighbors);
		performCollisionResponse << <blocks, threads >> > (particles, particlesNumber, move_wall_x, move_wall_y, move_wall_z, objPlanes, objFaceNum);
		objectCollision << <blocks, threads >> >(particles, particlesNumber, objPlanes, objFaceNum, objMaxX, objMaxY, objMaxZ, objMinX, objMinZ, objMinY);
		updatePredictedPositions << <blocks, threads >> > (particles, particlesNumber);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[5].push_back(milliseconds);

	cudaEventRecord(start);
	updateVelocity << <blocks, threads >> > (particles, particlesNumber, delta_time);
	applyViscosity << <blocks, threads >> > (particles, particlesNumber, neighbors, num_neighbors);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[6].push_back(milliseconds);

	cudaEventRecord(start);
	if (applyVorticity) {
		calculateVorticity << <blocks, threads >> > (particles, particlesNumber, neighbors, num_neighbors);
		applyVorticityConfinement << <blocks, threads >> > (particles, particlesNumber, inForceX, inForceZ);
		applyVorticity_old = true;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[7].push_back(milliseconds);

	cudaEventRecord(start);
	updatePositions << <blocks, threads >> > (particles, particlesNumber);
	updateVBO << <blocks, threads >> > (pos, particlesNumber, particles);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[8].push_back(milliseconds);

	cudaDeviceSynchronize();
}

// host method for calling from renderScene, generates particles
void initialize(float4* pos, int n, int particlesNumber, float current_Box_x, float current_Box_y, float current_Box_z, vector<facePlane>objFaces, int appEvent, vector<float>* vec) {
	dim3 blocks = dim3(n * n);
	dim3 threads = dim3(n);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);
	objFaceNum = objFaces.size();
	if (!objPlanes) {
		cudaMalloc((void**)&objPlanes, objFaceNum * sizeof(facePlane));
	}
	cudaMemcpy(objPlanes, objFaces.data(), objFaceNum * sizeof(facePlane), cudaMemcpyHostToDevice);

	if (appEvent == DAMBREAK) { inForceZ = 1.0f; }
	if (appEvent == CUBEFALL) { inForceZ = 0.0f; }
	if (appEvent == PYRAMIDFALL) { inForceZ = 0.0f; }
	if (appEvent == MUGFALL) { inForceZ = 0.0f; }
	
	initializeParticles << <blocks, threads >> > (particles, n, current_Box_x, current_Box_y, current_Box_z, inForceX, inForceZ, appEvent);
	updateVBO << <blocks, threads >> > (pos, particlesNumber, particles);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	vec[0].push_back(milliseconds);
}

// host method for calling from renderScene, generates additional particles
void initializeAdditionalParticles(float4* pos, int n, int particlesNumber, float current_Box_x, float current_Box_y, float current_Box_z) {
	dim3 blocks = dim3(n * n);
	dim3 threads = dim3(n);
	initializeAddParticles << <blocks, threads >> > (particles, n, particlesNumber, current_Box_x, current_Box_y, current_Box_z);
	blocks = dim3((particlesNumber + threadsPerBlock - 1) / threadsPerBlock);
	threads = dim3(threadsPerBlock);
	updateVBO << <blocks, threads >> > (pos, particlesNumber, particles);
}

// host method updating the object on the scene
void updateObject(std::vector<facePlane>objFaces) {
	objFaceNum = objFaces.size();
	if (objPlanes) {
		cudaFree(objPlanes);
		objPlanes = NULL;
	}
	if (!objPlanes) {
		cudaMalloc((void**)&objPlanes, objFaces.size() * sizeof(facePlane));
	}
	cudaMemcpy(objPlanes, objFaces.data(), objFaces.size() * sizeof(facePlane), cudaMemcpyHostToDevice);
}

// free memory on GPU
void freeCUDA(){
	cudaFree(objPlanes);
	cudaFree(particles);
	cudaFree(neighbors);
	cudaFree(num_neighbors);
	cudaFree(grid_ind);
	cudaFree(grid);
}
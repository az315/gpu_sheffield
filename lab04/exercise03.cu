#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 2048 //width
#define M 1000 // height
#define THREADS_PER_BLOCK 256
#define SQRT_THREADS_PER_BLOCK sqrt(THREADS_PER_BLOCK)

void checkCUDAError(const char*);
void random_ints(int *a);
void matrixAddCPU(int *a, int *b, int *c_ref);
int validate(int *a, int *c_ref);



__global__ void matrixAdd(int *a, int *b, int *c) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ((ix < N) && (iy < M)) {
    int i = iy*N + ix;
    c[i] = a[i] + b[i];
  }
	
}



int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
  // account for when threadsperblock does not divide N evenly = add extra block
  unsigned int blockWidth = (unsigned int)SQRT_THREADS_PER_BLOCK;
  unsigned int gridWidth = (unsigned int)ceil((double)N / blockWidth);
  unsigned int gridHeight = (unsigned int)ceil((double)M / blockWidth);
  dim3 blocksPerGrid(gridWidth, gridHeight, 1);
  dim3 threadsPerBlock(blockWidth, blockWidth, 1);
	// vectorAdd << <N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> >(d_a, d_b, d_c, N);
  matrixAdd <<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
	checkCUDAError("CUDA kernel");


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

  //CPU version
  matrixAddCPU(a, b, c_ref);

  //validate
  errors = validate(c, c_ref);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
	for (unsigned int ix = 0; ix < N; ix++){
    for (unsigned int iy = 0; iy < M; iy++) {
		  a[iy*N + ix] = rand();
    }
	}
}

void matrixAddCPU(int *a, int *b, int *c_ref) {
  for (unsigned int ix = 0; ix < N; ix++){
    for (unsigned int iy = 0; iy < M; iy++) {
		  int i = (iy*N) + ix;
      c_ref[i] = a[i] + b[i];
    }
	}
  
}

int validate(int *a, int *c_ref) {
  int errors = 0;
  // int len = sizeof(a) / sizeof(a[0]);

  for (unsigned int ix = 0; ix < N; ix++){
    for (unsigned int iy = 0; iy < M; iy++) {
		  int i = (iy*N) + ix;
      if (a[i] != c_ref[i]) {
        errors += 1;
        fprintf(stderr, "Error at index %d: The result %d from matrixAdd does not equal the result %d from matrixAddCPU \n", i, a[i], c_ref[i]);
        printf("Total errors so far: %d \n", errors);
      }
    }
  }

  printf("Total num errors: %d \n", errors);
  return errors;
}

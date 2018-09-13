#include <stdio.h> 

__global__ void print1DThreads() {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  printf("Overall Thread: %d ... Block: %d, Warp: %d, Thread: %d\n", 
         thread_idx, 
         blockIdx.x, 
         threadIdx.x / warpSize,
         threadIdx.x);
}

__global__ void print2DThreads() {
  const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y; 

  // (Number of threads in a row * y position) + x offset from start of row 
  const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

 printf("Overall Thread: %d ... xGrid: %d, yGrid: %d, xBlock: %d, yBlock: %d, Thread: %d\n", 
         thread_idx, 
         gridDim.x,
         gridDim.y,
         blockIdx.x, 
         blockIdx.y, 
         threadIdx.x);
}

int main() {
  const int num_blocks = 2;
  const int num_threads = 64; 

  print1DThreads<<<num_blocks, num_threads>>>();
  cudaDeviceSynchronize(); 

  // Number of blocks in a grid 
  const dim3 blocks(1, 4); 
  // Number of threads in a block 
  const dim3 threads(32, 4);

  print2DThreads<<<blocks, threads>>>();
  cudaDeviceSynchronize(); 
  return 0;
}

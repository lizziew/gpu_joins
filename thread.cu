#include <stdio.h> 

__global__ void printThreads() {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  printf("Overall Thread: %d ... Block: %d, Warp: %d, Thread: %d\n", 
         thread_idx, 
         blockIdx.x, 
         threadIdx.x / warpSize,
         threadIdx.x);
}

int main() {
  const int num_blocks = 2;
  const int num_threads = 64; 

  printThreads<<<num_blocks, num_threads>>>();
  cudaDeviceSynchronize(); 
  return 0;
}

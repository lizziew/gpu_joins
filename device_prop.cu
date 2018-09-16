#include <stdio.h> 

void printDeviceProperties(cudaDeviceProp prop) {
  printf("Device name: %s\n", prop.name);
  printf("Clock rate (KHz): %d\n", prop.clockRate);
  printf("Compute: %d.%d\n", prop.major, prop.minor);
  printf("Total number of SMs: %d\n", prop.multiProcessorCount);
  printf("Device shares CPU ram directly: %d\n", prop.integrated);
  printf("Device can map host memory into GPU virtual memory space: %d\n", prop.canMapHostMemory);
  printf("Total global memory (bytes): %zu\n", prop.totalGlobalMem);
  printf("Total constant memory (bytes): %zu\n", prop.totalConstMem); 
  
  for (int i = 0; i < 3; i++) printf("Blocks per dimension %d: %d\n", i, prop.maxGridSize[i]);
  printf("Shared memory per block (bytes): %zu\n", prop.sharedMemPerBlock);
  printf("Registers per block: %d\n", prop.regsPerBlock);
  printf("Threads per block: %d\n", prop.maxThreadsPerBlock);


  for (int i = 0; i < 3; i++) printf("Threads per dimension %d: %d\n", i, prop.maxThreadsDim[i]);
  printf("Warp size (threads): %d\n", prop.warpSize);
}

int main() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  printf("Number of GPUs: %d\n", num_devices);
  printf("=======================================\n");

  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop; 
    cudaGetDeviceProperties(&prop, i);
    printDeviceProperties(prop); 
    printf("=======================================\n");
  }
}

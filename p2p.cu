// Source: https://stackoverflow.com/questions/31628041/how-to-copy-memory-between-different-gpus-in-cuda

#include <math.h>
#include <stdio.h> 

#define SRC_DEV 0
#define DST_DEV 1

#define DSIZE (16*1048576)

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

int main(int argc, char *argv[]){
  // Any arguments will disable P2P
  int disablePeer = 0;
  if (argc > 1) disablePeer = 1;

  // Get number of devices 
  int devcount;
  cudaGetDeviceCount(&devcount);
  cudaCheckErrors("cuda failure");

  int srcdev = SRC_DEV;
  int dstdev = DST_DEV;

  if (devcount <= max(srcdev,dstdev)) {
    printf("not enough cuda devices for the requested operation\n"); 
    return 1;
  }

  // Allocate host array 
  int *d_s, *d_d, *h;
  int dsize = DSIZE*sizeof(int);
  h = (int *)malloc(dsize);
  if (h == NULL) {
    printf("malloc fail\n"); 
    return 1;
  }
  for (int i = 0; i < DSIZE; i++) 
    h[i] = i;

  // Check if we can get to other GPU
  int canAccessPeer = 0;
  if (!disablePeer) 
    cudaDeviceCanAccessPeer(&canAccessPeer, srcdev, dstdev);

  // Go to source device
  cudaSetDevice(srcdev);

  // Allocate source device array 
  cudaMalloc(&d_s, dsize);
  cudaMemcpy(d_s, h, dsize, cudaMemcpyHostToDevice);
  if (canAccessPeer) 
    cudaDeviceEnablePeerAccess(dstdev,0);

  // Go to destination device 
  cudaSetDevice(dstdev);
  cudaMalloc(&d_d, dsize);
  cudaMemset(d_d, 0, dsize);

  // Check if we can get to other GPU 
  if (canAccessPeer) 
    cudaDeviceEnablePeerAccess(srcdev, 0);
  cudaCheckErrors("cudaMalloc/cudaMemset fail");

  // Print statements 
  if (canAccessPeer) 
    printf("Timing P2P transfer");
  else 
    printf("Timing ordinary transfer");
  printf(" of %d bytes\n", dsize);

  // Transfer from source to device GPU
  cudaEvent_t start, stop;
  cudaEventCreate(&start); 
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaMemcpyPeer(d_d, dstdev, d_s, srcdev, dsize);
  cudaCheckErrors("cudaMemcpyPeer fail");
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Get elapsed time 
  float et;
  cudaEventElapsedTime(&et, start, stop);
  cudaSetDevice(dstdev);
  cudaMemcpy(h, d_d, dsize, cudaMemcpyDeviceToHost);

  // Check for failures 
  cudaCheckErrors("cudaMemcpy fail");
  for (int i = 0; i < DSIZE; i++)
    if (h[i] != i) {
      printf("transfer failure\n"); 
      return 1;
    }
  printf("transfer took %fms\n", et);
  return 0;
}

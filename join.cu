// make setup; make gpu_join; ./bin/gpu/join > out
// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cub/test/test_util.h"

#include "utils/generator.h"
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

#define DEBUG 1

__device__ __forceinline__
int HASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__global__
void build_hashtable_dev(int *d_dim_key, int *d_dim_val, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int key = d_dim_key[offset];
  int val = d_dim_val[offset];
  int hash = HASH(key, num_slots);

  hash_table[hash << 1] = key;
  hash_table[(hash << 1) + 1] = val;
  // printf("key and value are %d %d\n", key, val);
}

__global__
void probe_hashtable_dev(int *d_fact_fkey, int *d_fact_val, int num_tuples, int *hash_table, int num_slots, unsigned long long *res) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  unsigned long long checksum = 0;

  for (int i = offset; i < num_tuples; i += stride) {
    int key = d_fact_fkey[i];
    int val = d_fact_val[i];
    int hash = HASH(key, num_slots);

    int2 slot = reinterpret_cast<int2*>(hash_table)[hash];
    // printf("Hmmmm %d %d\n", slot.x, key);
    if (slot.x == key) {
      // printf("%d %d\n", slot.x, key);
      checksum += slot.y + val;
    }
  }

  atomicAdd(res, checksum);
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};

static int num_runs = 0;
static unsigned long long* h_res = 0;

TimeKeeper hashJoin(int* d_dim_key, int* d_dim_val, int* d_fact_fkey, int* d_fact_val, int num_dim, int num_fact, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  int* hash_table = NULL;
  unsigned long long* res;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  ALLOCATE(hash_table, sizeof(int) * 2 * num_dim);
  ALLOCATE(res, sizeof(long long));

  TIME_FUNC(cudaMemset(hash_table, 0, num_slots * sizeof(int) * 2), time_memset);
  TIME_FUNC(cudaMemset(res, 0, sizeof(long long)), time_memset2);

  TIME_FUNC((build_hashtable_dev<<<num_dim/128, 128>>>(d_dim_key, d_dim_val, num_dim, hash_table, num_slots)), time_build);

  cudaDeviceSynchronize(); 

  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_fkey, d_fact_val, num_fact, hash_table, num_slots, res)), time_probe);
  cudaDeviceSynchronize(); 

#if DEBUG
  cout << "{" << "\"time_memset\":" << time_memset
    << ",\"time_build\"" << time_build
    << ",\"time_probe\":" << time_probe << "}" << endl;
#endif

  num_runs += 1;
  if (num_runs == 3) {
    h_res = new unsigned long long[1];
    CubDebugExit(cudaMemcpy(h_res, res, sizeof(long long), cudaMemcpyDeviceToHost));
    cout << h_res[0] << endl;
  }

  CLEANUP(hash_table);
  CLEANUP(res);

  TimeKeeper t = {time_build, time_probe, time_memset, time_build + time_probe + time_memset};
  return t;
}

void RunHashJoinCPU(int *dim_key, int* dim_val, int* fact_fkey, int* fact_val, int* hash_table, long long* res, int num_dim, int num_fact, int num_slots) {
  for (int i = 0; i < num_dim; i++) {
    int key = dim_key[i];
    int val = dim_val[i];

    int hash = key & (num_slots - 1);

    hash_table[hash << 1] = key;
    hash_table[(hash << 1) + 1] = val;
  }

  for (int i = 0; i < num_fact; i++) {
    int key = fact_fkey[i];
    int val = fact_val[i];

    int hash = key & (num_slots - 1);

    if (hash_table[hash << 1] == key) {
      *res = *res + (hash_table[(hash << 1) + 1] + val);
    } 
  }
}

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


#define CLEANUP(vec) if(vec)CubDebugExit(g_allocator.DeviceFree(vec))

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
  int num_fact           = 256 * 1 << 20; // 256 * 1 << 20 , 1 << 28
  int num_dim            = 16 * 1 << 20; // 16 * 1 << 20 , 1 << 16
  int num_trials         = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("n", num_fact);
  args.GetCmdLineArgument("d", num_dim);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
        "[--n=<num fact>] "
        "[--d=<num dim>] "
        "[--t=<num trials>] "
        "[--device=<device-id>] "
        "[--v] "
        "\n", argv[0]);
    exit(0);
  }

  int log2 = 0;
  int num_dim_dup = num_dim >> 1;
  while (num_dim_dup) {
    num_dim_dup >>= 1;
    log2 += 1;
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Allocate problem device arrays
  int *d_dim_key = NULL;
  int *d_dim_val = NULL;
  int *d_fact_fkey = NULL;
  int *d_fact_val = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_key, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_val, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_fkey, sizeof(int) * num_fact));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_val, sizeof(int) * num_fact));

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  CubDebugExit(cudaMemcpy(d_dim_key, h_dim_key, sizeof(int) * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_val, h_dim_val, sizeof(int) * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_fkey, h_fact_fkey, sizeof(int) * num_fact, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_val, h_fact_val, sizeof(int) * num_fact, cudaMemcpyHostToDevice));

  /*cout << "DIM TABLE:" << endl;
  for (int i = 0; i < num_dim; i++) cout << h_dim_key[i] << "..." << h_dim_val[i] << endl;
  cout << endl;

  cout << "FACT TABLE:" << endl;
  for (int i = 0; i < num_fact; i++) cout << h_fact_fkey[i] << "..." << h_fact_val[i] << endl;
  cout << endl;*/

  for (int j = 0; j < num_trials; j++) {
    cout << "TRIAL " << j << endl;
    TimeKeeper t = hashJoin(d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact, g_allocator);
    cout<< "{"
      << "\"num_dim\":" << num_dim
      << ",\"num_fact\":" << num_fact
      << ",\"radix\":" << 0
      << ",\"time_partition_build\":" << 0
      << ",\"time_partition_probe\":" << 0
      << ",\"time_partition_total\":" << 0
      << ",\"time_build\":" << t.time_build
      << ",\"time_probe\":" << t.time_probe
      << ",\"time_extra\":" << t.time_extra
      << ",\"time_join_total\":" << t.time_total
      << "}" << endl;
    cout << endl;
  }

  int *d_fact_fkey_copy;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_fkey_copy, sizeof(int) * num_fact));
  CubDebugExit(cudaMemcpy(d_fact_fkey_copy, d_fact_fkey, sizeof(int) * num_fact, cudaMemcpyDeviceToDevice));

  int *d_fact_val_copy;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_val_copy, sizeof(int) * num_fact));
  CubDebugExit(cudaMemcpy(d_fact_val_copy, d_fact_val, sizeof(int) * num_fact, cudaMemcpyDeviceToDevice));

  int *d_buffer1;
  int *d_buffer2;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_buffer1, sizeof(int) * num_fact));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_buffer2, sizeof(int) * num_fact));

  // Checking against hash join on CPU
  int num_slots = num_dim;
  int *hash_table = new int[num_slots * 2];
  long long check_res = 0;
  RunHashJoinCPU(h_dim_key, h_dim_val, h_fact_fkey, h_fact_val, hash_table, &check_res, num_dim, num_fact, num_slots);
  cout << "CPU answer: " << check_res << endl; 

  CLEANUP(d_dim_key);
  CLEANUP(d_dim_val);
  CLEANUP(d_fact_fkey);
  CLEANUP(d_fact_val);

  return 0;
}


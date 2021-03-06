// make setup; make gpu_join; ./bin/gpu/join > out
// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <math.h> 

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cub/test/test_util.h"

#include "utils/generator.h"
#include "utils/gpu_utils.h"

using namespace std;
using namespace cub;

#define DEBUG 0
#define NGPU 2

__device__ __forceinline__
int HASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__forceinline__
int HHASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__global__
void build_hashtable_dev(int *d_dim_key, int *d_dim_val, int start, int num_tuples, 
    int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  int key = d_dim_key[offset];
  int val = d_dim_val[offset];
  int hash = HASH(key, num_slots);

  if (offset >= start && offset < num_tuples) {
#if DEBUG 
    printf("Start: %d < Offset: %d < Num_tuples: %d\n", start, offset, num_tuples);
#endif 
    hash_table[hash << 1] = key;
    hash_table[(hash << 1) + 1] = val;
#if DEBUG
    printf("Wrote %d to position %d in hashtable\n", key, hash << 1);
#endif 
  }
}

__global__
void probe_hashtable_dev(int *d_fact_fkey, int *d_fact_val, int start, int num_tuples, 
    int *hash_table, int num_slots, unsigned long long *res) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;

  unsigned long long checksum = 0;

  if (offset >= start && offset < num_tuples) {
    int key = d_fact_fkey[offset];
#if DEBUG
    printf("Fact key at %d is %d\n", offset, key);
#endif 
    int val = d_fact_val[offset];
    int hash = HASH(key, num_slots);

    int2 slot = reinterpret_cast<int2*>(hash_table)[hash];
#if DEBUG 
    printf("Key at hash %d is %d\n", hash, slot.x);
#endif 
    if (slot.x == key) {
#if DEBUG
      printf("%d matches! Adding %d and %d\n", key, slot.y, val);
#endif 
      checksum += slot.y + val;
    }

    atomicAdd(res, checksum);
  }
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};

static int num_runs = 0;
static unsigned long long* h_res = 0;

__global__
void printDeviceArray(int* key, int* val, int* count, int num_slots) {
  int offset = 0; 
  for (int i = 0; i < NGPU; i++) {
    for (int j = 0; j < count[i]; j++) {
      printf("%d:%d ", key[offset + j], val[offset + j]);
    }
    printf("\n");
    offset += count[i];
  }  
}

TimeKeeper hashJoin(int* h_dim_key, int* h_dim_val, int* h_fact_fkey, int* h_fact_val, int num_dim, int num_fact, CachingDeviceAllocator&  g_allocator) {
  ////////////
  // TIMING //
  ////////////
  SETUP_TIMING();
  float time_build, time_probe, time_memset, time_memset2;

  ///////////////
  // PARTITION //
  ///////////////
  // Set up dimension and fact partitions 
  int *d_dim_key_orig;
  int *d_dim_val_orig; 
  int *d_fact_key_orig;
  int *d_fact_val_orig; 

  ALLOCATE(d_dim_key_orig, sizeof(int) * NGPU * num_dim);
  ALLOCATE(d_dim_val_orig, sizeof(int) * NGPU * num_dim); 
  ALLOCATE(d_fact_key_orig, sizeof(int) * NGPU * num_fact); 
  ALLOCATE(d_fact_val_orig, sizeof(int) * NGPU * num_fact); 

  CubDebugExit(cudaMemcpy(d_dim_key_orig, h_dim_key, sizeof(int) * NGPU * num_dim,  
        cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_val_orig, h_dim_val, sizeof(int) * NGPU * num_dim, 
        cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_key_orig, h_fact_fkey, sizeof(int) * NGPU * num_fact, 
        cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_val_orig, h_fact_val, sizeof(int) * NGPU * num_fact, 
        cudaMemcpyHostToDevice));

  // Create histograms (# of elements in each partition) 
  int* h_dim_count = new int[NGPU]; 
  int* h_fact_count = new int[NGPU];

  memset(h_dim_count, 0, sizeof(int)*NGPU); 
  memset(h_fact_count, 0, sizeof(int)*NGPU); 

  int *d_dim_count;
  int *d_fact_count; 

  ALLOCATE(d_dim_count, sizeof(int) * NGPU);
  ALLOCATE(d_fact_count, sizeof(int) * NGPU); 

  for (int i = 0; i < num_dim; i++) {
    h_dim_count[HHASH(h_dim_key[i], num_dim) % NGPU]++; 
  }
  CubDebugExit(cudaMemcpy(d_dim_count, h_dim_count, sizeof(int) * NGPU, cudaMemcpyHostToDevice));

  for (int i = 0; i < num_fact; i++) {
    h_fact_count[HHASH(h_fact_fkey[i], num_fact) % NGPU]++; 
  }
  CubDebugExit(cudaMemcpy(d_fact_count, h_fact_count, sizeof(int) * NGPU, cudaMemcpyHostToDevice));

  // Radix (sort dim/fact key/value by last logNGPU bits of key)
  int start_bit = 0; 
  int end_bit = log2(NGPU); 
  printf("Radix sort %d %d\n", start_bit, end_bit);

  void* d_dim_temp_storage = NULL;
  size_t dim_temp_storage_bytes = 0;
  void* d_fact_temp_storage = NULL;
  size_t fact_temp_storage_bytes = 0;

  int* d_dim_temp0;
  int* d_dim_temp1;
  int* d_fact_temp0; 
  int* d_fact_temp1; 

  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_temp0, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_temp1, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_temp0, sizeof(int) * num_fact));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_temp1, sizeof(int) * num_fact));

  cub::DoubleBuffer<int> d_dim_key(d_dim_key_orig, d_dim_temp0);
  cub::DoubleBuffer<int> d_dim_val(d_dim_val_orig, d_dim_temp1); 
  cub::DoubleBuffer<int> d_fact_key(d_fact_key_orig, d_fact_temp0);
  cub::DoubleBuffer<int> d_fact_val(d_fact_val_orig, d_fact_temp1); 

#if DEBUG
  printf("Partitioning dim...\n");
#endif 
  cub::DeviceRadixSort::SortPairs(d_dim_temp_storage, dim_temp_storage_bytes, d_dim_key, d_dim_val, 
      num_dim, start_bit, end_bit);
  CubDebugExit(g_allocator.DeviceAllocate(&d_dim_temp_storage, dim_temp_storage_bytes));
  cub::DeviceRadixSort::SortPairs(d_dim_temp_storage, dim_temp_storage_bytes, d_dim_key, d_dim_val, 
      num_dim, start_bit, end_bit);
  cudaDeviceSynchronize(); 
#if DEBUG
  printDeviceArray<<<1, 1>>>(d_dim_key.Current(), d_dim_val.Current(), d_dim_count, num_dim);
  cudaDeviceSynchronize();  
#endif 

#if DEBUG
  printf("Partitioning fact...\n");
#endif 
  cub::DeviceRadixSort::SortPairs(d_fact_temp_storage, fact_temp_storage_bytes, d_fact_key, d_fact_val, 
      num_fact, start_bit, end_bit);
  CubDebugExit(g_allocator.DeviceAllocate(&d_fact_temp_storage, fact_temp_storage_bytes));
  cub::DeviceRadixSort::SortPairs(d_fact_temp_storage, fact_temp_storage_bytes, d_fact_key, d_fact_val, 
      num_fact, start_bit, end_bit);
  cudaDeviceSynchronize(); 
#if DEBUG
  printDeviceArray<<<1, 1>>>(d_fact_key.Current(), d_fact_val.Current(), d_fact_count, num_fact); 
  cudaDeviceSynchronize(); 
#endif 

  ///////////
  // BUILD //
  ///////////
  int* hash_table_0;
  int* hash_table_1; 
  unsigned long long* res;
  int num_slots = num_dim;

  ALLOCATE(hash_table_0, sizeof(int) * 2 * num_dim);
  ALLOCATE(hash_table_1, sizeof(int) * 2 * num_dim);
  ALLOCATE(res, sizeof(long long));

  TIME_FUNC(cudaMemset(hash_table_0, 0, 2 * num_slots * sizeof(int)), time_memset);
  TIME_FUNC(cudaMemset(hash_table_1, 0, 2 * num_slots * sizeof(int)), time_memset); 
  TIME_FUNC(cudaMemset(res, 0, sizeof(long long)), time_memset2);

#if DEBUG
  printf("\nBuilding hashtable 0...\n");
#endif 
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(d_dim_key.Current(), d_dim_val.Current(), 0, 
          h_dim_count[0], hash_table_0, num_slots)), time_build);
  cudaDeviceSynchronize();

#if DEBUG
  printf("Building hashtable 1...\n");
#endif 
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(d_dim_key.Current(), d_dim_val.Current(), h_dim_count[0], 
          h_dim_count[0] + h_dim_count[1], hash_table_1, num_slots)), time_build);
  cudaDeviceSynchronize(); 

  ///////////
  // PROBE //
  ///////////
#if DEBUG
  printf("\nProbing hashtable 0...\n");
#endif 
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_key.Current(), d_fact_val.Current(), 
          0, h_fact_count[0], hash_table_0, num_slots, res)), time_probe);
  cudaDeviceSynchronize(); 

#if DEBUG
  printf("Probing hashtable 1...\n");
#endif 
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_key.Current(), d_fact_val.Current(), 
          h_fact_count[0], h_fact_count[0] + h_fact_count[1], hash_table_1, num_slots, res)), 
      time_probe);
  cudaDeviceSynchronize(); 

  /////////////
  // CLEANUP //
  /////////////
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

  CLEANUP(hash_table_0);
  CLEANUP(hash_table_1); 
  CLEANUP(res);
  CLEANUP(d_dim_key.Current());
  CLEANUP(d_dim_val.Current()); 
  CLEANUP(d_fact_key.Current());
  CLEANUP(d_fact_val.Current()); 

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
#if DEBUG
      printf("%d matches! Adding %d and %d\n", key, hash_table[(hash << 1) + 1], val);
#endif 
      *res = *res + (hash_table[(hash << 1) + 1] + val);
#if DEBUG
      printf("res is now %lld\n", *res);
#endif 
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
  int num_fact           = 256 * 1 << 4; // 256 * 1 << 20 , 1 << 28
  int num_dim            = 16 * 1 << 4; // 16 * 1 << 20 , 1 << 16
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

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

#if DEBUG
  cout << "DIM TABLE:" << endl;
  for (int i = 0; i < num_dim; i++) cout << h_dim_key[i] << "..." << h_dim_val[i] << endl;
  cout << endl;

  cout << "FACT TABLE:" << endl;
  for (int i = 0; i < num_fact; i++) cout << h_fact_fkey[i] << "..." << h_fact_val[i] << endl;
  cout << endl;
#endif 

  for (int j = 0; j < num_trials; j++) {
    cout << "TRIAL " << j << endl;
    TimeKeeper t = hashJoin(h_dim_key, h_dim_val, h_fact_fkey, h_fact_val, num_dim, num_fact, g_allocator);
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

  // Checking against hash join on CPU
  int num_slots = num_dim;
  int *hash_table = new int[num_slots * 2];
  long long check_res = 0;
  RunHashJoinCPU(h_dim_key, h_dim_val, h_fact_fkey, h_fact_val, hash_table, &check_res, num_dim, 
      num_fact, num_slots);
  cout << "CPU answer: " << check_res << endl; 

  return 0;
}


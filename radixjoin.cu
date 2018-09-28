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

#define DEBUG 1
#define NGPU 2

__device__ __forceinline__
int HASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__forceinline__
int HHASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

// Partition based on log(NGPU) bits, into NGPU partitions
void partition_dev(int* h_key, int* d_key, int* d_val, int num_slots, int start_bit, int end_bit, CachingDeviceAllocator&  g_allocator) {
  int* d_buffer;
  ALLOCATE(d_buffer, sizeof(int) * num_slots * NGPU);
  cudaMemset(d_buffer, 0, sizeof(int) * num_slots * NGPU);

  cub::DoubleBuffer<int> d_key_buffer(d_key, &d_buffer[0]);
  cub::DoubleBuffer<int> d_val_buffer(d_val, &d_buffer[num_slots]); 

  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_key_buffer, d_val_buffer, num_slots, start_bit, end_bit);

  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Sort both keys and values by last logNGPU bits of key
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_key_buffer, d_val_buffer, num_slots, start_bit, end_bit);

  d_key = d_key_buffer.Current();
  d_val = d_val_buffer.Current(); 

  CLEANUP(d_temp_storage); 
}

__global__
void build_hashtable_dev(int *d_dim_key, int *d_dim_val, int num_tuples, int *hash_table, int num_slots) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  int key = d_dim_key[offset];
  int val = d_dim_val[offset];
  int hash = HASH(key, num_slots);

  if (offset < num_tuples) {
    hash_table[hash << 1] = key;
    hash_table[(hash << 1) + 1] = val;
#if DEBUG
    printf("Wrote %d to position %d in hashtable\n", key, hash << 1);
#endif 
  }
}

__global__
void probe_hashtable_dev(int *d_fact_fkey, int *d_fact_val, int num_tuples, int *hash_table, int num_slots, unsigned long long *res) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  unsigned long long checksum = 0;

  for (int i = offset; i < num_tuples; i += stride) {
    int key = d_fact_fkey[i];
#if DEBUG
    printf("Fact key at %d is %d\n", i, key);
#endif 
    int val = d_fact_val[i];
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

__global__
void printDeviceArray(int* key, int* val, int* count, int num_slots) {
  for (int i = 0; i < NGPU; i++) {
    for (int j = 0; j < count[i]; j++) {
      printf("%d:%d ", key[i*num_slots + j], val[i*num_slots + j]);
    }
    printf("\n");
  }  
}

TimeKeeper hashJoin(int* h_dim_key, int* h_dim_val, int* h_fact_fkey, int* h_fact_val, int num_dim, int num_fact, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();
  float time_build, time_probe, time_memset, time_memset2;

  // Partition
  int start_bit = sizeof(int) * 8 - log2(2);
  int end_bit = sizeof(int) * 8;

  printf("Radix sort %d %d\n", start_bit, end_bit);

  int* h_dim_count = new int[NGPU]; 
  int* h_fact_count = new int[NGPU];
  memset(h_dim_count, 0, sizeof(int)*NGPU); 
  memset(h_fact_count, 0, sizeof(int)*NGPU); 

  int *d_dim_key_partitions;
  int *d_dim_val_partitions; 
  int *d_fact_key_partitions;
  int *d_fact_val_partitions; 
  int *d_dim_count;
  int *d_fact_count; 

  ALLOCATE(d_dim_key_partitions, sizeof(int) * NGPU * num_dim);
  ALLOCATE(d_dim_val_partitions, sizeof(int) * NGPU * num_dim); 
  ALLOCATE(d_fact_key_partitions, sizeof(int) * NGPU * num_fact); 
  ALLOCATE(d_fact_val_partitions, sizeof(int) * NGPU * num_fact); 
  ALLOCATE(d_dim_count, sizeof(int) * NGPU);
  ALLOCATE(d_fact_count, sizeof(int) * NGPU); 

  CubDebugExit(cudaMemcpy(d_dim_key_partitions, h_dim_key, sizeof(int) * NGPU * num_dim,  cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_val_partitions, h_dim_val, sizeof(int) * NGPU * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_key_partitions, h_fact_fkey, sizeof(int) * NGPU * num_fact, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_val_partitions, h_fact_val, sizeof(int) * NGPU * num_fact, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_count, h_dim_count, sizeof(int) * NGPU, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_count, h_fact_count, sizeof(int) * NGPU, cudaMemcpyHostToDevice));

#if DEBUG
  printf("Partitioning dim...\n");
#endif 
  partition_dev(h_dim_key, d_dim_key_partitions, d_dim_val_partitions, num_dim, start_bit, end_bit, g_allocator); 

  // Count number of entries that'll be sent to each dim partition 
  for (int i = 0; i < num_dim; i++) {
    h_dim_count[HHASH(h_dim_key[i], num_dim) % NGPU]++; 
  }
#if DEBUG
  printDeviceArray<<<1, 1>>>(d_dim_key_partitions, d_dim_val_partitions, d_dim_count, num_dim);
  cudaDeviceSynchronize();  
#endif 

#if DEBUG
  printf("Partitioning fact...\n");
#endif 
  partition_dev(h_fact_fkey, d_fact_key_partitions, d_fact_val_partitions, num_fact, start_bit, end_bit, g_allocator);

  // Count number of entries that'll be sent to each partition 
  for (int i = 0; i < num_fact; i++) {
    h_fact_count[HHASH(h_fact_fkey[i], num_fact) % NGPU]++; 
  }
#if DEBUG
  printDeviceArray<<<1, 1>>>(d_fact_key_partitions, d_fact_val_partitions, d_fact_count, num_fact); 
  cudaDeviceSynchronize(); 
#endif 

  // Build hashtable (TODO: N hashtables) 
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
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(d_dim_key_partitions, d_dim_val_partitions, h_dim_count[0], hash_table_0, num_slots)), time_build);
  cudaDeviceSynchronize();

#if DEBUG
  printf("Building hashtable 1...\n");
#endif 
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(&d_dim_key_partitions[num_dim], &d_dim_val_partitions[num_dim], h_dim_count[1], hash_table_1, num_slots)), time_build);
  cudaDeviceSynchronize(); 

  // Probe hashtable
#if DEBUG
  printf("\nProbing hashtable 0...\n");
#endif 
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_key_partitions, d_fact_val_partitions, h_fact_count[0], hash_table_0, num_slots, res)), time_probe);
  cudaDeviceSynchronize(); 

#if DEBUG
  printf("Probing hashtable 1...\n");
#endif 
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(&d_fact_key_partitions[num_fact], &d_fact_val_partitions[num_fact], h_fact_count[1], hash_table_1, num_slots, res)), time_probe);
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

  CLEANUP(hash_table_0);
  CLEANUP(hash_table_1); 
  CLEANUP(res);
  CLEANUP(d_dim_key_partitions);
  CLEANUP(d_dim_val_partitions); 
  CLEANUP(d_fact_key_partitions);
  CLEANUP(d_fact_val_partitions); 

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
  int num_fact           = 16; // 256 * 1 << 20 , 1 << 28
  int num_dim            = 4; // 16 * 1 << 20 , 1 << 16
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
  RunHashJoinCPU(h_dim_key, h_dim_val, h_fact_fkey, h_fact_val, hash_table, &check_res, num_dim, num_fact, num_slots);
  cout << "CPU answer: " << check_res << endl; 

  return 0;
}


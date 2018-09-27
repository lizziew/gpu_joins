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
#define NGPU 2

__device__ __forceinline__
int HASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__forceinline__
int HHASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

void partition_dev(int* h_key, int* h_val, int* key_partitions, int* val_partitions, int num_slots, int count[NGPU]) {
  for (int i = 0; i < num_slots; i ++) {
    int key = h_key[i];
    int hash = (HHASH(key, num_slots) % NGPU);

    key_partitions[hash * num_slots + count[hash]] = key;
    val_partitions[hash * num_slots + count[hash]] = h_val[i];

    count[hash]++;
  }
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

TimeKeeper hashJoin(int* h_dim_key, int* h_dim_val, int* h_fact_fkey, int* h_fact_val, int num_dim, int num_fact, CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();
  float time_build, time_probe, time_memset, time_memset2;

  // Partition
  int* h_dim_key_partitions = new int[NGPU*num_dim] ;
  int* h_dim_val_partitions = new int[NGPU*num_dim];
  int* h_fact_key_partitions = new int[NGPU*num_fact];
  int* h_fact_val_partitions = new int[NGPU*num_fact];
  int* h_dim_count = new int[NGPU]; 
  int* h_fact_count = new int[NGPU];

  memset(h_dim_count, 0, sizeof(int)*NGPU); 
  memset(h_fact_count, 0, sizeof(int)*NGPU); 

#if DEBUG
  printf("Partitioning dim...\n");
#endif 
  partition_dev(h_dim_key, h_dim_val, h_dim_key_partitions, h_dim_val_partitions, num_dim, h_dim_count); 
#if DEBUG
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < h_dim_count[i]; j++) {
      printf("%d:%d ", h_dim_key_partitions[i*num_dim + j], h_dim_val_partitions[i*num_dim + j]);
    }
    printf("\n");
  } 
#endif 

#if DEBUG
  printf("Partitioning fact...\n");
#endif 
  partition_dev(h_fact_fkey, h_fact_val, h_fact_key_partitions, h_fact_val_partitions, num_fact, h_fact_count);
#if DEBUG
  for (int i = 0; i < NGPU; i++) {
    for (int j = 0; j < h_fact_count[i]; j++) {
      printf("%d:%d ", h_fact_key_partitions[i*num_fact + j], h_fact_val_partitions[i * num_fact + j]);
    }
    printf("\n");
  } 
#endif 

  int *d_dim_key_partitions;
  int *d_dim_val_partitions; 
  int *d_fact_key_partitions;
  int *d_fact_val_partitions; 

  ALLOCATE(d_dim_key_partitions, sizeof(int) * NGPU * num_dim);
  ALLOCATE(d_dim_val_partitions, sizeof(int) * NGPU * num_dim); 
  ALLOCATE(d_fact_key_partitions, sizeof(int) * NGPU * num_fact); 
  ALLOCATE(d_fact_val_partitions, sizeof(int) * NGPU * num_fact); 

  TIME_FUNC(cudaMemset(d_dim_key_partitions, 0, NGPU * num_dim * sizeof(int)), time_memset);
  TIME_FUNC(cudaMemset(d_dim_val_partitions, 0, NGPU * num_dim * sizeof(int)), time_memset); 

  CubDebugExit(cudaMemcpy(d_dim_key_partitions, h_dim_key_partitions, sizeof(int) * NGPU * num_dim,  cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_val_partitions, h_dim_val_partitions, sizeof(int) * NGPU * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_key_partitions, h_fact_key_partitions, sizeof(int) * NGPU * num_fact, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_val_partitions, h_fact_val_partitions, sizeof(int) * NGPU * num_fact, cudaMemcpyHostToDevice));

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
  int num_dim            = 2; // 16 * 1 << 20 , 1 << 16
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


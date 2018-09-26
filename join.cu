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
#define PRINT 1

__device__ __forceinline__
int HASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

__forceinline__
int HHASH(const int key, const int num_slots) {
  return key & (num_slots - 1);
}

void partition_dev(int* h_key, int* h_val, int** key_partitions, int** val_partitions, int num_slots, int count[NGPU]) {

  for (int i = 0; i < num_slots; i ++) {
    int key = h_key[i];
    int hash = (HHASH(key, num_slots) % NGPU);

    key_partitions[hash][count[hash]] = key;
    val_partitions[hash][count[hash]] = h_val[i];

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
    if (PRINT) printf("Wrote %d to position %d in hashtable\n", key, hash << 1);
  }
}

__global__
void probe_hashtable_dev(int *d_fact_fkey, int *d_fact_val, int num_tuples, int *hash_table, int num_slots, unsigned long long *res) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  unsigned long long checksum = 0;

  for (int i = offset; i < num_tuples; i += stride) {
    int key = d_fact_fkey[i];
    if (PRINT) printf("Fact key at %d is %d\n", i, key);
    int val = d_fact_val[i];
    int hash = HASH(key, num_slots);

    int2 slot = reinterpret_cast<int2*>(hash_table)[hash];
    if (PRINT) printf("Key at hash %d is %d\n", hash, slot.x);
    if (slot.x == key) {
      if (PRINT) printf("%d matches! Adding %d and %d\n", key, slot.y, val);
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

  // Partition
  int** h_dim_key_partitions = new int*[NGPU];
  for(int i = 0; i < NGPU; ++i) h_dim_key_partitions[i] = new int[num_dim];
  int** h_dim_val_partitions = new int*[NGPU];
  for(int i = 0; i < NGPU; ++i) h_dim_val_partitions[i] = new int[num_dim];
  int** h_fact_key_partitions = new int*[NGPU];
  for(int i = 0; i < NGPU; ++i) h_fact_key_partitions[i] = new int[num_fact];
  int** h_fact_val_partitions = new int*[NGPU];
  for(int i = 0; i < NGPU; ++i) h_fact_val_partitions[i] = new int[num_fact];
  int* h_dim_count = new int[NGPU]; 
  int* h_fact_count = new int[NGPU];

  memset(h_dim_count, 0, sizeof(int)*NGPU); 
  memset(h_fact_count, 0, sizeof(int)*NGPU); 

  if (PRINT) printf("Partitioning dim...\n");
  partition_dev(h_dim_key, h_dim_val, h_dim_key_partitions, h_dim_val_partitions, num_dim, h_dim_count); 
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < h_dim_count[i]; j++) {
      if (PRINT) printf("%d:%d ", h_dim_key_partitions[i][j], h_dim_val_partitions[i][j]);
    }
    if (PRINT) printf("\n");
  } 

  if (PRINT) printf("Partitioning fact...\n");
  partition_dev(h_fact_fkey, h_fact_val, h_fact_key_partitions, h_fact_val_partitions, num_fact, h_fact_count);
  for (int i = 0; i < NGPU; i++) {
    for (int j = 0; j < h_fact_count[i]; j++) {
      if (PRINT)  printf("%d:%d ", h_fact_key_partitions[i][j], h_fact_val_partitions[i][j]);
    }
    if (PRINT) printf("\n");
  } 

  // Build hashtable (TODO: N hashtables) 
  int* hash_table_0;
  int* hash_table_1; 
  unsigned long long* res;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  int (*d_dim_key_partitions)[NGPU];
  int (*d_dim_val_partitions)[NGPU]; 
  int (*d_fact_key_partitions)[NGPU];
  int (*d_fact_val_partitions)[NGPU]; 

  ALLOCATE(hash_table_0, sizeof(int) * 2 * num_dim);
  ALLOCATE(hash_table_1, sizeof(int) * 2 * num_dim);
  ALLOCATE(res, sizeof(long long));

  ALLOCATE(d_dim_key_partitions, sizeof(int) * NGPU * num_dim);
  ALLOCATE(d_dim_val_partitions, sizeof(int) * NGPU * num_dim); 
  ALLOCATE(d_fact_key_partitions, sizeof(int) * NGPU * num_fact); 
  ALLOCATE(d_fact_val_partitions, sizeof(int) * NGPU * num_fact); 

  TIME_FUNC(cudaMemset(d_dim_key_partitions, 0, NGPU * num_dim * sizeof(int)), time_memset);
  TIME_FUNC(cudaMemset(d_dim_val_partitions, 0, NGPU * num_dim * sizeof(int)), time_memset); 

  for (int i = 0; i < NGPU; i++) {
    CubDebugExit(cudaMemcpy(d_dim_key_partitions[i], h_dim_key_partitions[i], sizeof(int) * h_dim_count[i],  cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_dim_val_partitions[i], h_dim_val_partitions[i], sizeof(int) * h_dim_count[i], cudaMemcpyHostToDevice));

    printf("%d round: Copying 4 values to pointer %d = %d\n", i, d_fact_key_partitions + i*num_fact*sizeof(int), d_fact_key_partitions[i]);

    CubDebugExit(cudaMemcpy(d_fact_key_partitions + i * num_fact * sizeof(int), h_fact_key_partitions[i], sizeof(int) * h_fact_count[i], cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_fact_val_partitions + i * num_fact * sizeof(int), h_fact_val_partitions[i], sizeof(int) * h_fact_count[i], cudaMemcpyHostToDevice));
  }

  TIME_FUNC(cudaMemset(hash_table_0, 0, 2 * num_slots * sizeof(int)), time_memset);
  TIME_FUNC(cudaMemset(hash_table_1, 0, 2 * num_slots * sizeof(int)), time_memset); 
  TIME_FUNC(cudaMemset(res, 0, sizeof(long long)), time_memset2);

  if (PRINT) printf("\nBuilding hashtable 0...\n");
  // num_dim/128
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(d_dim_key_partitions[0], d_dim_val_partitions[0], h_dim_count[0], hash_table_0, num_slots)), time_build);
  cudaDeviceSynchronize(); 

  if (PRINT) printf("Building hashtable 1...\n");
  TIME_FUNC((build_hashtable_dev<<<128, 128>>>(d_dim_key_partitions[1], d_dim_val_partitions[1], h_dim_count[1], hash_table_1, num_slots)), time_build);
  cudaDeviceSynchronize(); 

  // Probe hashtable

  if (PRINT) printf("\nProbing hashtable 0...\n");
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_key_partitions[0], d_fact_val_partitions[0], h_fact_count[0], hash_table_0, num_slots, res)), time_probe);
  cudaDeviceSynchronize(); 

  if (PRINT) printf("Probing hashtable 1...\n");
  TIME_FUNC((probe_hashtable_dev<<<192, 256>>>(d_fact_key_partitions[1], d_fact_val_partitions[1], h_fact_count[1], hash_table_1, num_slots, res)), time_probe);
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
      printf("%d matches! Adding %d and %d\n", key, hash_table[(hash << 1) + 1], val);
      *res = *res + (hash_table[(hash << 1) + 1] + val);
      printf("res is now %d\n", *res);
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
  int num_fact           = 8; // 256 * 1 << 20 , 1 << 28
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

  int log2 = 0;
  int num_dim_dup = num_dim >> 1;
  while (num_dim_dup) {
    num_dim_dup >>= 1;
    log2 += 1;
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  if (PRINT) {
    cout << "DIM TABLE:" << endl;
    for (int i = 0; i < num_dim; i++) cout << h_dim_key[i] << "..." << h_dim_val[i] << endl;
    cout << endl;

    cout << "FACT TABLE:" << endl;
    for (int i = 0; i < num_fact; i++) cout << h_fact_fkey[i] << "..." << h_fact_val[i] << endl;
    cout << endl;
  }

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


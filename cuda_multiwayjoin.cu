#include <stdio.h> 

const int rSize = 4;
const int num_blocks = 2;
const int num_threads = 64; 

// Smallest power of 2 larger than n
int getHashTableSize(int n) {
  do {
    n--;
    n |= n >> 1;                                    
    n |= n >> 2;                                    
    n |= n >> 4;                                    
    n |= n >> 8;                                    
    n |= n >> 16;                                   
    n++; } while (0);

    return n; 
}

__global__ void calculateBucketSizes(int* rKeys, int rSize, int* bucketSizes, int hSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < rSize) {
    int rKey = rKeys[idx];
    int hKey = rKey & (hSize - 1);
    atomicAdd(&(bucketSizes[hKey]), 1);
    printf("%d: %d\n", rKey, bucketSizes[hKey]); 
  }
}

void calculateBucketPositions(int*bucketSizes, int* bucketPositions, int hSize) {
  bucketPositions[0] = 0; 
  for (int i = 1; i < hSize; i++) {
    bucketPositions[i] = bucketPositions[i-1] + 2 * bucketSizes[i-1]; 
  }
}

__global__ void buildPhase(int* rKeys, int rSize, int hSize, int* bucketSizes, int* bucketPositions, int* hashTable) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < rSize; i += stride) { 
    int rKey = rKeys[idx]; 
    int hKey = rKey & (hSize - 1);

    int pos = bucketPositions[hKey];
    // TODO: Handle duplicates 
    atomicAdd(&(bucketPositions[hKey]), 2); 
    hashTable[pos] = rKey;
    hashTable[pos+1] = rKey; // TODO: Assume value = key for now  
  }
}

__global__ void probePhase(int* joined, int* joinedSize, int* sKeys, int rSize, int hSize, int* hashTable, int* bucketPositions, int* bucketSizes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx < rSize) {
    int sKey = sKeys[idx]; 
    int hKey = sKey & (hSize - 1);
 
    int pos = bucketPositions[hKey]; 
    int len = bucketSizes[hKey];

    for (int i = pos; i < pos + 2 * len; i += 2) {
      if (hashTable[i] == sKey) {
        printf("Writing %d to %d\n", sKey, idx); 
        joined[idx] = sKey; // TODO: Figure out how to write to array in parallel  
      }
    }
  }
} 

void hashJoin(int** r, int** s) {
  // Get keys of r 
  int  h_rKeys[rSize];
  for (int i = 0; i < rSize; i++) h_rKeys[i] = r[i][0]; 

  int* d_rKeys;
  cudaMalloc((void**) &d_rKeys, sizeof(int) * rSize); 
  cudaMemcpy(d_rKeys, h_rKeys, sizeof(int) * rSize, cudaMemcpyHostToDevice); 

  // Get size of hashtable
  int hSize = getHashTableSize(rSize); 

  // Get size of each bucket in hash table
  int *d_bucketSizes; 
  cudaMalloc((void**) &d_bucketSizes, sizeof(int) * hSize); 
  cudaMemset(d_bucketSizes, 0, sizeof(int) * hSize); 

  printf("Bucket sizes:\n");
  calculateBucketSizes<<<num_blocks, num_threads>>>(d_rKeys, rSize, d_bucketSizes, hSize);
  cudaDeviceSynchronize(); 
  int h_bucketSizes[hSize]; 
  cudaMemcpy(h_bucketSizes, d_bucketSizes, sizeof(int) * hSize, cudaMemcpyDeviceToHost); 

  // Stores the position of each bucket in the hash table
  int h_bucketPositions[hSize];
  calculateBucketPositions(h_bucketSizes, h_bucketPositions, hSize); 
  printf("Bucket positions:\n");
  for (int i = 0; i < hSize; i++) printf("%d ", h_bucketPositions[i]);
  printf("\n");

  int* d_tempBucketPositions;
  cudaMalloc((void**) &d_tempBucketPositions, sizeof(int) * hSize); 
  cudaMemcpy(d_tempBucketPositions, h_bucketPositions, sizeof(int) * hSize, cudaMemcpyHostToDevice); 

  // Scan R and create in-memory hash table
  int* hashTable;
  cudaMalloc((void**) &hashTable, 2 * sizeof(int) * rSize); 
  buildPhase<<<num_blocks, num_threads>>>(d_rKeys, rSize, hSize, d_bucketSizes, d_tempBucketPositions, hashTable);  
  cudaDeviceSynchronize(); 

  // Get keys of s
  int h_sKeys[rSize]; // TODO: Assume two relations are the same size for now 
  for (int i = 0; i < rSize; i++) h_sKeys[i] = s[i][0];

  int* d_sKeys;
  cudaMalloc((void**) &d_sKeys, sizeof(int) * rSize); 
  cudaMemcpy(d_sKeys, h_sKeys, sizeof(int) * rSize, cudaMemcpyHostToDevice); 

  // Scan S, look up join key in hash table, and add tuple to output if match found
  int* d_joined; // TODO: just print out list of join keys for now
  cudaMalloc((void**) &d_joined, sizeof(int) * rSize);

  int* d_joinedSize; 
  cudaMalloc((void**) &d_joinedSize, sizeof(int)); 
  cudaMemset(d_joinedSize, 0, sizeof(int)); 

  int* d_bucketPositions; 
  cudaMalloc((void**) &d_bucketPositions, sizeof(int) * hSize); 
  cudaMemcpy(d_bucketPositions, h_bucketPositions, sizeof(int) * hSize, cudaMemcpyHostToDevice); 

  probePhase<<<num_blocks, num_threads>>>(d_joined, d_joinedSize, d_sKeys, rSize, hSize, hashTable, d_bucketPositions, d_bucketSizes);
  cudaDeviceSynchronize(); 

  int h_joined[rSize]; 
  cudaMemcpy(h_joined, d_joined, sizeof(int) * rSize, cudaMemcpyDeviceToHost); 

  int h_joinedSize[1]; 
  cudaMemcpy(h_joinedSize, d_joinedSize, sizeof(int), cudaMemcpyDeviceToHost); 

  printf("Final joined result:\n");
  for (int i = 0; i < rSize; i++) {
    if (h_joined[i] != 0) {
      printf("%d ", h_joined[i]);
    }
  }
  printf("\n");
}

void printRelation(int** relation, int relationSize) {
  printf("Relation:\n");

  for (int i = 0; i < relationSize; i++) { 
    for (int j = 0; j < 2; j++) {
      printf("%d ", relation[i][j]);
    }
    printf("\n");
  }
}

int** generateRelation(int relationSize) {
  int **relation = (int **)malloc(relationSize * sizeof(int *)); 
  for (int i = 0; i < relationSize; i++) 
    relation[i] = (int*) malloc(2 * sizeof(int)); 

  // TODO: randomize
  int count = rand() % 5;  
  for (int i = 0; i < relationSize; i++) {
    int value = ++count; 
    for (int j = 0; j < 2; j++) 
      relation[i][j] = value;
  }

  return relation; 
}

int main() {
  int** r = generateRelation(rSize);
  printRelation(r, rSize); 

  int** s = generateRelation(rSize);
  printRelation(s, rSize); 

  hashJoin(r, s); 
  return 0; 
}


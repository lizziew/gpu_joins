#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <unordered_map> 

using namespace std; 

const int rSize = 8; 
const int sSize = 32; 
const int maxValue = 100; 

typedef tuple<int, int> record; 
typedef vector<int> joinedRecord; 

void printRelation(vector<record> relation) {
  for (int i = 0; i < relation.size(); i++) {
    cout << get<0>(relation[i]) << ": " << get<1>(relation[i]) << endl; 
  }
  cout << endl; 
}

void printJoined(vector<joinedRecord> joined) {
  for (int i = 0; i < joined.size(); i++) {
    for (int j = 0; j < joined[i].size(); j++) {
      cout << joined[i][j] << " ";
    }
    cout << endl; 
  }
}

vector<record> generateRelation(int relationSize) {
  random_device randomDevice; 
  mt19937 gen(randomDevice());
  uniform_int_distribution<> uniformDistribution(1, maxValue);

  vector<record> relation;

  for (int i = 0; i < relationSize; i++) {
    int value = uniformDistribution(gen);
    relation.push_back(make_tuple(value, value)); 
  } 

  return relation; 
}

unordered_multimap<int, int> buildPhase(vector<record> r) {
  unordered_multimap<int, int> hashTable; 

  // Maps hash(join key) to row indices 
  for (int i = 0; i < r.size(); i++) {
    int rKey = get<0>(r[i]);
    hashTable.insert(make_pair(rKey, i));
  }

  return hashTable; 
}

vector<joinedRecord> probePhase(
    vector<record> r, 
    vector<record> s, 
    unordered_multimap<int, int> hashTable) {
  vector<joinedRecord> joined;

  for (int i = 0; i < s.size(); i++) {
    int joinKey = get<0>(s[i]);
    int sValue = get<1>(s[i]); 

    if (hashTable.find(joinKey) != hashTable.end()) {
      auto range = hashTable.equal_range(joinKey);
      for (auto it = range.first; it != range.second; ++it) {
        int rValue = get<1>(r[it->second]); 
        joined.push_back(vector<int>{rValue, joinKey, sValue});
      }
    }
  }

  return joined; 
}

vector<joinedRecord> hashJoin(vector<record> r, vector<record> s) {
  // Scan R and create in-memory hash table 
  unordered_multimap<int, int> hashTable = buildPhase(r);

  // Scan S, look up join key in hash table, and add tuple to output if match found 
  return probePhase(r, s, hashTable); 
}

int main() {
  // Initialize relations
  vector<record> rRelation = generateRelation(rSize);
  vector<record> sRelation = generateRelation(sSize);  

  cout << "R Relation: " << endl;
  printRelation(rRelation);
  cout << "S Relation: " << endl;
  printRelation(sRelation);

  vector<joinedRecord> joined = hashJoin(rRelation, sRelation);

  cout << "Hash Join result: " << endl; 
  printJoined(joined);

  return 0; 
}

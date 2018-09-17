#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <random>
#include <unordered_map> 

using namespace std; 

const int rSize = 4; 
const int maxValue = 10; 
const int numRelations = 2; 

typedef vector<int> record; 
typedef vector<int> record; 

void printRelation(vector<record> r) {
  for (int i = 0; i < r.size(); i++) {
    for (int j = 0; j < r[i].size(); j++) {
      cout << r[i][j] << " ";
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
    vector<int> row;
    for (int j = 0; j < 2; j++) {
      row.push_back(value);
    }
    relation.push_back(row); 
  } 

  return relation; 
}

unordered_multimap<int, int> buildPhase(vector<record> r) {
  unordered_multimap<int, int> hashTable; 

  // Maps hash(join key) to row indices 
  for (int i = 0; i < r.size(); i++) {
    int rKey = r[i][0];
    hashTable.insert(make_pair(rKey, i));
  }

  return hashTable; 
}

vector<record> probePhase(
    vector<record> r, 
    vector<record> s, 
    unordered_multimap<int, int> hashTable) {
  vector<record> joined;

  for (int i = 0; i < s.size(); i++) {
    int joinKey = s[i][0];
    int sValue = s[i][1]; 

    if (hashTable.find(joinKey) != hashTable.end()) {
      auto range = hashTable.equal_range(joinKey);
      for (auto it = range.first; it != range.second; ++it) {
        int rValue = r[it->second][1]; 
        joined.push_back(vector<int>{rValue, joinKey, sValue});
      }
    }
  }

  return joined; 
}

vector<record> hashJoin(vector<record> r, vector<record> s) {
  // Scan R and create in-memory hash table 
  unordered_multimap<int, int> hashTable = buildPhase(r);

  // Scan S, look up join key in hash table, and add tuple to output if match found 
  return probePhase(r, s, hashTable); 
}

vector<record> sortMergeJoin(vector<record> r, vector<record> s) {
  vector<record> joined;

  // Sort R and S based on join key 
  sort(r.begin(), r.end());
  sort(s.begin(), s.end()); 

  // Scan sorted relations and compare tuples
  int rIndex = 0, sIndex = 0;
  while (rIndex < r.size() && sIndex < s.size()) {
    int rKey = r[rIndex][0]; 
    int sKey = s[sIndex][0]; 

    if (rKey > sKey) {
      sIndex++; 
    } else if (rKey < sKey) {
      rIndex++; 
    } else {
      joined.push_back(vector<int>{r[rIndex][1], rKey, s[sIndex][1]});

      int sTempIndex = sIndex + 1;
      while (sTempIndex < s.size() && (rKey == s[sTempIndex][0])) {
        joined.push_back(vector<int>{r[rIndex][1], rKey, s[sTempIndex][1]});
        sTempIndex++; 
      }

      int rTempIndex = rIndex + 1; 
      while (rTempIndex < r.size() && (sKey == r[rTempIndex][0])) {
        joined.push_back(vector<int>{r[rTempIndex][1], rKey, s[sIndex][1]});
        rTempIndex++; 
      }

      rIndex++;
      sIndex++; 
    }
  }

  return joined;
}

vector<record> multiwayHashJoin(vector<vector<record>> relations) {
  vector<record> result = relations[0]; 

  for (int i = 1; i < relations.size(); i++) {
    result = hashJoin(relations[i], result); 
  }

  return result; 
} 

vector<record> multiwaySortMergeJoin(vector<vector<record>> relations) {
  vector<record> result = relations[0]; 

  for (int i = 1; i < relations.size(); i++) {
    result = sortMergeJoin(relations[i], result); 
  }

  return result; 
}

int main() {
  // Initialize relations
  vector<vector<record>> relations;
  for (int i = 0; i < numRelations; i++) {
    vector<record> relation = generateRelation(rSize);
    relations.push_back(relation);

    cout << "Input relation:" << endl;
    printRelation(relation); 
    cout << endl; 
  }

  // Hash join
  vector<record> hashJoinResult = multiwayHashJoin(relations);
  cout << "Hash Join result: " << endl; 
  printRelation(hashJoinResult);

  // Sort merge join
  vector<record> sortMergeJoinResult = multiwaySortMergeJoin(relations); 
  cout << "Sort Merge Join result: " << endl;
  printRelation(sortMergeJoinResult); 

  return 0; 
}

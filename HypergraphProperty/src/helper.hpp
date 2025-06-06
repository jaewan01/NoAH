#include <cmath>
#include <cassert>
#include <chrono>
#include <random>
#include <iterator>
#include <ctime>
#include <iomanip>
#include <stdio.h>
#include <cstdlib> 
#include <queue>
#include "graph.hpp"

#ifndef HELPER_HPP
#define HELPER_HPP

using namespace std;

class Helper {
public:
    HyperGraph *graph;
    vector<int> hypergraph_masking;
    vector<int> node_masking;
    vector<bool> check;
    vector<bool> check_node;
    int number_of_hedges;
    int number_of_nodes;
    int sum_of_hsizes;

    unordered_map<string, long long> pairdegree; // node pair -> degree
    set<int> neighbors;
    string outputdir;
    queue<int> visited;
    vector<int> nodes;

    Helper(set<int> subhypergraph, HyperGraph *graph, string outputdir);
    ~Helper(){
        hypergraph_masking.clear();
        node_masking.clear();
        check.clear();
        check_node.clear();
        pairdegree.clear();
        neighbors.clear();
    }

    void get_average_clustering_coef(void);
    void count_wcc(void);
    void get_dense_property(void);

    void update(set<int> deltaset, HyperGraph *graph);
    void save_properties(void);

private:
    int bfs(int start_node);
};
#endif
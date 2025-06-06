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

#ifndef HSET_HPP
#define HSET_HPP

using namespace std;

class HSet {
public:
    vector<int> hypergraph_masking;
    vector<int> node_masking; // node -> degree

    int number_of_hedges;
    int number_of_nodes;

    bool addflag = true;
    int timespent;
    int unit_numhedge;
    
    vector<int> hyperedge_order;
    string outputdir;
    
    // initial
    HSet(set<int> subhypergraph, HyperGraph *graph, string outputdir, double accuracy, string scoreflag);
    ~HSet(){
        hypergraph_masking.clear();
        node_masking.clear();
    }
    
    vector<int> get_hyperedgeset(void);
    void change_version(bool addflag);
    void update(vector<int> deltah, HyperGraph *graph, string sign);
    void save_as_txt(HyperGraph *graph);
};
#endif
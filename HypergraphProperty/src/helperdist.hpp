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

#ifndef HELPERDIST_HPP
#define HELPERDIST_HPP

using namespace std;

class HelperDist {
public:
    HyperGraph *graph;
    vector<bool> hypergraph_masking;
    vector<bool> node_masking;
    vector<int> check;
    vector<bool> check_node;
    vector<int> ancestor;
    vector<int> group_size;
    int number_of_hedges;
    int number_of_nodes;

    bool logflag = false;
    bool degree_flag = false;
    string degree_outname;
    bool size_flag = false;
    string size_outname;
    bool pairdeg_flag = false;
    string pairdeg_outname;
    bool its_flag = false;
    string its_outname;
    string dataname;

    unordered_map<int,long long> intersection_size;
    unordered_map<int,long long> nodedegree; // degree -> frequent
    unordered_map<int,long long> size_dist;
    unordered_map<string,long long> pairdegree_tmp;

    set<string> neighbor;
    string outputdir;
    queue<int> visited;
    vector<int> nodes;

    vector<int> vec;
    vector<int>::iterator it;
    vector<string> tmp;

    HelperDist(set<int> subhypergraph, HyperGraph *graph, string outputdir, string algo_opt, bool logflag);
    ~HelperDist(){
        hypergraph_masking.clear();
        node_masking.clear();
        check.clear();
        check_node.clear();
        neighbor.clear();

        nodedegree.clear();
        intersection_size.clear();
        size_dist.clear();
        pairdegree_tmp.clear();
    }
    void get_intersection_size_dist(void);
    void get_degree_dist(void);
    void get_pairdegree_dist(void);
    void get_size_dist(void);
    void update(set<int> deltaset, HyperGraph *graph);
    void save_properties(void);

};
#endif
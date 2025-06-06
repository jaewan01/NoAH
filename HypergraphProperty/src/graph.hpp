#include <fstream>
#include <unordered_map>
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <cassert>
#include "utils.hpp"

#ifndef GRAPH_HPP
#define GRAPH_HPP

using namespace std;

class HyperGraph {
public:
    string dataname;
    string inputpath;
    int number_of_hedges;
    int number_of_nodes;
    vector< vector<int> > node2hyperedge; 
	vector< vector<int> > hyperedge2node;
    unordered_map<int, int> nodename2index;
    unordered_map<string, int> hyperedge2index;
    unordered_map<int, int> index2nodename;
    vector<int> index2order;
    unordered_map<int, string> edgename;
    bool exist_edgename;

    HyperGraph(string inputpath, string dataname, bool dupflag);
    vector<vector<int>> get_incidence_matrix();
};
#endif
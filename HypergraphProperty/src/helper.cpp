#include "helper.hpp"

Helper::Helper(set<int> subhypergraph, HyperGraph *graph, string outputdir) {
    this->graph = graph;
    this->hypergraph_masking.resize(graph->number_of_hedges, 0);
    this->node_masking.resize(graph->number_of_nodes, 0);
    this->nodes.resize(graph->number_of_nodes, 0);
    number_of_nodes = 0;
    sum_of_hsizes=0;
    this->number_of_hedges = subhypergraph.size();
    this->number_of_nodes = number_of_nodes;
    this->outputdir = outputdir;

    this->check.resize(graph->number_of_hedges, false);
    this->check_node.resize(graph->number_of_nodes, false);

    // start with empty subhypergraph set
    assert((int)subhypergraph.size() == 0);
}

void Helper::get_average_clustering_coef(void){
    double average = 0;
    string key;

    for (int v = 0; v < (int)node_masking.size() ; v++){
        if (node_masking[v] == 0){
            continue;
        }
        int i = 0;
        for (int nv = 0 ; nv < (int)node_masking.size() ; nv++){
            if ((node_masking[nv] == 0) || (v == nv)){
                continue;
            }
            key = make_sortedkey(v, nv);
            if (pairdegree[key] > 0){
                nodes[i] = nv;
                i++;
            }
        }
        int vdeg = i;
        double cc = 0.0; // number of connected neighbor pairs
        double denominator = 0.0; // number of neighbor pairs
        if (vdeg < 2){
            continue;
        }
        // neighbor pair
        for(int va_idx = 0 ; va_idx < vdeg ; va_idx++){
            for(int vb_idx = va_idx + 1 ; vb_idx < vdeg ; vb_idx++){
                int va = nodes[va_idx];
                int vb = nodes[vb_idx];
                key = make_sortedkey(va, vb);
                if (pairdegree[key] > 0){
                    cc++;
                }
                denominator++;
            }
        }
        average += (cc / denominator) / number_of_nodes;
    }
    string writeFile = outputdir + "clusteringcoef.txt";
    ofstream resultFile(writeFile.c_str(), fstream::out | fstream::app);
    resultFile << to_string(average) << endl;
    resultFile.close();
}

int Helper::bfs(int start_node){
    int number_of_visited = 0;
    visited.push(start_node);
    check_node[start_node] = true;
    number_of_visited++;
    while((int)visited.size() > 0){
        int v = visited.front();
        visited.pop();
        for (int nv = 0 ; nv < (int)node_masking.size() ; nv++){
            if ((node_masking[nv] == 0) || (v == nv)){
                continue;
            }
            string key = make_sortedkey(v, nv);
            if (pairdegree[key] > 0){
                if (check_node[nv] == false){
                    visited.push(nv);
                    check_node[nv] = true;
                    number_of_visited++;
                }
            }
        }
    }
    return number_of_visited;
}

void Helper::count_wcc(void){
    vector<int> wcc_sizes;
    fill(check_node.begin(), check_node.end(), false);
    for(int start = 0 ; start < number_of_nodes ; start++){
        if(check_node[start] == false){
            int size_wcc = bfs(start);
            wcc_sizes.push_back(size_wcc);
        }
    }
    sort(wcc_sizes.begin(), wcc_sizes.end(), greater<int>());
    string writeFile = outputdir + "sizewcc.txt";
    ofstream resultFile(writeFile.c_str());
    int cumul_nodes = 0;
    for(auto wcc_sz : wcc_sizes){
        cumul_nodes += wcc_sz;
        resultFile << to_string(double(cumul_nodes) / number_of_nodes) << endl;
    }
    assert (cumul_nodes <= number_of_nodes);
    resultFile.close();
}

void Helper::get_dense_property(void){
    double density = double(number_of_hedges) / number_of_nodes;
    double overlapness = double(sum_of_hsizes) / number_of_nodes;

    string writeFile1 = outputdir + "density.txt";
    ofstream resultFile1(writeFile1.c_str(),  fstream::out | fstream::app);
    resultFile1 << to_string(density) << endl;
    resultFile1.close();

    string writeFile2 = outputdir + "overlapness.txt";
    ofstream resultFile2(writeFile2.c_str(),  fstream::out | fstream::app);
    resultFile2 << to_string(overlapness) << endl;
    resultFile2.close();
}

void Helper::update(set<int> deltaset, HyperGraph *graph){
    map<int, int> delta_deg;

    for (auto h : deltaset){
        assert(hypergraph_masking[h] == 0);
        hypergraph_masking[h] = 1;
        int hsize = (int)graph->hyperedge2node[h].size();
        sum_of_hsizes += hsize;
        number_of_hedges++;
        
        // neighbors
        for (int i = 0 ; i < hsize ; i++){
            int v = graph->hyperedge2node[h][i];
            if (node_masking[v] == 0){
                number_of_nodes += 1;
            }
            node_masking[v] += 1;
            for ( int j = i+1 ; j < hsize ; j++){
                int nv = graph->hyperedge2node[h][j];
                string key = make_sortedkey(v, nv);
                pairdegree[key] += 1;
                assert(pairdegree[key] >= 0);
            }
        }
    }
    return;
}

void Helper::save_properties(void){
    get_average_clustering_coef();
    count_wcc();
    get_dense_property();
}

#include "helperdist.hpp"

HelperDist::HelperDist(set<int> subhypergraph, HyperGraph *graph, string outputdir, string algo_opt, bool logflag) {
    this->graph = graph;
    this->logflag = logflag;
    this->hypergraph_masking.resize(graph->number_of_hedges, false);
    this->node_masking.resize(graph->number_of_nodes, false);
    this->nodes.resize(graph->number_of_nodes, 0);
    this->ancestor.resize(graph->number_of_nodes);
    this->group_size.resize(graph->number_of_nodes, 1);
    for (int v = 0 ; v < graph->number_of_nodes ; v++){
        this->ancestor[v] = v;
    }
    number_of_nodes = 0;
    this->number_of_hedges = subhypergraph.size();
    this->number_of_nodes = number_of_nodes;
    this->outputdir = outputdir;

    this->check.resize(graph->number_of_hedges, 0);
    this->check_node.resize(graph->number_of_nodes, false);

    // setting flag
    assert ((int)algo_opt.size() > 0);

    tmp = split(algo_opt, ',');
    for (int i = 0 ; i < (int)tmp.size() ; i++){
        if (tmp[i].compare(0, 6, "degree") == 0){
            degree_flag = true;
            degree_outname = tmp[i];
        }
        else if (tmp[i].compare(0, 4, "size") == 0){
            size_flag = true;
            size_outname = tmp[i];
        }
        else if (tmp[i].compare(0, 7, "pairdeg") == 0){
            pairdeg_flag = true;
            pairdeg_outname = tmp[i];
        }
        else if (tmp[i].compare(0, 12, "intersection") == 0){
            its_flag = true;
            its_outname = tmp[i];
        }
    }

    // update by subhypergraph
    if ((int)subhypergraph.size() > 0){
        for (int h : subhypergraph){
            this->number_of_hedges++;
            fill(check.begin(), check.end(), 0);
            this->hypergraph_masking[h] = true;
            int hsize = (int)graph->hyperedge2node[h].size();
            if (size_flag){
                size_dist[hsize]++;
            }
            for (int i = 0; i < hsize ; i++){
                int vi = graph->hyperedge2node[h][i];
                if (!this->node_masking[vi]){
                    this->number_of_nodes++;
                    this->node_masking[vi] = true; // update node_masking
                }
                if (degree_flag){
                    nodedegree[vi]++;
                }
                if (pairdeg_flag){
                    for (int j = i+1 ; j < hsize ; j++){
                        int vj = graph->hyperedge2node[h][j];
                        if (pairdeg_flag){
                            string key = make_sortedkey(vi, vj);
                            this->pairdegree_tmp[key]++;
                        }
                    }
                    
                }
                if (its_flag){
                    int vdeg = (int)graph->node2hyperedge[vi].size();
                    for (int j = 0 ; j < vdeg ; j++){
                        int nh = graph->node2hyperedge[vi][j];
                        if (!hypergraph_masking[nh]){
                            continue;
                        }
                        if (h != nh){
                            int prev = check[nh];
                            check[nh] += 1;
                            if (prev > 0){
                                assert (intersection_size[prev] > 0);
                                intersection_size[prev]--;
                                int cur = check[nh];
                                intersection_size[cur]++;
                            }
                        }
                    }
                }
            }
        }
    }
}

void HelperDist::get_intersection_size_dist(void){
    long long total = 0;
    for (auto elem : intersection_size){
        total += elem.second;
    }
    string writeFile = outputdir + its_outname + ".txt";
    ofstream resultFile(writeFile.c_str(), fstream::out); //  | fstream::app
    for (auto elem : intersection_size){
        double p = ((double)elem.second / (double)total);
        resultFile << to_string(elem.first) << "," << to_string(p) << endl;
    }
    resultFile.close();

    string writeFile_tot = outputdir + its_outname + "_total.txt";
    ofstream resultTFile(writeFile_tot.c_str(), fstream::out);
    resultTFile << to_string(total) << endl;
    resultTFile.close();
}

void HelperDist::get_degree_dist(void){
    unordered_map<int, long long> degree_dist;
    for (int v = 0 ; v < graph->number_of_nodes ; v++){
        int deg = nodedegree[v];
        if (deg > 0){
            degree_dist[deg]++;
        }
    }
    string writeFile = outputdir + degree_outname + ".txt";
    ofstream resultFile(writeFile.c_str(), fstream::out); // | fstream::app
    for (auto elem : degree_dist){
        double p = ((double)elem.second / (double)this->number_of_nodes);
        resultFile << to_string(elem.first) << "," << to_string(p) << endl;
    }
    resultFile.close();
    
    string writeFile_tot = outputdir + degree_outname + "_total.txt";
    ofstream resultTFile(writeFile_tot.c_str(), fstream::out);
    resultTFile << to_string(this->number_of_nodes) << endl;
    resultTFile.close();
}

void HelperDist::get_pairdegree_dist(void){
    unordered_map<int, long long> pairdegree_dist;
    double total = 0.0;
    for (auto elem : pairdegree_tmp){
        int deg = elem.second;
        pairdegree_dist[deg]++;
        total++;        
    }
    string writeFile = outputdir + pairdeg_outname + ".txt";
    ofstream resultFile(writeFile.c_str(), fstream::out); //  | fstream::app
    for (auto elem : pairdegree_dist){
        double p = ((double)elem.second / (double)total);
        resultFile << to_string(elem.first) << "," << to_string(p) << endl;
    }
    resultFile.close();
    string writeFile_tot = outputdir + pairdeg_outname + "_total.txt";
    ofstream resultTFile(writeFile_tot.c_str(), fstream::out);
    resultTFile << to_string(total) << endl;
    resultTFile.close();
}

void HelperDist::get_size_dist(void){
    string writeFile = outputdir + size_outname + ".txt";
    ofstream resultFile(writeFile.c_str(), fstream::out);
    for (auto elem : size_dist){
        double p = ((double)elem.second / (double)this->number_of_hedges);
        resultFile << to_string(elem.first) << "," << to_string(p) << endl;
    }
    resultFile.close();
    
    string writeFile_tot = outputdir + size_outname + "_total.txt";
    ofstream resultTFile(writeFile_tot.c_str(), fstream::out);
    resultTFile << to_string(this->number_of_hedges) << endl;
    resultTFile.close();
}

void HelperDist::update(set<int> deltaset, HyperGraph *graph){
    for (auto h : deltaset){
        this->number_of_hedges++;
        assert(!hypergraph_masking[h]);
        this->hypergraph_masking[h] = true;
        int hsize = (int)graph->hyperedge2node[h].size();
        if (size_flag){
            size_dist[hsize]++;
        }

        for (int i = 0; i < hsize ; i++){
            int vi = graph->hyperedge2node[h][i];
            if (!this->node_masking[vi]){
                this->number_of_nodes++;
                this->node_masking[vi] = true; // update node_masking
            }
            if (degree_flag){
                nodedegree[vi]++;
            }
        }

        // intersect
        if (its_flag){
            fill(check.begin(), check.end(), false);
            for (int i = 0 ; i < hsize ; i++){
                int v = graph->hyperedge2node[h][i];
                int vdeg = (int)graph->node2hyperedge[v].size();
                for (int j = 0 ; j < vdeg ; j++){
                    int nh = graph->node2hyperedge[v][j]; // neighbor hyperedge
                    if (!hypergraph_masking[nh]){
                        continue;
                    }
                    if (h != nh){
                        int prev = check[nh];
                        check[nh] += 1;
                        if (prev > 0){
                            assert (intersection_size[prev] > 0);
                            intersection_size[prev]--;
                        }
                        int cur = check[nh];
                        intersection_size[cur]++;
                    }
                }
            }
        }

        // neighbors
        if (pairdeg_flag){
            for (int i = 0 ; i < hsize ; i++){
                for ( int j = i+1 ; j < hsize ; j++){
                    int v = graph->hyperedge2node[h][i];
                    int nv = graph->hyperedge2node[h][j];
                    if (pairdeg_flag){
                        string key = make_sortedkey(v, nv);
                        this->pairdegree_tmp[key]++;
                    }
                }
            }
        }
    }
    return;
}

void HelperDist::save_properties(void){
    if (degree_flag) get_degree_dist();
    if (size_flag) get_size_dist();
    if (pairdeg_flag) get_pairdegree_dist();
    if (its_flag)  get_intersection_size_dist();
}
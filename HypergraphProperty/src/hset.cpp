#include "hset.hpp"

HSet::HSet(set<int> subhypergraph, HyperGraph *graph, string outputdir, double accuracy, string scoreflag) {
    this->hypergraph_masking.resize(graph->number_of_hedges, 0);
    this->node_masking.resize(graph->number_of_nodes, 0);
    number_of_nodes = 0;
    // assert ((int)subhypergraph.size() == 0);
    this->number_of_hedges = subhypergraph.size();
    this->number_of_nodes = number_of_nodes;
    this->outputdir = outputdir;

    if ((int)subhypergraph.size() > 0){
        set<int> nodeset;
        for (int h : subhypergraph){
            this->hypergraph_masking[h] = 1;
            int hsize = (int)graph->hyperedge2node[h].size();
            for (int i = 0; i < hsize ; i++){
                int vi = graph->hyperedge2node[h][i];
                nodeset.insert(vi);
                this->node_masking[vi] += 1; // update node_masking
            }
        }
        this->number_of_nodes = (int)nodeset.size();
    }
    string output = outputdir + "sampled_graph.txt";
    remove(output.c_str());
    output = outputdir + "sampled_hindexes.txt";
    remove(output.c_str());
}

void HSet::change_version(bool addflag){
    // Add hyperedge or Remove hyperedge
    this->addflag = addflag;
}

vector<int> HSet::get_hyperedgeset(void){
    vector<int> hyperedgeset;
    for (int h = 0 ; h < (int)hypergraph_masking.size() ; h++){
        if (hypergraph_masking[h]) hyperedgeset.push_back(h);
    }
    sort(hyperedgeset.begin(), hyperedgeset.end());
    return hyperedgeset;
}

void HSet::update(vector<int> deltaset, HyperGraph *graph, string sign){
    for (auto h : deltaset){
        // change node_masking & number of nodes    
        int hsize = (int)graph->hyperedge2node[h].size();
        for (int i = 0 ; i < hsize ; i++){
            int v = graph->hyperedge2node[h][i];
            if (sign.compare("-") == 0){
                node_masking[v] -= 1;
                if (node_masking[v] == 0){
                    number_of_nodes--;
                }
            }
            else if (sign.compare("+") == 0){
                if (node_masking[v] == 0){
                    number_of_nodes++;
                }
                node_masking[v] += 1;
            }
        }
        // update hypergraph_masking and hyperedge_order
        if (sign.compare("-") == 0){
            assert(hypergraph_masking[h] == 1);
            hypergraph_masking[h] = 0;
            int hsize = (int)graph->hyperedge2node[h].size();
            // sum_of_hsizes -= hsize;
            if (addflag){
                hyperedge_order.erase(remove(hyperedge_order.begin(), hyperedge_order.end(), h), hyperedge_order.end());
            }
            else{
                hyperedge_order.push_back(h);
            }
            number_of_hedges--;

        }
        else if (sign.compare("+") == 0){
            assert(hypergraph_masking[h] == 0);
            hypergraph_masking[h] = 1;
            int hsize = (int)graph->hyperedge2node[h].size();
            // sum_of_hsizes += hsize;
            if (addflag){
                hyperedge_order.push_back(h);
            }
            else{
                hyperedge_order.erase(remove(hyperedge_order.begin(), hyperedge_order.end(), h), hyperedge_order.end());
            }
            number_of_hedges++;
        }
    }
    return;
}

void HSet::save_as_txt(HyperGraph *graph){
    // Save all by hyperedge_order  
    string sampledgraphFName = outputdir + "sampled_graph.txt";
    ofstream sampledgraphFile(sampledgraphFName.c_str());
    cout << (int)hyperedge_order.size() << endl;
    int c = 0;
    int h;
    while (c < graph->number_of_hedges){
        if (addflag){
            h = hyperedge_order[c];
        }
        else{
            h = hyperedge_order[graph->number_of_hedges - 1 - c];
        }
        int hsize = (int)graph->hyperedge2node[h].size();
        for( int i = 0 ; i < hsize ; i++){
            int v = graph->hyperedge2node[h][i];
            sampledgraphFile << graph->index2nodename[v]; 
            if(i != (hsize -1)){
                sampledgraphFile << ",";
            }
        }
        sampledgraphFile << endl;
        c++;
    }
    sampledgraphFile.close();

    string sampledgraphFName2 = outputdir + "sampled_hindexes.txt";
    ofstream sampledgraphFile2(sampledgraphFName2.c_str());
    c = 0;
    while (c < graph->number_of_hedges){
        if (addflag){
            h = hyperedge_order[c];
        }
        else{
            h = hyperedge_order[graph->number_of_hedges - 1 - c];
        }
        sampledgraphFile2 << to_string(h) << endl;
        c++;
    }
    sampledgraphFile2.close();
}

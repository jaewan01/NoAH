#include "graph.hpp"

HyperGraph::HyperGraph(string inputpath, string dataname, bool dupflag){
    string path;
    this->inputpath = inputpath;
    this->dataname = dataname;
    
    cout << path << endl;
    path = inputpath + ".txt";
    this->exist_edgename = false;
    
    ifstream graphFile(path.c_str());
	string line;
	int num_hyperedge = 0;
    int num_nodes = 0;
    set<int> node_set;
    cout << "Start Read Dataset " << path << endl;
	while (getline(graphFile, line)){
        string ename;
		vector<string> nodes = split(line, ',');
		set<int> token_set;
        vector<int> tokens;
        // node reindexing
		for (int i = 0; i < (int)nodes.size(); i++){
            int cur_node = stoi(nodes[i]);
            node_set.insert(cur_node);
            if (nodename2index.find(cur_node) == nodename2index.end()){
                int index = num_nodes++;
                nodename2index[cur_node] = index;
                index2nodename[index] = cur_node;
                this->node2hyperedge.push_back(vector<int>());
            }
            int node_index = nodename2index[cur_node];
            token_set.insert(node_index);
        }
        for (int v : token_set){
            tokens.push_back(v);
        }
        sort(tokens.begin(), tokens.end());
        string token_str = "";
        for (int v : tokens){
            token_str += to_string(v) + "_";
        }
        // add hyperedges
        if (dupflag || (hyperedge2index.find(token_str) == hyperedge2index.end())){
            hyperedge2index[token_str] = num_hyperedge;
            this->hyperedge2node.push_back(tokens);
            if (this->exist_edgename){
                edgename[num_hyperedge] = ename;
            }
            for (int i = 0; i < (int)tokens.size(); i++){
                int node_index = tokens[i];
                this->node2hyperedge[node_index].push_back(num_hyperedge);
            }
            num_hyperedge++;
        }
	}
    this->number_of_hedges = num_hyperedge;
    this->number_of_nodes = (int)this->node2hyperedge.size();

    assert(number_of_nodes == *node_set.rbegin() + 1);
    cout << "|V| : " << number_of_nodes << " nodes" << endl;
    cout << "|E| : " << number_of_hedges << " hyperedges" << endl;

    index2order.resize(number_of_nodes, -1);
    for (int h =  0 ; h < number_of_hedges ; h++){
        int hsize = hyperedge2node[h].size();
        for (int vi= 0 ; vi < hsize; vi++){
            int v = hyperedge2node[h][vi];
            if (index2order[v] == -1){
                index2order[v] = h;
            }
        }
    }
}
vector<vector<int>> HyperGraph::get_incidence_matrix(){
    int row = this->number_of_hedges;
    int col = this->number_of_nodes;
    vector<vector<int>> inc_mat(row, std::vector<int>(col, 0));
    for (int h = 0 ; h < row ; h++){
        for (auto v : this->hyperedge2node[h]){
            inc_mat[h][v] = 1;
        }
    }
    return inc_mat;
}
#include <sys/types.h>
#include <ctime>

#include "helper.hpp"
#include "helperdist.hpp"

using namespace std;

int main(int argc, char* argv[]){
    string inputpath;
    string outputdir;
    bool logflag = false;
    bool dupflag = true;
    string dataname = "";
    string inputdir = "";
    string algoname = "";

    for(int i=1; i<argc ; i++){
        string input = argv[i];
        if(input.compare("--inputpath") == 0) inputpath = argv[++i];
        else if(input.compare("--outputdir") == 0) outputdir = argv[++i];
        else if(input.compare("--log") == 0) logflag = true;
        else if(input.compare("--dupflag") == 0) dupflag = true; // dupflag true: accept duplicate ; false: reject duplicate
    }
    HyperGraph *graph;
    graph = new HyperGraph(inputpath, dataname, dupflag);
    
    make_directory_by_name(outputdir);
    cout << outputdir << endl;
    
    // Organize past outputs
    vector<int> hyperedges;
    set<int> hedge_set;
    set<int> init_set;
    int unit_numhedge = (int)ceil((double)graph->number_of_hedges * 0.001);
    
    for (int h = 0 ; h < graph->number_of_hedges; h++){
        hyperedges.push_back(h);
        hedge_set.insert(h);
    }
    // remove outputs
    string algo_opt = "degree,size,pairdeg,intersection";
    string output;
    vector<string> tmp = split(algo_opt, ',');
    for (int i = 0 ; i < (int)tmp.size() ; i++){
        output = outputdir + tmp[i] + ".txt";
        remove(output.c_str());
    }

    // Evaluation: degree, size, pairdeg, intersection distribution
    int num_hedges = 0; 
    HelperDist *algo = new HelperDist(init_set, graph, outputdir, algo_opt, logflag);
    algo->update(hedge_set, graph);
    algo->save_properties();

    // Evaluation: sizewcc, clusteringcoef, density, overlapness
    Helper *algo2 = new Helper(init_set, graph, outputdir);
    algo2->update(hedge_set, graph);
    algo2->save_properties();
    hedge_set.clear();
    
    cout << "End" << endl;
    return 0;
}
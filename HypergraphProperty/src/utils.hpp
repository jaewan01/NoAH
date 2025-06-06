#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef UTILS_HPP
#define UTILS_HPP

using namespace std;

# define EPSILON 1e-6

inline vector<string> split(string str, char delimiter){
	vector<string> internal;
	stringstream ss(str);
	string temp;
	while (getline(ss, temp, delimiter)){
		internal.push_back(temp);
	}
	return internal;
}

inline double get_Dstat(unordered_map<int,long long> &dist, unordered_map<int,long long> &dist_target){
    vector<int> key_list;
    vector<int>::iterator it;
    double cumul_dist_v = 0.0, cumul_dist_t_v = 0.0;
    double sum_dist = 0.0, sum_dist_t = 0.0;
    
    for (auto d : dist){
        if (d.second == 0) continue;
        sum_dist += d.second;
        it = find(key_list.begin(), key_list.end(), d.first);
        if (it == key_list.end()){
            key_list.push_back(d.first);
        }
    }
    for (auto d : dist_target){
        if (d.second == 0) continue;
        sum_dist_t += d.second;
        it = find(key_list.begin(), key_list.end(), d.first);
        if (it == key_list.end()){
            key_list.push_back(d.first);
        }
    }
    sort(key_list.begin(), key_list.end());

    double d_stat = 0.0;
    for (auto key : key_list){
        if (sum_dist != 0){
            cumul_dist_v += ((double)dist[key] / sum_dist);
        }
        if (sum_dist_t != 0){
            cumul_dist_t_v += ((double)dist_target[key] / sum_dist_t);
        }

        if (d_stat < fabs(cumul_dist_v - cumul_dist_t_v)){
            d_stat = fabs(cumul_dist_v - cumul_dist_t_v);
        }
    }
    return d_stat;
}

inline string make_sortedkey(int vi, int vj){
    string key;
    if (vi < vj){
        key = to_string(vi) + "_" + to_string(vj);
    }
    else{
        key = to_string(vj) + "_" + to_string(vi);
    }
    return key;
}

inline string convertToString(char* a, int size) 
{ 
    int i; 
    string s = ""; 
    for (i = 0; i < size; i++) { 
        s = s + a[i]; 
    } 
    return s; 
} 

inline void currentDateTime(string &date_str, string &time_str) {
    time_t     now = time(0); 
    struct tm  tstruct;
    char       date_buf[11];
    char       time_buf[9];
    tstruct = *localtime(&now);
    strftime(date_buf, sizeof(date_buf), "%Y-%m-%d", &tstruct);
    strftime(time_buf, sizeof(time_buf), "%H-%M-%S", &tstruct);

    date_str = convertToString(date_buf, 10);
    time_str = convertToString(time_buf, 8);

    return;
}

inline bool file_exist (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

inline string make_directory (vector<string> args){
    // string outputdir = "/home/dmlab/minyoung/HypergraphProperty/results/";
    string outputdir = "/home/dmlab/minyoung/HypergraphProperty/";
    for (int i = 0 ; i < (int)args.size() ; i++){
        outputdir += args[i] + "/";
        mkdir(outputdir.c_str(), 0776);
    }
    // string degree_outputdir = outputdir + "degree/";
    // mkdir(degree_outputdir.c_str(), 0776);
    return outputdir;
}

inline void make_directory_by_name (string dirname){
    string outputdir = "";
    vector<string> tmp = split(dirname, '/');
    for (int i = 0 ; i < (int)tmp.size() ; i++){
        outputdir += tmp[i] + "/";
        if (!file_exist(outputdir)){
            mkdir(outputdir.c_str(), 0776);
        }
    }
    // string degree_outputdir = outputdir + "degree/";
    // mkdir(degree_outputdir.c_str(), 0776);
    return;
}

inline bool already_run (const std::string& name, int numhedges){
    if (!file_exist(name)){
        return false;
    }
    ifstream File(name.c_str());
    int num_sampledhedges = 0;
    string line;
    while (getline(File, line)){
        num_sampledhedges += 1;
    }
    if (numhedges == num_sampledhedges){
        return true;
    }
    else{
        return false;
    }
}

#endif
#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include <utility>

std::pair<double, double> compute_avg_stddev(const std::vector<double>& times, int start, int len){
    double sum = 0.0;
    for(int i=start; i<len+start; i++){
        sum += times[i];
    }
    double avg = sum / len;
    double sq_sum = 0.0;
    for(int i=start; i<len+start; i++){
        sq_sum += (times[i] - avg) * (times[i] - avg);
    }
    double stddev = std::sqrt(sq_sum / len);
    
    return std::make_pair(avg, stddev);
}

#endif //UTILS_HPP
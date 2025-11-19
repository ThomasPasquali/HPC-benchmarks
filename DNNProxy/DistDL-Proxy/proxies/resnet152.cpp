/*********************************************************************
 *
 * Description: C++/MPI proxy for ResNet-152 distributed training 
 *              with data parallelism
 * Author: Shigang Li
 * Email: shigangli.cs@gmail.com
 *
 *********************************************************************/

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <cmath>
#include <utility>

#include "utils.hpp"

#define WARM_UP 8
#define RUNS 8

// allreduce sizes for gradients with message aggregation
#define NUM_B 10
int allreduce_sizes[NUM_B] = {6511592, 6567936, 5905920, 6113280, 6176256, 6112768, 6176256, 6112768, 5321216, 5194816};

std::vector<double> allreduce_times = std::vector<double>(WARM_UP + RUNS);

// batchsize = 128
// Suggest world_size <= 256, which is corresponding to a global batch_size <= 32 K
// A100 GPU
// runtime in us (10E-6) for each iteration 
int fwd_rt_whole_model = 119000;
int bwd_rt_per_B = 23800; //considering 10 buckets

int run_data_parallel(float** grad_ptrs, float** sum_grad_ptrs, int new_num_b, int* gradient_sizes, int run_id){
    
    //forward
    usleep(fwd_rt_whole_model); //compute

    //backward
    MPI_Request grad_allreduce_reqs[new_num_b];
    //must initialize with MPI_REQUEST_NULL
    for(int i=0; i<new_num_b; i++)
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;

    int index, flag;
    for(int i=0; i<new_num_b; i++){
	if(i > 1)
            MPI_Testany(new_num_b, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); //advancing MPI in the background

        usleep(bwd_rt_per_B); //compute

        MPI_Iallreduce(grad_ptrs[i], sum_grad_ptrs[i], gradient_sizes[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i]);	
    }


    double start_time = MPI_Wtime();
    MPI_Waitall(new_num_b, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    double end_time = MPI_Wtime();

    allreduce_times[run_id] = end_time - start_time;

    return 0;
}

// adapt allreduce_sizes and bwd_rt_per_B when changing the number of buckets (can be smaller or greater than NUM_B)
std::pair<int, int*> change_num_bucket(uint new_num_b){
    assert(new_num_b > 0);

    float new_bwd_rt_per_B = bwd_rt_per_B * (NUM_B / (float)new_num_b);

    // Calculate total size
    // Calculate total size
    long long total_size = 0;
    for (int i = 0; i < NUM_B; i++) {
        total_size += allreduce_sizes[i];
    }
    
    // Allocate new array for bucket sizes
    int* new_allreduce_sizes = (int*)calloc(new_num_b, sizeof(int));
    
    // Step 1: Sample the distribution pattern (without preserving total yet)
    double* proportions = (double*)malloc(new_num_b * sizeof(double));
    double sum_proportions = 0.0;
    
    for (uint i = 0; i < new_num_b; i++) {
        // Map new bucket index to original bucket space
        float original_pos = i * (NUM_B - 1) / (float)(new_num_b - 1);
        int base_idx = (int)original_pos;
        float fraction = original_pos - base_idx;
        
        if (base_idx >= NUM_B - 1) {
            proportions[i] = allreduce_sizes[NUM_B - 1];
        } else {
            // Interpolate between two adjacent buckets
            proportions[i] = allreduce_sizes[base_idx] * (1.0 - fraction) +
                                allreduce_sizes[base_idx + 1] * fraction;
        }
        sum_proportions += proportions[i];
    }
    
    // Step 2: Scale proportions to match total_size
    long long assigned_total = 0;
    for (uint i = 0; i < new_num_b - 1; i++) {
        new_allreduce_sizes[i] = (int)std::round((proportions[i] / sum_proportions) * total_size);
        assigned_total += new_allreduce_sizes[i];
    }
    
    // Step 3: Assign remainder to last bucket to ensure exact total
    new_allreduce_sizes[new_num_b - 1] = total_size - assigned_total;
    
    free(proportions);
    
    return std::make_pair(new_bwd_rt_per_B, new_allreduce_sizes);
}


int main(int argc, char *argv[]){
    int rank, world_size;
    uint old_num_b = NUM_B;
    uint new_num_b = NUM_B;

    int* gradient_sizes = allreduce_sizes;

    if(argc == 2) new_num_b = std::stoi(argv[1]);
        
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (new_num_b != old_num_b){
        std::pair<int, int*> res = change_num_bucket(new_num_b);
        bwd_rt_per_B = res.first;
        gradient_sizes = res.second;
   
    } 

    float* grad_ptrs[new_num_b];
    float* sum_grad_ptrs[new_num_b];
    for(int i=0; i<new_num_b; i++){
        grad_ptrs[i] = (float *)calloc(gradient_sizes[i], sizeof(float));
        sum_grad_ptrs[i] = (float *)calloc(gradient_sizes[i], sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int run_id = 0;

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, new_num_b, gradient_sizes, run_id++);
    }

    std::vector<double> run_times = std::vector<double>(RUNS);

    double begin;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        begin = MPI_Wtime();
        run_data_parallel(grad_ptrs, sum_grad_ptrs, new_num_b, gradient_sizes, run_id++);
        run_times[iter] = (MPI_Wtime()-begin);
    }

    std::pair<double, double> avg_std = compute_avg_stddev(run_times, 0, RUNS);
    double runtime_avg = avg_std.first;
    double runtime_stddev = avg_std.second;

    // int total_params = 0;
    // for(int i=0; i<new_num_b; i++){
	//     total_params += gradient_sizes[i];	
    // }

    //avg and std of allreduce times
    avg_std = compute_avg_stddev(allreduce_times, WARM_UP, RUNS);
    double barrier_avg = avg_std.first;
    double barrier_stddev = avg_std.second;

    std::vector<double> msg_sizes(gradient_sizes, gradient_sizes + new_num_b);
    avg_std = compute_avg_stddev(msg_sizes, 0, new_num_b);
    double msg_size_avg = avg_std.first;
    double msg_size_stddev = avg_std.second;

    
    if(rank == 0){
        //TODO: remove this thing and make a useful log with (packet size, bucket size, runtime, compute time, comm time, etc.)
        printf("Avg_runtime(s):%f\n", runtime_avg);
        printf("Stddev_runtime(s):%f\n", runtime_stddev);
        printf("Fwd_compute(us):%d\n", fwd_rt_whole_model);
        printf("Bwd_compute_per_B(us):%d\n", bwd_rt_per_B);
        printf("Num_buckets:%d\n", new_num_b);
        printf("Avg_barrier_time(s):%f\n", barrier_avg);
        printf("Stddev_barrier_time(s):%f\n", barrier_stddev);
        printf("Avg_msg_size(bytes):%f\n", msg_size_avg * sizeof(float));
        printf("Stddev_msg_size(bytes):%f\n", msg_size_stddev * sizeof(float));
    }

    MPI_Finalize();
}

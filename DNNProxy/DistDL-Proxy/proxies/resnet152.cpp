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

#include <ccutils/mpi/mpi_macros.h>
#include <ccutils/mpi/mpi_timers.h>
#include <ccutils/timers.h>

//TODO: use the ccutils MPI macros and timers

MPI_TIMER_DEF(runtime)
MPI_TIMER_DEF(barrier)

#define WARM_UP 8
#define RUNS 8

// allreduce sizes for gradients with message aggregation
#define NUM_B 10
int allreduce_sizes[NUM_B] = {
    6019281, 6019281, 6019281, 6019281, 
    6019281, 6019281, 6019281, 6019281,
    6019280, 6019280
};

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


    MPI_TIMER_START(barrier)
    MPI_Waitall(new_num_b, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    MPI_TIMER_STOP(barrier)

    return 0;
}

// adapt allreduce_sizes and bwd_rt_per_B when changing the number of buckets (can be smaller or greater than NUM_B)
std::pair<int, int*> change_num_bucket(uint new_num_b){
    assert(new_num_b > 0);

    float new_bwd_rt_per_B = bwd_rt_per_B * (NUM_B / (float)new_num_b);

    long long total_size = 0;
    for (uint32_t i = 0; i < NUM_B; i++) {
        total_size += allreduce_sizes[i];
    }

    // Alloca nuovo array per i bucket
    int* new_allreduce_sizes = (int*)calloc(new_num_b, sizeof(int));
    if (!new_allreduce_sizes) return {new_bwd_rt_per_B, nullptr};

    // Divisione uniforme
    int base = total_size / new_num_b;
    int remainder = total_size % new_num_b;

    for (uint32_t i = 0; i < new_num_b; i++) {
        new_allreduce_sizes[i] = base + (i < remainder ? 1 : 0);
    }

    return {new_bwd_rt_per_B, new_allreduce_sizes};
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

    std::vector<float> run_times = std::vector<float>(RUNS);

    for(int iter = 0; iter < RUNS; iter++){
        MPI_TIMER_START(runtime)
        run_data_parallel(grad_ptrs, sum_grad_ptrs, new_num_b, gradient_sizes, run_id++);
        MPI_TIMER_STOP(runtime)
    }

    ccutils_timers::TimerStats runtime_stats;
    ccutils_timers::TimerStats barrier_stats;

    runtime_stats = ccutils_timers::compute_stats(__timer_vals_runtime);
    barrier_stats = ccutils_timers::compute_stats(__timer_vals_barrier);

    float runtime_avg = runtime_stats.avg;
    float runtime_stddev = runtime_stats.stddev;
    float barrier_avg = barrier_stats.avg;
    float barrier_stddev = barrier_stats.stddev;

    float msg_size_avg = 0.0f;
    for(int i=0; i<new_num_b; i++){
        msg_size_avg += gradient_sizes[i];
    }
    msg_size_avg = msg_size_avg / new_num_b;

    float msg_size_std = 0.0f;
    for(int i=0; i<new_num_b; i++){
        msg_size_std += (gradient_sizes[i] - msg_size_avg) * (gradient_sizes[i] - msg_size_avg);
    }
    msg_size_std = std::sqrt(msg_size_std / new_num_b);

    MPI_PRINT_ONCE("Avg_runtime(s):%f\n", runtime_avg);
    MPI_PRINT_ONCE("Stddev_runtime(s):%f\n", runtime_stddev);
    MPI_PRINT_ONCE("Fwd_compute(us):%d\n", fwd_rt_whole_model);
    MPI_PRINT_ONCE("Bwd_compute_per_B(us):%d\n", bwd_rt_per_B);
    MPI_PRINT_ONCE("Num_buckets:%d\n", new_num_b);
    MPI_PRINT_ONCE("Avg_barrier_time(s):%f\n", barrier_avg);
    MPI_PRINT_ONCE("Stddev_barrier_time(s):%f\n", barrier_stddev);
    MPI_PRINT_ONCE("Avg_msg_size(bytes):%f\n", msg_size_avg * sizeof(float));
    MPI_PRINT_ONCE("Stddev_msg_size(bytes):%f\n", msg_size_std*sizeof(float)); //all buckets have similar size

    MPI_Finalize();
}

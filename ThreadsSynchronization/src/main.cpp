#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <cstring>  // for strcmp
#include <cstdlib>  // for atoi

std::atomic<bool> flag(false);
std::atomic<bool> stop_flag(false);
std::atomic<uint64_t> total_ops(0);

// This will be defined differently by each benchmark
extern void* Worker(void* arg);

void print_help(const char* prog_name) {
    printf("Usage: %s [num_threads] [duration_sec] [runs]\n", prog_name);
    printf("  num_threads   Number of threads to launch (default: hardware concurrency)\n");
    printf("  duration_sec  Time (in seconds) to run each test (default: 5)\n");
    printf("  runs          Number of runs to repeat the experiment (default: 10)\n");
    printf("  -h, --help    Show this help message\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    int num_threads = std::thread::hardware_concurrency();
    int test_duration_sec = 5;
    int runs = 10;

    // Parse CLI arguments
    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
        num_threads = std::atoi(argv[1]);
    }
    if (argc >= 3) test_duration_sec = std::atoi(argv[2]);
    if (argc >= 4) runs = std::atoi(argv[3]);

    // CSV header
    printf("run,threads,duration_sec,total_ops,throughput_mops\n");

    for (int run = 1; run <= runs; ++run) {
        flag.store(false);
        stop_flag.store(false);
        total_ops.store(0);

        std::vector<pthread_t> threads(num_threads);

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_threads; ++i)
            pthread_create(&threads[i], nullptr, Worker, (void*)(intptr_t)i);

        std::this_thread::sleep_for(std::chrono::seconds(test_duration_sec));
        stop_flag.store(true);

        for (int i = 0; i < num_threads; ++i)
            pthread_join(threads[i], nullptr);

        auto end_time = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end_time - start_time).count();
        uint64_t ops = total_ops.load();
        double mops = ops / (duration * 1e6);

        printf("%d,%d,%.3f,%lu,%.3f\n", run, num_threads, duration, ops, mops);
    }

    return 0;
}

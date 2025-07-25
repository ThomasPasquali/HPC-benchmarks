#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <cmath>
#include <pthread.h>
#include <algorithm>

extern std::atomic<bool> stop_flag;
extern std::atomic<uint64_t> total_ops;

// --------------------------
// Mutex Lock (pthreads)
// --------------------------
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Simulated critical section
int CS(int id) {
    return id * 2 + 1; // Dummy computation
}

// Worker thread function
void* Worker(void* arg) {
    int id = (intptr_t)arg;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        pthread_mutex_lock(&mutex);
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        pthread_mutex_unlock(&mutex);
    }

    return nullptr;
}

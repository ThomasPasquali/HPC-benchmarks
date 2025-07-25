#include <vector>
#include <thread>
#include <atomic>
#include <pthread.h>

extern std::atomic<bool> flag;
extern std::atomic<bool> stop_flag;
extern std::atomic<uint64_t> total_ops;

int CS(int id) {
    return id * 2 + 1;
}

void* Worker(void* arg) {
    int id = (intptr_t)arg;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        bool acquired = false;
        while (!acquired) {
            bool expected = false;
            // Try to acquire the lock
            acquired = flag.compare_exchange_strong(expected, true, std::memory_order_acquire);
            if (!acquired) {
                // Optional: yield CPU to avoid busy-waiting too aggressively
                std::this_thread::yield();
            }
        }

        // Critical section
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);

        // Release lock
        flag.store(false, std::memory_order_release);
    }

    return nullptr;
}

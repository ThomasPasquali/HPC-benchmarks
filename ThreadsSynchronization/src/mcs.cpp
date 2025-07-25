#include <vector>
#include <thread>
#include <atomic>
#include <pthread.h>
#include <algorithm>

extern std::atomic<bool> stop_flag;
extern std::atomic<uint64_t> total_ops;

// --------------------------
// MCS Lock implementation
// --------------------------
class MCSLock {
public:
    struct Node {
        std::atomic<Node*> next{nullptr};
        std::atomic<bool> locked{false};
    };

private:
    std::atomic<Node*> tail{nullptr};

public:
    void lock(Node* node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        node->locked.store(true, std::memory_order_relaxed);

        Node* pred = tail.exchange(node, std::memory_order_acq_rel);
        if (pred != nullptr) {
            pred->next.store(node, std::memory_order_release);
            while (node->locked.load(std::memory_order_acquire)) {
                std::this_thread::yield();  // reduce busy spinning
            }
        }
    }

    void unlock(Node* node) {
        Node* succ = node->next.load(std::memory_order_acquire);
        if (!succ) {
            Node* expected = node;
            if (tail.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) {
                return;  // no successor
            }
            // Wait until successor sets next
            while ((succ = node->next.load(std::memory_order_acquire)) == nullptr) {
                std::this_thread::yield();
            }
        }
        succ->locked.store(false, std::memory_order_release);
    }
};

// Global MCS lock instance
MCSLock lock;

// Simulated critical section
int CS(int id) {
    return id * 2 + 1; // Dummy computation
}

// Worker thread function
void* Worker(void* arg) {
    int id = (intptr_t)arg;
    MCSLock::Node node;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        lock.lock(&node);
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        lock.unlock(&node);
    }

    return nullptr;
}

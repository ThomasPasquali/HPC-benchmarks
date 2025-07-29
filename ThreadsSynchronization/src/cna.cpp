#include <vector>
#include <thread>
#include <atomic>
#include <pthread.h>
#include <algorithm>
#include <numa.h>
#include <sched.h>

extern std::atomic<bool> stop_flag;
extern std::atomic<uint64_t> total_ops;

// --------------------------
// CNA Lock implementation
// --------------------------
typedef struct cna_node {
    std::atomic<uintptr_t> spin;
    std::atomic<int> socket;
    std::atomic<struct cna_node *> secTail;
    std::atomic<struct cna_node *> next;
} cna_node_t;

typedef struct {
    std::atomic<cna_node_t *> tail;
} cna_lock_t;

// Fixed NUMA node detection
static int current_numa_node() {
    int core = sched_getcpu();
    return (core / 16);
}

#define THRESHOLD (0xffff)
#define UNLOCK_COUNT_THRESHOLD 1024

static inline uint32_t xor_random() {
    static thread_local uint32_t rv = 1; // Simplified initialization
    uint32_t v = rv;
    v ^= v << 6;
    v ^= (uint32_t)(v) >> 21;
    v ^= v << 7;
    rv = v;
    return v & (UNLOCK_COUNT_THRESHOLD - 1);
}

static inline bool keep_lock_local() { return xor_random() & THRESHOLD; }

static inline cna_node_t *find_successor(cna_node_t *me) {
    cna_node_t *next = me->next.load(std::memory_order_relaxed);
    int mySocket = me->socket.load(std::memory_order_relaxed);
    if (mySocket == -1)
        mySocket = current_numa_node();
    if (next && next->socket.load(std::memory_order_relaxed) == mySocket) {
        return next;
    }

    cna_node_t *secHead = next;
    cna_node_t *secTail = next;
    if (!next)
        return NULL;

    cna_node_t *cur = next->next.load(std::memory_order_acquire);
    while (cur) {
        if (cur->socket.load(std::memory_order_relaxed) == mySocket) {
            if (me->spin.load(std::memory_order_relaxed) > 1) {
                cna_node_t *_spin = (cna_node_t *)me->spin.load(std::memory_order_relaxed);
                cna_node_t *_secTail = _spin->secTail.load(std::memory_order_relaxed);
                _secTail->next.store(secHead, std::memory_order_relaxed);
            } else {
                me->spin.store((uintptr_t)secHead, std::memory_order_relaxed);
            }
            secTail->next.store(NULL, std::memory_order_relaxed);
            cna_node_t *_spin = (cna_node_t *)me->spin.load(std::memory_order_relaxed);
            _spin->secTail.store(secTail, std::memory_order_relaxed);
            return cur;
        }
        secTail = cur;
        cur = cur->next.load(std::memory_order_acquire);
    }
    return NULL;
}

static inline void cna_lock(cna_lock_t *lock, cna_node_t *me) {
    me->next.store(NULL, std::memory_order_relaxed);
    me->socket.store(-1, std::memory_order_relaxed);
    me->spin.store(0, std::memory_order_relaxed);

    cna_node_t *tail = lock->tail.exchange(me, std::memory_order_seq_cst);

    if (!tail) {
        me->spin.store(1, std::memory_order_relaxed);
        return;
    }

    me->socket.store(current_numa_node(), std::memory_order_relaxed);
    tail->next.store(me, std::memory_order_release);

    while (!me->spin.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static inline void cna_unlock(cna_lock_t *lock, cna_node_t *me) {
    cna_node_t *next = me->next.load(std::memory_order_acquire);

    if (!next) {
        uintptr_t spin = me->spin.load(std::memory_order_relaxed);
        if (spin == 1) {
            cna_node_t *expected = me;
            if (lock->tail.compare_exchange_strong(expected, NULL, 
                std::memory_order_seq_cst, std::memory_order_relaxed)) {
                return;
            }
        } else {
            cna_node_t *secHead = (cna_node_t *)spin;
            cna_node_t *expected = me;
            if (lock->tail.compare_exchange_strong(expected,
                secHead->secTail.load(std::memory_order_relaxed),
                std::memory_order_seq_cst, std::memory_order_relaxed)) {
                secHead->spin.store(1, std::memory_order_release);
                return;
            }
        }

        while (!(next = me->next.load(std::memory_order_acquire))) {
            std::this_thread::yield();
        }
    }

    cna_node_t *succ = NULL;
    if (keep_lock_local() && (succ = find_successor(me))) {
        succ->spin.store(me->spin.load(std::memory_order_relaxed), std::memory_order_release);
    } else if (me->spin.load(std::memory_order_relaxed) > 1) {
        succ = (cna_node_t *)me->spin.load(std::memory_order_relaxed);
        cna_node_t *secTail = succ->secTail.load(std::memory_order_relaxed);
        secTail->next.store(next, std::memory_order_relaxed);
        succ->spin.store(1, std::memory_order_release);
    } else {
        next->spin.store(1, std::memory_order_release);
    }
}

// Global CNA lock instance
cna_lock_t lock;

// Simulated critical section
int CS(int id) {
    return id * 2 + 1; // Dummy computation
}

// Worker thread function
void* Worker(void* arg) {
    int id = (intptr_t)arg;
    cna_node_t node;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        cna_lock(&lock, &node);
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        cna_unlock(&lock, &node);
    }

    return nullptr;
}

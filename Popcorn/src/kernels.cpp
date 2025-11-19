#include <bits/stdc++.h>
#include <cblas.h>
#include <omp.h>
#include <ccutils/colors.h>
#include "kernels.h"

// CPU equivalent of Pair structure
Pair cpu_argmin(Pair a, Pair b) {
  return a.v <= b.v ? a : b;
}

/**
 * @brief CPU version: finds the closest centroid for each point
 *
 * @param n number of points
 * @param k number of clusters
 * @param distances distance matrix (n x k)
 * @param points_clusters point-cluster associations
 * @param clusters_len length of clusters
 * @param is_row_major memory layout flag
 */
void clusters_argmin_cpu(const uint32_t n, const uint32_t k, 
                         DATA_TYPE* distances, uint32_t* points_clusters,  
                         uint32_t* clusters_len, bool is_row_major) {
  
  // Reset cluster lengths
  std::fill(clusters_len, clusters_len + k, 0);
  
  #pragma omp parallel for
  for (uint32_t point = 0; point < n; point++) {
    DATA_TYPE min_dist = std::numeric_limits<DATA_TYPE>::max();
    uint32_t min_cluster = 0;
    
    for (uint32_t cluster = 0; cluster < k; cluster++) {
      uint32_t idx;
      if (is_row_major) {
        idx = point * k + cluster;
      } else {
        idx = cluster * n + point;
      }
      
      if (distances[idx] < min_dist) {
        min_dist = distances[idx];
        min_cluster = cluster;
      }
    }
    
    points_clusters[point] = min_cluster;
    
    // Thread-safe increment
    #pragma omp atomic
    clusters_len[min_cluster]++;
  }
}

/**
 * @brief Computes centroids using shuffle-like reduction (CPU version)
 */
void compute_centroids_cpu(DATA_TYPE* centroids, const DATA_TYPE* points, 
                           const uint32_t* points_clusters, 
                           const uint32_t* clusters_len, 
                           const uint64_t n, const uint32_t d, 
                           const uint32_t k) {
  
  // Initialize centroids to zero
  std::fill(centroids, centroids + k * d, 0.0);
  
  // Sum all points in each cluster
  #pragma omp parallel
  {
    // Thread-local accumulator
    std::vector<DATA_TYPE> local_centroids(k * d, 0.0);
    
    #pragma omp for nowait
    for (uint64_t i = 0; i < n; i++) {
      uint32_t cluster = points_clusters[i];
      for (uint32_t j = 0; j < d; j++) {
        local_centroids[cluster * d + j] += points[i * d + j];
      }
    }
    
    // Reduce thread-local results
    #pragma omp critical
    {
      for (uint32_t cluster = 0; cluster < k; cluster++) {
        for (uint32_t j = 0; j < d; j++) {
          centroids[cluster * d + j] += local_centroids[cluster * d + j];
        }
      }
    }
  }
  
  // Divide by cluster size
  #pragma omp parallel for
  for (uint32_t cluster = 0; cluster < k; cluster++) {
    uint32_t count = clusters_len[cluster] > 0 ? clusters_len[cluster] : 1;
    DATA_TYPE scale = 1.0 / static_cast<DATA_TYPE>(count);
    for (uint32_t j = 0; j < d; j++) {
      centroids[cluster * d + j] *= scale;
    }
  }
}

/**
 * @brief Computes V matrix (cluster membership matrix) in CSR format
 */
void compute_v_sparse_csr_cpu(DATA_TYPE* vals,
                              int32_t* colinds,
                              int32_t* row_offsets,
                              const uint32_t* points_clusters,
                              const uint32_t* clusters_len,
                              const size_t n,
                              const uint32_t k) {
  
  // Count elements per row
  std::vector<uint32_t> row_counts(k, 0);
  for (size_t i = 0; i < n; i++) {
    row_counts[points_clusters[i]]++;
  }
  
  // Compute row offsets
  row_offsets[0] = 0;
  for (uint32_t i = 0; i < k; i++) {
    row_offsets[i + 1] = row_offsets[i] + row_counts[i];
  }
  
  // Fill values and column indices
  std::vector<uint32_t> current_positions(k, 0);
  for (uint32_t i = 0; i < k; i++) {
    current_positions[i] = row_offsets[i];
  }
  
  for (size_t i = 0; i < n; i++) {
    uint32_t cluster = points_clusters[i];
    uint32_t pos = current_positions[cluster]++;
    vals[pos] = 1.0 / static_cast<DATA_TYPE>(clusters_len[cluster]);
    colinds[pos] = static_cast<int32_t>(i);
  }
}

/**
 * @brief Compute kernel matrix using naive method
 */
void compute_kernel_matrix_cpu(DATA_TYPE* K, 
                               const DATA_TYPE* P, 
                               const unsigned long long n, 
                               const uint32_t d) {
  
  #pragma omp parallel for collapse(2)
  for (unsigned long long i = 0; i < n; i++) {
    for (unsigned long long j = 0; j < n; j++) {
      DATA_TYPE result = 0.0;
      for (uint32_t dim = 0; dim < d; dim++) {
        result += P[i * d + dim] * P[j * d + dim];
      }
      K[i * n + j] = result;
    }
  }
}

/**
 * @brief Initialize kernel matrix using GEMM (BLAS)
 */
void init_kernel_mtx_gemm_cpu(const unsigned long long n,
                              const uint32_t d,
                              const DATA_TYPE* points,
                              DATA_TYPE* B) {
  
  // B = P * P^T using BLAS
  // P is n x d (row major)
  // Result is n x n
  
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n, n, d,
              1.0,           // alpha
              points, d,     // A
              points, d,     // B
              0.0,           // beta
              B, n);         // C
}

/**
 * @brief Apply linear kernel transformation
 */
void apply_linear_kernel_cpu(const uint32_t n, DATA_TYPE* B) {
  #pragma omp parallel for
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] *= -2.0;
  }
}

/**
 * @brief Apply polynomial kernel transformation
 */
void apply_polynomial_kernel_cpu(const uint32_t n, DATA_TYPE* B) {
  #pragma omp parallel for
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] = -2.0 * std::pow(B[i] + 1.0, 2);
  }
}

/**
 * @brief Apply sigmoid kernel transformation
 */
void apply_sigmoid_kernel_cpu(const uint32_t n, DATA_TYPE* B) {
  #pragma omp parallel for
  for (unsigned long long i = 0; i < (unsigned long long)n * n; i++) {
    B[i] = -2.0 * std::tanh(B[i] + 1.0);
  }
}

/**
 * @brief Copy diagonal elements with scaling
 */
void copy_diag_scal_cpu(const DATA_TYPE* M, DATA_TYPE* output,
                        const int m, const int n,
                        const DATA_TYPE alpha) {
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    output[i] = M[i * n + i] / alpha;
  }
}

/**
 * @brief Compute distances using sparse matrix multiplication (CPU version)
 */
void compute_distances_spmm_cpu(const uint32_t n, const uint32_t k,
                                const DATA_TYPE* B,
                                const DATA_TYPE* V_vals,
                                const int32_t* V_colinds,
                                const int32_t* V_rowptrs,
                                DATA_TYPE* distances) {
  
  // D = V * B (sparse-dense matrix multiplication)
  // V is k x n (CSR format)
  // B is n x n (dense)
  // D is k x n (dense, column major for compatibility)
  
  #pragma omp parallel for
  for (uint32_t i = 0; i < k; i++) {
    for (uint32_t j = 0; j < n; j++) {
      DATA_TYPE sum = 0.0;
      for (int32_t idx = V_rowptrs[i]; idx < V_rowptrs[i + 1]; idx++) {
        int32_t col = V_colinds[idx];
        sum += V_vals[idx] * B[col * n + j];
      }
      distances[i + j * k] = sum; // Column major
    }
  }
}

/**
 * @brief Sum points for naive distance computation
 */
void sum_points_cpu(const DATA_TYPE* K,
                   const int32_t* clusters,
                   const uint32_t* clusters_len,
                   DATA_TYPE* distances,
                   const uint32_t n, const uint32_t k) {
  
  #pragma omp parallel for
  for (uint32_t point_id = 0; point_id < n; point_id++) {
    for (uint32_t cluster = 0; cluster < k; cluster++) {
      DATA_TYPE sum = 0.0;
      for (uint32_t j = 0; j < n; j++) {
        if (clusters[j] == static_cast<int32_t>(cluster)) {
          sum += K[point_id * n + j];
        }
      }
      distances[point_id * k + cluster] = sum / static_cast<DATA_TYPE>(clusters_len[cluster]);
    }
  }
}

/**
 * @brief Sum centroids for naive distance computation
 */
void sum_centroids_cpu(const DATA_TYPE* tmp,
                      const int32_t* clusters,
                      const uint32_t* clusters_len,
                      DATA_TYPE* centroids,
                      const uint32_t n, const uint32_t k) {
  
  std::fill(centroids, centroids + k, 0.0);
  
  #pragma omp parallel
  {
    std::vector<DATA_TYPE> local_centroids(k, 0.0);
    
    #pragma omp for nowait
    for (uint32_t i = 0; i < n; i++) {
      uint32_t c = clusters[i];
      uint32_t len = clusters_len[c];
      DATA_TYPE thread_data = tmp[c + k * i] / -2.0;
      thread_data /= static_cast<DATA_TYPE>(len);
      local_centroids[c] += thread_data;
    }
    
    #pragma omp critical
    {
      for (uint32_t c = 0; c < k; c++) {
        centroids[c] += local_centroids[c];
      }
    }
  }
}

/**
 * @brief Compute final distances in naive method
 */
void compute_distances_naive_cpu(const DATA_TYPE* K,
                                 const DATA_TYPE* centroids,
                                 const DATA_TYPE* tmp,
                                 DATA_TYPE* distances,
                                 const uint32_t n, const uint32_t k) {
  
  #pragma omp parallel for collapse(2)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      distances[j + i * k] = tmp[j + i * k] + centroids[j];
    }
  }
}
// pairwise_clock_sync.c
// Estimate pairwise local-clock differences across MPI processes in C
// Usage: mpirun -n P ./sync [rounds]

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts); // system clock
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rounds = 10;
    if (argc > 1) {
        rounds = atoi(argv[1]);
        if (rounds <= 0) rounds = 1;
    }

    double* offsets_to = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) offsets_to[i] = NAN;
    offsets_to[rank] = 0.0;

    for (int other = 0; other < size; other++) {
        if (other == rank) continue;

        int a = rank < other ? rank : other;
        int b = rank < other ? other : rank;
        int base_tag = 10000 + a * size + b;
        int reply_base = base_tag + 100000;

        if (rank < other) {
            double best_rtt = DBL_MAX;
            double best_offset = NAN;

            for (int r = 0; r < rounds; r++) {
                double t1 = now_seconds();
                MPI_Send(&t1, 1, MPI_DOUBLE, other, base_tag + r, MPI_COMM_WORLD);

                double t2t3[2];
                MPI_Recv(t2t3, 2, MPI_DOUBLE, other, reply_base + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double t4 = now_seconds();

                double t2 = t2t3[0];
                double t3 = t2t3[1];

                double offset = ((t2 - t1) + (t3 - t4)) / 2.0;
                double rtt = (t4 - t1) - (t3 - t2);

                if (rtt < best_rtt) {
                    best_rtt = rtt;
                    best_offset = offset;
                }
            }

            offsets_to[other] = best_offset;
        } else {
            for (int r = 0; r < rounds; r++) {
                double t1recv;
                MPI_Recv(&t1recv, 1, MPI_DOUBLE, other, base_tag + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double t2 = now_seconds();
                double t3 = now_seconds();
                double t2t3[2] = { t2, t3 };
                MPI_Send(t2t3, 2, MPI_DOUBLE, other, reply_base + r, MPI_COMM_WORLD);
            }
        }
    }

    // Gather results at root
    double* gathered = NULL;
    if (rank == 0) gathered = (double*)malloc(size * size * sizeof(double));

    MPI_Gather(offsets_to, size, MPI_DOUBLE,
               gathered, size, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        double** matrix = (double**)malloc(size * sizeof(double*));
        for (int i = 0; i < size; i++) {
            matrix[i] = (double*)malloc(size * sizeof(double));
            for (int j = 0; j < size; j++) {
                matrix[i][j] = gathered[i * size + j];
            }
        }

        // Symmetrize
        for (int i = 0; i < size; i++) {
            matrix[i][i] = 0.0;
        }
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                int ij = !isnan(matrix[i][j]);
                int ji = !isnan(matrix[j][i]);
                if (ij && !ji) matrix[j][i] = -matrix[i][j];
                else if (!ij && ji) matrix[i][j] = -matrix[j][i];
                else if (ij && ji) {
                    double avg = 0.5 * (matrix[i][j] - matrix[j][i]);
                    matrix[i][j] = avg;
                    matrix[j][i] = -avg;
                }
            }
        }

        // Print matrix
        printf("Pairwise clock offset matrix (seconds): offset[i][j] ~= clock_j - clock_i\n");
        printf("Rows = i, Columns = j  (positive => j's clock ahead of i)\n\n");

        printf("%8s", "proc\\to");
        for (int j = 0; j < size; j++) printf("%16d", j);
        printf("\n");

        for (int i = 0; i < size; i++) {
            printf("%8d", i);
            for (int j = 0; j < size; j++) {
                if (isnan(matrix[i][j])) printf("%16s", "NaN");
                else printf("%16.9f", matrix[i][j]);
            }
            printf("\n");
        }

        printf("\nNotes:\n- Values are estimates in seconds.\n");
        printf("- Positive means column process's clock is ahead of row's clock.\n");
        printf("- Estimates chosen from %d rounds per pair by minimizing RTT.\n", rounds);

        for (int i = 0; i < size; i++) free(matrix[i]);
        free(matrix);
        free(gathered);
    }

    free(offsets_to);
    MPI_Finalize();
    return 0;
}

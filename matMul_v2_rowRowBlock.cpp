#include "helper.h"
#include "mpi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

void divide_AB(int rank, int nprcs, int M, int P, int N, int& Mr, int& Pr,
    float* A, float* B, float*& A_sub, float*& B_sub, float*& C_sub)
{
    Mr = M / nprcs, Pr = P / nprcs;

    A_sub = new float[Mr * P], B_sub = new float[Pr * N], C_sub = new float[Mr * N];

    for (int i = 0; i < Mr * N; i++)
        C_sub[i] = 0.0f;

    MPI_Scatter(A, Mr * P, MPI_FLOAT, A_sub, Mr * P, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatter(B, Pr * N, MPI_FLOAT, B_sub, Pr * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void schedule(int rank, int nprcs, int P, int N, int Pr, int Mr, float* A_sub, float* B_sub, float* C_sub)
{
    int dest = (rank - 1 + nprcs) % nprcs;
    int src = (rank + 1) % nprcs;
    MPI_Status status;

    for (int i = 0; i < nprcs; i++) {

        int l = (i + rank) % nprcs;
        matMul(Mr, Pr, N, A_sub + l * Pr, P, B_sub, N, C_sub, N);

        if (i < nprcs - 1) {
            MPI_Sendrecv_replace(B_sub, Pr * N, MPI_FLOAT, dest, 777, src, 777, MPI_COMM_WORLD, &status);
        }
    }
}

int main(int argc, char** argv)
{
    int rank, nprcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprcs);

    char file[128];
    int M, P, N, Mr, Pr;

    float *A, *B, *C, *C_ref;
    float *A_sub, *B_sub, *C_sub;

    A = B = C = C_ref = nullptr;
    A_sub = B_sub = C_sub = nullptr;

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: executable M P N file\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        strcpy(file, argv[1]);
        init_element(file, M, P, N, A, B, C, C_ref);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    divide_AB(rank, nprcs, M, P, N, Mr, Pr, A, B, A_sub, B_sub, C_sub);

    double elapsed_time, start, end;
    start = MPI_Wtime();

    schedule(rank, nprcs, P, N, Pr, Mr, A_sub, B_sub, C_sub);

    MPI_Gather(C_sub, Mr * N, MPI_FLOAT, C, Mr * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();
    elapsed_time = end - start;

    if (rank == 0) {
        if (check(M * N, C, C_ref, 1e-6)) {
            printf("The answer is right !!!\n");
            printf("Elapsed time is %f s, %lf GFLOPS\n", elapsed_time, M * P * N * 2.0 / elapsed_time / 1e9);
        } else {
            printf("The answer is wrong !!!\n");
        }
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
    }
    delete[] A_sub;
    delete[] B_sub;
    delete[] C_sub;

    MPI_Finalize();
    return 0;
}
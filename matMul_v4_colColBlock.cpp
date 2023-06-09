#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "helper.h"
#include "mpi.h"
using namespace std;

void divide_AB(int rank,
    int nprcs,
    int M,
    int P,
    int N,
    int& Pc,
    int& Nc,
    float* A,
    float* B,
    float*& A_sub,
    float*& B_sub,
    float*& C_sub)
{
    Pc = P / nprcs, Nc = N / nprcs;
    A_sub = new float[M * Pc], B_sub = new float[P * Nc],
    C_sub = new float[M * Nc];
    for (int i = 0; i < M * Nc; i++)
        C_sub[i] = 0.0f;

    MPI_Datatype colA_t, colB_t;
    MPI_Status status;

    MPI_Type_vector(M, Pc, P, MPI_FLOAT, &colA_t);
    MPI_Type_vector(P, Nc, N, MPI_FLOAT, &colB_t);

    MPI_Type_commit(&colA_t);
    MPI_Type_commit(&colB_t);

    if (rank == 0) {
        MPI_Request req1, req2;
        for (int i = 0; i < nprcs; i++) {
            MPI_Isend(A + i * Pc, 1, colA_t, i, i + 1, MPI_COMM_WORLD, &req1);
            MPI_Isend(B + i * Nc, 1, colB_t, i, i + 2, MPI_COMM_WORLD, &req2);
        }
    }

    MPI_Recv(A_sub, M * Pc, MPI_FLOAT, 0, rank + 1, MPI_COMM_WORLD, &status);
    MPI_Recv(B_sub, P * Nc, MPI_FLOAT, 0, rank + 2, MPI_COMM_WORLD, &status);

    MPI_Type_free(&colA_t);
    MPI_Type_free(&colB_t);
}

void schedule(int rank,
    int nprcs,
    int M,
    int P,
    int N,
    int Pc,
    int Nc,
    float* A_sub,
    float* B_sub,
    float* C_sub)
{
    int dest = (rank - 1 + nprcs) % nprcs;
    int src = (rank + 1) % nprcs;
    MPI_Status status;

    for (int i = 0; i < nprcs; i++) {

        int l = (rank + i) % nprcs;
        matMul(M, Pc, Nc, A_sub, Pc, B_sub + l * Pc * Nc, Nc, C_sub, Nc);

        if (i < nprcs - 1) {
            MPI_Sendrecv_replace(
                A_sub, M * Pc, MPI_FLOAT, dest, 777, src, 777, MPI_COMM_WORLD, &status);
        }
    }
}

void gatherResult(int rank,
    int nprcs,
    int M,
    int N,
    int Nc,
    float* C_sub,
    float* C)
{
    MPI_Datatype colBlock_t;
    MPI_Status status;

    MPI_Type_vector(M, Nc, N, MPI_FLOAT, &colBlock_t);
    MPI_Type_commit(&colBlock_t);

    MPI_Request req[nprcs];
    MPI_Status sta[nprcs];

    if (rank == 0) {
        for (int i = 0; i < nprcs; i++) {
            MPI_Irecv(C + i * Nc, 1, colBlock_t, i, i, MPI_COMM_WORLD, &req[i]);
        }
    }

    MPI_Send(C_sub, M * Nc, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);

    if (rank == 0)
        MPI_Waitall(nprcs, req, sta);

    MPI_Type_free(&colBlock_t);
}

int main(int argc, char** argv)
{
    int rank, nprcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprcs);

    char file[128];
    int M, P, N, Pc, Nc;

    float *A, *B, *C, *C_ref;
    float *A_sub, *B_sub, *C_sub;

    A = B = C = C_ref = nullptr;
    A_sub = B_sub = C_sub = nullptr;

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: executable file\n");
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

    divide_AB(rank,
        nprcs,
        M,
        P,
        N,
        Pc,
        Nc,
        A,
        B,
        A_sub,
        B_sub,
        C_sub);

    double elapsed_time, start, end;
    start = MPI_Wtime();

    schedule(rank, nprcs, M, P, N, Pc, Nc, A_sub, B_sub, C_sub);

    gatherResult(rank, nprcs, M, N, Nc, C_sub, C);

    end = MPI_Wtime();
    elapsed_time = end - start;

    if (rank == 0) {
        if (check(M * N, C, C_ref, 1e-6)) {
            printf("The answer is right !!!\n");
            printf("Elapsed time is %f s, %lf GFLOPS\n",
                elapsed_time,
                M * P * N * 2.0 / elapsed_time / 1e9);
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
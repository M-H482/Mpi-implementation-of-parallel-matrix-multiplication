#include "helper.h"
#include "mpi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

void divide_AB(int rank, int size, int M, int P, int N, int& Mr, int& Pr, int& Nc, float* A, float* B, float*& A_sub, float*& A_tmp, float*& B_sub, float*& C_sub)
{
    Mr = M / size, Pr = P / size, Nc = N / size;

    A_sub = new float[Mr * Pr];
    A_tmp = new float[Mr * Pr];
    B_sub = new float[Pr * Nc];
    C_sub = new float[Mr * Nc];

    for (int i = 0; i < Mr * Nc; i++)
        C_sub[i] = 0.0f;

    MPI_Datatype blockA_t, blockB_t;

    MPI_Type_vector(Mr, Pr, P, MPI_FLOAT, &blockA_t);
    MPI_Type_vector(Pr, Nc, N, MPI_FLOAT, &blockB_t);

    MPI_Type_commit(&blockA_t);
    MPI_Type_commit(&blockB_t);

    if (rank == 0) {
        MPI_Request req;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int dest = i * size + j;

                MPI_Isend(A + i * Mr * P + j * Pr, 1, blockA_t,
                    dest, dest, MPI_COMM_WORLD, &req);

                MPI_Isend(B + i * Pr * N + j * Nc, 1, blockB_t,
                    dest, dest + 77, MPI_COMM_WORLD, &req);
            }
        }
    }

    MPI_Status sta;

    MPI_Recv(A_sub, Mr * Pr, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &sta);
    MPI_Recv(B_sub, Pr * Nc, MPI_FLOAT, 0, rank + 77, MPI_COMM_WORLD, &sta);

    MPI_Type_free(&blockA_t);
    MPI_Type_free(&blockB_t);
}

void schedule(int rank, int size, int P, int N, int Mr, int Pr, int Nc, float* A_sub, float* A_tmp, float* B_sub, float* C_sub)
{
    int row = rank / size;
    int col = rank % size;

    int rm1 = (row - 1 + size) % size, rp1 = (row + 1) % size;
    int cm1 = (col - 1 + size) % size, cp1 = (col + 1) % size;

    MPI_Status sta;
    MPI_Request req;

    for (int i = 0; i < size; i++) {
        int l = (i + row) % size;

        if (col == l) { // row bcast
            MPI_Isend(A_sub, Mr * Pr, MPI_FLOAT,
                row * size + cp1, 777, MPI_COMM_WORLD, &req);

            matMul(Mr, Pr, Nc, A_sub, Pr, B_sub, Nc, C_sub, Nc);

            MPI_Wait(&req, &sta);
        } else {
            MPI_Recv(A_tmp, Mr * Pr, MPI_FLOAT,
                row * size + cm1, 777, MPI_COMM_WORLD, &sta);

            if (cp1 != l)
                MPI_Isend(A_tmp, Mr * Pr, MPI_FLOAT,
                    row * size + cp1, 777, MPI_COMM_WORLD, &req);

            matMul(Mr, Pr, Nc, A_tmp, Pr, B_sub, Nc, C_sub, Nc);

            if (cp1 != l)
                MPI_Wait(&req, &sta);
        }

        if (i < size - 1) {
            MPI_Sendrecv_replace(B_sub, Pr * Nc, MPI_FLOAT,
                rm1 * size + col, 777, rm1 * size + col, 777, MPI_COMM_WORLD, &sta);
        }
    }
}

void gatherResult(int rank, int size, int N, int Mr, int Nc, float* C_sub, float* C)
{
    MPI_Datatype blockC_t;
    MPI_Type_vector(Mr, Nc, N, MPI_FLOAT, &blockC_t);
    MPI_Type_commit(&blockC_t);

    MPI_Request req;
    MPI_Status sta;

    MPI_Isend(C_sub, Mr * Nc, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &req);

    if (rank == 0) {
        MPI_Status s;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int src = i * size + j;
                MPI_Recv(C + i * Mr * N + j * Nc, 1, blockC_t, src, src, MPI_COMM_WORLD, &s);
            }
        }
    }
    MPI_Wait(&req, &sta);

    MPI_Type_free(&blockC_t);
}

int main(int argc, char** argv)
{
    int rank, nprcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprcs);

    char file[128];
    int M, P, N, Mr, Pr, Nc;

    float *A, *B, *C, *C_ref;
    float *A_sub, *A_tmp, *B_sub, *C_sub;

    A = B = C = C_ref = nullptr;
    A_sub = A_tmp = B_sub = C_sub = nullptr;

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

    int size = sqrt(nprcs);

    divide_AB(rank, size, M, P, N, Mr, Pr, Nc, A, B, A_sub, A_tmp, B_sub, C_sub);

    double elapsed_time, start, end;
    start = MPI_Wtime();

    schedule(rank, size, P, N, Mr, Pr, Nc, A_sub, A_tmp, B_sub, C_sub);

    gatherResult(rank, size, N, Mr, Nc, C_sub, C);

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
    delete[] A_tmp;

    MPI_Finalize();
    return 0;
}
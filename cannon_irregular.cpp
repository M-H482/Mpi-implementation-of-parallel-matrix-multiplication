#include "helper.h"
#include "mpi.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

void creatComm(int rank, int p, MPI_Comm* row_comm, MPI_Comm* col_comm)
{
    int color, key;

    color = rank / p;
    key = rank % p;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, row_comm);

    color = rank % p;
    key = rank / p;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, col_comm);
}

void myScatter(int& x, int& y, float* work, int num_of_rows, int num_of_cols,
    MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rid, cid, np, cur_len, nex_len;
    MPI_Datatype block_t;
    MPI_Status sta;

    MPI_Comm_size(col_comm, &np); // np must be power of 2

    MPI_Comm_rank(row_comm, &cid);
    MPI_Comm_rank(col_comm, &rid);

    if (cid == 0) { // scatter rows in  col_comm 0
        cur_len = num_of_rows;

        for (int stride = np / 2; stride > 0; stride /= 2) {
            if (rid % stride == 0 && ((rid / stride) & 1) == 0) {
                int dest = rid + stride;
                nex_len = cur_len / 2;

                MPI_Send(&nex_len, 1, MPI_INT, dest, 777, col_comm);
                MPI_Send(work + (cur_len - nex_len) * num_of_cols, nex_len * num_of_cols, MPI_FLOAT,
                    dest, 888, col_comm);

                cur_len = cur_len - nex_len;
            }

            if (rid % stride == 0 && ((rid / stride) & 1) == 1) {
                int src = rid - stride;
                MPI_Recv(&cur_len, 1, MPI_INT, src, 777, col_comm, &sta);
                MPI_Recv(work, cur_len * num_of_cols, MPI_FLOAT, src, 888, col_comm, &sta);
            }
        }
        x = cur_len;
    }

    MPI_Bcast(&x, 1, MPI_INT, 0, row_comm);

    cur_len = num_of_cols;

    for (int stride = np / 2; stride > 0; stride /= 2) { // scatter blocks in row_comm
        if (cid % stride == 0 && ((cid / stride) & 1) == 0) {
            int dest = cid + stride;
            nex_len = cur_len / 2;
            MPI_Send(&nex_len, 1, MPI_INT, dest, 333, row_comm);

            MPI_Type_vector(x, nex_len, num_of_cols, MPI_FLOAT, &block_t);
            MPI_Type_commit(&block_t);

            MPI_Send(work + (cur_len - nex_len), 1, block_t, dest, 555, row_comm);
            cur_len = cur_len - nex_len;
        }

        if (cid % stride == 0 && ((cid / stride) & 1) == 1) {
            int src = cid - stride;
            MPI_Recv(&cur_len, 1, MPI_INT, src, 333, row_comm, &sta);

            MPI_Type_vector(x, cur_len, num_of_cols, MPI_FLOAT, &block_t);
            MPI_Type_commit(&block_t);

            MPI_Recv(work, 1, block_t, src, 555, row_comm, &sta);
        }
    }

    y = cur_len;

    MPI_Type_free(&block_t);
}

void myGather(int x, int y, float* work, int ld,
    MPI_Comm row_comm, MPI_Comm col_comm)
{
    int rid, cid, np, cur_len, nex_len;
    MPI_Datatype block_t;
    MPI_Status sta;

    MPI_Comm_size(col_comm, &np);

    MPI_Comm_rank(row_comm, &cid);
    MPI_Comm_rank(col_comm, &rid);

    cur_len = y;

    for (int stride = 1; stride < np; stride *= 2) { // gather blocks in row_comm
        if (cid % stride == 0 && ((cid / stride) & 1) == 0) {
            int src = cid + stride;

            MPI_Recv(&nex_len, 1, MPI_INT, src, 123, row_comm, &sta);

            MPI_Type_vector(x, nex_len, ld, MPI_FLOAT, &block_t);
            MPI_Type_commit(&block_t);

            MPI_Recv(work + cur_len, 1, block_t, src, 581, row_comm, &sta);

            cur_len = cur_len + nex_len;
        }

        if (cid % stride == 0 && ((cid / stride) & 1) == 1) {
            int dest = cid - stride;
            MPI_Send(&cur_len, 1, MPI_INT, dest, 123, row_comm);

            MPI_Type_vector(x, cur_len, ld, MPI_FLOAT, &block_t);
            MPI_Type_commit(&block_t);

            MPI_Send(work, 1, block_t, dest, 581, row_comm);
        }
    }
    MPI_Type_free(&block_t);

    if (cid == 0) {
        cur_len = x;

        for (int stride = 1; stride < np; stride *= 2) {
            if (rid % stride == 0 && ((rid / stride) & 1) == 0) {
                int src = rid + stride;

                MPI_Recv(&nex_len, 1, MPI_INT, src, 678, col_comm, &sta);
                MPI_Recv(work + cur_len * ld, nex_len * ld, MPI_FLOAT, src, 1314, col_comm, &sta);

                cur_len = cur_len + nex_len;
            }

            if (rid % stride == 0 && ((rid / stride) & 1) == 1) {
                int dest = rid - stride;
                MPI_Send(&cur_len, 1, MPI_INT, dest, 678, col_comm);
                MPI_Send(work, cur_len * ld, MPI_FLOAT, dest, 1314, col_comm);
            }
        }
    }
}

void schedule(int p, MPI_Comm row_comm, MPI_Comm col_comm,
    int Mr, int Pc, int Pr, int Nc,
    float* A, float* A_tmp, int lda, float* B, float* B_tmp, int ldb, float* C, int ldc)
{
    int rid, cid, l, k, prev_k;
    MPI_Comm_rank(row_comm, &cid);
    MPI_Comm_rank(col_comm, &rid);

    MPI_Status sta;
    MPI_Datatype block_a_t, block_b1_t, block_b2_t;

    int dest = (rid - 1 + p) % p;
    int src = (rid + 1) % p;

    float* B_ptr[2] = { B, B_tmp };
    float* A_ptr[2] = { A_tmp, A };

    for (int i = 0; i < p; i++) {

        l = (i + rid) % p;

        int isRoot = (cid == l);
        int cur = i & 1, pre = (i + 1) & 1;

        if (isRoot) {
            k = Pc;
        }

        MPI_Bcast(&k, 1, MPI_INT, l, row_comm);

        MPI_Type_vector(Mr, k, lda, MPI_FLOAT, &block_a_t);
        MPI_Type_commit(&block_a_t);

        MPI_Bcast(A_ptr[isRoot], 1, block_a_t, l, row_comm);

        if (i > 0) {
            MPI_Type_vector(prev_k, Nc, ldb, MPI_FLOAT, &block_b1_t);
            MPI_Type_vector(k, Nc, ldb, MPI_FLOAT, &block_b2_t);

            MPI_Type_commit(&block_b1_t);
            MPI_Type_commit(&block_b2_t);

            MPI_Sendrecv(B_ptr[pre], 1, block_b1_t, dest, 999,
                B_ptr[cur], 1, block_b2_t, src, 999, col_comm, &sta);
        }

        matMul(Mr, k, Nc, A_ptr[isRoot], lda, B_ptr[cur], ldb, C, ldc);

        prev_k = k;
    }
    MPI_Type_free(&block_a_t);
    MPI_Type_free(&block_b1_t);
    MPI_Type_free(&block_b2_t);
}

void print_matrix(float* a, int ld, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.0f ", a[i * ld + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    int rank, nprcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprcs);

    char file[128];
    int M, P, N, Mr, Pc, Pr, Nc;

    float *A, *A_tmp, *B, *B_tmp, *C, *C_ref;

    A = B = C = C_ref = nullptr;
    A_tmp = B_tmp = nullptr;

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

    A_tmp = new float[M * P];
    B_tmp = new float[P * N];

    if (rank != 0)
        A = new float[M * P], B = new float[P * N], C = new float[M * N];

    int p = sqrt(nprcs);
    // if (rank == 0) {
    //     printf("A:\n");
    //     print_matrix(A, P, M, P);

    //     printf("B:\n");
    //     print_matrix(B, N, P, N);

    //     printf("C_ref:\n");
    //     print_matrix(C_ref, N, M, N);
    // }

    MPI_Comm row_com, col_com;
    creatComm(rank, p, &row_com, &col_com);

    myScatter(Mr, Pc, A, M, P, row_com, col_com);
    myScatter(Pr, Nc, B, P, N, row_com, col_com);
    myScatter(Mr, Nc, C, M, N, row_com, col_com);

    double elapsed_time, start, end;
    start = MPI_Wtime();

    schedule(p, row_com, col_com, Mr, Pc, Pr, Nc, A, A_tmp, P, B, B_tmp, N, C, N);

    myGather(Mr, Nc, C, N, row_com, col_com);

    end = MPI_Wtime();
    elapsed_time = end - start;

    if (rank == 0) {
        if (check(M * N, C, C_ref, 1e-6)) {
            printf("The answer is right !!!\n");
            printf("Elapsed time is %f s, %lf GFLOPS\n", elapsed_time, M * P * N * 2.0 / elapsed_time / 1e9);
        } else {
            printf("The answer is wrong !!!\n");
        }
        delete[] C_ref;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A_tmp;
    delete[] B_tmp;

    MPI_Finalize();
    return 0;
}
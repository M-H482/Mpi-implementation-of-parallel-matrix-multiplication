#ifndef __HELPER_H__
#define __HELPER_H__

#include <cmath>
#include <iostream>

void matMul(int m, // row num of A
    int p, // col num of A
    int n, // col num of B
    float* A_sub,
    int lda, // leading dimension size of A
    float* B_sub,
    int ldb, // leading dimension size of B
    float* C_sub,
    int ldc) // leading dimension size of C
{
    float* pC = C_sub;

    for (int i = 0; i < m; i++) {
        float* pB = B_sub;

        for (int k = 0; k < p; k++) {
            float a = A_sub[i * lda + k];

            for (int j = 0; j < n; j++) {
                pC[j] += a * pB[j];
            }
            pB = pB + ldb;
        }
        pC = pC + ldc;
    }
}

void init_element(char* file, int& M, int& P, int& N, float*& A, float*& B, float*& C, float*& C_ref)
{

    FILE* fp = fopen(file, "r");
    int a, b, c, r;

    r = fscanf(fp, "%d %d %d\n", &a, &b, &c);
    M = a, P = b, N = c;

    printf("M = %d, P = %d, N = %d\n", M, P, N);
    A = new float[M * P], B = new float[P * N], C = new float[M * N];
    C_ref = new float[M * N];

    for (int i = 0; i < M * P; i++)
        r = fscanf(fp, "%f", &A[i]);

    for (int i = 0; i < P * N; i++)
        r = fscanf(fp, "%f", &B[i]);

    for (int i = 0; i < M * N; i++) {
        r = fscanf(fp, "%f", &C_ref[i]);
        C[i] = 0.0f;
    }

    fclose(fp);
}

bool check(int cnt, float* ans, float* ref, float threshold)
{
    int j = 0;
    for (int i = 0; i < cnt; i++) {
        if (fabsf(ans[i] - ref[i]) > threshold) {
            printf("Error on index %d, ans = %f and ref = %f\n", i, ans[i], ref[i]);
            if (++j == 3) {
                printf("...\n");
                break;
            }
        }
    }
    return j == 0;
}

#endif
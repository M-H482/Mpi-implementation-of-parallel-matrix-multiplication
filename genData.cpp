#include "helper.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char** argv)
{
    int M, P, N;
    char file[128];

    if (argc != 4) {
        printf("Usage: executable M P N\n");
        return 0;
    }

    M = atoi(argv[1]), P = atoi(argv[2]), N = atoi(argv[3]);

    sprintf(file, "./data/matrix_%d_%d_%d.txt", M, P, N);
    vector<float> A(M * P);
    vector<float> B(P * N);
    vector<float> C(M * N);

    srand(time(NULL));
    for (int i = 0; i < M * P; i++)
        A[i] = rand() % 10;
    for (int i = 0; i < P * N; i++)
        B[i] = rand() % 10;

    matMul(M, P, N, A.data(), P, B.data(), N, C.data(), N);

    FILE* fp = fopen(file, "w");

    fprintf(fp, "%d %d %d\n", M, P, N);

    for (int i = 0; i < M * P; i++)
        fprintf(fp, "%f\n", A[i]);
    for (int i = 0; i < P * N; i++)
        fprintf(fp, "%f\n", B[i]);
    for (int i = 0; i < M * N; i++)
        fprintf(fp, "%f\n", C[i]);

    fclose(fp);
    return 0;
}
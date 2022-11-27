#ifndef __MATRIX_H__
#include "matrix.h"
#endif

// clock() Time
#define TIME_START start = clock();
#define TIME_FINISH(NAME)                                 \
    finish = clock();                                     \
    duration = (double)(finish - start) / CLOCKS_PER_SEC; \
    printf("%s: The computing time = %fs\n", (NAME), duration);

// omp_get_wtime() Time
#define TIME_BEGIN begin = omp_get_wtime();
#define TIME_END(NAME)                \
    end = omp_get_wtime();            \
    duration = (double)(end - begin); \
    printf("%s: The computing time = %fs\n", (NAME), duration);

// The row and colomn of matrix
size_t TESTSIZE = 8000;

int main()
{
    // Initialization
    Matrix *matrixA = (Matrix *)malloc(sizeof(Matrix));
    Matrix *matrixB = (Matrix *)malloc(sizeof(Matrix));
    Matrix *matrixC = (Matrix *)malloc(sizeof(Matrix));
    createMatrix(TESTSIZE, TESTSIZE, matrixA);
    createMatrix(TESTSIZE, TESTSIZE, matrixB);
    createMatrix(TESTSIZE, TESTSIZE, matrixC);
    size_t TESTSIZESQUARE = TESTSIZE * TESTSIZE;
    srand((unsigned)time(NULL));
    float *matrixAData = (float *)malloc(sizeof(float) * TESTSIZESQUARE);
    float *matrixBData = (float *)malloc(sizeof(float) * TESTSIZESQUARE);
    randomData(TESTSIZESQUARE, matrixAData);
    randomData(TESTSIZESQUARE, matrixBData);
    valueMatrix(matrixA, matrixAData);
    valueMatrix(matrixB, matrixBData);
    // printf("matrixA:  \n");
    // printMatrix(matrixA);
    // printf("matrixB:  \n");
    // printMatrix(matrixB);
    // printf("matrixC:  \n");
    // printMatrix(matrixC);

    clock_t start, finish;
    double begin, end;
    double duration;

    // // Calculation
    // TIME_BEGIN
    // multiplyMatrix_plain(matrixA, matrixB, matrixC);
    // // printf("matrixC  matmul_plain():  \n");
    // // printMatrix(matrixC);
    // TIME_END("Plain")

    // TIME_BEGIN
    // multiplyMatrix_plain_OpenMP(matrixA, matrixB, matrixC);
    // // printf("matrixC matmul_plain_OpenMP:  \n");
    // // printMatrix(matrixC);
    // TIME_END("Plain_OpenMP")

    TIME_BEGIN
    multiplyMatrix_improved_SIMD(matrixA, matrixB, matrixC);
    // printf("matrixC  matmul_improved_SIMD:  \n");
    // printMatrix(matrixC);
    TIME_END("Improved_SIMD")

    TIME_BEGIN
    multiplyMatrix_improved_SIMD_OpenMP(matrixA, matrixB, matrixC);
    // printf("matrixC  matmul_improved_SIMD_OpenMP:  \n");
    // printMatrix(matrixC);
    TIME_END("Improved_SIMD_OpenMP")

    TIME_BEGIN
    multiplyMatrix_OpenBLAS(matrixA, matrixB, matrixC);
    // printf("matrixC  matmul_OpenBLAS:  \n");
    // printMatrix(matrixC);
    TIME_END("OpenBLAS")

    deleteMatrix(&matrixA);
    deleteMatrix(&matrixB);
    deleteMatrix(&matrixC);

    free(matrixAData);
    free(matrixBData);
    return 0;
}
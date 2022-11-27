#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#ifdef WITH_AVX2
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_OPENBLAS
#include <cblas.h>
#endif

typedef struct
{
    size_t row, col;
    float *matrixData;
} Matrix;

bool randomData(const size_t length, float *data);
size_t sizeMatrix(const Matrix *matrix);
bool printMatrix(const Matrix *matrix);
bool createMatrix(const size_t row, const size_t col, Matrix *matrix);
bool valueMatrix(Matrix *matrix, const float *valuedMatrixData);
bool transMatrix(const Matrix *matrix, Matrix *matrixTrans);
Matrix *leftSupplementZero(const Matrix *matrix);
Matrix *rightSupplementZero(const Matrix *matrix);
bool deleteMatrix(Matrix **matrix);

// 矩阵乘法
bool multiplyMatrix_plain(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool multiplyMatrix_plain_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool multiplyMatrix_improved_SIMD(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool multiplyMatrix_improved_SIMD_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool multiplyMatrix_OpenBLAS(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);

bool matmul_plain(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool matmul_plain_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool matmul_improved_SIMD(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool matmul_improved_SIMD_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);
bool matmul_OpenBLAS(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC);

#endif
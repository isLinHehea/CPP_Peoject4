#ifndef __MATRIX_H__
#include "matrix.h"
#endif

const int RANDOM_MIN = 0.0;
const int RANDOM_MAX = 10.0;

bool randomData(const size_t length, float *data)
{
    if (data == NULL)
    {
        fprintf(stderr, "Error: The data is NULL. (randomData)\n");
        return false;
    }
    else
    {
        for (size_t i = 0; i < length; i++)
        {
            float random = RANDOM_MIN + 1.0 * rand() / RAND_MAX * (RANDOM_MAX - RANDOM_MIN);
            data[i] = random;
        }
    }
    return true;
}

size_t sizeMatrix(const Matrix *matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: The matrix is NULL. (sizeMatrix)\n");
        return 1;
    }
    else
        return matrix->row * matrix->col;
}

bool printMatrix(const Matrix *matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: The matrix to be printed is NULL. (printMatrix)\n");
        return false;
    }
    else if (matrix->matrixData == NULL)
    {
        fprintf(stderr, "Error: The data of matrix to be printed is NULL. (printMatrix)\n");
        return false;
    }
    else
    {
        printf("Matrix\n");
        if (matrix->row == 1 && matrix->col == 1)
        {
            printf("【 ");
            printf("%-12f  ", matrix->matrixData[0]);
            printf("】");
        }
        else if (matrix->row == 1)
        {
            printf("【 ");
            for (int i = 0; i < sizeMatrix(matrix); i++)
                printf("%-12f  ", matrix->matrixData[i]);
            printf("】");
        }
        else
        {
            for (size_t i = 0; i < sizeMatrix(matrix); i++)
            {
                if (i == 0)
                    printf("┏ ");
                else if (i == sizeMatrix(matrix) - matrix->col)
                    printf("┗ ");
                else if (i % matrix->col == 0)
                    printf("┃ ");

                printf("%-12f  ", matrix->matrixData[i]);
                if (i == matrix->col - 1)
                    printf(" ┓");
                else if (i == sizeMatrix(matrix) - 1)
                    printf(" ┛");
                else if ((i + 1) % matrix->col == 0)
                    printf(" ┃");
                if ((i + 1) % matrix->col == 0)
                    printf("\n");
            }
        }
        return true;
    }
}

bool createMatrix(const size_t row, const size_t col, Matrix *matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "The pointer of the matrix to be created is NULL. (createMatrix)\n");
        return false;
    }
    else if (row <= 0 || col <= 0)
    {
        fprintf(stderr, "Error: The row and col of the initialized matrix should be positive. (createMatrix)\n");
        return false;
    }
    else
    {
        matrix->row = row;
        matrix->col = col;
        matrix->matrixData = (float *)malloc(sizeof(float) * row * col);
        // 初始化为0
        memset(matrix->matrixData, .0, sizeof(float) * row * col);
        return true;
    }
}

bool valueMatrix(Matrix *matrix, const float *valuedMatrixData)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: The matrix to be valued is NULL. (valueMatrix)\n");
        return false;
    }
    else if (matrix->matrixData == NULL)
    {
        fprintf(stderr, "Error: The data of matrix to be valued is NULL. (valueMatrix)\n");
        return false;
    }
    else if (valuedMatrixData == NULL)
    {
        fprintf(stderr, "Error: The data to be assigned is NULL. (valueMatrix)\n");
        return false;
    }
    else
        memcpy(matrix->matrixData, valuedMatrixData, sizeof(float) * matrix->row * matrix->col);
    return true;
}

bool transMatrix(const Matrix *matrix, Matrix *matrixTrans)
{
    if (matrixTrans == NULL)
    {
        fprintf(stderr, "Error: The transposed matrix is NULL. (transMatrix)\n");
        return false;
    }
    else if (matrix == NULL)
    {
        fprintf(stderr, "Error: The matrix to be transposed is NULL. (transMatrix)\n");
        return false;
    }
    else if (matrix->matrixData == NULL)
    {
        fprintf(stderr, "Error: The data of matrix to be transposed is NULL. (transMatrix)\n");
        return false;
    }
    else
    {
        matrixTrans->col = matrix->row;
        matrixTrans->row = matrix->col;
        for (size_t i = 0; i < matrix->row; i++)
        {
            size_t matrixIndexMinus = i * matrix->col;
            for (size_t j = 0; j < matrix->col; j++)
                matrixTrans->matrixData[j * matrixTrans->col + i] = matrix->matrixData[matrixIndexMinus + j];
        }
        return true;
    }
}

Matrix *leftSupplementZero(const Matrix *matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "The pointer of the matrix to be supplemented is NULL. (createMatrix)\n");
        return NULL;
    }
    else
    {
        size_t remainder = matrix->col % 8;
        Matrix *matrix_sup = (Matrix *)malloc(sizeof(Matrix));
        createMatrix(matrix->row, matrix->col - remainder + 8, matrix_sup);
        for (size_t i = 0; i < matrix_sup->row; i++)
        {
            size_t matrixIndexMinus = i * matrix->col;
            size_t matrix_supIndexMinus = i * matrix_sup->col;
            for (size_t j = 0; j < matrix_sup->col; j++)
            {
                if (j < matrix->col)
                    matrix_sup->matrixData[matrix_supIndexMinus + j] = matrix->matrixData[matrixIndexMinus + j];
                else
                    matrix_sup->matrixData[matrix_supIndexMinus + j] = 0.0;
            }
        }
        return matrix_sup;
    }
}
Matrix *rightSupplementZero(const Matrix *matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "The pointer of the matrix to be supplemented is NULL. (createMatrix)\n");
        return NULL;
    }
    else
    {
        size_t remainder = matrix->row % 8;
        Matrix *matrix_sup = (Matrix *)malloc(sizeof(Matrix));
        createMatrix(matrix->row - remainder + 8, matrix->col, matrix_sup);
        memcpy(matrix_sup->matrixData, matrix->matrixData, sizeof(float) * matrix->row * matrix->col);
        memset(matrix_sup->matrixData + sizeof(float) * matrix->row * matrix->col, .0, sizeof(float) * remainder * matrix->col);
        // for (size_t i = 0; i < matrix_sup->row; i++)
        // {
        //     for (size_t j = 0; j < matrix_sup->col; j++)
        //     {
        //         if (j < matrix->row)
        //             matrix_sup->matrixData[i * matrix_sup->col + j] = matrix->matrixData[i * matrix->col + j];
        //         else
        //             matrix_sup->matrixData[i * matrix_sup->col + j] = 0.0;
        //     }
        // }
        return matrix_sup;
    }
}

bool deleteMatrix(Matrix **matrix)
{
    if (matrix == NULL)
    {
        fprintf(stderr, "Error:The pointer of Matrix pointer matrix is NULL.(deleteMatrix)\n");
        return false;
    }
    else if ((*matrix) == NULL)
    {
        fprintf(stderr, "Error: The Matrix is already NULL.(deleteMatrix)\n");
        return false;
    }
    else
    {
        free((*matrix)->matrixData);
        (*matrix)->matrixData = NULL;
        free(*matrix);
        *matrix = NULL;
        printf("Matrix memory released successfully!\n");
        return true;
    }
}

bool multiplyMatrix_plain(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    if (matrixC == NULL)
    {
        printf("Error: The multiplied matrix is NULL. (multiplyMatrix)\n");
        return false;
    }
    else if (matrixA == NULL || matrixB == NULL)
    {
        fprintf(stderr, "Error: The matrix to be multiplied is NULL. (multiplyMatrix)\n");
        return false;
    }
    else
    {
        if (matrixA->matrixData == NULL || matrixB->matrixData == NULL)
        {
            fprintf(stderr, "Error: The data of matrix to be multiplied is NULL. (multiplyMatrix)\n");
            return false;
        }
        else
        {
            if (matrixA->col != matrixB->row)
            {
                fprintf(stderr, "Error: The sizes of two matrices to be multiplied do not match. (multiplyMatrix)\n");
                return false;
            }
            else if (matrixC->row != matrixA->row || matrixC->col != matrixB->col)
            {
                fprintf(stderr, "Error: The size of multiplied matrices does not match. (multiplyMatrix)\n");
                return false;
            }
            else
            {
                matmul_plain(matrixA, matrixB, matrixC);
                return true;
            }
        }
    }
}
bool multiplyMatrix_plain_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    if (matrixC == NULL)
    {
        printf("Error: The multiplied matrix is NULL. (multiplyMatrix)\n");
        return false;
    }
    else if (matrixA == NULL || matrixB == NULL)
    {
        fprintf(stderr, "Error: The matrix to be multiplied is NULL. (multiplyMatrix)\n");
        return false;
    }
    else
    {
        if (matrixA->matrixData == NULL || matrixB->matrixData == NULL)
        {
            fprintf(stderr, "Error: The data of matrix to be multiplied is NULL. (multiplyMatrix)\n");
            return false;
        }
        else
        {
            if (matrixA->col != matrixB->row)
            {
                fprintf(stderr, "Error: The sizes of two matrices to be multiplied do not match. (multiplyMatrix)\n");
                return false;
            }
            else if (matrixC->row != matrixA->row || matrixC->col != matrixB->col)
            {
                fprintf(stderr, "Error: The size of multiplied matrices does not match. (multiplyMatrix)\n");
                return false;
            }
            else
            {
                matmul_plain_OpenMP(matrixA, matrixB, matrixC);
                return true;
            }
        }
    }
}
bool multiplyMatrix_improved_SIMD(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    if (matrixC == NULL)
    {
        printf("Error: The multiplied matrix is NULL. (multiplyMatrix)\n");
        return false;
    }
    else if (matrixA == NULL || matrixB == NULL)
    {
        fprintf(stderr, "Error: The matrix to be multiplied is NULL. (multiplyMatrix)\n");
        return false;
    }
    else
    {
        if (matrixA->matrixData == NULL || matrixB->matrixData == NULL)
        {
            fprintf(stderr, "Error: The data of matrix to be multiplied is NULL. (multiplyMatrix)\n");
            return false;
        }
        else
        {
            if (matrixA->col != matrixB->row)
            {
                fprintf(stderr, "Error: The sizes of two matrices to be multiplied do not match. (multiplyMatrix)\n");
                return false;
            }
            else if (matrixC->row != matrixA->row || matrixC->col != matrixB->col)
            {
                fprintf(stderr, "Error: The size of multiplied matrices does not match. (multiplyMatrix)\n");
                return false;
            }
            else
            {
                matmul_improved_SIMD(matrixA, matrixB, matrixC);
                return true;
            }
        }
    }
}
bool multiplyMatrix_improved_SIMD_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    if (matrixC == NULL)
    {
        printf("Error: The multiplied matrix is NULL. (multiplyMatrix)\n");
        return false;
    }
    else if (matrixA == NULL || matrixB == NULL)
    {
        fprintf(stderr, "Error: The matrix to be multiplied is NULL. (multiplyMatrix)\n");
        return false;
    }
    else
    {
        if (matrixA->matrixData == NULL || matrixB->matrixData == NULL)
        {
            fprintf(stderr, "Error: The data of matrix to be multiplied is NULL. (multiplyMatrix)\n");
            return false;
        }
        else
        {
            if (matrixA->col != matrixB->row)
            {
                fprintf(stderr, "Error: The sizes of two matrices to be multiplied do not match. (multiplyMatrix)\n");
                return false;
            }
            else if (matrixC->row != matrixA->row || matrixC->col != matrixB->col)
            {
                fprintf(stderr, "Error: The size of multiplied matrices does not match. (multiplyMatrix)\n");
                return false;
            }
            else
            {
                matmul_improved_SIMD_OpenMP(matrixA, matrixB, matrixC);
                return true;
            }
        }
    }
}
bool multiplyMatrix_OpenBLAS(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    if (matrixC == NULL)
    {
        printf("Error: The multiplied matrix is NULL. (multiplyMatrix)\n");
        return false;
    }
    else if (matrixA == NULL || matrixB == NULL)
    {
        fprintf(stderr, "Error: The matrix to be multiplied is NULL. (multiplyMatrix)\n");
        return false;
    }
    else
    {
        if (matrixA->matrixData == NULL || matrixB->matrixData == NULL)
        {
            fprintf(stderr, "Error: The data of matrix to be multiplied is NULL. (multiplyMatrix)\n");
            return false;
        }
        else
        {
            if (matrixA->col != matrixB->row)
            {
                fprintf(stderr, "Error: The sizes of two matrices to be multiplied do not match. (multiplyMatrix)\n");
                return false;
            }
            else if (matrixC->row != matrixA->row || matrixC->col != matrixB->col)
            {
                fprintf(stderr, "Error: The size of multiplied matrices does not match. (multiplyMatrix)\n");
                return false;
            }
            else
            {
                matmul_OpenBLAS(matrixA, matrixB, matrixC);
                return true;
            }
        }
    }
}

bool matmul_plain(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    for (size_t i = 0; i < matrixA->row; i++)
    {
        for (size_t j = 0; j < matrixB->col; j++)
        {
            size_t matrixCIndex = i * matrixC->col + j;
            size_t matrixAIndexMinus = i * matrixA->col;
            float ans = 0.0;
            for (size_t k = 0; k < matrixA->col; k++)
                ans += matrixA->matrixData[matrixAIndexMinus + k] * matrixB->matrixData[k * matrixB->col + j];
            matrixC->matrixData[matrixCIndex] = ans;
        }
    }
    return true;
}
bool matmul_plain_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
    omp_set_num_threads(8);
#pragma omp parallel for
    for (size_t i = 0; i < matrixA->row; i++)
    {
        for (size_t j = 0; j < matrixB->col; j++)
        {
            size_t matrixCIndex = i * matrixC->col + j;
            size_t matrixAIndexMinus = i * matrixA->col;
            float ans = 0.0;
            for (size_t k = 0; k < matrixA->col; k++)
                ans += matrixA->matrixData[matrixAIndexMinus + k] * matrixB->matrixData[k * matrixB->col + j];
            matrixC->matrixData[matrixCIndex] = ans;
        }
    }
    return true;
}
bool matmul_improved_SIMD(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
#ifdef WITH_AVX2
    if (matrixA->col % 8 != 0)
    {
        matrixA = leftSupplementZero(matrixA);
        matrixB = rightSupplementZero(matrixB);
    }
    Matrix *matrixB_trans = (Matrix *)malloc(sizeof(Matrix));
    createMatrix(matrixB->col, matrixB->row, matrixB_trans);
    transMatrix(matrixB, matrixB_trans);
    __m256 a, b;
    float sum[8] = {0.0};
    for (size_t i = 0; i < matrixA->row; i++)
    {
        size_t matrixAIndexMinus = i * matrixA->col;
        size_t matrixCIndexMinus = i * matrixC->col;
        for (size_t j = 0; j < matrixB_trans->row; j++)
        {
            float ans = 0.0;
            __m256 c = _mm256_setzero_ps();
            size_t matrixB_transIndexMinus = j * matrixB_trans->col;
            for (size_t k = 0; k < matrixA->col; k += 8)
            {
                a = _mm256_loadu_ps(matrixA->matrixData + matrixAIndexMinus + k);
                b = _mm256_loadu_ps(matrixB_trans->matrixData + matrixB_transIndexMinus + k);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(sum, c);
            for (size_t i = 0; i < 8; i++)
                ans += sum[i];
            matrixC->matrixData[matrixCIndexMinus + j] = ans;
        }
    }
    return true;
#else
    printf("AVX2 is not supported");
    return false;
#endif
}
bool matmul_improved_SIMD_OpenMP(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
#ifdef WITH_AVX2
    if (matrixA->col % 8 != 0)
    {
        matrixA = leftSupplementZero(matrixA);
        matrixB = rightSupplementZero(matrixB);
    }
    Matrix *matrixB_trans = (Matrix *)malloc(sizeof(Matrix));
    createMatrix(matrixB->col, matrixB->row, matrixB_trans);
    transMatrix(matrixB, matrixB_trans);
    __m256 a, b;
    float sum[8] = {0.0};
    omp_set_num_threads(8);
#pragma omp parallel for
    for (size_t i = 0; i < matrixA->row; i++)
    {
        size_t matrixAIndexMinus = i * matrixA->col;
        size_t matrixCIndexMinus = i * matrixC->col;
        for (size_t j = 0; j < matrixB_trans->row; j++)
        {
            float ans = 0.0;
            __m256 c = _mm256_setzero_ps();
            size_t matrixB_transIndexMinus = j * matrixB_trans->col;
            for (size_t k = 0; k < matrixA->col; k += 8)
            {
                a = _mm256_loadu_ps(matrixA->matrixData + matrixAIndexMinus + k);
                b = _mm256_loadu_ps(matrixB_trans->matrixData + matrixB_transIndexMinus + k);
                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(sum, c);
            for (size_t i = 0; i < 8; i++)
                ans += sum[i];
            matrixC->matrixData[matrixCIndexMinus + j] = ans;
        }
    }
    return true;
#else
    printf("AVX2 is not supported");
    return false;
#endif
}
bool matmul_OpenBLAS(const Matrix *matrixA, const Matrix *matrixB, Matrix *matrixC)
{
#ifdef WITH_OPENBLAS
    const size_t M = matrixA->row;
    const size_t N = matrixB->col;
    const size_t K = matrixB->row;
    const double alpha = 1.0;
    const double beta = 0.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, matrixA->matrixData, K, matrixB->matrixData, N, beta, matrixC->matrixData, N);
    return true;
#else
    printf("OPENBLAS is not supported");
    return false;
#endif
}

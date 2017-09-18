// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

#include <iostream>
#include <stdio.h>
#include "array_utils.h"
using namespace std;


typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32

// Forward declaration of the matrix multiplication kernel
// __global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

__global__
void print_arr(float * x, int N){
    for (int i=0; i<N; ++i){
        printf("%f ",x[i]);
    }
    printf("\n");

}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row>=C.height||col>=C.width)
        return;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width+dimBlock.x-1) / dimBlock.x, (A.height+dimBlock.y-1) / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);


    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}



int main(){
    Matrix A, B, C;
    A.height=40; A.width=100;
    size_t N_A=A.width * A.height;
    size_t sizeA = A.width * A.height * sizeof(float);
    A.elements=(float *) malloc(sizeA);

    B.height=A.width; B.width=50;
    size_t N_B=B.width * B.height;
    size_t sizeB = B.width * B.height * sizeof(float);
    B.elements=(float *) malloc(sizeB);

    C.height=A.height; C.width=B.width;
    size_t N_C=C.width * C.height;
    // size_t sizeC = C.width * C.height * sizeof(float);
    // C.elements=(float *) malloc(sizeC);
    C.elements = new float[N_C]();

    //initialize
    for(int i=0;i<N_A;++i){
        A.elements[i]=i;
    }
    for(int i=0;i<N_B;++i){
        B.elements[i]=1.0f;
    }

    JL::print_array1d<float *>(A.elements,N_A);
    JL::print_array1d<float *>(B.elements,N_B);
    JL::print_array1d<float *>(C.elements,N_C);

    MatMul(A, B, C);

    JL::print_array1d<float *>(C.elements,N_C);

    free(A.elements);free(B.elements);
    delete[] C.elements;

}
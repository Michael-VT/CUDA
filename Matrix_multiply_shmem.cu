// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)

#include <iostream>
#include <stdio.h>
#include "array_utils.h"
using namespace std;


__global__
void print_arr(float * x, int N){
    for (int i=0; i<N; ++i){
        printf("%f ",x[i]);
    }
    printf("\n");

}

// Thread block size
#define BLOCK_SIZE 32 

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;  
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}



// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
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

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;  
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results

    for (int m = 0; m < ((A.width+BLOCK_SIZE-1) / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        int col_end=(m+1)*BLOCK_SIZE>A.width ? A.width-m*BLOCK_SIZE : BLOCK_SIZE;


        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Limit the index from going out of scope
        if(col<col_end)
            As[row][col] = GetElement(Asub, row, col);
        if(row<col_end)
            Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Limit the index from going out of scope
        if(blockRow*BLOCK_SIZE+row<C.height&&blockCol*BLOCK_SIZE+col<C.width){
                    // Multiply Asub and Bsub together
            for (int e = 0; e < col_end; ++e){
                Cvalue += As[row][e] * Bs[e][col];
            }
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();

    }

    // Write Csub to device memory
    // Each thread writes one element
    // Limit the index from going out of scope
    if(blockRow*BLOCK_SIZE+row<C.height&&blockCol*BLOCK_SIZE+col<C.width)
        SetElement(Csub, row, col, Cvalue);
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
/*
1d stencil, compute local sum within RADIUS
create shared memory for each block for faster access
*/
#include <stdio.h>

// #define N_in 86
// #define RADIUS 3
// #define N_out (N_in - 2*RADIUS)
// #define Nblock 8
// #define Nthread 10

// Constant values are prefered over Macro, 
// Something unexpected would happen when using Macro
const int N_in=86, RADIUS=3, N_out=N_in-2*RADIUS;
const int Nblock=8, Nthread=10;

void ones_ints(int* a, int N){
    for (int i=0;i<N; ++i){
        a[i]=1;
    }
}

__global__ void stencil_1d(int *in, int *out){
    // Copy all the data used by on block into the shared memory
    // Shared memory is much faster than global memory on the Device
    __shared__ int temp[Nthread+2*RADIUS];
    int gidx=threadIdx.x + blockIdx.x * blockDim.x + RADIUS; // glocal memory index
    int lidx=threadIdx.x + RADIUS; // local memory index

    // printf("%d %d\n", gidx, lidx);
    temp[lidx]=in[gidx];
    if (threadIdx.x<RADIUS){
        temp[lidx - RADIUS]=in[gidx - RADIUS];
        temp[lidx + Nthread] = in[gidx + Nthread];
    }

// Synchronize all the threads to make sure the data has been transfered before using
    __syncthreads();

// Apply the stencil 
    int result=0;
    for(int offset=-RADIUS;offset<=RADIUS;++offset){
        result += temp[lidx+offset];
    }
    out[gidx - RADIUS]=result;

    // if(threadIdx.x==0)
    //     printf("hello\n");
}

int main(){
    int * h_in, *h_out;
    int* d_in, *d_out;
    int in_size=N_in*sizeof(int);
    int out_size=N_out*sizeof(int);

    h_in=(int *)malloc(in_size);
    ones_ints(h_in,N_in);
    h_out=(int *)malloc(out_size);

// Print input
    for(int i=0;i<N_in;++i){
        printf("%d ",h_in[i]);
    }
    printf("\n");

    cudaMalloc((void **) &d_in, in_size);
    cudaMalloc((void **) &d_out, out_size);

    cudaMemcpy(d_in,h_in,in_size,cudaMemcpyHostToDevice);

    stencil_1d<<<Nblock,Nthread>>>(d_in,d_out);

    cudaMemcpy(h_out,d_out,out_size,cudaMemcpyDeviceToHost);

// Print output
    for(int i=0;i<N_out;++i){
        printf("%d ",h_out[i]);
    }
    printf("\n");

    free(h_in);free(h_out);
    cudaFree(d_in);cudaFree(d_out);

}
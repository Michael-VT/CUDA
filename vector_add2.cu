/*
Two vector add together, a+b=c
get familar with the thread and block indexing
Here only use several blocks and several threads
*/
#include <stdio.h>

void random_ints(int* a, int N){
    for (int i=0;i<N; ++i){
        a[i]=rand();
    }
}

void ones_ints(int* a, int N){
    for (int i=0;i<N; ++i){
        a[i]=1;
    }
}

// Kernal, call on Host, run on Device
__global__ 
void vectoradd2(int* a, int *b, int *c, int N){
    int idx=threadIdx.x + blockIdx.x*blockDim.x;
    if(idx<N)
    c[idx]=a[idx]+b[idx];
}



int main(){
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    // None of the three variables below has to be constant
    const int N=45;
    // const int Nblock=1;
    const int Nthread=20;

    int size=N*sizeof(int);

// Allocate space for host copy of variables, and initialize
    h_a=(int *)malloc(size);
    ones_ints(h_a,N);
    h_b=(int *)malloc(size);
    ones_ints(h_b,N);
    h_c=(int *)malloc(size);

// Allocate space for device copy
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    for(int i=0;i<N;++i){
        printf("%d ",h_a[i]);
    }
    printf("\n");
    for(int i=0;i<N;++i){
        printf("%d ",h_b[i]);
    }
    printf("\n");

// Copy from Host to Device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

// run kernal
    vectoradd2<<<(N+Nthread-1)/Nthread,Nthread>>>(d_a,d_b,d_c,N);

// Copy from Device to Host
    cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;++i){
        printf("%d ",h_c[i]);
    }
    printf("\n");

// free host memory
    free(h_a); free(h_b); free(h_c);
// free device memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);



}
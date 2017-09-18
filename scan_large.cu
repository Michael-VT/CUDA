#include <iostream>
using namespace std;

__host__ __device__
void swap(int & a, int &b){
    a=a^b;
    b=a^b;
    a=b^a;
}

// Scan, limited to 1 block, upto 1024 threads; 
__global__ 
void scan(unsigned int *g_data, unsigned int * g_intermediate, int n, int flag)  {
    // flag =0 inclusive; flag =1 Exclusive
    extern __shared__ unsigned int temp[]; // allocated on invocation  
    int gid=threadIdx.x + blockIdx.x*blockDim.x;
    int Ndim=blockDim.x;
    int thid = threadIdx.x;
    int ln= (blockDim.x*(blockIdx.x+1)>n)? (n - blockDim.x*blockIdx.x) : blockDim.x;
    int pin=0,pout=1;
    unsigned int data_end;

    // if(threadIdx.x==0) printf("in Scan %d %d\n", d_bins[0],d_bins[1]);

    // Load input into shared memory.  
     // This is exclusive scan, so shift right by one  
     // and set first element to 0  
    if (gid>=n)
        return;     
        // temp[thid+pin*Ndim]=g_idata[gid]; // Inclusive
    if (flag ==0){
        temp[thid+pout*Ndim]=g_data[gid]; // Inclusive
    }else{
        temp[thid+pout*Ndim] = (thid > 0) ? g_data[gid-1] : 0;   // Exclusive
        data_end=g_data[gid];
    }

    // if(threadIdx.x==0) printf("in Scan %d %d\n", d_bins[0],d_bins[1]);

    __syncthreads();  
    // printf("%d\n",thid);
    for (int offset = 1; offset < ln; offset *= 2)  
    { 

      swap(pin,pout);
      if (thid >= offset)  
        temp[pout*Ndim+thid] = temp[pin*Ndim+thid]+ temp[pin*Ndim+thid - offset];  
      else  
        temp[pout*Ndim+thid] = temp[pin*Ndim+thid];  

    __syncthreads();  
    }  

    g_data[gid] = temp[pout*Ndim+thid]; // write output 

    if(thid==ln-1){
        if(flag == 0){
            g_intermediate[blockIdx.x]=temp[pout*Ndim+thid];
        } else{
            g_intermediate[blockIdx.x]=temp[pout*Ndim+thid]+data_end; // Exclusive
        }
    }

 }

 __global__
 void scan_extra(unsigned int *g_io, unsigned int * g_intermediate, int n){
    int gid=threadIdx.x + blockIdx.x*blockDim.x;
    int interid=blockIdx.x;
    // int thid = threadIdx.x;
    if(gid<n)
        g_io[gid] +=g_intermediate[interid];
 }

 void scan_large(unsigned int * d_in,const int N){
    unsigned int * d_intermediate;
    int Nthread=1024;
    int Nblock=(N+Nthread-1)/Nthread;
    int Nblock_s=Nblock;
    int flag =1; // 0 inclusive; flag =1 Exclusive

    // h_intermediate=(unsigned int *) malloc(Nblock*sizeof(unsigned int));
    cudaMalloc(&d_intermediate,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_in,d_intermediate,N, flag);

    Nthread=Nblock;
    Nblock=1;
    flag =1; // 0 inclusive; flag =1 Exclusive
    unsigned int * d_junk;
    cudaMalloc(&d_junk,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_intermediate,d_junk,Nthread,flag);

    Nthread=1024;
    Nblock=(N+Nthread-1)/Nthread;
    scan_extra<<<Nblock,Nthread>>>(d_in,d_intermediate,N);

    cudaFree(d_intermediate); cudaFree(d_junk);
 }



 int main(){
    const int N=10000;
    unsigned int sizeN=N*sizeof(unsigned int);
    unsigned int *h_input=new unsigned int[N];
    unsigned int *h_cdf=new unsigned int[N]();

    for(int i=0;i<N;++i){
        h_input[i]=1;
    }

    unsigned int * d_input, *d_cdf;
    cudaMalloc(&d_input, sizeN);
    cudaMalloc(&d_cdf, sizeN);

    cudaMemcpy(d_input,h_input,sizeN,cudaMemcpyHostToDevice);

    cudaMemcpy(d_cdf,d_input,sizeN,cudaMemcpyDeviceToDevice);

    // scan_small(d_cdf,d_input,N);
    scan_large(d_cdf, N);

    cudaMemcpy(h_cdf,d_cdf,sizeN,cudaMemcpyDeviceToHost);

    unsigned int acc=0;
    for(int i=0;i<N;++i){
        printf("%u ", acc);
        acc += h_input[i];
    }
    printf("\n");

    for(int i=0;i<N;++i){
        printf("%u ", h_cdf[i]);
    }

    cudaFree(d_input); cudaFree(d_cdf);
    delete[] h_input; delete[] h_cdf;

 }
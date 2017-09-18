#include <iostream>
using namespace std;

// Scan, limited to 1 block, upto 1024 threads; 
__global__ 
void scan(unsigned int *g_odata, unsigned int *g_idata, int n)  {
  extern __shared__ unsigned int temp[]; // allocated on invocation  
  int thid = threadIdx.x;  
  int pout = 0, pin = 1;  
  int Ndim=n;
  // Load input into shared memory.  
   // This is exclusive scan, so shift right by one  
   // and set first element to 0  

  if(thid>=n)
    return;

  temp[pout*Ndim + thid] = (thid > 0) ? g_idata[thid-1] : 0;  // Exclusive scan
  // temp[pout*n + thid]=g_idata[thid]; // Inclusive

  __syncthreads();  
  for (int offset = 1; offset < n; offset *= 2)  
  {  
    pout = 1 - pout; // swap double buffer indices  
    pin = 1 - pout;  
    if (thid >= offset) 
      temp[pout*Ndim+thid] = temp[pin*Ndim+thid] + temp[pin*Ndim+thid - offset];  // Code on CUDA tutorial page is Wrong!
    else 
      temp[pout*Ndim+thid] = temp[pin*Ndim+thid];  

    __syncthreads();  
  }  
  g_odata[thid] = temp[pout*Ndim+thid]; // write output  
 }

 void scan_small(unsigned int * d_cdf, unsigned int * d_input, int N){
  int Nblock=1;
  int Nthread=N;
  unsigned int sizeN=N*sizeof(unsigned int);
  scan<<<Nblock,Nthread,2*sizeN>>>(d_cdf,d_input,N);
 }



 int main(){
    const int N=100;
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

    scan_small(d_cdf,d_input,N);

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
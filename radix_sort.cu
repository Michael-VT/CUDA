#include <stdio.h>
#include <iostream>
#include "array_utils.h"

__host__ __device__
void swap(int & a, int &b){
    a=a^b;
    b=a^b;
    a=b^a;
}

// __device__ unsigned int d_bins[2];

__global__
void print_bins(unsigned int* d_bins, int i){
    printf("Device: %d %d total %d where: %d\n",d_bins[0],d_bins[1], d_bins[0]+d_bins[1],i);
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
    int flag =1; //inclusive; flag =1 Exclusive

    // h_intermediate=(unsigned int *) malloc(Nblock*sizeof(unsigned int));
    cudaMalloc(&d_intermediate,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_in,d_intermediate,N, flag);

    Nthread=Nblock;
    Nblock=1;
    flag =1; //inclusive; flag =1 Exclusive
    unsigned int * d_junk;
    cudaMalloc(&d_junk,Nblock*sizeof(unsigned int));

    scan<<<Nblock,Nthread,2*Nthread*sizeof(unsigned int)>>>(d_intermediate,d_junk,Nthread,flag);

    Nthread=1024;
    Nblock=(N+Nthread-1)/Nthread;
    scan_extra<<<Nblock,Nthread>>>(d_in,d_intermediate,N);

    cudaFree(d_intermediate); cudaFree(d_junk);
 }


__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned const int*  d_input, 
                      const int size) {  
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= size)
        return;

    // reset_hist(d_bins);
    unsigned int one = 1;
    int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    if(bin) 
         atomicAdd(&d_bins[1], 1);
    else
         atomicAdd(&d_bins[0], 1);
}

__global__
void digit_identify(unsigned const int * d_input, 
                unsigned int * d_out,const int N, const int pass, int flag=0){
    // flag == 0 for 0, or, for 1
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid < N){
        unsigned int one = 1;
        unsigned int label = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;

        d_out[gid]=label;
    }

}

// __global__
// void move(unsigned int *d_output,unsigned const int *d_input,unsigned const int *d_digitloc,unsigned const int *d_bins,
//     const int N, const int pass, int flag=0){
//         // flag == 0 for 0, or, for 1
//     int gid = threadIdx.x + blockDim.x * blockIdx.x;
//     if(gid >= N)
//         return;
//     unsigned int one = 1;
//     int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;
//     // printf("Here %d %d %d\n", bin, d_input[gid],flag);
//     if(bin) {
//         int newloc=d_digitloc[gid]+d_bins[flag];
//         // printf("Here %d %d %d\n", bin, d_input[gid],flag);
//         // printf("Move  %d %d %d %d\n",newloc,d_input[gid], d_bins[flag], flag2);
//         // std::cout<<"Move "<< newloc<<d_input[gid]<<d_bins[flag]<<std::endl;
//         d_output[newloc]=d_input[gid];
//     }
// }

__global__  // Move both values and positions
void move(unsigned int *d_output, unsigned int *d_output_pos,unsigned const int *d_input, 
            unsigned const int* d_input_pos, unsigned const int *d_digitloc,unsigned const int *d_bins,
            const int N, const int pass, int flag=0){
        // flag == 0 for 0, or, for 1
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if(gid >= N)
        return;
    unsigned int one = 1;
    int bin = ((d_input[gid] & (one<<pass)) == (one<<pass)) ? flag : 1-flag;
    // printf("Here %d %d %d\n", bin, d_input[gid],flag);
    if(bin) {
        int newloc=d_digitloc[gid]+d_bins[flag];
        d_output[newloc]=d_input[gid];
        d_output_pos[newloc]=d_input_pos[gid];
    }
}

__global__
void print_digit(unsigned int* d_digitloc, int n){
    printf("Here: ");
    for(int i=0;i<n;++i){
        printf("%d ",d_digitloc[i]);
    }
    printf("\n");
}

__global__
void reset_bins(unsigned int *bin){
    bin[0]=0;
    bin[1]=0;
}


void radix_sort(unsigned const int * d_input_const,  unsigned const int * d_input_pos_const, 
                unsigned int * d_out, unsigned int * d_out_pos, const int N){
    unsigned int * d_digitloc;//,* h_digitloc;
    int sizeN= N * sizeof(unsigned int);

    // h_digitloc=(unsigned int *) malloc(sizeN);
    cudaMalloc(&d_digitloc, sizeN);

    unsigned int * d_input;
    cudaMalloc(&d_input,sizeN);
    cudaMemcpy(d_input,d_input_const,sizeN,cudaMemcpyDeviceToDevice);

    unsigned int* d_input_pos;
    cudaMalloc(&d_input_pos,sizeN);
    cudaMemcpy(d_input_pos,d_input_pos_const,sizeN,cudaMemcpyDeviceToDevice);

    unsigned int * d_bins;
    int Nblock, Nthread;

    cudaMalloc(&d_bins, 2*sizeof(unsigned int));
    unsigned int h_bins[2]={0};
    cudaMemcpy(d_bins,h_bins,2*sizeof(unsigned int),cudaMemcpyHostToDevice);


    for (int pass=0;pass<32;pass++){
        Nthread=1024;
        Nblock=(N+Nthread-1)/Nthread;



        histogram_kernel<<<Nblock,Nthread>>>(pass, d_bins, d_input, N);

        scan_large(d_bins, 2);


        Nthread=1024;
        Nblock=(N+Nthread-1)/Nthread;

        digit_identify<<<Nthread,Nblock>>>(d_input,  d_digitloc, N, pass, 0);
        scan_large(d_digitloc, N);
        move<<<Nthread,Nblock>>>(d_out,d_out_pos,d_input,d_input_pos, d_digitloc,d_bins,N,pass,0);


        digit_identify<<<Nthread,Nblock>>>(d_input,  d_digitloc, N, pass, 1);
        scan_large(d_digitloc, N);
        move<<<Nthread,Nblock>>>(d_out,d_out_pos,d_input,d_input_pos, d_digitloc,d_bins,N,pass,1);

        cudaMemcpy(d_input,d_out,sizeN,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_input_pos,d_out_pos,sizeN,cudaMemcpyDeviceToDevice);

    }

    cudaFree(d_digitloc);  cudaFree(d_bins);
    cudaFree(d_input); cudaFree(d_input_pos);
    // free(h_digitloc);
}

int main(){
    const int N=3000;
    unsigned int * h_inputVals, *h_inputPos;
    unsigned int * h_outputVals, *h_outputPos;
    unsigned int * d_inputVals, *d_inputPos;
    unsigned int * d_outputVals, * d_outputPos;

    int sizeN = N*sizeof(unsigned int);

    h_inputVals=(unsigned int *) malloc(sizeN);
    h_inputPos=(unsigned int *) malloc(sizeN);
    h_outputVals=(unsigned int *) malloc(sizeN);
    h_outputPos=(unsigned int *) malloc(sizeN);



// Initialize
    for (unsigned int i=0;i<N; ++i){
        h_inputPos[i]=i;
        // h_inputVals[i]=rand();
        h_inputVals[i]=N-i;
    }
    JL::print_array1d(h_inputVals, N);
    // JL::print_array1d(h_inputPos, N);

    cudaMalloc(&d_inputVals, sizeN);
    cudaMalloc(&d_inputPos, sizeN);
    cudaMalloc(&d_outputVals, sizeN);
    cudaMalloc(&d_outputPos, sizeN);


    cudaMemcpy(d_inputVals,h_inputVals,sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputPos,h_inputPos,sizeN, cudaMemcpyHostToDevice);

    radix_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, N);

    cudaMemcpy(h_outputVals,d_outputVals,sizeN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputPos,d_outputPos,sizeN, cudaMemcpyDeviceToHost);

    

    JL::print_array1d(h_outputVals,N);
    // JL::print_array1d(h_outputPos, N);



    cudaFree(d_inputVals); cudaFree(d_inputPos);
    cudaFree(d_outputVals); cudaFree(d_outputPos);
    free(h_inputVals); free(h_inputPos);
    free(h_outputVals); free(h_outputPos);
}
#include <stdio.h>

// Reduce
__global__ 
void reduce_kernel(float * d_out, const float * d_in, int n, int op)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    if(myId>=n)
      return;

    sdata[tid] = d_in[myId];
    
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if ((myId+s<n) && (tid < s) )
        {
          if (op==0){
        sdata[tid]=min(sdata[tid], sdata[tid+s]);
          }else if (op==1){
        sdata[tid]=max(sdata[tid], sdata[tid+s]);
          }else{
            sdata[tid] += sdata[tid + s];
          }           
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void print_vals2(float *d_min, float * d_max){
  printf("Naive %f %f\n", *d_min, *d_max);
}

__global__
void print_arr(float *arr, int N){
  for(int i=0; i<N; ++i){
    printf("%d %f ",i, arr[i]);
  }
  printf("\n");
}

__host__ __device__
unsigned int round2power(unsigned int v){
  // unsigned int v; // compute the next highest power of 2 of 32-bit v
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}


void reduce(const float* const d_in,float & h_reduce, int h_op, const size_t N){

  unsigned int Nblock, Nthread;
  float *d_reduce;
  cudaMalloc(&d_reduce,sizeof(float));
  // int h_op=0;  // 0 Min, 1 Max, else Sum
  Nblock=1;
  Nthread=round2power(N); // need to be power of 2

  cudaMemcpy(d_reduce,&h_reduce,sizeof(float),cudaMemcpyHostToDevice);
  // cudaMemcpy(d_maxlum,&h_maxlum,sizeof(float),cudaMemcpyHostToDevice);
  reduce_kernel<<<Nblock, Nthread, Nthread * sizeof(float)>>>(d_reduce,d_in,N,h_op);
  cudaMemcpy(&h_reduce, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_reduce);

}

// __global__
// void search_minmax(const float* const d_lumin, float *d_min, float *d_max, const int N){
//   for (int i=0;i<N;i++){
//     // if (*d_min>d_lumin[i]) *d_min=d_lumin[i];
//     // if (*d_max<d_lumin[i]) *d_max=d_lumin[i];
//     *d_min = min(*d_min, d_lumin[i]);
//     *d_max = max(*d_max, d_lumin[i]);
//   }
// }


// void naive_minmax(const float* const d_lumin, float &h_min, float &h_max, const int N){

//   float * d_min, * d_max;
//   cudaMalloc(&d_min, sizeof(float));
//   cudaMalloc(&d_max, sizeof(float));

//   cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice);

//   search_minmax<<<1,1>>>(d_lumin, d_min, d_max, N);

//   // print_vals2<<<1,1>>>(d_min, d_max);

//   cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
//   cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);

//   cudaFree(d_min); cudaFree(d_max);


// }


int main(){

  const int N=300; // Need to be smaller than 1024
  int sizeN=N*sizeof(float);
  // int h_op;

  float * h_input = new float[N];
  float *d_input;
  float h_min= 99999.0f, h_max= -99999.0f;

  for(int i=0;i<N;++i){
    h_input[i]=(rand())%N/10.0f;//-50.0f;
  }

  cudaMalloc(&d_input, sizeN);

  cudaMemcpy(d_input,h_input,sizeN,cudaMemcpyHostToDevice);

  // naive_minmax(d_input, h_min, h_max, N);
  // printf("%f %f \n", h_min, h_max);


  reduce(d_input, h_min, 0,N);
  reduce(d_input, h_max, 1,N);
  printf("%f %f \n", h_min, h_max);

  cudaFree(d_input);
  delete [] h_input;

}


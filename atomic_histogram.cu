#include <iostream>
#include "gputimer.h"
using namespace std;

template<class T>
void random_init(T *a, int N){
    for (int i=0;i<N; ++i){
        a[i]=rand();
    }
}

template<class T>
void uniform_init(T *a, T value, int N){
    for (int i=0;i<N; ++i){
        a[i]=value;
    }
}
__global__
void hist_naive(char * letters, int * hist, int n){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        int id_h=letters[id]-65;
        hist[id_h] +=1;
    }
}

__global__
void hist_atomic(char * letters, int * hist, int n){
    int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        int id_h=letters[id]-65;
        // hist[id_h] +=1;
        atomicAdd(& hist[id_h], 1); 
    }
}

int main(){
    GpuTimer timer;
    const int N=100000;
    const int NL=26;

    char *h_letters,*d_letters;
    int *h_hist, *d_hist;
    int sizeN=N*sizeof(char);
    int sizeNL=NL*sizeof(int);

    h_hist=new int[NL];
    uniform_init(h_hist,0,NL);
    for (int i=0;i<NL;++i){
        cout<<h_hist[i]<<' ';
    }
    printf("\n");

    h_letters = new char[N];
    for (int i=0;i<N;++i){
        h_letters[i]=rand()%26+65;
    }

    cudaMalloc(&d_hist,sizeNL);
    cudaMalloc(&d_letters,sizeN);

    cudaMemcpy(d_hist,h_hist,sizeNL,cudaMemcpyHostToDevice);
    cudaMemcpy(d_letters,h_letters,sizeN,cudaMemcpyHostToDevice);

    int Nthread=512;
    int Nblock=(N+Nthread-1)/Nthread;

    timer.Start();
    // hist_naive<<<Nblock,Nthread>>>(d_letters,d_hist,N);
    /* Output: (Wrong histogram)
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    Time elapsed = 0.091616 ms
    14 13 14 14 14 13 15 13 13 15 13 13 14 13 14 14 15 13 16 16 15 14 15 14 13 14 
    */
    hist_atomic<<<Nblock,Nthread>>>(d_letters,d_hist,N);
    /* Output:
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    Time elapsed = 0.065312 ms
    3882 3823 3829 3780 3713 3908 3795 3796 3850 3915 3861 3823 3877 3816 3863 3911 3961 3886 3865 3864 3772 3900 3869 3806 3718 3917 
    */
    timer.Stop();

    cudaMemcpy(h_hist,d_hist,sizeNL,cudaMemcpyDeviceToHost);

    printf("Time elapsed = %g ms\n", timer.Elapsed());

    for (int i=0;i<NL;++i){
        cout<<h_hist[i]<<' ';
    }
    printf("\n");


    cudaFree(d_hist); cudaFree(d_letters);
    delete[] h_hist; delete[] h_letters;

}
/*
Print Hello World
also print the block id and thread id within the block
*/
#include <stdio.h> 

const int Nthread = 3; 
const int Nblock = 2; 
 
__global__ void hello(void){
    printf("Hello world! block ID %d, thread ID %d\n",blockIdx.x,threadIdx.x);

}

int main() {
    hello<<<Nblock,Nthread>>>();

}